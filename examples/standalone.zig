const std = @import("std");
const builtin = @import("builtin");
const dvui = @import("dvui");
const vk = @import("vulkan");

const DvuiVkBackend = dvui.backend;
const win32 = if (builtin.target.os.tag == .windows) DvuiVkBackend.win32 else void;
const win = if (builtin.target.os.tag == .windows) DvuiVkBackend.win else void;
const vk_dll = DvuiVkBackend.vk_dll;
const SyncObjects = DvuiVkBackend.SyncObjects;

pub const AppState = struct {
    backend: *DvuiVkBackend.VkBackend,
    render_pass: vk.RenderPass,
    sync: SyncObjects,
    command_buffers: []vk.CommandBuffer,
};
pub var g_app_state: AppState = undefined;

pub const max_frames_in_flight = 2;
pub const vsync = true;

pub fn main() !void {
    dvui.Backend.Common.windowsAttachConsole() catch {};

    var gpa_instance = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = gpa_instance.allocator();
    defer _ = gpa_instance.deinit();

    const window_class = win32.L("DvuiWindow");
    win.RegisterClass(window_class, .{}) catch win32.panicWin32(
        "RegisterClass",
        win32.GetLastError(),
    );

    vk_dll.init() catch |err| std.debug.panic("Failed to init Vulkan: {}", .{err});
    defer vk_dll.deinit();
    const loader = vk_dll.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse @panic("Failed to lookup Vulkan VkGetInstanceProcAddr: {}");

    var b = DvuiVkBackend.VkBackend.init(gpa, undefined);
    defer b.deinit();

    // init backend (creates and owns OS window)
    var window_context: *DvuiVkBackend.Context = try b.allocContext();
    window_context.* = .{
        .backend = &b,
        .dvui_window = try dvui.Window.init(@src(), gpa, DvuiVkBackend.dvuiBackend(window_context), .{}),
        .hwnd = undefined,
    };
    try win.initWindow(window_context, .{
        .registered_class = window_class,
        .dvui_gpa = gpa,
        .gpa = gpa,
        .title = "My window",
    });

    b.vkc = try DvuiVkBackend.VkContext.init(gpa, loader, window_context);

    window_context.swapchain_state = try DvuiVkBackend.Context.SwapchainState.init(window_context, .{
        .graphics_queue_index = b.vkc.physical_device.graphics_queue_index,
        .present_queue_index = b.vkc.physical_device.present_queue_index,
        .desired_extent = vk.Extent2D{ .width = @intFromFloat(window_context.last_pixel_size.w), .height = @intFromFloat(window_context.last_pixel_size.h) },
        .desired_formats = &.{
            // NOTE: all dvui examples as far as I can tell expect all color transformations to happen directly in srgb space, so we request unorm not srgb backend. To support linear rendering this will be an issue.
            // TODO: add support for both linear and srgb render targets
            // similar issue: https://github.com/ocornut/imgui/issues/578
            .{ .format = .a2b10g10r10_unorm_pack32, .color_space = .srgb_nonlinear_khr },
            .{ .format = .b8g8r8a8_unorm, .color_space = .srgb_nonlinear_khr },
        },
        .desired_present_modes = if (!vsync) &.{.immediate_khr} else &.{.fifo_khr},
    });

    const render_pass = try DvuiVkBackend.createRenderPass(b.vkc.device, window_context.swapchain_state.?.swapchain.image_format);
    defer b.vkc.device.destroyRenderPass(render_pass, null);

    const sync = try DvuiVkBackend.SyncObjects.init(b.vkc.device);
    defer sync.deinit(b.vkc.device);

    const command_buffers = try DvuiVkBackend.createCommandBuffers(gpa, b.vkc.device, b.vkc.cmd_pool, max_frames_in_flight);
    defer gpa.free(command_buffers);

    g_app_state = .{
        .backend = &b,
        .command_buffers = command_buffers,
        .render_pass = render_pass,
        .sync = sync,
    };

    b.renderer = try DvuiVkBackend.VkRenderer.init(b.gpa, .{
        .dev = b.vkc.device,
        .comamnd_pool = b.vkc.cmd_pool,
        .queue = b.vkc.graphics_queue.handle,
        .pdev = b.vkc.physical_device.handle,
        .mem_props = b.vkc.physical_device.memory_properties,
        .render_pass = render_pass,
        .max_frames_in_flight = max_frames_in_flight,
    });

    var current_frame_in_flight: u32 = 0;
    defer b.vkc.device.queueWaitIdle(b.vkc.graphics_queue.handle) catch {}; // let gpu finish its work on exit, otherwise we will get validation errors
    main_loop: while (b.contexts.items.len > 0) {
        defer current_frame_in_flight = (current_frame_in_flight + 1) % max_frames_in_flight;
        switch (win.serviceMessageQueue()) {
            .queue_empty => {
                for (b.contexts.items) |ctx| {
                    try paint(g_app_state, ctx, current_frame_in_flight);
                    b.prev_frame_stats = b.renderer.?.stats;
                    if (ctx.received_close) {
                        _ = win32.PostMessageA(ctx.hwnd, win32.WM_CLOSE, 0, 0);
                        continue;
                    }
                }
            },
            .quit => break :main_loop,
        }
    }
}

pub fn drawFrame(ctx: *DvuiVkBackend.Context) void {
    {
        // _ = dvui.windowHeader("settings", "", null);
        const m = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .background = true, .gravity_y = 0 });
        defer m.deinit();
        _ = dvui.checkbox(@src(), &dvui.Examples.show_demo_window, "Show dvui demo", .{});
    }

    {
        const stats_box = dvui.box(@src(), .{ .dir = .vertical }, .{ .gravity_x = 1, .background = true });
        ctx.drawStats();
        stats_box.deinit();
    }

    dvui.Examples.demo();
}

pub fn paint(app_state: AppState, ctx: *DvuiVkBackend.Context, current_frame_in_flight: usize) !void {
    const b = ctx.backend;
    const gpa = b.gpa;
    const render_pass = app_state.render_pass;
    const sync = app_state.sync;

    if (ctx.last_pixel_size.w < 1 or ctx.last_pixel_size.h < 1) return;

    { // check/wait for previous frame to finish
        const result = try b.vkc.device.waitForFences(1, @ptrCast(&sync.in_flight_fences[current_frame_in_flight]), vk.TRUE, std.math.maxInt(u64));
        std.debug.assert(result == .success);
    }

    const image_index = blk: while (true) {
        if (ctx.swapchain_state.?.framebuffers.len == 0) {
            ctx.swapchain_state.?.framebuffers = try DvuiVkBackend.createFramebuffers(
                gpa,
                b.vkc.device,
                ctx.swapchain_state.?.swapchain.extent,
                ctx.swapchain_state.?.swapchain.image_count,
                ctx.swapchain_state.?.image_views,
                render_pass,
            );
        }
        const next_image_result = b.vkc.device.acquireNextImageKHR(
            ctx.swapchain_state.?.swapchain.handle,
            std.math.maxInt(u64),
            sync.image_available_semaphores[current_frame_in_flight],
            .null_handle,
        ) catch |err| {
            if (err == error.OutOfDateKHR) {
                try ctx.swapchain_state.?.recreate(ctx);
                continue; // need framebuffer
            }
            return err;
        };
        switch (next_image_result.result) {
            .success => {},
            .suboptimal_khr => {
                try ctx.swapchain_state.?.recreate(ctx);
                continue; // need framebuffer
            },
            else => |err| std.debug.panic("Failed to acquire next frame: {}", .{err}),
        }
        break :blk next_image_result.image_index;
    };

    const command_buffer = app_state.command_buffers[current_frame_in_flight];
    const framebuffer = ctx.swapchain_state.?.framebuffers[image_index];
    try b.vkc.device.beginCommandBuffer(command_buffer, &.{ .flags = .{} });
    const cmd = vk.CommandBufferProxy.init(command_buffer, b.vkc.device.wrapper);

    const clear_values = [_]vk.ClearValue{
        .{ .color = .{ .float_32 = .{ 0.1, 0.1, 0.1, 1 } } },
    };
    const extent = vk.Extent2D{ .width = @intFromFloat(ctx.last_pixel_size.w), .height = @intFromFloat(ctx.last_pixel_size.h) };
    const render_pass_begin_info = vk.RenderPassBeginInfo{
        .render_pass = render_pass,
        .framebuffer = framebuffer,
        .render_area = .{
            .offset = .{ .x = 0, .y = 0 },
            .extent = extent,
        },
        .clear_value_count = clear_values.len,
        .p_clear_values = &clear_values,
    };

    cmd.beginRenderPass(&render_pass_begin_info, .@"inline");

    b.renderer.?.beginFrame(cmd.handle, extent);
    // defer _ = b.renderer.?.endFrame();

    // beginWait coordinates with waitTime below to run frames only when needed
    const nstime = ctx.dvui_window.beginWait(false);

    // marks the beginning of a frame for dvui, can call dvui functions after thisz
    try ctx.dvui_window.begin(nstime);

    drawFrame(ctx);

    // marks end of dvui frame, don't call dvui functions after this
    // - sends all dvui stuff to backend for rendering, must be called before renderPresent()
    _ = try ctx.dvui_window.end(.{});

    // cursor management
    // TODO: reenable
    // b.setCursor(win.cursorRequested());

    const frame_sync_objects = DvuiVkBackend.FrameSyncObjects{
        .image_available_semaphore = sync.image_available_semaphores[current_frame_in_flight],
        .render_finished_semaphore = sync.render_finished_semaphores[current_frame_in_flight],
        .in_flight_fence = sync.in_flight_fences[current_frame_in_flight],
    };
    cmd.endRenderPass();
    cmd.endCommandBuffer() catch |err| std.debug.panic("Failed to end vulkan cmd buffer: {}", .{err});

    if (!try DvuiVkBackend.present(
        ctx,
        app_state.command_buffers[current_frame_in_flight],
        frame_sync_objects,
        ctx.swapchain_state.?.swapchain.handle,
        image_index,
    )) {
        // should_recreate_swapchain = true;
    }
}
