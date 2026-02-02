const std = @import("std");
const builtin = @import("builtin");
const dvui = @import("dvui");
const vk = @import("vulkan");

const DvuiVkBackend = dvui.backend;
const win32 = if (builtin.target.os.tag == .windows) DvuiVkBackend.win32 else void;
const win = if (builtin.target.os.tag == .windows) DvuiVkBackend.win else void;
const vk_dll = DvuiVkBackend.vk_dll;
const slog = std.log.scoped(.main);
const FrameSync = DvuiVkBackend.FrameSync;

pub const AppState = struct {
    backend: *DvuiVkBackend.VkBackend,
    render_pass: vk.RenderPass,
    sync: FrameSync,
    command_buffers: []vk.CommandBuffer,

    pub fn init(gpa: std.mem.Allocator) !AppState {
        vk_dll.init() catch |err| std.debug.panic("Failed to init Vulkan: {}", .{err});
        errdefer vk_dll.deinit();
        const loader = vk_dll.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse @panic("Failed to lookup Vulkan VkGetInstanceProcAddr: {}");

        // init backend (creates and owns OS window)
        const iw = try DvuiVkBackend.initWindow(loader, .{
            .title = "standalone",
            .dvui_gpa = gpa,
            .gpa = gpa,
            .max_frames_in_flight = max_frames_in_flight,
            .vsync = vsync,
        });
        const b = iw.backend;
        const window_context = iw.window;

        const render_pass = try DvuiVkBackend.createRenderPass(b.vkc.device, window_context.swapchain_state.?.swapchain.image_format);
        errdefer b.vkc.device.destroyRenderPass(render_pass, b.vkc.alloc);

        const sync = try FrameSync.init(gpa, max_frames_in_flight, b.vkc);
        errdefer sync.deinit(gpa, b.vkc.device);

        const command_buffers = try DvuiVkBackend.createCommandBuffers(gpa, b.vkc.device, b.vkc.cmd_pool, max_frames_in_flight);
        errdefer gpa.free(command_buffers);

        b.renderer = try DvuiVkBackend.VkRenderer.init(b.gpa, .{
            .dev = b.vkc.device,
            .comamnd_pool = b.vkc.cmd_pool,
            .queue = b.vkc.graphics_queue.handle,
            .pdev = b.vkc.physical_device.handle,
            .memory = DvuiVkBackend.VkRenderer.VkMemory.init(b.vkc.physical_device.memory_properties) orelse @panic("invalid vulkan memory"),
            .render_pass = render_pass,
            .max_frames_in_flight = max_frames_in_flight,
        });

        return .{
            .backend = b,
            .command_buffers = command_buffers,
            .render_pass = render_pass,
            .sync = sync,
        };
    }

    pub fn deinit(self: AppState, gpa: std.mem.Allocator) void {
        defer vk_dll.deinit();
        defer gpa.destroy(self.backend);
        defer self.backend.deinit();
        defer self.backend.vkc.device.destroyRenderPass(self.render_pass, self.backend.vkc.alloc);
        defer self.sync.deinit(gpa, self.backend.vkc.device);
        defer gpa.free(self.command_buffers);
    }
};
pub var g_app_state: AppState = undefined;

pub const max_frames_in_flight = 2;
pub const vsync = false;

pub fn main() !void {
    if (builtin.target.os.tag == .windows) dvui.Backend.Common.windowsAttachConsole() catch {};
    dvui.Examples.show_demo_window = true;

    var gpa_instance = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = gpa_instance.allocator();
    defer _ = gpa_instance.deinit();

    g_app_state = try AppState.init(gpa);
    defer g_app_state.deinit(gpa);
    defer g_app_state.backend.vkc.device.queueWaitIdle(g_app_state.backend.vkc.graphics_queue.handle) catch {}; // let gpu finish its work on exit, otherwise we will get validation errors

    main_loop: while (g_app_state.backend.contexts.items.len > 0) {
        // slog.info("frame: {}", .{current_frame_in_flight});
        switch (win.serviceMessageQueue()) {
            .queue_empty => {
                for (g_app_state.backend.contexts.items) |ctx| {
                    try paint(&g_app_state, ctx);
                    g_app_state.backend.prev_frame_stats = g_app_state.backend.renderer.?.stats;
                    if (ctx.received_close) {
                        _ = win32.PostMessageA(@ptrCast(ctx.hwnd), win32.WM_CLOSE, 0, 0);
                        continue;
                    }
                }
            },
            .quit => break :main_loop,
        }
    }
}

pub fn drawGUI(ctx: *DvuiVkBackend.WindowContext) void {
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

pub fn paint(app_state: *AppState, ctx: *DvuiVkBackend.WindowContext) !void {
    const b = ctx.backend;
    const gpa = b.gpa;
    const render_pass = app_state.render_pass;
    const sync = &app_state.sync;
    const device = b.vkc.device;

    if (ctx.last_pixel_size.w < 1 or ctx.last_pixel_size.h < 1) return;

    try sync.begin(device);
    defer sync.end();
    const image_index = try ctx.swapchain_state.?.acquireImageMaybeRecreate(gpa, ctx, sync.*, &.{}, render_pass);

    const command_buffer = app_state.command_buffers[sync.current_frame];
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

    drawGUI(ctx);

    // marks end of dvui frame, don't call dvui functions after this
    // - sends all dvui stuff to backend for rendering, must be called before renderPresent()
    _ = try ctx.dvui_window.end(.{});

    // cursor management
    // TODO: reenable
    // b.setCursor(win.cursorRequested());

    cmd.endRenderPass();
    cmd.endCommandBuffer() catch |err| std.debug.panic("Failed to end vulkan cmd buffer: {}", .{err});

    if (!try DvuiVkBackend.present(
        ctx,
        command_buffer,
        sync.items[sync.current_frame],
        ctx.swapchain_state.?.swapchain.handle,
        image_index,
    )) {
        // should_recreate_swapchain = true;
    }
}
