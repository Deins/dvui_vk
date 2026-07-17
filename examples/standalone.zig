const std = @import("std");
const builtin = @import("builtin");
const dvui = @import("dvui");
const vk = @import("vulkan");

const DvuiVkBackend = dvui.backend;
const win = if (@hasDecl(DvuiVkBackend, "win")) DvuiVkBackend.win else void;
const vk_dll = DvuiVkBackend.vk_dll;

pub const AppState = struct {
    backend: *DvuiVkBackend.VkBackend,

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

        const color_format: vk.Format = @enumFromInt(window_context.render_target.?.colorFormat());

        b.renderer = try DvuiVkBackend.VkRenderer.init(b.gpa, .{
            .dev = b.vkc.device,
            .comamnd_pool = b.vkc.cmd_pool,
            .queue = b.vkc.graphics_queue.handle,
            .pdev = b.vkc.physical_device.handle,
            .memory = DvuiVkBackend.VkRenderer.VkMemory.init(b.vkc.physical_device.memory_properties) orelse @panic("invalid vulkan memory"),
            .render_pass = .{ .dynamic = .{
                .view_mask = 0,
                .color_attachment_count = 1,
                .p_color_attachment_formats = &.{color_format},
                .depth_attachment_format = .undefined,
                .stencil_attachment_format = .undefined,
            } },
            .max_frames_in_flight = max_frames_in_flight,
        });

        return .{ .backend = b };
    }

    pub fn deinit(self: AppState, gpa: std.mem.Allocator) void {
        defer vk_dll.deinit();
        defer gpa.destroy(self.backend);
        defer self.backend.deinit();
    }
};
pub var g_app_state: AppState = undefined;

pub const max_frames_in_flight = 2;
pub const vsync = false;

pub fn main() !void {
    if (builtin.target.os.tag == .windows) dvui.Backend.Common.windowsAttachConsole() catch {};
    dvui.Examples.show_demo_window = true;

    var gpa_instance = std.heap.DebugAllocator(.{}){};
    const gpa = gpa_instance.allocator();
    defer _ = gpa_instance.deinit();

    g_app_state = try AppState.init(gpa);
    defer g_app_state.deinit(gpa);
    defer g_app_state.backend.vkc.device.queueWaitIdle(g_app_state.backend.vkc.graphics_queue.handle) catch {}; // let gpu finish its work on exit, otherwise we will get validation errors

    main_loop: while (g_app_state.backend.contexts.items.len > 0) {
        // slog.info("frame: {}", .{current_frame_in_flight});
        switch (win.serviceMessageQueue()) {
            .queue_empty => {
                var i: usize = 0;
                while (i < g_app_state.backend.contexts.items.len) {
                    const ctx = g_app_state.backend.contexts.items[i];
                    try paint(&g_app_state, ctx);
                    g_app_state.backend.prev_frame_stats = g_app_state.backend.renderer.?.stats;
                    if (ctx.received_close) {
                        g_app_state.backend.destroyContext(ctx);
                        continue;
                    }
                    i += 1;
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

    dvui.Examples.demo(.full);
}

pub fn paint(app_state: *AppState, ctx: *DvuiVkBackend.WindowContext) !void {
    const b = ctx.backend;
    const device = b.vkc.device;
    _ = app_state;

    if (ctx.last_pixel_size.w < 1 or ctx.last_pixel_size.h < 1) return;

    var frame = ctx.render_target.?.acquire() catch |err| switch (err) {
        error.FrameSkipped, error.FrameOutOfDate => return,
        else => return err,
    };
    defer frame.abort();
    const command_buffer = frame.commandBuffer(vk.CommandBuffer);
    const cmd = vk.CommandBufferProxy.init(command_buffer, device.wrapper);
    const extent: vk.Extent2D = .{ .width = frame.extent.width, .height = frame.extent.height };
    cmd.beginRendering(&.{
        .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = extent },
        .view_mask = 0,
        .layer_count = 1,
        .color_attachment_count = 1,
        .p_color_attachments = &.{.{
            .image_view = frame.imageView(vk.ImageView),
            .image_layout = .color_attachment_optimal,
            .resolve_mode = .{},
            .resolve_image_view = .null_handle,
            .resolve_image_layout = .undefined,
            .load_op = .clear,
            .store_op = .store,
            .clear_value = .{ .color = .{ .float_32 = .{ 0.1, 0.1, 0.1, 1 } } },
        }},
    });

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

    cmd.endRendering();
    try frame.submitAndPresent(.{});
}
