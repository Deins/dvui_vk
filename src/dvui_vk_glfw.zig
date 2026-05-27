const std = @import("std");
const builtin = @import("builtin");
pub const dvui = @import("dvui");
pub const glfw = @import("zglfw");
pub const kind: dvui.enums.Backend = .custom;
const slog = std.log.scoped(.dvu_vk_glfw);
comptime {
    _ = @import("dvui_c.zig");
}

pub const VkRenderer = @import("dvui_vk_renderer.zig");

pub const InitOptions = struct {
    dvui_gpa: std.mem.Allocator,
    /// The allocator used for temporary allocations used during init()
    gpa: std.mem.Allocator,
    /// The initial size of the application window
    size: ?dvui.Size = null,

    /// The application title to display
    title: [:0]const u8,
    /// content of a PNG image (or any other format stb_image can load)
    /// tip: use @embedFile
    icon: ?[]const u8 = null,
};

pub const vk = @import("vk");
pub const dvui_vk_common = @import("dvui_vk_common.zig");
pub const VkContext = dvui_vk_common.VkContext;
pub const VkBackend = dvui_vk_common.VkBackend;
pub const WindowContext = dvui_vk_common.WindowContext;
pub const createRenderPass = dvui_vk_common.createRenderPass;
pub const createFramebuffers = dvui_vk_common.createFramebuffers;
pub const destroyFramebuffers = dvui_vk_common.destroyFramebuffers;
pub const FrameSync = dvui_vk_common.FrameSync;
pub const createCommandBuffers = dvui_vk_common.createCommandBuffers;
pub const present = dvui_vk_common.present;
pub const max_frames_in_flight = 3;

pub const GenericError = dvui.Backend.GenericError;
pub const TextureError = dvui.Backend.TextureError;

const is_windows = @import("builtin").target.os.tag == .windows;

pub fn dvuiBackend(context: *WindowContext) dvui.Backend {
    return dvui.Backend.init(@ptrCast(@alignCast(context)));
}

/// to support multiple windows @This pointer is used as per dvui.window context (*dvui_vk_common.WindowContext)
pub const ContextHandle = *@This();
comptime {
    if (@sizeOf(@This()) != 0) unreachable;
}
/// get real context from handle
pub inline fn get(ch: ContextHandle) *WindowContext {
    return @as(*WindowContext, @ptrCast(@alignCast(ch)));
}
// shortcuts
pub inline fn backend(ch: ContextHandle) *VkBackend {
    return ch.get().backend;
}
pub inline fn renderer(ch: ContextHandle) *VkRenderer {
    return &ch.backend().renderer.?;
}

fn glfwWin(context: *WindowContext) *glfw.Window {
    return @ptrCast(@alignCast(context.glfw_win.?));
}

pub fn createVkSurfaceGLFW(self: *WindowContext, vk_instance: vk.InstanceProxy) bool {
    const vk_alloc = null; //  @ptrCast(self.backend.vkc.alloc)
    glfw.createWindowSurface(@ptrFromInt(@intFromEnum(vk_instance.handle)), glfwWin(self), vk_alloc, @ptrCast(&self.surface)) catch return false;
    return true;
}
pub const createVkSurface = createVkSurfaceGLFW;

//
//   Dvui backend implementation
//

/// Get monotonic nanosecond timestamp. Doesn't have to be system time.
pub fn nanoTime(_: ContextHandle) i128 {
    return @intFromFloat(glfw.getTime() * std.time.ns_per_s);
}

pub fn sleep(_: ContextHandle, ns: u64) void {
    glfw.waitEventsTimeout(@as(f64, @floatFromInt(ns)) / std.time.ns_per_s);
}

/// Called by dvui during `dvui.Window.begin`, so prior to any dvui
/// rendering.  Use to setup anything needed for this frame.  The arena
/// arg is cleared before `dvui.Window.begin` is called next, useful for any
/// temporary allocations needed only for this frame.
pub fn begin(context_handle: ContextHandle, arena_: std.mem.Allocator) GenericError!void {
    get(context_handle).arena = arena_;
    context_handle.renderer().begin(arena_, context_handle.pixelSize());
}

/// Called during `dvui.Window.end` before freeing any memory for the current frame.
pub fn end(context_handle: ContextHandle) GenericError!void {
    context_handle.renderer().end();
    get(context_handle).arena = undefined;
}

/// Return size of the window in physical pixels.  For a 300x200 retina
/// window (so actually 600x400), this should return 600x400.
pub fn pixelSize(context_handle: ContextHandle) dvui.Size.Physical {
    return context_handle.get().last_pixel_size;
}

/// Return size of the window in logical pixels.  For a 300x200 retina
/// window (so actually 600x400), this should return 300x200.
pub fn windowSize(context_handle: ContextHandle) dvui.Size.Natural {
    return context_handle.get().last_window_size;
}

/// Return the detected additional scaling.  This represents the user's
/// additional display scaling (usually set in their window system's
/// settings).  Currently only called during `dvui.Window.init`, so currently
/// this sets the initial content scale.
pub fn contentScale(self: ContextHandle) f32 {
    _ = self; // autofix
    return 1.0;
}

/// Render a triangle list using the idx indexes into the vtx vertexes
/// clipped to to `clipr` (if given).  Vertex positions and `clipr` are in
/// physical pixels.  If `texture` is given, the vertexes uv coords are
/// normalized (0-1). `clipr` (if given) has whole pixel values.
pub fn drawClippedTriangles(ch: ContextHandle, texture: ?dvui.Texture, vtx: []const dvui.Vertex, idx: []const u16, clipr: ?dvui.Rect.Physical) GenericError!void {
    return ch.renderer().drawClippedTriangles(texture, vtx, idx, clipr);
}

/// Create a `dvui.Texture` from premultiplied alpha `pixels` in RGBA.  The
/// returned pointer is what will later be passed to `drawClippedTriangles`.
pub fn textureCreate(ch: ContextHandle, pixels: [*]const u8, width: u32, height: u32, interpolation: dvui.enums.TextureInterpolation, _: dvui.enums.TexturePixelFormat) TextureError!dvui.Texture {
    return ch.renderer().textureCreate(pixels, width, height, interpolation);
}

/// Update a `dvui.Texture` from premultiplied alpha `pixels` in RGBA.  The
/// passed in texture must be created  with textureCreate
// pub fn textureUpdate(self: ContextHandle, texture: dvui.Texture, pixels: [*]const u8) TextureError!void {
//     // we can handle backends that dont support textureUpdate by using destroy and create again!
//     if (comptime !@hasDecl(Implementation, "textureUpdate")) return TextureError.NotImplemented else {
//         return self.base_backend.textureUpdate(texture, pixels);
//     }
// }

/// Destroy `texture` made with `textureCreate`. After this call, this texture
/// pointer will not be used by dvui.
pub fn textureDestroy(ch: ContextHandle, texture: dvui.Texture) void {
    ch.renderer().textureDestroy(texture);
}

/// Create a `dvui.Texture` that can be rendered to with `renderTarget`.  The
/// returned pointer is what will later be passed to `drawClippedTriangles`.
pub fn textureCreateTarget(ch: ContextHandle, width: u32, height: u32, interpolation: dvui.enums.TextureInterpolation, _: dvui.enums.TexturePixelFormat) TextureError!dvui.TextureTarget {
    return ch.renderer().textureCreateTarget(width, height, interpolation);
}

/// Read pixel data (RGBA) from `texture` into `pixels_out`.
pub fn textureReadTarget(ch: ContextHandle, texture: dvui.TextureTarget, pixels_out: [*]u8) TextureError!void {
    return ch.renderer().textureReadTarget(texture, pixels_out);
}

/// Clear a texture target without destroying it.
pub fn textureClearTarget(ch: ContextHandle, texture: dvui.Texture.Target) void {
    _ = ch;
    _ = texture;
}

/// Destroy `texture` made with `textureCreateTarget`. After this call, this
/// texture pointer will not be used by dvui.
pub fn textureDestroyTarget(ch: ContextHandle, texture: dvui.Texture.Target) void {
    ch.renderer().textureDestroy(.{ .ptr = texture.ptr, .width = texture.width, .height = texture.height, .format = texture.format });
}

/// Convert texture target made with `textureCreateTarget` into return texture
/// as if made by `textureCreate`.  After this call, texture target will not be
/// used by dvui.
pub fn textureFromTarget(ch: ContextHandle, texture: dvui.TextureTarget) TextureError!dvui.Texture {
    return ch.renderer().textureFromTarget(texture);
}

/// Get a temporary drawable texture from this target.  target is not destroyed.
pub fn textureFromTargetTemp(ch: ContextHandle, target: dvui.TextureTarget) TextureError!dvui.Texture {
    return ch.renderer().textureFromTarget(target);
}

/// Render future `drawClippedTriangles` to the passed `texture` (or screen
/// if null).
pub fn renderTarget(ch: ContextHandle, texture: ?dvui.TextureTarget) GenericError!void {
    return ch.renderer().renderTarget(texture);
}

/// Get clipboard content (text only)
pub fn clipboardText(self: ContextHandle) GenericError![]const u8 {
    return glfwWin(get(self)).getClipboardString() orelse return error.BackendError;
}

/// Set clipboard content (text only)
pub fn clipboardTextSet(self: ContextHandle, text: []const u8) GenericError!void {
    const ctx = get(self);
    if (text.len == 0) return;
    const c_text = try ctx.arena.dupeSentinel(u8, text, 0);
    defer ctx.arena.free(c_text);
    glfwWin(ctx).setClipboardString(c_text);
}

/// Open URL in system browser
pub fn openURL(self: ContextHandle, url: []const u8, new_window: bool) GenericError!void {
    _ = new_window; // autofix
    return dvui_vk_common.openURL(get(self).arena, url);
}

/// Get the preferredColorScheme if available
pub fn preferredColorScheme(self: ContextHandle) ?dvui.enums.ColorScheme {
    _ = self; // autofix
    if (builtin.os.tag == .windows) {
        return dvui.Backend.Common.windowsGetPreferredColorScheme();
    }
    return null;
}

/// Show/hide the cursor.
///
/// Returns the previous state of the cursor, `true` meaning shown
pub fn cursorShow(self: ContextHandle, value: ?bool) GenericError!bool {
    _ = self; // autofix
    _ = value; // autofix
}

/// Called by `dvui.refresh` when it is called from a background
/// thread.  Used to wake up the gui thread.  It only has effect if you
/// are using `dvui.Window.waitTime` or some other method of waiting until
/// a new event comes in.
pub fn refresh(self: ContextHandle) void {
    _ = self; // autofix
}

test {
    @import("std").testing.refAllDecls(@This());
}

pub fn initWindow(context: *WindowContext, options: InitOptions) !void {
    const size = if (options.size) |s| s else dvui.Size{ .w = context.last_pixel_size.w, .h = context.last_pixel_size.h };
    context.last_pixel_size = .{ .w = size.w, .h = size.h };
    context.last_window_size = .{ .w = size.w, .h = size.h };
    glfw.windowHint(.client_api, .no_api);
    context.glfw_win = try glfw.Window.create(@intFromFloat(size.w), @intFromFloat(size.h), options.title, null);
    glfwWin(context).setUserPointer(context);
    _ = glfwWin(context).setSizeCallback(&resizeCB);
}

//
//  APP
//
pub const vkk = @import("vk_kickstart");

pub const AppState = dvui_vk_common.AppState;
pub var g_app_state: AppState = undefined;
pub const paint = dvui_vk_common.paint;

pub fn getInstanceProcAddress(instance: vk.Instance, procname: [*:0]const u8) vk.PfnVoidFunction {
    return @ptrCast(glfw.getInstanceProcAddress(@ptrFromInt(@intFromEnum(instance)), procname));
}

pub fn main() !void {
    if (builtin.target.os.tag == .windows) dvui.Backend.Common.windowsAttachConsole() catch {};

    const app = dvui.App.get() orelse return error.DvuiAppNotDefined;

    var gpa_instance = std.heap.DebugAllocator(.{}){};
    const gpa = gpa_instance.allocator();
    defer _ = gpa_instance.deinit();

    {
        try glfw.init();
        slog.info("GLFW vk_support: {}", .{glfw.isVulkanSupported()});
    }
    defer glfw.terminate();

    const loader = getInstanceProcAddress;
    const init_opts = app.config.get();
    var b = VkBackend.init(gpa, undefined); // the undefined sucks here, see comment about it at assignment
    defer b.deinit();

    // init backend (creates and owns OS window)
    var window_context: *WindowContext = try b.allocContext();
    window_context.* = .{
        .backend = &b,
        .dvui_window = undefined,
        .hwnd = undefined,
    };
    window_context.dvui_window = try dvui.Window.init(@src(), gpa, dvuiBackend(window_context), .{}); // this uses context, thats why called separately from constructor
    initWindow(window_context, .{
        .dvui_gpa = gpa,
        .gpa = gpa,
        .size = init_opts.size,
        .title = init_opts.title,
        .icon = init_opts.icon,
    }) catch |err| {
        slog.err("initWindow failed: {}", .{err});
        return err;
    };
    const window = glfwWin(window_context);
    defer window.destroy();
    // TODO: this sucks, because to select vk.device we need window, it creates this nasty circular partial initialization nastiness.

    b.vkc = VkContext.init(gpa, loader, window_context, &createVkSurfaceGLFW, .{
        .device_select_settings = .{ .required_extensions = &.{
            vk.extensions.khr_swapchain.name,
        } },
    }) catch |err| {
        slog.err("VkContext.init failed: {}", .{err});
        return err;
    };

    window_context.swapchain_state = WindowContext.SwapchainState.init(window_context, .{
        .graphics_queue_index = b.vkc.physical_device.graphics_queue_index,
        .present_queue_index = b.vkc.physical_device.present_queue_index orelse b.vkc.physical_device.graphics_queue_index,
        .desired_min_image_count = max_frames_in_flight,
        .desired_extent = vk.Extent2D{ .width = @intFromFloat(window_context.last_pixel_size.w), .height = @intFromFloat(window_context.last_pixel_size.h) },
        .desired_formats = &.{
            // NOTE: all dvui examples as far as I can tell expect all color transformations to happen directly in srgb space, so we request unorm not srgb backend. To support linear rendering this will be an issue.
            // TODO: add support for both linear and srgb render targets
            // similar issue: https://github.com/ocornut/imgui/issues/578
            .{ .format = .a2b10g10r10_unorm_pack32, .color_space = .srgb_nonlinear_khr },
            .{ .format = .b8g8r8a8_unorm, .color_space = .srgb_nonlinear_khr },
            .{ .format = .r8g8b8a8_unorm, .color_space = .srgb_nonlinear_khr },
        },
        .desired_present_modes = if (!init_opts.vsync) &.{ .immediate_khr, .mailbox_khr } else &.{ .fifo_khr, .mailbox_khr },
    }) catch |err| {
        slog.err("SwapchainState.init failed: {}", .{err});
        return err;
    };

    const render_pass = try createRenderPass(b.vkc.device, window_context.swapchain_state.?.swapchain.image_format);
    defer b.vkc.device.destroyRenderPass(render_pass, null);

    const sync = try FrameSync.init(gpa, max_frames_in_flight, b.vkc);
    defer sync.deinit(gpa, b.vkc.device);

    const command_buffers = try createCommandBuffers(gpa, b.vkc.device, b.vkc.cmd_pool, max_frames_in_flight);
    defer gpa.free(command_buffers);

    g_app_state = .{
        .backend = &b,
        .command_buffers = command_buffers,
        .render_pass = render_pass,
        .sync = sync,
    };

    if (app.initFn) |initFn| {
        // try ctx.dvui_window.begin(ctx.dvui_window.frame_time_ns);
        try initFn(&b.contexts.items[0].dvui_window);
        // _ = try ctx.dvui_window.end(.{});
    }
    defer if (app.deinitFn) |deinitFn| deinitFn();

    b.renderer = try VkRenderer.init(b.gpa, .{
        .dev = b.vkc.device,
        .comamnd_pool = b.vkc.cmd_pool,
        .queue = b.vkc.graphics_queue.handle,
        .pdev = b.vkc.physical_device.handle,
        .memory = VkRenderer.VkMemory.init(b.vkc.physical_device.memory_properties) orelse @panic("invalid vulkan memory"),
        .render_pass = .{ .static = render_pass },
        .max_frames_in_flight = max_frames_in_flight,
    });

    registerDvuiIO(glfwWin(window_context));

    defer b.vkc.device.queueWaitIdle(b.vkc.graphics_queue.handle) catch {}; // let gpu finish its work on exit, otherwise we will get validation errors
    //extern fn glfwSetWindowRefreshCallback(window: ?*Window, callback: WindowRefreshFun) WindowRefreshFun;
    if (builtin.os.tag == .windows) { // windows blocks event loop while resizing, use separate callback to keep rendering
        _ = setWindowRefreshCallback(window, &refreshCB);
    }

    while (!window.shouldClose()) {
        if (window.getKey(.escape) == .press) {
            window.setShouldClose(true);
        }

        // TODO: fixme: implement actual multi-window implementation, this won't work right
        for (b.contexts.items, 0..) |ctx, ctx_i| {
            _ = ctx_i; // autofix
            try paint(app, &g_app_state, ctx);
            b.prev_frame_stats = b.renderer.?.stats;
        }

        glfw.pollEvents();
    }
}

/// Window damage and refresh
pub fn refreshCB(window: *glfw.Window) callconv(.c) void {
    disable_wait_event = true;
    const ctx = glfwWindowContext(window);
    if (dvui.App.get()) |app| {
        paint(app, &g_app_state, ctx) catch |err| slog.debug("paint on refresh failed: {}", .{err});
    } else slog.debug("Can't refresh - missing dvui.getApp()", .{});
    disable_wait_event = false;
}

pub fn resizeCB(window: *glfw.Window, w: c_int, h: c_int) callconv(.c) void {
    const ctx = glfwWindowContext(window);
    ctx.recreate_swapchain_requested = true;
    ctx.last_pixel_size.w = @floatFromInt(w);
    ctx.last_pixel_size.h = @floatFromInt(h);
    ctx.last_window_size.w = @floatFromInt(w);
    ctx.last_window_size.h = @floatFromInt(h);
}

//
//      INPUT
//
pub fn glfwWindowContext(win: *glfw.Window) *WindowContext {
    return win.getUserPointer(WindowContext).?;
}

pub var disable_wait_event: bool = false;
/// Return true if interrupted by event
pub fn waitEventTimeout(self: *@This(), timeout_micros: u32) !bool {
    // at least on windows when rendering from damage callback waitEvents can get stuck forever, so it must be disabled
    if (disable_wait_event or get(self).recreate_swapchain_requested) return true;

    if (timeout_micros == std.math.maxInt(u32)) {
        glfw.waitEvents();
        return true;
    }

    if (timeout_micros > 0) {
        // wait with a timeout
        const timeout = @min((timeout_micros + 999) / 1000, std.math.maxInt(c_int));
        const timeout_s = @as(f64, @floatFromInt(timeout)) / std.time.us_per_s;
        glfw.waitEventsTimeout(timeout_s);
        return true;
    }

    // don't wait at all
    return false;
}

pub fn registerDvuiIO(win: *glfw.Window) void {
    _ = win.setCursorPosCallback(&CursorPosCB);
    _ = win.setCursorEnterCallback(&CursorEnterCB);
    _ = win.setScrollCallback(&ScrollCB);
    _ = win.setKeyCallback(&KeyCB);
    _ = win.setCharCallback(&CharCB);
    _ = win.setDropCallback(&DropCB);
    _ = win.setMouseButtonCallback(&MouseButtonCB);
}

//Mods is bitfield of modifiers, button is enum of mouse buttons, and action is enum of keystates.
pub fn MouseButtonCB(window: *glfw.Window, button: glfw.MouseButton, action: glfw.Action, mods: glfw.Mods) callconv(.c) void {
    _ = mods; // autofix
    _ = glfwWindowContext(window).dvui_window.addEventMouseButton(glfwMouseButtonToDvui(button), if (action == .press) .press else .release) catch unreachable;
}

pub fn CursorPosCB(window: *glfw.Window, xpos: f64, ypos: f64) callconv(.c) void {
    _ = glfwWindowContext(window).dvui_window.addEventMouseMotion(.{
        .pt = .{
            .x = @floatCast(xpos),
            .y = @floatCast(ypos),
        },
    }) catch {};
}
//Entered is true or false
pub fn CursorEnterCB(window: *glfw.Window, entered: c_int) callconv(.c) void {
    _ = window; // autofix
    _ = entered; // autofix
}
pub fn ScrollCB(window: *glfw.Window, xoffset: f64, yoffset: f64) callconv(.c) void {
    if (xoffset != 0) _ = glfwWindowContext(window).dvui_window.addEventMouseWheel(@as(f32, @floatCast(xoffset)) * dvui.scroll_speed, .horizontal, null) catch unreachable;
    if (yoffset != 0) _ = glfwWindowContext(window).dvui_window.addEventMouseWheel(@as(f32, @floatCast(yoffset)) * dvui.scroll_speed, .vertical, null) catch unreachable;
}
//Mods refers to the bitfield of Modifiers
pub fn KeyCB(window: *glfw.Window, key: glfw.Key, scancode: c_int, action: glfw.Action, mods: glfw.Mods) callconv(.c) void {
    _ = scancode; // autofix
    _ = glfwWindowContext(window).dvui_window.addEventKey(.{
        .code = glfwKeyCodeToDvui(key),
        .action = switch (action) {
            .press => .down,
            .repeat => .repeat,
            else => .up,
        },
        .mod = glfwModDvui(mods),
    }) catch {};
}
pub fn CharCB(window: *glfw.Window, codepoint: c_uint) callconv(.c) void {
    var buf: [4]u8 = undefined;
    const len = std.unicode.utf8Encode(@intCast(codepoint), &buf) catch unreachable;
    _ = glfwWindowContext(window).dvui_window.addEventText(.{ .text = buf[0..len] }) catch {};
}
pub fn DropCB(window: *glfw.Window, path_count: i32, paths: [*][*:0]const u8) callconv(.c) void {
    _ = window; // autofix
    _ = path_count; // autofix
    _ = paths; // autofix
}
//Event is one of two states defined by the enum 'Connection'
pub fn MonitorCB(monitor: *glfw.Monitor, event: c_int) callconv(.c) void {
    _ = monitor; // autofix
    _ = event; // autofix
}
//Event is one of two states defined by the enum 'Connection'
pub fn JoystickCB(id: c_int, event: c_int) callconv(.c) void {
    _ = id; // autofix
    _ = event; // autofix
}

pub fn glfwModDvui(mod: glfw.Mods) dvui.enums.Mod {
    if (mod.shift) return .lshift;
    if (mod.control) return .lcontrol;
    if (mod.alt) return .lalt;
    if (mod.super) return .lcommand;
    // glfw.ModifierCapsLock => ,
    // glfw.ModifierNumLock => ,
    return .none;
}

pub fn glfwMouseButtonToDvui(button: glfw.MouseButton) dvui.enums.Button {
    return switch (button) {
        .left => .left,
        .right => .right,
        .middle => .middle,
        .four => .four,
        .five => .five,
        .six => .six,
        .seven => .seven,
        .eight => .eight,
    };
}

pub fn glfwKeyCodeToDvui(keycode: glfw.Key) dvui.enums.Key {
    return switch (keycode) {
        .a => .a,
        .b => .b,
        .c => .c,
        .d => .d,
        .e => .e,
        .f => .f,
        .g => .g,
        .h => .h,
        .i => .i,
        .j => .j,
        .k => .k,
        .l => .l,
        .m => .m,
        .n => .n,
        .o => .o,
        .p => .p,
        .q => .q,
        .r => .r,
        .s => .s,
        .t => .t,
        .u => .u,
        .v => .v,
        .w => .w,
        .x => .x,
        .y => .y,
        .z => .z,
        .zero => .zero,
        .one => .one,
        .two => .two,
        .three => .three,
        .four => .four,
        .five => .five,
        .six => .six,
        .seven => .seven,
        .eight => .eight,
        .nine => .nine,
        .F1 => .f1,
        .F2 => .f2,
        .F3 => .f3,
        .F4 => .f4,
        .F5 => .f5,
        .F6 => .f6,
        .F7 => .f7,
        .F8 => .f8,
        .F9 => .f9,
        .F10 => .f10,
        .F11 => .f11,
        .F12 => .f12,
        .F13 => .f13,
        .F14 => .f14,
        .F15 => .f15,
        .F16 => .f16,
        .F17 => .f17,
        .F18 => .f18,
        .F19 => .f19,
        .F20 => .f20,
        .F21 => .f21,
        .F22 => .f22,
        .F23 => .f23,
        .F24 => .f24,
        .F25 => .f25,
        .kp_divide => .kp_divide,
        .kp_multiply => .kp_multiply,
        .kp_subtract => .kp_subtract,
        .kp_add => .kp_add,
        .kp_0 => .kp_0,
        .kp_1 => .kp_1,
        .kp_2 => .kp_2,
        .kp_3 => .kp_3,
        .kp_4 => .kp_4,
        .kp_5 => .kp_5,
        .kp_6 => .kp_6,
        .kp_7 => .kp_7,
        .kp_8 => .kp_8,
        .kp_9 => .kp_9,
        .kp_decimal => .kp_decimal,
        .kp_equal => .kp_equal,
        .kp_enter => .kp_enter,
        .enter => .enter,
        .escape => .escape,
        .tab => .tab,
        .left_shift => .left_shift,
        .right_shift => .right_shift,
        .left_control => .left_control,
        .right_control => .right_control,
        .left_alt => .left_alt,
        .right_alt => .right_alt,
        .left_super => .left_command,
        .right_super => .right_command,
        .menu => .menu,
        .num_lock => .num_lock,
        .caps_lock => .caps_lock,
        .print_screen => .print,
        .scroll_lock => .scroll_lock,
        .pause => .pause,
        .delete => .delete,
        .home => .home,
        .end => .end,
        .page_up => .page_up,
        .page_down => .page_down,
        .insert => .insert,
        .left => .left,
        .right => .right,
        .up => .up,
        .down => .down,
        .backspace => .backspace,
        .space => .space,
        .minus => .minus,
        .equal => .equal,
        .left_bracket => .left_bracket,
        .right_bracket => .right_bracket,
        .backslash => .backslash,
        .semicolon => .semicolon,
        .apostrophe => .apostrophe,
        .comma => .comma,
        .period => .period,
        .slash => .slash,
        .grave_accent => .grave,

        else => {
            slog.warn("Unknown key code {}", .{keycode});
            return .unknown;
        },
    };
}

const WindowRefreshFn = *const fn (*glfw.Window) callconv(.c) void;
extern fn glfwSetWindowRefreshCallback(*glfw.Window, ?WindowRefreshFn) ?WindowRefreshFn;
pub fn setWindowRefreshCallback(window: *glfw.Window, callback: ?WindowRefreshFn) ?WindowRefreshFn {
    return glfwSetWindowRefreshCallback(window, callback);
}
