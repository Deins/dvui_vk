const std = @import("std");
const builtin = @import("builtin");
pub const dvui = @import("dvui");
pub const low = @import("low");
pub const win32 = if (builtin.target.os.tag == .windows) @import("win32").everything else void;
pub const kind: dvui.enums.Backend = .custom;
const slog = std.log.scoped(.dvu_vk_low);
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

    vsync: bool,

    max_frames_in_flight: u8 = 2,
    desired_surface_formats: []const vk.SurfaceFormatKHR = &.{
        // NOTE: all dvui examples as far as I can tell expect all color transformations to happen directly in srgb space, so we request unorm not srgb format. To support linear rendering this will be an issue.
        // TODO: add support for both linear and srgb render targets
        // similar issue: https://github.com/ocornut/imgui/issues/578
        .{ .format = .a2b10g10r10_unorm_pack32, .color_space = .srgb_nonlinear_khr },
        .{ .format = .b8g8r8a8_unorm, .color_space = .srgb_nonlinear_khr },
        .{ .format = .r8g8b8a8_unorm, .color_space = .srgb_nonlinear_khr },
    },

    backend: low.BackendRequest = .auto,
    /// Wayland decoration preference. `server_side` asks for a native titlebar.
    titlebar: low.DecorationMode = .server_side,
    /// Initial OS window state.
    window_state: low.WindowState = .normal,
    app_name: [:0]const u8 = "dvui_vk",
};

pub const InitWindowResult = struct {
    backend: *VkBackend,
    window: *WindowContext,

    pub fn deinit(self: InitWindowResult) void {
        self.backend.deinit();
        self.backend.gpa.destroy(self.backend);
    }
};

pub const vk = @import("vk");
pub const dvui_vk_common = @import("dvui_vk_common.zig");
pub const DecorationMode = low.DecorationMode;
pub const WindowState = low.WindowState;
pub const VkContext = dvui_vk_common.VkContext;
pub const VkBackend = dvui_vk_common.VkBackend;
pub const WindowContext = dvui_vk_common.WindowContext;
pub const createRenderPass = dvui_vk_common.createRenderPass;
pub const createFramebuffers = dvui_vk_common.createFramebuffers;
pub const destroyFramebuffers = dvui_vk_common.destroyFramebuffers;
pub const FrameSync = dvui_vk_common.FrameSync;
pub const createCommandBuffers = dvui_vk_common.createCommandBuffers;
pub const present = dvui_vk_common.present;
pub const AppState = dvui_vk_common.AppState;
pub var g_app_state: AppState = undefined;
pub const paint = dvui_vk_common.paint;
pub const max_frames_in_flight = 2;

pub const win = struct {
    pub const ServiceResult = enum {
        queue_empty,
        quit,
    };

    pub fn serviceMessageQueue() ServiceResult {
        if (g_low_context) |*ctx| {
            ctx.pollEvents();
        }
        for (g_low_window_contexts.items) |ctx| syncWindowState(ctx);
        return .queue_empty;
    }
};

pub const GenericError = dvui.Backend.GenericError;
pub const TextureError = dvui.Backend.TextureError;
pub const vk_dll = @import("vk_dll.zig");
pub const vkk = @import("vk_kickstart");

var g_low_context: ?low.Context = null;
var g_low_context_refs: usize = 0;
var g_low_window_contexts: std.ArrayListUnmanaged(*WindowContext) = .empty;

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

fn lowContext() *low.Context {
    if (g_low_context) |*ctx| return ctx;
    @panic("low context not initialized");
}

fn ensureLowContext(gpa: std.mem.Allocator, init_opts: InitOptions) !*low.Context {
    if (g_low_context) |*ctx| {
        g_low_context_refs += 1;
        return ctx;
    }

    g_low_context = try low.Context.init(gpa, .{
        .backend = init_opts.backend,
        .app_name = init_opts.app_name,
    });
    g_low_context_refs = 1;
    return lowContext();
}

fn releaseLowContext(gpa: std.mem.Allocator) void {
    if (g_low_context == null or g_low_context_refs == 0) return;
    g_low_context_refs -= 1;
    if (g_low_context_refs == 0) {
        if (g_low_context) |*ctx| {
            ctx.deinit();
        }
        g_low_context = null;
        g_low_window_contexts.deinit(gpa);
    }
}

fn lowWindow(context: *WindowContext) *low.Window {
    return @as(*low.Window, @ptrCast(@alignCast(context.glfw_win.?)));
}

fn lowSizeToNatural(size: low.Size) dvui.Size.Natural {
    return .{ .w = @floatFromInt(size.width), .h = @floatFromInt(size.height) };
}

fn lowSizeToPhysical(size: low.Size) dvui.Size.Physical {
    return .{ .w = @floatFromInt(size.width), .h = @floatFromInt(size.height) };
}

fn lowWindowSizeFromDvui(size: dvui.Size) low.Size {
    return .{
        .width = @max(1, @as(i32, @intFromFloat(@round(size.w)))),
        .height = @max(1, @as(i32, @intFromFloat(@round(size.h)))),
    };
}

pub fn createVkSurfaceLow(self: *WindowContext, vk_instance: vk.InstanceProxy) bool {
    const low_win = lowWindow(self);
    if (builtin.target.os.tag == .windows) {
        const ci = vk.Win32SurfaceCreateInfoKHR{
            .hinstance = @ptrCast(low_win.nativeDisplay()),
            .hwnd = @ptrFromInt(low_win.nativeSurface()),
        };
        self.surface = vk_instance.createWin32SurfaceKHR(&ci, self.backend.vkc.alloc) catch |err| {
            slog.err("Failed to create surface: {}", .{err});
            return false;
        };
        return true;
    }
    return switch (low_win.ctx.backendKind()) {
        .wayland => blk: {
            const ci = vk.WaylandSurfaceCreateInfoKHR{
                .display = @ptrCast(@alignCast(low_win.nativeDisplay())),
                .surface = @ptrFromInt(low_win.nativeSurface()),
            };
            self.surface = vk_instance.createWaylandSurfaceKHR(&ci, null) catch |err| {
                slog.err("Failed to create surface: {}", .{err});
                return false;
            };
            break :blk true;
        },
        .x11 => blk: {
            const ci = vk.XlibSurfaceCreateInfoKHR{
                .dpy = @ptrCast(@alignCast(low_win.nativeDisplay())),
                .window = @as(vk.Window, @intCast(low_win.nativeSurface())),
            };
            self.surface = vk_instance.createXlibSurfaceKHR(&ci, null) catch |err| {
                slog.err("Failed to create surface: {}", .{err});
                return false;
            };
            break :blk true;
        },
        // Neither backend exposes an OS Vulkan surface. Windows is handled
        // above, while offscreen rendering intentionally has no surface.
        .offscreen, .windows => false,
    };
}
pub const createVkSurface = createVkSurfaceLow;

//
//   Dvui backend implementation
//

/// Get monotonic nanosecond timestamp. Doesn't have to be system time.
pub fn nanoTime(_: ContextHandle) i128 {
    return std.Io.Timestamp.now(std.Options.debug_io, .awake).nanoseconds;
}

pub fn sleep(_: ContextHandle, ns: u64) void {
    (std.Io.Clock.Duration{
        .raw = std.Io.Duration.fromNanoseconds(@intCast(ns)),
        .clock = .awake,
    }).sleep(std.Options.debug_io) catch {};
}

/// Called by dvui during `dvui.Window.begin`, so prior to any dvui
/// rendering.  Use to setup anything needed for this frame.  The arena
/// arg is cleared before `dvui.Window.begin` is called next, useful for any
/// temporary allocations needed only for this frame.
pub fn begin(context_handle: ContextHandle, arena: std.mem.Allocator) GenericError!void {
    get(context_handle).arena = arena;
    context_handle.renderer().begin(arena, context_handle.pixelSize());
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
    const ctx = get(self);
    if (ctx.glfw_win) |ptr| {
        const low_win: *low.Window = @ptrCast(@alignCast(ptr));
        return low_win.getContentScale().x;
    }
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
pub fn textureCreate(ch: ContextHandle, pixels: [*]const u8, options: dvui.Texture.CreateOptions) TextureError!dvui.Texture {
    return ch.renderer().textureCreate(pixels, options);
}

/// Destroy `texture` made with `textureCreate`. After this call, this texture
/// pointer will not be used by dvui.
pub fn textureDestroy(ch: ContextHandle, texture: dvui.Texture) void {
    ch.renderer().textureDestroy(texture);
}

/// Create a `dvui.Texture` that can be rendered to with `renderTarget`.  The
/// returned pointer is what will later be passed to `drawClippedTriangles`.
pub fn textureCreateTarget(ch: ContextHandle, options: dvui.Texture.CreateOptions) TextureError!dvui.TextureTarget {
    return ch.renderer().textureCreateTarget(options);
}

/// Read pixel data (RGBA) from `texture` into `pixels_out`.
pub fn textureReadTarget(ch: ContextHandle, texture: dvui.TextureTarget, pixels_out: [*]u8) TextureError!void {
    return ch.renderer().textureReadTarget(texture, pixels_out);
}

/// Clear a texture target without destroying it.
pub fn textureClearTarget(ch: ContextHandle, texture: dvui.Texture.Target) void {
    // Opening a target starts an offscreen render pass with a transparent
    // clear value; closing it submits that clear before the target is used.
    ch.renderer().renderTarget(texture) catch |err| {
        slog.err("Failed to clear render target: {}", .{err});
        return;
    };
    ch.renderer().renderTarget(null) catch |err| {
        slog.err("Failed to finish render target clear: {}", .{err});
    };
}

/// Destroy `texture` made with `textureCreateTarget`. After this call, this
/// texture pointer will not be used by dvui.
pub fn textureDestroyTarget(ch: ContextHandle, texture: dvui.Texture.Target) void {
    ch.renderer().textureDestroy(dvui.Texture.cast(texture));
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
    return lowContext().clipboardText(get(self).arena);
}

/// Set clipboard content (text only)
pub fn clipboardTextSet(self: ContextHandle, text: []const u8) GenericError!void {
    _ = self;
    return lowContext().clipboardTextSet(text);
}

/// Open URL in system browser
pub fn openURL(self: ContextHandle, url: []const u8, new_window: bool) GenericError!void {
    _ = new_window;
    return dvui_vk_common.openURL(get(self).backend.gpa, url);
}

/// Get the preferredColorScheme if available
pub fn preferredColorScheme(self: ContextHandle) ?dvui.enums.ColorScheme {
    _ = self;
    return switch (lowContext().preferredColorScheme() orelse return null) {
        .light => .light,
        .dark => .dark,
    };
}

/// Desktop low-level window systems do not consistently expose this setting.
pub fn prefersReducedMotion(_: ContextHandle) bool {
    return false;
}

fn cursorShape(cursor: dvui.enums.Cursor) low.CursorShape {
    return switch (cursor) {
        .arrow => .arrow,
        .arrow_all => .resize_all,
        .arrow_n_s => .resize_ns,
        .arrow_ne_sw => .resize_nesw,
        .arrow_nw_se => .resize_nwse,
        .arrow_w_e => .resize_ew,
        .bad => .not_allowed,
        .crosshair => .crosshair,
        .hand => .hand,
        .ibeam => .ibeam,
        .wait, .wait_arrow => .arrow,
        .hidden => .hidden,
    };
}

/// Show/hide the cursor.
///
/// Returns the previous state of the cursor, `true` meaning shown
pub fn cursorShow(self: ContextHandle, value: ?bool) GenericError!bool {
    const low_win = lowWindow(get(self));
    const prev = low_win.cursor_visible;
    if (value) |shown| {
        low_win.setCursorVisible(shown);
    }
    return prev;
}

pub fn title(self: ContextHandle, _: *dvui.Window, new_title: []const u8) void {
    var buf: [512]u8 = undefined;
    const title_text = std.fmt.bufPrintSentinel(&buf, "{s}", .{new_title}, 0) catch {
        slog.warn("Window title too long", .{});
        return;
    };
    lowWindow(get(self)).setTitle(title_text);
}

pub fn windowStateSet(self: ContextHandle, _: *dvui.Window, state: dvui.enums.WindowState) void {
    const low_win = lowWindow(get(self));
    switch (state) {
        .normal => low_win.setState(.normal),
        .maximize => low_win.setState(.maximize),
        .fullscreen => low_win.setState(.fullscreen),
    }
    if (g_low_context) |*ctx| ctx.wake();
}

/// Set the cursor shape for the window.
pub fn setCursor(self: ContextHandle, cursor: dvui.enums.Cursor) void {
    const low_win = lowWindow(get(self));
    if (cursor == .hidden) {
        _ = self.cursorShow(false) catch |err| {
            slog.err("Failed to hide cursor: {}", .{err});
        };
        return;
    }

    _ = self.cursorShow(true) catch |err| {
        slog.err("Failed to show cursor: {}", .{err});
        return;
    };

    low_win.setCursor(cursorShape(cursor));
}

/// Manage text input.
pub fn textInputRect(self: ContextHandle, rect: ?dvui.Rect.Natural) void {
    const scale = contentScale(self);
    lowWindow(get(self)).setTextInputRect(if (rect) |r| .{
        .x = r.x * scale,
        .y = r.y * scale,
        .width = r.w * scale,
        .height = r.h * scale,
    } else null);
}

/// Render/present is performed by the Vulkan frame loop in `paint`.
pub fn renderPresent(_: ContextHandle) void {}

pub fn deinitWindow(self: *WindowContext) void {
    if (self.glfw_win) |ptr| {
        const window: *low.Window = @ptrCast(@alignCast(ptr));
        self.glfw_win = null;
        unregisterWindowContext(self);
        window.deinit();
    }
    releaseLowContext(self.backend.gpa);
}

/// Release backend resources.
pub fn deinit(_: ContextHandle) void {}

/// Called by `dvui.refresh` when it is called from a background
/// thread.  Used to wake up the gui thread.  It only has effect if you
/// are using `dvui.Window.waitTime` or some other method of waiting until
/// a new event comes in.
pub fn refresh(_: ContextHandle) void {
    if (g_low_context) |*ctx| {
        ctx.wake();
    }
}

pub fn registerDvuiIO(window: *low.Window) void {
    window.callbacks = .{
        .mouse_button = MouseButtonCB,
        .scroll = ScrollCB,
        .key = KeyCB,
        .text = CharCB,
    };
}

fn windowContext(window: *low.Window) *WindowContext {
    return @as(*WindowContext, @ptrCast(@alignCast(window.getUserData().?)));
}

pub fn ScrollCB(window: *low.Window, xoffset: f64, yoffset: f64) void {
    if (xoffset != 0) _ = windowContext(window).dvui_window.addEventMouseWheel(@as(f32, @floatCast(xoffset)) * dvui.scroll_speed, .horizontal, null) catch unreachable;
    if (yoffset != 0) _ = windowContext(window).dvui_window.addEventMouseWheel(@as(f32, @floatCast(yoffset)) * dvui.scroll_speed, .vertical, null) catch unreachable;
}

fn lowModDvui(mods: low.Modifiers) dvui.enums.Mod {
    var mod = dvui.enums.Mod.none;
    if (mods.shift) mod.combine(.lshift);
    if (mods.control) mod.combine(.lcontrol);
    if (mods.alt) mod.combine(.lalt);
    if (mods.super) mod.combine(.lcommand);
    return mod;
}

pub fn MouseButtonCB(window: *low.Window, button: low.MouseButton, action: low.Action, mods: low.Modifiers) void {
    _ = mods;
    _ = windowContext(window).dvui_window.addEventMouseButton(lowMouseButtonToDvui(button), if (action == .press) .press else .release) catch unreachable;
}

pub fn KeyCB(window: *low.Window, key: low.Key, raw_keycode: u32, action: low.Action, mods: low.Modifiers) void {
    _ = windowContext(window).dvui_window.addEventKey(.{
        .code = lowKeyCodeToDvui(key),
        .action = switch (action) {
            .press => .down,
            .repeat => .repeat,
            .release => .up,
        },
        .mod = lowModDvui(mods),
    }) catch unreachable;
    _ = raw_keycode;
}

pub fn CharCB(window: *low.Window, bytes: []const u8) void {
    _ = windowContext(window).dvui_window.addEventText(.{ .text = bytes }) catch unreachable;
}

pub fn lowMouseButtonToDvui(button: low.MouseButton) dvui.enums.Button {
    return switch (button) {
        inline else => |b| @field(dvui.enums.Button, @tagName(b)),
    };
}

pub fn lowKeyCodeToDvui(keycode: low.Key) dvui.enums.Key {
    return switch (keycode) {
        inline else => |k| @field(dvui.enums.Key, @tagName(k)),
    };
}

pub fn waitEventTimeout(self: *@This(), timeout_micros: u32) !bool {
    _ = self;
    if (timeout_micros == 0) return false;
    const timeout_ns: u64 = if (timeout_micros == std.math.maxInt(u32))
        std.math.maxInt(u64)
    else
        @as(u64, timeout_micros) * std.time.ns_per_us;
    return lowContext().waitEventsTimeout(timeout_ns);
}

fn syncWindowState(ctx: *WindowContext) void {
    const window = lowWindow(ctx);
    ctx.received_close = window.shouldClose();

    const window_size = lowSizeToNatural(window.getSize());
    ctx.last_window_size = window_size;

    const pixel_size = lowSizeToPhysical(window.getFramebufferSize());
    if (pixel_size.w != ctx.last_pixel_size.w or pixel_size.h != ctx.last_pixel_size.h) {
        ctx.last_pixel_size = pixel_size;
        ctx.recreate_swapchain_requested = true;
    }

    ctx.dvui_window.content_scale = window.getContentScale().x;

    const cursor_pos = window.getCursorPos();
    if (cursor_pos.x != ctx.last_cursor_x or cursor_pos.y != ctx.last_cursor_y) {
        ctx.last_cursor_x = cursor_pos.x;
        ctx.last_cursor_y = cursor_pos.y;
        _ = ctx.dvui_window.addEventMouseMotion(.{
            .pt = .{
                .x = @floatCast(cursor_pos.x),
                .y = @floatCast(cursor_pos.y),
            },
        }) catch unreachable;
    }
}

fn registerWindowContext(ctx: *WindowContext, gpa: std.mem.Allocator) !void {
    try g_low_window_contexts.append(gpa, ctx);
}

fn unregisterWindowContext(ctx: *WindowContext) void {
    const index = std.mem.indexOfScalar(*WindowContext, g_low_window_contexts.items, ctx) orelse return;
    _ = g_low_window_contexts.swapRemove(index);
}

pub fn initWindow(loader: vk.PfnGetInstanceProcAddr, init_opts: InitOptions) !InitWindowResult {
    const gpa = init_opts.dvui_gpa;

    const b = try gpa.create(VkBackend);
    errdefer gpa.destroy(b);
    b.* = VkBackend.init(gpa, undefined);
    var backend_containers_need_cleanup = true;
    errdefer if (backend_containers_need_cleanup) {
        b.contexts.deinit(gpa);
        b.contexts_pool.deinit(gpa);
    };

    const window_context: *WindowContext = try b.allocContext();
    var low_ctx_acquired = false;
    var window: ?*low.Window = null;
    var dvui_window_init = false;
    var manual_cleanup = true;
    errdefer if (manual_cleanup) {
        if (dvui_window_init) {
            window_context.dvui_window.deinit();
        }
        if (window) |w| {
            w.deinit();
        }
        if (low_ctx_acquired) {
            releaseLowContext(gpa);
        }
        b.freeContext(window_context);
    };

    const low_ctx = try ensureLowContext(gpa, init_opts);
    low_ctx_acquired = true;

    const created_window = try low_ctx.createWindow(.{
        .title = init_opts.title,
        .size = if (init_opts.size) |size| lowWindowSizeFromDvui(size) else .{ .width = 1280, .height = 720 },
        .app_id = null,
        .resizable = true,
        .decorated = true,
        .titlebar = init_opts.titlebar,
        .state = init_opts.window_state,
        .visible = true,
    });
    window = created_window;
    created_window.setUserData(window_context);
    registerDvuiIO(created_window);

    window_context.* = .{
        .backend = b,
        .dvui_window = undefined,
        .hwnd = if (builtin.os.tag != .windows) {} else undefined,
        .glfw_win = @ptrCast(created_window),
        .received_close = false,
        .last_pixel_size = lowSizeToPhysical(created_window.getFramebufferSize()),
        .last_window_size = lowSizeToNatural(created_window.getSize()),
        .last_cursor_x = created_window.getCursorPos().x,
        .last_cursor_y = created_window.getCursorPos().y,
    };

    window_context.dvui_window = try dvui.Window.init(@src(), gpa, dvuiBackend(window_context), .{});
    dvui_window_init = true;
    try registerWindowContext(window_context, gpa);

    var ext_dynamic_state_features = vk.PhysicalDeviceExtendedDynamicStateFeaturesEXT{ .extended_dynamic_state = .true };
    b.vkc = try VkContext.init(gpa, loader, window_context, &createVkSurfaceLow, .{
        .device_select_settings = .{
            .required_extensions = &.{
                vk.extensions.khr_swapchain.name,
                vk.extensions.ext_extended_dynamic_state.name,
            },
            .required_features_13 = .{
                .synchronization_2 = .true,
                .dynamic_rendering = .true,
            },
        },
        .device_create_info_p_next = @ptrCast(&ext_dynamic_state_features),
    });
    backend_containers_need_cleanup = false;
    manual_cleanup = false;
    errdefer b.deinit();

    window_context.swapchain_state = try WindowContext.SwapchainState.init(window_context, .{
        .graphics_queue_index = b.vkc.physical_device.graphics_queue_index,
        .present_queue_index = if (b.vkc.physical_device.present_queue_index) |q| q else b.vkc.physical_device.graphics_queue_index,
        .desired_extent = vk.Extent2D{ .width = @intFromFloat(window_context.last_pixel_size.w), .height = @intFromFloat(window_context.last_pixel_size.h) },
        .desired_min_image_count = init_opts.max_frames_in_flight,
        .desired_formats = init_opts.desired_surface_formats,
        .desired_present_modes = if (!init_opts.vsync) &.{ .immediate_khr, .mailbox_khr } else &.{ .fifo_khr, .mailbox_khr },
    });
    return .{ .backend = b, .window = window_context };
}

pub fn main() !void {
    const app = dvui.App.get() orelse return error.DvuiAppNotDefined;

    var gpa_instance = std.heap.DebugAllocator(.{}){};
    const gpa = gpa_instance.allocator();
    defer _ = gpa_instance.deinit();

    const init_opts = app.config.get();

    vk_dll.init() catch |err| std.debug.panic("Failed to init Vulkan: {}", .{err});
    defer vk_dll.deinit();
    const loader = vk_dll.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse @panic("Failed to lookup Vulkan VkGetInstanceProcAddr");

    const iw = try initWindow(loader, .{
        .title = init_opts.title,
        .icon = init_opts.icon,
        .size = init_opts.size,
        .max_frames_in_flight = max_frames_in_flight,
        .dvui_gpa = gpa,
        .gpa = gpa,
        .vsync = true,
    });
    defer iw.deinit();
    const b = iw.backend;

    const render_pass = try createRenderPass(b.vkc.device, iw.window.swapchain_state.?.swapchain.image_format);
    defer b.vkc.device.destroyRenderPass(render_pass, b.vkc.alloc);

    const sync = try FrameSync.init(gpa, max_frames_in_flight, b.vkc);
    defer sync.deinit(gpa, b.vkc.device);

    const command_buffers = try createCommandBuffers(gpa, b.vkc.device, b.vkc.cmd_pool, max_frames_in_flight);
    defer gpa.free(command_buffers);

    g_app_state = .{
        .backend = b,
        .command_buffers = command_buffers,
        .render_pass = render_pass,
        .sync = sync,
    };

    if (app.initFn) |initFn| {
        try initFn(&b.contexts.items[0].dvui_window);
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

    defer b.vkc.device.queueWaitIdle(b.vkc.graphics_queue.handle) catch {};

    while (b.contexts.items.len > 0) {
        lowContext().pollEvents();
        var i: usize = 0;
        while (i < b.contexts.items.len) {
            const ctx = b.contexts.items[i];
            syncWindowState(ctx);
            try paint(app, &g_app_state, ctx);
            b.prev_frame_stats = b.renderer.?.stats;
            if (ctx.received_close) {
                b.destroyContext(ctx);
                continue;
            }
            i += 1;
        }
    }
}

test {
    std.testing.refAllDecls(@This());
}
