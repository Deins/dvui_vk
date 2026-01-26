const std = @import("std");
const builtin = @import("builtin");
pub const dvui = @import("dvui");
pub const kind: dvui.enums.Backend = .custom;
const slog = std.log.scoped(.dvu_vk_win32);
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
};

pub const InitWindowResult = struct {
    backend: *VkBackend,
    window: *WindowContext,
    pub fn deinit(self: InitWindowResult) void {
        self.backend.deinit(); // will clean up all windows, don't destroy window as its done from win32 callbacks
        self.backend.gpa.destroy(self.backend);
    }
};

// init backend (creates and owns OS window)
pub fn initWindow(loader: vk.PfnGetInstanceProcAddr, init_opts: InitOptions) !InitWindowResult {
    const gpa = init_opts.dvui_gpa;

    const window_class = win32.L("DvuiWindow");
    win.RegisterClass(window_class, .{}) catch win32.panicWin32(
        "RegisterClass",
        win32.GetLastError(),
    );

    // TODO: on error cleanup here is messy because we need to get surface to init vk. More likely than not if error happens we will leak or crash here.
    const b = try gpa.create(VkBackend);
    errdefer gpa.destroy(b);
    b.* = VkBackend.init(gpa, undefined);
    const window_context: *WindowContext = try b.allocContext();
    window_context.* = .{
        .backend = b,
        .dvui_window = try dvui.Window.init(@src(), gpa, dvuiBackend(window_context), .{}),
        .hwnd = undefined,
    };
    errdefer window_context.dvui_window.deinit();
    try win.initWindow(window_context, window_class, init_opts);
    b.vkc = try VkContext.init(gpa, loader, window_context, &createVkSurfaceWin32, .{});
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

pub const GenericError = dvui.Backend.GenericError;
pub const TextureError = dvui.Backend.TextureError;

const is_windows = @import("builtin").target.os.tag == .windows;
// pub const dvui_win = if (is_windows) @import("dvui_win") else void;
// pub const win32 = if (is_windows) dvui_win.win32 else void;
pub const win32 = @import("win32").everything;

pub fn dvuiBackend(context: *WindowContext) dvui.Backend {
    return dvui.Backend.init(@ptrCast(@alignCast(context)));
}

/// to support multiple windows @This pointer is used as per dvui.window context from
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

pub fn createVkSurfaceWin32(self: *WindowContext, vk_instance: vk.InstanceProxy) bool {
    const ci = vk.Win32SurfaceCreateInfoKHR{
        .hwnd = @ptrCast(self.hwnd),
        .hinstance = @ptrCast(win32.GetModuleHandleW(null)),
    };
    self.surface = vk_instance.createWin32SurfaceKHR(&ci, self.backend.vkc.alloc) catch |err| {
        slog.err("Failed to create surface: {}", .{err});
        return false;
    };
    return true;
}
pub const createVkSurface = createVkSurfaceWin32;

//
//   Dvui backend implementation
//

/// Get monotonic nanosecond timestamp. Doesn't have to be system time.
pub fn nanoTime(_: ContextHandle) i128 {
    return std.time.nanoTimestamp();
}

pub fn sleep(_: ContextHandle, ns: u64) void {
    std.Thread.sleep(ns);
}

/// Called by dvui during `dvui.Window.begin`, so prior to any dvui
/// rendering.  Use to setup anything needed for this frame.  The arena
/// arg is cleared before `dvui.Window.begin` is called next, useful for any
/// temporary allocations needed only for this frame.
pub fn begin(context_handle: ContextHandle, arena: std.mem.Allocator) GenericError!void {
    context_handle.renderer().begin(arena, context_handle.pixelSize());
}

/// Called during `dvui.Window.end` before freeing any memory for the current frame.
pub fn end(context_handle: ContextHandle) GenericError!void {
    context_handle.renderer().end();
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
pub fn textureCreate(ch: ContextHandle, pixels: [*]const u8, width: u32, height: u32, interpolation: dvui.enums.TextureInterpolation) TextureError!dvui.Texture {
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
pub fn textureCreateTarget(ch: ContextHandle, width: u32, height: u32, interpolation: dvui.enums.TextureInterpolation) TextureError!dvui.TextureTarget {
    return ch.renderer().textureCreateTarget(width, height, interpolation);
}

/// Read pixel data (RGBA) from `texture` into `pixels_out`.
pub fn textureReadTarget(ch: ContextHandle, texture: dvui.TextureTarget, pixels_out: [*]u8) TextureError!void {
    return ch.renderer().textureReadTarget(texture, pixels_out);
}

/// Convert texture target made with `textureCreateTarget` into return texture
/// as if made by `textureCreate`.  After this call, texture target will not be
/// used by dvui.
pub fn textureFromTarget(ch: ContextHandle, texture: dvui.TextureTarget) TextureError!dvui.Texture {
    return ch.renderer().textureFromTarget(texture);
}

/// Render future `drawClippedTriangles` to the passed `texture` (or screen
/// if null).
pub fn renderTarget(ch: ContextHandle, texture: ?dvui.TextureTarget) GenericError!void {
    return ch.renderer().renderTarget(texture);
}

/// Get clipboard content (text only)
pub fn clipboardText(self: ContextHandle) GenericError![]const u8 {
    _ = self; // autofix
    return "";
}

/// Set clipboard content (text only)
pub fn clipboardTextSet(self: ContextHandle, text: []const u8) GenericError!void {
    _ = self; // autofix
    _ = text; // autofix

}

/// Open URL in system browser
pub fn openURL(self: ContextHandle, url: []const u8, new_window: bool) GenericError!void {
    _ = new_window; // autofix
    _ = self; // autofix
    _ = url; // autofix
}

/// Get the preferredColorScheme if available
pub fn preferredColorScheme(self: ContextHandle) ?dvui.enums.ColorScheme {
    _ = self; // autofix
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

//
//  APP
//
pub const vkk = @import("vk_kickstart");
pub const vk_dll = @import("vk_dll.zig");

pub const AppState = dvui_vk_common.AppState;
pub var g_app_state: AppState = undefined;
pub const paint = dvui_vk_common.paint;

pub fn main() !void {
    if (builtin.target.os.tag == .windows) dvui.Backend.Common.windowsAttachConsole() catch {};

    const app = dvui.App.get() orelse return error.DvuiAppNotDefined;

    var gpa_instance = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = gpa_instance.allocator();
    defer _ = gpa_instance.deinit();

    const init_opts = app.config.get();

    vk_dll.init() catch |err| std.debug.panic("Failed to init Vulkan: {}", .{err});
    defer vk_dll.deinit();
    const loader = vk_dll.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse @panic("Failed to lookup Vulkan VkGetInstanceProcAddr: {}");

    const max_frames_in_flight = 2;
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
    const window_context = iw.window;

    const render_pass = try createRenderPass(b.vkc.device, window_context.swapchain_state.?.swapchain.image_format);
    defer b.vkc.device.destroyRenderPass(render_pass, null);

    const sync = try FrameSync.init(gpa, max_frames_in_flight, b.vkc.device);
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
        .mem_props = b.vkc.physical_device.memory_properties,
        .render_pass = render_pass,
        .max_frames_in_flight = max_frames_in_flight,
    });

    defer b.vkc.device.queueWaitIdle(b.vkc.graphics_queue.handle) catch {}; // let gpu finish its work on exit, otherwise we will get validation errors
    main_loop: while (b.contexts.items.len > 0) {
        switch (win.serviceMessageQueue()) {
            .queue_empty => {
                for (b.contexts.items, 0..) |ctx, ctx_i| {
                    _ = ctx_i; // autofix
                    try paint(app, &g_app_state, ctx);
                    b.prev_frame_stats = b.renderer.?.stats;
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

pub const win = if (is_windows) struct {
    const log = std.log.scoped(.winapi);
    // most stuff here borrowed from dvui/src/backend/dx11
    fn resToErr(res: win32.HRESULT, what: []const u8) !void {
        if (win32.SUCCEEDED(res)) return;
        slog.err("{s} failed, hresult=0x{x}", .{ what, res });
        return dvui.Backend.GenericError.BackendError;
    }

    /// Check the return value and prints `win32.GetLastError()` on failure
    fn boolToErr(res: win32.BOOL, what: []const u8) !void {
        if (res != win32.FALSE) return;
        return lastErr(what);
    }

    /// prints `win32.GetLastError()`
    fn lastErr(what: []const u8) !void {
        const err = win32.GetLastError();
        return win32ToErr(err, what);
    }

    fn win32ToErr(err: win32.WIN32_ERROR, what: []const u8) !void {
        if (err == win32.NO_ERROR) return;
        slog.err("{s} failed, error={f}", .{ what, err });
        return dvui.Backend.GenericError.BackendError;
    }

    pub const RegisterClassOptions = struct {
        /// styles in addition to DBLCLICKS
        style: win32.WNDCLASS_STYLES = .{},
        // NOTE: we could allow the user to provide their own wndproc which we could
        //       call before or after ours
        //wndproc: ...,
        class_extra: c_int = 0,
        // NOTE: the dx11 backend uses the first @sizeOf(*anyopaque) bytes, any length
        //       added here will be offset by that many bytes
        window_extra_after_sizeof_ptr: c_int = 0,
        instance: union(enum) { this_module, custom: ?win32.HINSTANCE } = .this_module,
        cursor: union(enum) { arrow, custom: ?win32.HICON } = .arrow,
        icon: ?win32.HICON = null,
        icon_small: ?win32.HICON = null,
        bg_brush: ?win32.HBRUSH = null,
        menu_name: ?[*:0]const u16 = null,
    };

    /// A wrapper for win32.RegisterClass that registers a window class compatible
    /// with initWindow. Returns error.Win32 on failure, call win32.GetLastError()
    /// for the error code.
    ///
    /// RegisterClass can only be called once for a given name (unless it's been unregistered
    /// via UnregisterClass). Typically there's no reason to unregister a window class.
    pub fn RegisterClass(name: [*:0]const u16, opt: RegisterClassOptions) error{Win32}!void {
        const wc: win32.WNDCLASSEXW = .{
            .cbSize = @sizeOf(win32.WNDCLASSEXW),
            .style = @bitCast(@as(u32, @bitCast(win32.WNDCLASS_STYLES{ .DBLCLKS = 1 })) | @as(u32, @bitCast(opt.style))),
            .lpfnWndProc = wndProc,
            .cbClsExtra = opt.class_extra,
            .cbWndExtra = @sizeOf(usize) + opt.window_extra_after_sizeof_ptr,
            .hInstance = switch (opt.instance) {
                .this_module => win32.GetModuleHandleW(null),
                .custom => |i| i,
            },
            .hIcon = opt.icon,
            .hIconSm = opt.icon_small,
            .hCursor = switch (opt.cursor) {
                .arrow => win32.LoadCursorW(null, win32.IDC_ARROW),
                .custom => |c| c,
            },
            .hbrBackground = opt.bg_brush,
            .lpszMenuName = opt.menu_name,
            .lpszClassName = name,
        };
        if (0 == win32.RegisterClassExW(&wc)) return error.Win32;
    }

    const CreateWindowArgs = struct {
        context: *WindowContext,
        dvui_gpa: std.mem.Allocator,
        err: ?anyerror = null,
    };

    pub fn initWindow(context: *WindowContext, window_class: [*:0]const u16, options: InitOptions) !void {
        const style = win32.WS_OVERLAPPEDWINDOW;
        const style_ex: win32.WINDOW_EX_STYLE = .{ .APPWINDOW = 1, .WINDOWEDGE = 1 };

        const create_args: CreateWindowArgs = .{
            .context = context,
            .dvui_gpa = options.dvui_gpa,
        };
        const hwnd = blk: {
            const wnd_title = try std.unicode.utf8ToUtf16LeAllocZ(options.gpa, options.title);
            defer options.gpa.free(wnd_title);
            break :blk win32.CreateWindowExW(
                style_ex,
                window_class,
                wnd_title,
                style,
                win32.CW_USEDEFAULT, // x
                win32.CW_USEDEFAULT, // y
                win32.CW_USEDEFAULT, // w
                win32.CW_USEDEFAULT, // h
                null, // hWndParent
                null, // hMenu
                win32.GetModuleHandleW(null), // This message is sent to the created window by this function before it returns.
                @ptrCast(@constCast(&create_args)),
            ) orelse switch (win32.GetLastError()) {
                win32.ERROR_CANNOT_FIND_WND_CLASS => switch (builtin.mode) {
                    .Debug => std.debug.panic(
                        "did you forget to call RegisterClass? (class_name='{f}')",
                        .{std.unicode.fmtUtf16Le(std.mem.span(window_class))},
                    ),
                    else => unreachable,
                },
                else => |win32Err| {
                    if (create_args.err) |err| return err;
                    win32.panicWin32("CreateWindow", win32Err);
                },
            };
        };
        context.hwnd = hwnd;

        switch (dvui.Backend.Common.windowsGetPreferredColorScheme() orelse .light) {
            .dark => resToErr(
                win32.DwmSetWindowAttribute(hwnd, win32.DWMWA_USE_IMMERSIVE_DARK_MODE, &win32.TRUE, @sizeOf(win32.BOOL)),
                "DwmSetWindowAttribute dark window in initWindow",
            ) catch {},
            .light => {},
        }

        if (options.size) |size| {
            const dpi = win32.GetDpiForWindow(hwnd);
            try boolToErr(@intCast(dpi), "GetDpiForWindow in initWindow");
            const screen_width = win32.GetSystemMetricsForDpi(@intFromEnum(win32.SM_CXSCREEN), dpi);
            const screen_height = win32.GetSystemMetricsForDpi(@intFromEnum(win32.SM_CYSCREEN), dpi);
            var wnd_size: win32.RECT = .{
                .left = 0,
                .top = 0,
                .right = @min(screen_width, @as(i32, @intFromFloat(@round(win32.scaleDpi(f32, size.w, dpi))))),
                .bottom = @min(screen_height, @as(i32, @intFromFloat(@round(win32.scaleDpi(f32, size.h, dpi))))),
            };
            try boolToErr(
                win32.AdjustWindowRectEx(&wnd_size, style, 0, style_ex),
                "AdjustWindowRectEx in initWindow",
            );

            const wnd_width = wnd_size.right - wnd_size.left;
            const wnd_height = wnd_size.bottom - wnd_size.top;
            try boolToErr(win32.SetWindowPos(
                hwnd,
                null,
                @divFloor(screen_width - wnd_width, 2),
                @divFloor(screen_height - wnd_height, 2),
                wnd_width,
                wnd_height,
                win32.SWP_NOCOPYBITS,
            ), "SetWindowPos in initWindow");
        }
        // Returns 0 if the window was previously hidden
        _ = win32.ShowWindow(hwnd, .{ .SHOWNORMAL = 1 });
        try boolToErr(win32.UpdateWindow(hwnd), "UpdateWindow in initWindow");
    }

    pub const ServiceResult = union(enum) {
        queue_empty,
        quit,
    };
    /// Dispatches messages to any/all native OS windows until either the
    /// queue is empty or WM_QUIT/WM_CLOSE are encountered.
    pub fn serviceMessageQueue() ServiceResult {
        var msg: win32.MSG = undefined;
        // https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-peekmessagea#return-value
        while (win32.PeekMessageA(&msg, null, 0, 0, win32.PM_REMOVE) != 0) {
            _ = win32.TranslateMessage(&msg);
            // ignore return value, https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-dispatchmessagew#return-value
            _ = win32.DispatchMessageW(&msg);
            if (msg.message == win32.WM_QUIT) {
                @branchHint(.cold);
                return .quit;
            }
        }
        return .queue_empty;
    }

    fn hwndFromContext(ctx: WindowContext) win32.HWND {
        return @ptrCast(ctx);
    }

    pub fn contextFromHwnd(hwnd: win32.HWND) ?*WindowContext {
        const addr: usize = @bitCast(win32.GetWindowLongPtrW(hwnd, win32.WINDOW_LONG_PTR_INDEX._USERDATA));
        if (addr != 0) return @ptrFromInt(addr) else {
            @branchHint(.unlikely);
            return null;
        }
    }

    pub fn wndProc(
        hwnd: win32.HWND,
        umsg: u32,
        wparam: win32.WPARAM,
        lparam: win32.LPARAM,
    ) callconv(.winapi) win32.LRESULT {
        const maybe_context = contextFromHwnd(hwnd);
        if (maybe_context) |ctx| {
            @branchHint(.likely); // only during init state is not available
            if (handleInputEvents(umsg, wparam, lparam, &ctx.dvui_window))
                return 0;
        }

        switch (umsg) {
            win32.WM_CREATE => {
                slog.debug("WM_CREATE", .{});
                const create_struct: *win32.CREATESTRUCTW = @ptrFromInt(@as(usize, @bitCast(lparam)));
                const args: *CreateWindowArgs = @ptrCast(@alignCast(create_struct.lpCreateParams));

                // attach our data to window
                if (0 != win32.SetWindowLongPtrW(hwnd, win32.WINDOW_LONG_PTR_INDEX._USERDATA, @bitCast(@intFromPtr(args.context))))
                    unreachable;

                const addr: usize = @bitCast(win32.GetWindowLongPtrW(hwnd, win32.WINDOW_LONG_PTR_INDEX._USERDATA));
                if (addr == 0) @panic("unable to attach window state pointer to HWND, did you set cbWndExtra to be >= to @sizeof(usize)?");
                std.debug.assert(contextFromHwnd(hwnd).? == args.context);

                const psize = win.pixelSize(hwnd);
                args.context.last_pixel_size = .{ .w = @floatFromInt(psize[0]), .h = @floatFromInt(psize[1]) };
                args.context.last_window_size = win.windowSize(hwnd, psize);

                return 0;
            },
            win32.WM_DESTROY => {
                slog.debug("WM_DESTROY", .{});
                if (maybe_context) |ctx| {
                    _ = win32.SetWindowLongPtrW(@ptrCast(ctx.hwnd), win32.WINDOW_LONG_PTR_INDEX._USERDATA, 0); // deatach context
                    ctx.backend.vkc.device.queueWaitIdle(ctx.backend.vkc.graphics_queue.handle) catch {};
                    ctx.backend.destroyContext(ctx);
                    if (0 != win32.DestroyWindow(@ptrCast(ctx.hwnd))) slog.err("Failed to close window!", .{});
                }
                return 0;
            },
            // win32.WM_PAINT => {
            //     var ps: win32.PAINTSTRUCT = undefined;
            //     if (win32.BeginPaint(hwnd, &ps) == null) lastErr("BeginPaint") catch return -1;
            //     boolToErr(win32.EndPaint(hwnd, &ps), "EndPaint") catch return -1;
            //     return 0;
            // },
            win32.WM_CLOSE => {
                _ = win32.DefWindowProcW(hwnd, umsg, wparam, lparam);
                return 0;
            },
            win32.WM_WINDOWPOSCHANGED, win32.WM_SIZE => {
                if (contextFromHwnd(hwnd)) |ctx| {
                    const psize = win.pixelSize(hwnd);
                    ctx.last_pixel_size = .{ .w = @floatFromInt(psize[0]), .h = @floatFromInt(psize[1]) };
                    ctx.last_window_size = win.windowSize(hwnd, psize);
                    ctx.recreate_swapchain_requested = true;
                    slog.debug("WM_SIZE {any}", .{psize});
                    // if (ctx.swapchain_state != null) {
                    //     if (dvui.App.get()) |app| {
                    //         paint(app, &g_app_state, ctx, 0) catch |err| {
                    //             ctx.received_close = true;
                    //             slog.warn("paint error during resize: {}", .{err});
                    //         };
                    //     }
                    // }
                } else slog.warn("WM_SIZE: missing hwnd", .{});
                // if (contextFromHwnd(hwnd)) |ctx| {
                //     slog.info("WM_SIZE", .{});
                //     if (ctx.swapchain_state) |*s| {
                //         s.recreate(ctx) catch unreachable;
                //     }
                // }
                return 0;
            },
            else => return win32.DefWindowProcW(hwnd, umsg, wparam, lparam),
        }
    }

    pub fn pixelSize(hwnd: win32.HWND) @Vector(2, i32) {
        var rect: win32.RECT = undefined;
        resToErr(win32.GetClientRect(hwnd, &rect), "GetClientRect in pixelSize") catch unreachable;
        std.debug.assert(rect.left == 0);
        std.debug.assert(rect.top == 0);
        return @Vector(2, i32){ rect.right, rect.bottom };
    }

    pub fn windowSize(hwnd: win32.HWND, pixel_size: @Vector(2, i32)) dvui.Size.Natural {
        // apply dpi scaling manually as there is no convenient api to get the window
        // size of the client size. `win32.GetWindowRect` includes window decorations
        const dpi = win32.GetDpiForWindow(hwnd);
        boolToErr(@intCast(dpi), "GetDpiForWindow in windowSize") catch unreachable;
        const scaling: f32 = 1.0 / win32.scaleFromDpi(f32, dpi);
        return .{
            .w = @as(f32, @floatFromInt(pixel_size[0])) * scaling,
            .h = @as(f32, @floatFromInt(pixel_size[1])) * scaling,
        };
    }

    /// handles wndProc keyboard/mouse etc input events and passes them to dvui
    /// returns true if event was consumed or if defaultEventHandling might be needed (WM_SYSKEYDOWN, WM_SYSKEYUP)
    pub fn handleInputEvents(umsg: u32, wparam: win32.WPARAM, lparam: win32.LPARAM, dvui_window: *dvui.Window) bool {
        switch (umsg) {
            // All mouse events
            win32.WM_LBUTTONDOWN,
            win32.WM_LBUTTONDBLCLK,
            win32.WM_RBUTTONDOWN,
            win32.WM_MBUTTONDOWN,
            win32.WM_XBUTTONDOWN,
            win32.WM_LBUTTONUP,
            win32.WM_RBUTTONUP,
            win32.WM_MBUTTONUP,
            win32.WM_XBUTTONUP,
            => |msg| {
                const button: dvui.enums.Button = switch (msg) {
                    win32.WM_LBUTTONDOWN, win32.WM_LBUTTONDBLCLK, win32.WM_LBUTTONUP => .left,
                    win32.WM_RBUTTONDOWN, win32.WM_RBUTTONUP => .right,
                    win32.WM_MBUTTONDOWN, win32.WM_MBUTTONUP => .middle,
                    win32.WM_XBUTTONDOWN, win32.WM_XBUTTONUP => switch (win32.hiword(wparam)) {
                        0x0001 => .four,
                        0x0002 => .five,
                        else => unreachable,
                    },
                    else => unreachable,
                };
                _ = dvui_window.addEventMouseButton(
                    button,
                    switch (msg) {
                        win32.WM_LBUTTONDOWN, win32.WM_LBUTTONDBLCLK, win32.WM_RBUTTONDOWN, win32.WM_MBUTTONDOWN, win32.WM_XBUTTONDOWN => .press,
                        win32.WM_LBUTTONUP, win32.WM_RBUTTONUP, win32.WM_MBUTTONUP, win32.WM_XBUTTONUP => .release,
                        else => unreachable,
                    },
                ) catch {};
                return true;
            },
            win32.WM_MOUSEMOVE => {
                const x = win32.xFromLparam(lparam);
                const y = win32.yFromLparam(lparam);
                _ = dvui_window.addEventMouseMotion(
                    .{ .pt = .{ .x = @floatFromInt(x), .y = @floatFromInt(y) } },
                ) catch {};
                return true;
            },
            win32.WM_MOUSEWHEEL,
            win32.WM_MOUSEHWHEEL,
            => |msg| {
                const delta: i16 = @bitCast(win32.hiword(wparam));
                const float_delta: f32 = @floatFromInt(delta);
                const wheel_delta: f32 = @floatFromInt(win32.WHEEL_DELTA);
                _ = dvui_window.addEventMouseWheel(
                    float_delta / wheel_delta * dvui.scroll_speed,
                    switch (msg) {
                        win32.WM_MOUSEWHEEL => .vertical,
                        win32.WM_MOUSEHWHEEL => .horizontal,
                        else => unreachable,
                    },
                ) catch {};
                return true;
            },
            // All key events
            win32.WM_KEYUP,
            win32.WM_SYSKEYUP,
            win32.WM_KEYDOWN,
            win32.WM_SYSKEYDOWN,
            => |msg| {
                // https://learn.microsoft.com/en-us/windows/win32/inputdev/about-keyboard-input#keystroke-message-flags
                const KeystrokeMessageFlags = packed struct(u32) {
                    /// The repeat count for the current message. The value is the number of times
                    /// the keystroke is autorepeated as a result of the user holding down the key.
                    /// The repeat count is always 1 for a WM_KEYUP message.
                    repeat_count: u16,
                    /// The scan code. The value depends on the OEM.
                    scan_code: u8,
                    /// Indicates whether the key is an extended key, such as the right-hand ALT
                    /// and CTRL keys that appear on an enhanced 101- or 102-key keyboard. The value
                    /// is 1 if it is an extended key; otherwise, it is 0.
                    is_extended_key: bool,
                    _reserved: u4,
                    /// The context code. The value is always 0 for a WM_KEYUP message.
                    has_alt_down: bool,
                    /// The previous key state. The value is always 1 for a WM_KEYUP message.
                    was_key_down: bool,
                    /// The transition state. The value is always 1 for a WM_KEYUP message.
                    is_key_released: bool,
                };
                const info: KeystrokeMessageFlags = @bitCast(@as(i32, @truncate(lparam)));

                if (std.meta.intToEnum(win32.VIRTUAL_KEY, wparam)) |as_vkey| {
                    // https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getasynckeystate
                    // NOTE: If the key is pressed, the most significant bit is set.
                    //       For a signed integer that means it's a negative number
                    //       if the key is currently down.
                    var mods = dvui.enums.Mod.none;
                    if (win32.GetAsyncKeyState(@intFromEnum(win32.VK_LSHIFT)) < 0) mods.combine(.lshift);
                    if (win32.GetAsyncKeyState(@intFromEnum(win32.VK_RSHIFT)) < 0) mods.combine(.rshift);
                    if (win32.GetAsyncKeyState(@intFromEnum(win32.VK_LCONTROL)) < 0) mods.combine(.lcontrol);
                    if (win32.GetAsyncKeyState(@intFromEnum(win32.VK_RCONTROL)) < 0) mods.combine(.rcontrol);
                    if (win32.GetAsyncKeyState(@intFromEnum(win32.VK_LMENU)) < 0) mods.combine(.lalt);
                    if (win32.GetAsyncKeyState(@intFromEnum(win32.VK_RMENU)) < 0) mods.combine(.ralt);
                    // Command mods would be the windows key, which we do not handle

                    const code = convertVKeyToDvuiKey(as_vkey);

                    _ = dvui_window.addEventKey(.{
                        .code = code,
                        .action = switch (msg) {
                            win32.WM_KEYDOWN, win32.WM_SYSKEYDOWN => if (info.was_key_down) .repeat else .down,
                            win32.WM_KEYUP, win32.WM_SYSKEYUP => .up,
                            else => unreachable,
                        },
                        .mod = mods,
                    }) catch {};
                    // Repeats are counted, so we produce an event for each additional repeat
                    for (1..info.repeat_count) |_| {
                        _ = dvui_window.addEventKey(.{
                            .code = code,
                            .action = .repeat,
                            .mod = mods,
                        }) catch {};
                    }
                } else |err| {
                    slog.err("invalid key found: {}", .{err});
                }
                return switch (msg) {
                    // default expected behaviour:
                    // win32.DefWindowProcW(hwnd, umsg, wparam, lparam)
                    // but unsure what it does
                    win32.WM_SYSKEYDOWN, win32.WM_SYSKEYUP => false,
                    else => true,
                };
            },
            win32.WM_CHAR => {
                const ascii_char: u8 = @truncate(wparam);
                if (std.ascii.isPrint(ascii_char)) {
                    const string: []const u8 = &.{ascii_char};
                    _ = dvui_window.addEventText(.{ .text = string }) catch {};
                }
                return true;
            },
            else => return false,
        }
    }

    pub fn convertVKeyToDvuiKey(vkey: win32.VIRTUAL_KEY) dvui.enums.Key {
        const K = dvui.enums.Key;
        return switch (vkey) {
            .@"0" => .zero,
            .@"1" => .one,
            .@"2" => .two,
            .@"3" => .three,
            .@"4" => .four,
            .@"5" => .five,
            .@"6" => .six,
            .@"7" => .seven,
            .@"8" => .eight,
            .@"9" => .nine,
            .NUMPAD0 => K.kp_0,
            .NUMPAD1 => K.kp_1,
            .NUMPAD2 => K.kp_2,
            .NUMPAD3 => K.kp_3,
            .NUMPAD4 => K.kp_4,
            .NUMPAD5 => K.kp_5,
            .NUMPAD6 => K.kp_6,
            .NUMPAD7 => K.kp_7,
            .NUMPAD8 => K.kp_8,
            .NUMPAD9 => K.kp_9,
            .A => K.a,
            .B => K.b,
            .C => K.c,
            .D => K.d,
            .E => K.e,
            .F => K.f,
            .G => K.g,
            .H => K.h,
            .I => K.i,
            .J => K.j,
            .K => K.k,
            .L => K.l,
            .M => K.m,
            .N => K.n,
            .O => K.o,
            .P => K.p,
            .Q => K.q,
            .R => K.r,
            .S => K.s,
            .T => K.t,
            .U => K.u,
            .V => K.v,
            .W => K.w,
            .X => K.x,
            .Y => K.y,
            .Z => K.z,
            .BACK => K.backspace,
            .TAB => K.tab,
            .RETURN => K.enter,
            .F1 => K.f1,
            .F2 => K.f2,
            .F3 => K.f3,
            .F4 => K.f4,
            .F5 => K.f5,
            .F6 => K.f6,
            .F7 => K.f7,
            .F8 => K.f8,
            .F9 => K.f9,
            .F10 => K.f10,
            .F11 => K.f11,
            .F12 => K.f12,
            .F13 => K.f13,
            .F14 => K.f14,
            .F15 => K.f15,
            .F16 => K.f16,
            .F17 => K.f17,
            .F18 => K.f18,
            .F19 => K.f19,
            .F20 => K.f20,
            .F21 => K.f21,
            .F22 => K.f22,
            .F23 => K.f23,
            .F24 => K.f24,
            .SHIFT, .LSHIFT => K.left_shift,
            .RSHIFT => K.right_shift,
            .CONTROL, .LCONTROL => K.left_control,
            .RCONTROL => K.right_control,
            .MENU => K.menu,
            .PAUSE => K.pause,
            .ESCAPE => K.escape,
            .SPACE => K.space,
            .END => K.end,
            .HOME => K.home,
            .LEFT => K.left,
            .RIGHT => K.right,
            .UP => K.up,
            .DOWN => K.down,
            .PRINT => K.print,
            .INSERT => K.insert,
            .DELETE => K.delete,
            .LWIN => K.left_command,
            .RWIN => K.right_command,
            .PRIOR => K.page_up,
            .NEXT => K.page_down,
            .MULTIPLY => K.kp_multiply,
            .ADD => K.kp_add,
            .SUBTRACT => K.kp_subtract,
            .DIVIDE => K.kp_divide,
            .NUMLOCK => K.num_lock,
            .OEM_1 => K.semicolon,
            .OEM_2 => K.slash,
            .OEM_3 => K.grave,
            .OEM_4 => K.left_bracket,
            .OEM_5 => K.backslash,
            .OEM_6 => K.right_bracket,
            .OEM_7 => K.apostrophe,
            .CAPITAL => K.caps_lock,
            .OEM_PLUS => K.kp_equal,
            .OEM_MINUS => K.minus,
            else => |e| {
                slog.warn("Key {s} not supported.", .{@tagName(e)});
                return K.unknown;
            },
        };
    }
} else void;
