const std = @import("std");
const builtin = @import("builtin");
pub const dvui = @import("dvui");
pub const kind: dvui.enums.Backend = .custom;
const slog = std.log.scoped(.dvu_vk_backend);

pub const max_frames_in_flight = 2;

pub const InitOptions = struct {
    /// [windows os only] A windows class that has previously been registered via RegisterClass.
    registered_class: [*:0]const u16,

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

/// to support multiple windows @This pointer is used as per dvui.window context from
pub const ContextHandle = *@This();
comptime {
    if (@sizeOf(@This()) != 0) unreachable;
}
/// get real context from handle
pub inline fn get(ch: ContextHandle) *Context {
    return @as(*Context, @alignCast(@ptrCast(ch)));
}
// shortcuts
pub inline fn backend(ch: ContextHandle) *VkBackend {
    return ch.get().backend;
}
pub inline fn renderer(ch: ContextHandle) *VkRenderer {
    return &ch.backend().renderer.?;
}

/// context links each dvui.window with os window and holds per window vulkan backend state
pub const Context = struct {
    backend: *VkBackend, // kindof unnecessary, for common cases single backend could be just global
    dvui_window: dvui.Window,
    received_close: bool = false,
    resized: bool = false,

    last_pixel_size: dvui.Size.Physical = .{ .w = 800, .h = 600 },
    last_window_size: dvui.Size.Natural = .{ .w = 800, .h = 600 },

    surface: vk.SurfaceKHR = vk.SurfaceKHR.null_handle,
    swapchain_state: ?SwapchainState = null,

    hwnd: if (builtin.os.tag != .windows) void else win32.HWND,

    pub const SwapchainState = struct {
        swapchain: vkk.Swapchain,
        images: []vk.Image,
        image_views: []vk.ImageView,
        framebuffers: []vk.Framebuffer = &.{},

        pub fn init(ctx: *Context, options: vkk.Swapchain.CreateOptions) !SwapchainState {
            const gpa = ctx.backend.gpa;
            const vkc = ctx.backend.vkc;
            var swapchain = try vkk.Swapchain.create(
                gpa,
                vkc.instance,
                vkc.device,
                vkc.physical_device.handle,
                ctx.surface,
                options,
                null,
            );
            errdefer vkc.device.destroySwapchainKHR(swapchain.handle, null);
            slog.debug("created swapchain: {}X {}x{} {}", .{ swapchain.image_count, swapchain.extent.width, swapchain.extent.height, swapchain.image_format });

            const images: []vk.Image = try gpa.alloc(vk.Image, swapchain.image_count);
            try swapchain.getImages(images);
            const image_views = try swapchain.getImageViewsAlloc(gpa, images, vkc.alloc);
            return .{
                .swapchain = swapchain,
                .images = images,
                .image_views = image_views,
            };
        }

        pub fn deinit(self: *@This(), ctx: *Context) void {
            const gpa = ctx.backend.gpa;
            const vkc = ctx.backend.vkc;
            for (self.framebuffers) |fb| vkc.device.destroyFramebuffer(fb, vkc.alloc);
            gpa.free(self.framebuffers);
            self.framebuffers = &.{};
            for (self.image_views) |view| vkc.device.destroyImageView(view, vkc.alloc);
            gpa.free(self.image_views);
            self.image_views = &.{};
            gpa.free(self.images);
            self.images = &.{};
        }

        pub fn recreate(self: *@This(), ctx: *Context) !void {
            const gpa = ctx.backend.gpa;
            const vkc = ctx.backend.vkc;
            try vkc.device.deviceWaitIdle();
            const extent = vk.Extent2D{
                .width = @max(1, @as(u32, @intFromFloat(ctx.last_pixel_size.w))),
                .height = @max(1, @as(u32, @intFromFloat(ctx.last_pixel_size.h))),
            };
            const old_swapchain = self.swapchain;
            const new_swapchain = try vkk.Swapchain.create(
                gpa,
                vkc.instance,
                vkc.device,
                vkc.physical_device.handle,
                ctx.surface,
                .{
                    .graphics_queue_index = vkc.physical_device.graphics_queue_index,
                    .present_queue_index = vkc.physical_device.present_queue_index,
                    .desired_extent = extent,
                    .old_swapchain = old_swapchain.handle,
                    .desired_min_image_count = old_swapchain.image_count,
                    .desired_formats = &.{.{ .format = old_swapchain.image_format, .color_space = old_swapchain.color_space }},
                },
                null,
            );
            slog.debug("recreated swapchain: {}X {}x{} {}", .{ new_swapchain.image_count, new_swapchain.extent.width, new_swapchain.extent.height, new_swapchain.image_format });
            ctx.last_pixel_size.w = @floatFromInt(new_swapchain.extent.width);
            ctx.last_pixel_size.h = @floatFromInt(new_swapchain.extent.height);
            vkc.device.destroySwapchainKHR(old_swapchain.handle, vkc.alloc);

            self.deinit(ctx);
            self.images = try gpa.alloc(vk.Image, self.swapchain.image_count);
            self.swapchain = new_swapchain;
            try self.swapchain.getImages(self.images);
            self.image_views = try self.swapchain.getImageViewsAlloc(gpa, self.images, vkc.alloc);
            self.framebuffers = &.{};
        }
    };

    pub fn createVkSurface(self: *@This(), vk_instance: vk.InstanceProxy) !void {
        const ci = vk.Win32SurfaceCreateInfoKHR{
            .hwnd = @ptrCast(self.hwnd),
            .hinstance = @ptrCast(win32.GetModuleHandleW(null)),
        };
        self.surface = try vk_instance.createWin32SurfaceKHR(&ci, self.backend.vkc.alloc);
    }

    pub fn deinit(self: *@This()) void {
        self.dvui_window.deinit();
        self.backend.vkc.instance.destroySurfaceKHR(self.surface, self.backend.vkc.alloc);
        if (self.swapchain_state) |*s| s.deinit(self);
        self.* = undefined;
    }

    pub fn drawStats(self: *@This()) void {
        const stats = self.backend.prev_frame_stats;

        // const overlay = dvui.overlay(@src(), .{ .expand = null, .rect = dvui.windowRect().?, .min_size_content = .{ .w = 300, .h = 300 } });
        // defer overlay.deinit();

        // var m = dvui.box(@src(), .vertical, .{ .background = true, .expand = null, .gravity_y = 0.5, .min_size_content = .{ .w = 300, .h = 0 } });
        // defer m.deinit();
        var prc: f32 = 0; // progress bar percent [0..1]

        dvui.labelNoFmt(@src(), "DVUI VK Backend stats", .{}, .{ .expand = .horizontal, .gravity_x = 0.5, .font_style = .heading });
        dvui.label(@src(), "draw_calls:  {}", .{stats.draw_calls}, .{ .expand = .horizontal });

        const idx_max = self.backend.renderer.?.current_frame.idx_data.len / @sizeOf(VkRenderer.Indice);
        dvui.label(@src(), "indices: {} / {}", .{ stats.indices, idx_max }, .{ .expand = .horizontal });
        prc = @as(f32, @floatFromInt(stats.indices)) / @as(f32, @floatFromInt(idx_max));
        dvui.progress(@src(), .{ .percent = prc }, .{ .expand = .horizontal, .color_accent = dvui.Color.fromHSLuv(@max(12, (1 - prc * prc) * 155), 99, 50, 100) });

        const verts_max = self.backend.renderer.?.current_frame.vtx_data.len / @sizeOf(VkRenderer.Vertex);
        dvui.label(@src(), "vertices:  {} / {}", .{ stats.verts, verts_max }, .{ .expand = .horizontal });
        prc = @as(f32, @floatFromInt(stats.verts)) / @as(f32, @floatFromInt(verts_max));
        dvui.progress(@src(), .{ .percent = prc }, .{ .expand = .horizontal, .color_accent = dvui.Color.fromHSLuv(@max(12, (1 - prc * prc) * 155), 99, 50, 100) });

        dvui.label(@src(), "Textures:", .{}, .{ .expand = .horizontal, .font_style = .caption_heading });
        dvui.label(@src(), "count:  {}", .{stats.textures_alive}, .{ .expand = .horizontal });
        dvui.label(@src(), "mem (gpu): {:.1}", .{std.fmt.fmtIntSizeBin(stats.textures_mem)}, .{ .expand = .horizontal });

        dvui.label(@src(), "Static/Preallocated memory (gpu):", .{}, .{ .expand = .horizontal, .font_style = .caption_heading });
        const prealloc_mem = self.backend.renderer.?.host_vis_data.len;
        dvui.label(@src(), "total:  {:.1}", .{std.fmt.fmtIntSizeBin(prealloc_mem)}, .{ .expand = .horizontal });
        const prealloc_mem_frame = prealloc_mem / self.backend.renderer.?.frames.len;
        const prealloc_mem_frame_used = stats.indices * @sizeOf(VkRenderer.Indice) + stats.verts * @sizeOf(VkRenderer.Vertex);
        dvui.label(@src(), "current frame:  {:.1} / {:.1}", .{ std.fmt.fmtIntSizeBin(prealloc_mem_frame_used), std.fmt.fmtIntSizeBin(prealloc_mem_frame) }, .{ .expand = .horizontal });
        prc = @as(f32, @floatFromInt(prealloc_mem_frame_used)) / @as(f32, @floatFromInt(prealloc_mem_frame));
        dvui.progress(@src(), .{ .percent = prc }, .{ .expand = .horizontal, .color_accent = dvui.Color.fromHSLuv(@max(12, (1 - prc * prc) * 155), 99, 50, 100) });
    }
};

pub const VkRenderer = @import("dvui_vulkan_renderer.zig");
pub const VkBackend = struct {
    gpa: std.mem.Allocator,
    contexts: std.ArrayListUnmanaged(*Context) = .{},
    contexts_pool: std.heap.MemoryPool(Context),

    vkc: VkContext,
    renderer: ?VkRenderer = null, // dvui renderer
    prev_frame_stats: VkRenderer.Stats = .{},

    pub fn init(gpa: std.mem.Allocator, vkc: VkContext) VkBackend {
        return .{
            .gpa = gpa,
            .contexts_pool = std.heap.MemoryPool(Context).init(gpa),
            .vkc = vkc,
        };
    }

    pub fn deinit(self: *@This()) void {
        for (self.contexts.items) |ctx| ctx.deinit();
        self.contexts.deinit(self.gpa);
        if (self.renderer) |*r| r.deinit(self.gpa);
        self.contexts_pool.deinit();
        self.vkc.deinit(self.gpa);
    }

    /// alloc context without init
    pub fn allocContext(self: *@This()) !*Context {
        const v = try self.contexts_pool.create();
        errdefer self.contexts_pool.destroy(v);
        try self.contexts.append(self.gpa, v);
        return v;
    }

    pub fn destroyContext(self: *@This(), c: *Context) void {
        c.deinit();
        freeContext(self, c);
    }

    pub fn freeContext(self: *@This(), c: *Context) void {
        _ = self.contexts.swapRemove(std.mem.indexOfScalar(*Context, self.contexts.items, c).?);
        self.contexts_pool.destroy(c);
    }
};

pub const VkContext = struct {
    alloc: ?*vk.AllocationCallbacks = null,
    instance: vk.InstanceProxy,
    physical_device: vkk.PhysicalDevice,
    device: vk.DeviceProxy,
    graphics_queue: vk.QueueProxy,
    present_queue: vk.QueueProxy,
    cmd_pool: vk.CommandPool,

    pub fn deinit(self: VkContext, alloc: std.mem.Allocator) void {
        self.physical_device.deinit();
        alloc.destroy(self.instance.wrapper);
        alloc.destroy(self.device.wrapper);
    }

    pub fn init(
        allocator: std.mem.Allocator,
        loader: anytype,
        window_context: *Context,
    ) !VkContext {
        const instance = try vkk.instance.create(
            allocator,
            loader,
            .{ .required_api_version = @bitCast(vk.API_VERSION_1_3) },
            null,
        );
        errdefer instance.destroyInstance(null);

        // const debug_messenger = try vkk.instance.createDebugMessenger(instance_handle, .{}, null);
        // errdefer vkk.instance.destroyDebugMessenger(instance_handle, debug_messenger, null);

        try window_context.createVkSurface(instance);

        const physical_device = try vkk.PhysicalDevice.select(allocator, instance, .{
            .surface = window_context.surface,
            .transfer_queue = .dedicated,
            .required_api_version = @bitCast(vk.API_VERSION_1_2),
            .required_extensions = &.{
                // vk.extensions.khr_ray_tracing_pipeline.name,
                // vk.extensions.khr_acceleration_structure.name,
                // vk.extensions.khr_deferred_host_operations.name,
                // vk.extensions.khr_buffer_device_address.name,
                // vk.extensions.ext_descriptor_indexing.name,
            },
            .required_features = .{
                .sampler_anisotropy = vk.TRUE,
            },
            .required_features_12 = .{
                .descriptor_indexing = vk.TRUE,
            },
        });

        std.log.info("selected {s}", .{physical_device.name()});

        var features = vk.PhysicalDeviceRayTracingPipelineFeaturesKHR{};

        const device = try vkk.device.create(allocator, instance, &physical_device, @ptrCast(&features), null);
        errdefer device.destroyDevice(null);

        const graphics_queue_index = physical_device.graphics_queue_index;
        const present_queue_index = physical_device.present_queue_index;
        const graphics_queue_handle = device.getDeviceQueue(graphics_queue_index, 0);
        const present_queue_handle = device.getDeviceQueue(present_queue_index, 0);
        const graphics_queue = vk.QueueProxy.init(graphics_queue_handle, device.wrapper);
        const present_queue = vk.QueueProxy.init(present_queue_handle, device.wrapper);

        const cmd_pool = try device.createCommandPool(&.{
            .queue_family_index = graphics_queue_index,
        }, null);
        errdefer device.destroyCommandPool(cmd_pool, null);

        return .{
            .instance = instance,
            // .debug_messenger = debug_messenger,
            .device = device,
            .physical_device = physical_device,
            .graphics_queue = graphics_queue,
            .present_queue = present_queue,
            .cmd_pool = cmd_pool,
        };
    }
};

pub const vk = @import("vk");

pub const GenericError = dvui.Backend.GenericError;
pub const TextureError = dvui.Backend.TextureError;

const is_windows = @import("builtin").target.os.tag == .windows;
// pub const dvui_win = if (is_windows) @import("dvui_win") else void;
// pub const win32 = if (is_windows) dvui_win.win32 else void;
pub const win32 = @import("win32").everything;

pub fn dvuiBackend(context: *Context) dvui.Backend {
    return dvui.Backend.init(@alignCast(@ptrCast(context)));
}

//
//   Dvui backend implementation
//

/// Get monotonic nanosecond timestamp. Doesn't have to be system time.
pub fn nanoTime(_: ContextHandle) i128 {
    return std.time.nanoTimestamp();
}

pub fn sleep(_: ContextHandle, ns: u64) void {
    std.time.sleep(ns);
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
pub fn textureCreateTarget(self: ContextHandle, width: u32, height: u32, interpolation: dvui.enums.TextureInterpolation) TextureError!dvui.TextureTarget {
    _ = self; // autofix
    _ = width; // autofix
    _ = height; // autofix
    _ = interpolation; // autofix
    return TextureError.NotImplemented;
}

/// Read pixel data (RGBA) from `texture` into `pixels_out`.
pub fn textureReadTarget(self: ContextHandle, texture: dvui.TextureTarget, pixels_out: [*]u8) TextureError!void {
    _ = self; // autofix
    _ = texture; // autofix
    _ = pixels_out; // autofix
    return TextureError.NotImplemented;
}

/// Convert texture target made with `textureCreateTarget` into return texture
/// as if made by `textureCreate`.  After this call, texture target will not be
/// used by dvui.
pub fn textureFromTarget(self: ContextHandle, texture: dvui.TextureTarget) TextureError!dvui.Texture {
    _ = self; // autofix
    _ = texture; // autofix
    return TextureError.NotImplemented;
}

/// Render future `drawClippedTriangles` to the passed `texture` (or screen
/// if null).
pub fn renderTarget(self: ContextHandle, texture: ?dvui.TextureTarget) GenericError!void {
    _ = self; // autofix
    _ = texture; // autofix
    return error.BackendError;
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
pub fn openURL(self: ContextHandle, url: []const u8) GenericError!void {
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

pub fn hasEvent() bool {
    return false; // TODO:
}

pub const AppState = struct {
    backend: *VkBackend,
    render_pass: vk.RenderPass,
    sync: SyncObjects,
    command_buffers: []vk.CommandBuffer,
};
pub var g_app_state: AppState = undefined;

pub fn main() !void {
    dvui.Backend.Common.windowsAttachConsole() catch {};

    const app = dvui.App.get() orelse return error.DvuiAppNotDefined;

    var gpa_instance = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = gpa_instance.allocator();
    defer _ = gpa_instance.deinit();

    const window_class = win32.L("DvuiWindow");
    win.RegisterClass(window_class, .{}) catch win32.panicWin32(
        "RegisterClass",
        win32.GetLastError(),
    );

    const init_opts = app.config.get();

    vk_dll.init() catch |err| std.debug.panic("Failed to init Vulkan: {}", .{err});
    defer vk_dll.deinit();
    const loader = vk_dll.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse @panic("Failed to lookup Vulkan VkGetInstanceProcAddr: {}");

    var b = VkBackend.init(gpa, undefined);
    defer b.deinit();

    // init backend (creates and owns OS window)
    var window_context: *Context = try b.allocContext();
    window_context.* = .{
        .backend = &b,
        .dvui_window = try dvui.Window.init(@src(), gpa, dvuiBackend(window_context), .{}),
        .hwnd = undefined,
    };
    try win.initWindow(window_context, .{
        .registered_class = window_class,
        .dvui_gpa = gpa,
        .gpa = gpa,
        .size = init_opts.size,
        .title = init_opts.title,
        .icon = init_opts.icon,
    });

    b.vkc = try VkContext.init(gpa, loader, window_context);

    window_context.swapchain_state = try Context.SwapchainState.init(window_context, .{
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
        .desired_present_modes = if (!init_opts.vsync) &.{.immediate_khr} else &.{.fifo_khr},
    });

    const render_pass = try createRenderPass(b.vkc.device, window_context.swapchain_state.?.swapchain.image_format);
    defer b.vkc.device.destroyRenderPass(render_pass, null);

    const sync = try SyncObjects.init(b.vkc.device);
    defer sync.deinit(b.vkc.device);

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
                for (b.contexts.items, 0..) |ctx, ctx_i| {
                    _ = ctx_i; // autofix
                    // slog.info("frame {} ctx {}", .{ current_frame_in_flight, ctx_i });
                    try paint(app, g_app_state, ctx, current_frame_in_flight);
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

pub fn paint(app: dvui.App, app_state: AppState, ctx: *Context, current_frame_in_flight: usize) !void {
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
            ctx.swapchain_state.?.framebuffers = try createFramebuffers(
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
    const nstime = ctx.dvui_window.beginWait(hasEvent());

    // marks the beginning of a frame for dvui, can call dvui functions after thisz
    try ctx.dvui_window.begin(nstime);
    const res = try app.frameFn();
    // marks end of dvui frame, don't call dvui functions after this
    // - sends all dvui stuff to backend for rendering, must be called before renderPresent()
    _ = try ctx.dvui_window.end(.{});

    if (res != .ok) ctx.received_close = true;

    // cursor management
    // TODO: reenable
    // b.setCursor(win.cursorRequested());

    const frame_sync_objects = FrameSyncObjects{
        .image_available_semaphore = sync.image_available_semaphores[current_frame_in_flight],
        .render_finished_semaphore = sync.render_finished_semaphores[current_frame_in_flight],
        .in_flight_fence = sync.in_flight_fences[current_frame_in_flight],
    };
    cmd.endRenderPass();
    cmd.endCommandBuffer() catch |err| std.debug.panic("Failed to end vulkan cmd buffer: {}", .{err});

    if (!try present(
        ctx,
        app_state.command_buffers[current_frame_in_flight],
        frame_sync_objects,
        ctx.swapchain_state.?.swapchain.handle,
        image_index,
    )) {
        // should_recreate_swapchain = true;
    }
}

pub fn createRenderPass(device: vk.DeviceProxy, image_format: vk.Format) !vk.RenderPass {
    const color_attachment = vk.AttachmentDescription{
        .format = image_format,
        .samples = .{ .@"1_bit" = true },
        .load_op = .clear,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .present_src_khr,
    };

    const color_attachment_refs = [_]vk.AttachmentReference{.{
        .attachment = 0,
        .layout = .color_attachment_optimal,
    }};

    const subpasses = [_]vk.SubpassDescription{.{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = color_attachment_refs.len,
        .p_color_attachments = &color_attachment_refs,
    }};

    const dependencies = [_]vk.SubpassDependency{.{
        .src_subpass = vk.SUBPASS_EXTERNAL,
        .dst_subpass = 0,
        .src_stage_mask = .{ .color_attachment_output_bit = true, .early_fragment_tests_bit = true },
        .src_access_mask = .{},
        .dst_stage_mask = .{ .color_attachment_output_bit = true, .early_fragment_tests_bit = true },
        .dst_access_mask = .{ .color_attachment_write_bit = true },
    }};

    const attachments = [_]vk.AttachmentDescription{color_attachment};
    const renderpass_info = vk.RenderPassCreateInfo{
        .attachment_count = attachments.len,
        .p_attachments = &attachments,
        .subpass_count = subpasses.len,
        .p_subpasses = &subpasses,
        .dependency_count = dependencies.len,
        .p_dependencies = &dependencies,
    };

    return device.createRenderPass(&renderpass_info, null);
}

pub fn createFramebuffers(
    allocator: std.mem.Allocator,
    device: vk.DeviceProxy,
    extent: vk.Extent2D,
    image_count: u32,
    image_views: []vk.ImageView,
    render_pass: vk.RenderPass,
) ![]vk.Framebuffer {
    var framebuffers = try std.ArrayList(vk.Framebuffer).initCapacity(allocator, image_count);
    errdefer {
        for (framebuffers.items) |framebuffer| {
            device.destroyFramebuffer(framebuffer, null);
        }
        framebuffers.deinit();
    }

    for (0..image_count) |i| {
        const attachments = [_]vk.ImageView{image_views[i]};
        const framebuffer_info = vk.FramebufferCreateInfo{
            .render_pass = render_pass,
            .attachment_count = attachments.len,
            .p_attachments = &attachments,
            .width = extent.width,
            .height = extent.height,
            .layers = 1,
        };

        const framebuffer = try device.createFramebuffer(&framebuffer_info, null);
        try framebuffers.append(framebuffer);
    }

    return framebuffers.toOwnedSlice();
}

pub fn destroyFramebuffers(allocator: std.mem.Allocator, device: vk.DeviceProxy, framebuffers: []const vk.Framebuffer) void {
    for (framebuffers) |framebuffer| {
        device.destroyFramebuffer(framebuffer, null);
    }
    allocator.free(framebuffers);
}

pub const SyncObjects = struct {
    image_available_semaphores: [max_frames_in_flight]vk.Semaphore,
    render_finished_semaphores: [max_frames_in_flight]vk.Semaphore,
    in_flight_fences: [max_frames_in_flight]vk.Fence,

    pub fn init(device: vk.DeviceProxy) !SyncObjects {
        var image_available_semaphores = [_]vk.Semaphore{.null_handle} ** max_frames_in_flight;
        var render_finished_semaphores = [_]vk.Semaphore{.null_handle} ** max_frames_in_flight;
        var in_flight_fences = [_]vk.Fence{.null_handle} ** max_frames_in_flight;
        errdefer {
            for (image_available_semaphores) |semaphore| {
                if (semaphore == .null_handle) continue;
                device.destroySemaphore(semaphore, null);
            }
            for (render_finished_semaphores) |semaphore| {
                if (semaphore == .null_handle) continue;
                device.destroySemaphore(semaphore, null);
            }
            for (in_flight_fences) |fence| {
                if (fence == .null_handle) continue;
                device.destroyFence(fence, null);
            }
        }

        const semaphore_info = vk.SemaphoreCreateInfo{};
        const fence_info = vk.FenceCreateInfo{ .flags = .{ .signaled_bit = true } };
        for (0..max_frames_in_flight) |i| {
            image_available_semaphores[i] = try device.createSemaphore(&semaphore_info, null);
            render_finished_semaphores[i] = try device.createSemaphore(&semaphore_info, null);
            in_flight_fences[i] = try device.createFence(&fence_info, null);
        }

        return .{
            .image_available_semaphores = image_available_semaphores,
            .render_finished_semaphores = render_finished_semaphores,
            .in_flight_fences = in_flight_fences,
        };
    }

    pub fn deinit(sync: SyncObjects, device: vk.DeviceProxy) void {
        for (sync.image_available_semaphores) |semaphore| {
            device.destroySemaphore(semaphore, null);
        }
        for (sync.render_finished_semaphores) |semaphore| {
            device.destroySemaphore(semaphore, null);
        }
        for (sync.in_flight_fences) |fence| {
            device.destroyFence(fence, null);
        }
    }
};

pub fn createCommandPool(device: vk.DeviceProxy, queue_family_index: u32) !vk.CommandPool {
    const create_info = vk.CommandPoolCreateInfo{
        .flags = .{ .reset_command_buffer_bit = true },
        .queue_family_index = queue_family_index,
    };
    return device.createCommandPool(&create_info, null);
}

pub fn createCommandBuffers(
    allocator: std.mem.Allocator,
    device: vk.DeviceProxy,
    command_pool: vk.CommandPool,
    count: u32,
) ![]vk.CommandBuffer {
    const command_buffers = try allocator.alloc(vk.CommandBuffer, count);
    errdefer allocator.free(command_buffers);

    const command_buffer_info = vk.CommandBufferAllocateInfo{
        .command_pool = command_pool,
        .level = .primary,
        .command_buffer_count = count,
    };
    try device.allocateCommandBuffers(&command_buffer_info, command_buffers.ptr);

    return command_buffers;
}

pub const FrameSyncObjects = struct {
    image_available_semaphore: vk.Semaphore,
    render_finished_semaphore: vk.Semaphore,
    in_flight_fence: vk.Fence,
};

pub fn present(
    ctx: *const Context,
    command_buffer: vk.CommandBuffer,
    sync: FrameSyncObjects,
    swapchain: vk.SwapchainKHR,
    image_index: u32,
) !bool {
    const vkc = ctx.backend.vkc;
    const wait_semaphores = [_]vk.Semaphore{sync.image_available_semaphore};
    const wait_stages = [_]vk.PipelineStageFlags{.{ .color_attachment_output_bit = true }};
    const signal_semaphores = [_]vk.Semaphore{sync.render_finished_semaphore};
    const command_buffers = [_]vk.CommandBuffer{command_buffer};
    const submit_info = vk.SubmitInfo{
        .wait_semaphore_count = wait_semaphores.len,
        .p_wait_semaphores = &wait_semaphores,
        .p_wait_dst_stage_mask = &wait_stages,
        .command_buffer_count = command_buffers.len,
        .p_command_buffers = &command_buffers,
        .signal_semaphore_count = signal_semaphores.len,
        .p_signal_semaphores = &signal_semaphores,
    };

    const fences = [_]vk.Fence{sync.in_flight_fence};
    try vkc.device.resetFences(fences.len, &fences);

    const submits = [_]vk.SubmitInfo{submit_info};
    try vkc.graphics_queue.submit(submits.len, &submits, sync.in_flight_fence);

    const indices = [_]u32{image_index};
    const swapchains = [_]vk.SwapchainKHR{swapchain};
    const present_info = vk.PresentInfoKHR{
        .wait_semaphore_count = signal_semaphores.len,
        .p_wait_semaphores = &signal_semaphores,
        .swapchain_count = swapchains.len,
        .p_swapchains = &swapchains,
        .p_image_indices = &indices,
    };

    const present_result = vkc.present_queue.presentKHR(&present_info) catch |err| {
        if (err == error.OutOfDateKHR) {
            return false;
        }
        return err;
    };

    if (present_result == .suboptimal_khr) {
        return false;
    }

    return true;
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
        slog.err("{s} failed, error={}", .{ what, err });
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
        context: *Context,
        dvui_gpa: std.mem.Allocator,
        err: ?anyerror = null,
    };

    pub fn initWindow(context: *Context, options: InitOptions) !void {
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
                options.registered_class,
                wnd_title,
                style,
                win32.CW_USEDEFAULT, // x
                win32.CW_USEDEFAULT, // y
                win32.CW_USEDEFAULT, // w
                win32.CW_USEDEFAULT, // h
                null, // hWndParent
                null, // hMenu
                win32.GetModuleHandleW(null), // This message is sent to the created window by this function before it returns.
                @constCast(@ptrCast(&create_args)),
            ) orelse switch (win32.GetLastError()) {
                win32.ERROR_CANNOT_FIND_WND_CLASS => switch (builtin.mode) {
                    .Debug => std.debug.panic(
                        "did you forget to call RegisterClass? (class_name='{}')",
                        .{std.unicode.fmtUtf16Le(std.mem.span(options.registered_class))},
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

    fn hwndFromContext(ctx: Context) win32.HWND {
        return @ptrCast(ctx);
    }

    pub fn contextFromHwnd(hwnd: win32.HWND) ?*Context {
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
    ) callconv(std.os.windows.WINAPI) win32.LRESULT {
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
                const args: *CreateWindowArgs = @alignCast(@ptrCast(create_struct.lpCreateParams));

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
                    _ = win32.SetWindowLongPtrW(ctx.hwnd, win32.WINDOW_LONG_PTR_INDEX._USERDATA, 0); // deatach context
                    ctx.backend.destroyContext(ctx);
                    if (0 != win32.DestroyWindow(ctx.hwnd)) slog.err("Failed to close window!", .{});
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
                    if (ctx.swapchain_state != null) {
                        if (dvui.App.get()) |app| {
                            paint(app, g_app_state, ctx, 0) catch |err| {
                                ctx.received_close = true;
                                slog.warn("paint error during resize: {}", .{err});
                            };
                        }
                    }
                }
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
                    .{ .x = @floatFromInt(x), .y = @floatFromInt(y) },
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
                    _ = dvui_window.addEventText(string) catch {};
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
