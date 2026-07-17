const std = @import("std");
const builtin = @import("builtin");
const vk = @import("vk");
const low = @import("low");
const dvui = @import("dvui");
const VkRenderer = @import("dvui_vk_renderer.zig");
const slog = std.log.scoped(.dvui_vk_common);
const Targets = low.vulkan.targets();
/// global vulkan context
pub const VkContext = struct {
    alloc: ?*vk.AllocationCallbacks = null,
    instance: vk.InstanceProxy,
    physical_device: PhysicalDevice,
    device: vk.DeviceProxy,
    graphics_queue: vk.QueueProxy,
    present_queue: vk.QueueProxy,
    cmd_pool: vk.CommandPool,
    low_loader: ?low.vulkan.Loader = null,
    low_instance: low.vulkan.Instance = undefined,
    low_device: low.vulkan.Device = undefined,
    instance_wrapper: ?*vk.InstanceWrapper = null,
    device_wrapper: ?*vk.DeviceWrapper = null,
    owns_device: bool = true,
    owns_instance: bool = true,

    /// Binding-native physical-device information retained by the backend.
    /// It intentionally contains no allocator-owned selection state.
    pub const PhysicalDevice = struct {
        handle: vk.PhysicalDevice,
        properties: vk.PhysicalDeviceProperties,
        memory_properties: vk.PhysicalDeviceMemoryProperties,
        graphics_queue_index: u32,
        present_queue_index: ?u32,
    };

    /// Resources supplied by an embedding application. The backend creates
    /// only its command pool and never destroys the instance, device, queues,
    /// or dispatch wrappers.
    pub const ExternalDevice = struct {
        instance: vk.InstanceProxy,
        physical_device: PhysicalDevice,
        device: vk.DeviceProxy,
        graphics_queue: vk.Queue,
        present_queue: vk.Queue,
    };

    pub fn deinit(self: *@This(), alloc: std.mem.Allocator) void {
        if (self.owns_device) self.device.deviceWaitIdle() catch {};
        self.device.destroyCommandPool(self.cmd_pool, self.alloc);
        // low's device helper currently creates the device without allocation
        // callbacks, so destruction must use the same allocator contract.
        if (self.owns_device) self.device.destroyDevice(null);
        if (self.device_wrapper) |wrapper| alloc.destroy(wrapper);
        if (self.owns_instance) self.instance.destroyInstance(self.alloc);
        if (self.instance_wrapper) |wrapper| alloc.destroy(wrapper);
        if (self.low_loader) |*loader| loader.deinit();
        self.* = undefined;
    }

    pub const Options = struct {
        /// Shared by the instance, window surfaces, device, and all child
        /// Vulkan objects created by this backend.
        vk_alloc: ?*vk.AllocationCallbacks = null,
        instance_extensions: []const [*:0]const u8 = &.{},
        device_extensions: []const [*:0]const u8 = &.{vk.extensions.khr_swapchain.name},
        required_api_version: vk.Version = vk.API_VERSION_1_3,
        required_features: vk.PhysicalDeviceFeatures = .{},
        required_features_13: ?vk.PhysicalDeviceVulkan13Features = .{
            .synchronization_2 = .true,
            .dynamic_rendering = .true,
        },
        /// use to attach extension feature bits etc. to VkDeviceCreateInfo chain
        device_create_info_p_next: ?*anyopaque = null,
    };

    pub const Instance = struct {
        allocator: std.mem.Allocator,
        wrapper: *vk.InstanceWrapper,
        proxy: vk.InstanceProxy,

        pub fn deinit(self: *@This(), vk_alloc: ?*vk.AllocationCallbacks) void {
            self.proxy.destroyInstance(vk_alloc);
            self.allocator.destroy(self.wrapper);
            self.* = undefined;
        }
    };

    /// Creates an owned, binding-native Vulkan instance. The supplied loader
    /// must remain valid while the resulting instance is in use.
    pub fn createInstance(
        allocator: std.mem.Allocator,
        loader: anytype,
        opt: Options,
    ) !Instance {
        const base = vk.BaseWrapper.load(loader);
        const app_info = vk.ApplicationInfo{
            .p_application_name = "dvui_vk",
            .application_version = 0,
            .p_engine_name = "dvui_vk",
            .engine_version = 0,
            .api_version = @bitCast(opt.required_api_version),
        };
        const wrapper = try allocator.create(vk.InstanceWrapper);
        errdefer allocator.destroy(wrapper);
        const handle = try base.createInstance(&.{
            .p_application_info = &app_info,
            .enabled_extension_count = @intCast(opt.instance_extensions.len),
            .pp_enabled_extension_names = opt.instance_extensions.ptr,
        }, opt.vk_alloc);
        wrapper.* = vk.InstanceWrapper.load(handle, loader);
        return .{ .allocator = allocator, .wrapper = wrapper, .proxy = vk.InstanceProxy.init(handle, wrapper) };
    }

    /// Creates the device and queues for an existing Vulkan instance. The
    /// returned context borrows `instance`; the caller keeps ownership of it.
    pub fn initDevice(
        allocator: std.mem.Allocator,
        instance: vk.InstanceProxy,
        presentation: ?low.vulkan.PresentationSupport,
        opt: Options,
    ) !VkContext {
        // low owns the binding-aware capability checks, queue selection, and
        // device-create feature chain. It returns the same native vulkan-zig
        // handles used by the rest of this backend.
        var low_loader = try low.vulkan.Loader.init();
        errdefer low_loader.deinit();
        const low_instance = try low_loader.loadInstanceApi(low.vulkan.toInstance(instance.handle));
        const selection = try Targets.findDevice(vk, allocator, instance, &low_instance, .{
            .presentation = presentation,
            .requirements = .{
                .required_api_version = opt.required_api_version,
                .required_extensions = opt.device_extensions,
                .required_features = opt.required_features,
                .required_features_13 = opt.required_features_13,
                .extra_features = opt.device_create_info_p_next,
            },
        });
        var device_resources = try Targets.createDevice(vk, allocator, instance, &low_instance, selection);
        errdefer device_resources.deinit();
        const device = device_resources.device;
        const wrapper = device_resources.wrapper;
        const graphics_queue_handle = device_resources.graphics_queue;
        const present_queue_handle = device_resources.present_queue;
        const physical_device = PhysicalDevice{
            .handle = selection.physical_device,
            .properties = instance.getPhysicalDeviceProperties(selection.physical_device),
            .memory_properties = instance.getPhysicalDeviceMemoryProperties(selection.physical_device),
            .graphics_queue_index = selection.graphics_queue_family,
            .present_queue_index = selection.present_queue_family,
        };
        const graphics_queue = vk.QueueProxy.init(graphics_queue_handle, device.wrapper);
        const present_queue = vk.QueueProxy.init(present_queue_handle, device.wrapper);

        const cmd_pool = try device.createCommandPool(&.{
            .flags = .{ .reset_command_buffer_bit = true },
            .queue_family_index = selection.graphics_queue_family,
        }, opt.vk_alloc);

        return .{
            .instance = instance,
            .alloc = opt.vk_alloc,
            .device = device,
            .device_wrapper = wrapper,
            .physical_device = physical_device,
            .graphics_queue = graphics_queue,
            .present_queue = present_queue,
            .cmd_pool = cmd_pool,
            .low_loader = low_loader,
            .low_instance = low_instance,
            .low_device = device_resources.low_device,
            .owns_instance = false,
        };
    }

    pub fn init(
        allocator: std.mem.Allocator,
        loader: anytype,
        presentation: ?low.vulkan.PresentationSupport,
        opt: Options,
    ) !VkContext {
        var instance = try createInstance(allocator, loader, opt);
        errdefer instance.deinit(opt.vk_alloc);

        var context = try initDevice(allocator, instance.proxy, presentation, opt);
        context.owns_instance = true;
        context.instance_wrapper = instance.wrapper;
        return context;
    }

    /// Borrows a host application's Vulkan instance, device, and queues.
    /// The host retains ownership and must keep the proxies and wrappers alive
    /// until this backend has been deinitialized.
    pub fn initExternal(resources: ExternalDevice, vk_alloc: ?*vk.AllocationCallbacks) !VkContext {
        var low_loader = try low.vulkan.Loader.init();
        errdefer low_loader.deinit();
        const low_instance = try low_loader.loadInstanceApi(low.vulkan.toInstance(resources.instance.handle));
        const low_device = try low.vulkan.Device.init(&low_instance, low.vulkan.toDevice(resources.device.handle));
        const cmd_pool = try resources.device.createCommandPool(&.{
            .flags = .{ .reset_command_buffer_bit = true },
            .queue_family_index = resources.physical_device.graphics_queue_index,
        }, vk_alloc);
        return .{
            .alloc = vk_alloc,
            .instance = resources.instance,
            .physical_device = resources.physical_device,
            .device = resources.device,
            .graphics_queue = vk.QueueProxy.init(resources.graphics_queue, resources.device.wrapper),
            .present_queue = vk.QueueProxy.init(resources.present_queue, resources.device.wrapper),
            .cmd_pool = cmd_pool,
            .low_loader = low_loader,
            .low_instance = low_instance,
            .low_device = low_device,
            .owns_device = false,
            .owns_instance = false,
        };
    }
};

/// Shared Vulkan resources used by all low-backed windows.
pub const VkBackend = struct {
    gpa: std.mem.Allocator,
    vk_alloc: ?*vk.AllocationCallbacks = null,
    contexts: std.ArrayListUnmanaged(*WindowContext) = .empty,
    contexts_pool: std.heap.MemoryPool(WindowContext),

    vkc: VkContext,
    renderer: ?VkRenderer = null, // dvui renderer
    prev_frame_stats: VkRenderer.Stats = .{},

    pub fn init(gpa: std.mem.Allocator, vkc: VkContext, vk_alloc: ?*vk.AllocationCallbacks) VkBackend {
        return .{
            .gpa = gpa,
            .vk_alloc = vk_alloc,
            .contexts_pool = .empty,
            .vkc = vkc,
        };
    }

    pub fn deinit(self: *@This()) void {
        for (self.contexts.items) |ctx| ctx.deinit();
        self.contexts.deinit(self.gpa);
        if (self.renderer) |*r| r.deinit(self.gpa);
        self.contexts_pool.deinit(self.gpa);
        self.vkc.deinit(self.gpa);
    }

    /// alloc context without init
    pub fn allocContext(self: *@This()) !*WindowContext {
        const v = try self.contexts_pool.create(self.gpa);
        errdefer self.contexts_pool.destroy(v);
        try self.contexts.append(self.gpa, v);
        return v;
    }

    pub fn destroyContext(self: *@This(), c: *WindowContext) void {
        c.deinit();
        freeContext(self, c);
    }

    pub fn freeContext(self: *@This(), c: *WindowContext) void {
        _ = self.contexts.swapRemove(std.mem.indexOfScalar(*WindowContext, self.contexts.items, c).?);
        self.contexts_pool.destroy(c);
    }
};

/// context links each dvui.window with os window and holds per window vulkan backend state
pub const WindowContext = struct {
    backend: *VkBackend, // kindof unnecessary, for common cases single backend could be just global
    dvui_window: dvui.Window,
    dvui_interrupted: bool = true, // interrupt dvui sleep if its used
    received_close: bool = false,

    last_pixel_size: dvui.Size.Physical = .{ .w = 800, .h = 600 },
    last_window_size: dvui.Size.Natural = .{ .w = 800, .h = 600 },
    last_cursor_x: f64 = 0,
    last_cursor_y: f64 = 0,

    render_context: Targets.RenderContext = undefined,
    render_target: ?Targets.RenderTarget = null,

    arena: std.mem.Allocator = undefined,

    hwnd: if (builtin.os.tag != .windows) void else *anyopaque, // win32.HWND
    low_window: ?*anyopaque = null,
    pub fn deinit(self: *@This()) void {
        // Presentation may be running on a distinct queue.
        self.backend.vkc.device.deviceWaitIdle() catch {};
        if (self.render_target) |*target| target.deinit();
        self.dvui_window.deinit();
        if (@hasDecl(dvui.backend, "deinitWindow")) {
            dvui.backend.deinitWindow(self);
        }
        self.* = undefined;
    }

    pub fn drawStats(self: *@This()) void {
        const stats = self.backend.prev_frame_stats;

        // const overlay = dvui.overlay(@src(), .{ .expand = null, .rect = dvui.windowRect().?, .min_size_content = .{ .w = 300, .h = 300 } });
        // defer overlay.deinit();

        // var m = dvui.box(@src(), .vertical, .{ .background = true, .expand = null, .gravity_y = 0.5, .min_size_content = .{ .w = 300, .h = 0 } });
        // defer m.deinit();
        var prc: f32 = 0; // progress bar percent [0..1]

        const h1 = dvui.Font.theme(.body).larger(-1).withWeight(.bold).withLineHeight(1.1);
        const h2 = dvui.Font.theme(.body).larger(-2).withWeight(.bold).withLineHeight(1.1);
        dvui.labelNoFmt(@src(), "DVUI VK Backend stats", .{}, .{ .font = h1, .expand = .horizontal, .gravity_x = 0.5 });
        dvui.label(@src(), "draw_calls:  {}", .{stats.draw_calls}, .{ .expand = .horizontal });

        const idx_max = self.backend.renderer.?.current_frame.idx_data.len / @sizeOf(VkRenderer.Indice);
        dvui.label(@src(), "indices: {} / {}", .{ stats.indices, idx_max }, .{ .expand = .horizontal });
        prc = @as(f32, @floatFromInt(stats.indices)) / @as(f32, @floatFromInt(idx_max));

        // we modify highlight color to change progress bar inner color
        // TODO: custom progress bar?
        const original_highlight = dvui.themeGet().highlight.fill;
        defer dvui.currentWindow().theme.highlight.fill = original_highlight;

        dvui.currentWindow().theme.highlight.fill = dvui.Color.fromHSLuv(@max(12, (1 - prc * prc) * 155), 99, 50, 100);
        dvui.progress(@src(), .{ .percent = prc }, .{ .expand = .horizontal });

        const verts_max = self.backend.renderer.?.current_frame.vtx_data.len / @sizeOf(VkRenderer.Vertex);
        dvui.label(@src(), "vertices:  {} / {}", .{ stats.verts, verts_max }, .{ .expand = .horizontal });
        prc = @as(f32, @floatFromInt(stats.verts)) / @as(f32, @floatFromInt(verts_max));
        dvui.currentWindow().theme.highlight.fill = dvui.Color.fromHSLuv(@max(12, (1 - prc * prc) * 155), 99, 50, 100);
        dvui.progress(@src(), .{ .percent = prc }, .{ .expand = .horizontal });

        dvui.label(@src(), "Textures:", .{}, .{ .font = h2, .expand = .horizontal });
        dvui.label(@src(), "count:  {}", .{stats.textures_alive}, .{ .expand = .horizontal });
        dvui.label(@src(), "mem (gpu): {Bi:.1}", .{stats.textures_mem}, .{ .expand = .horizontal });

        dvui.label(@src(), "Static/Preallocated memory (gpu):", .{}, .{ .font = h2, .expand = .horizontal });
        const prealloc_mem = self.backend.renderer.?.host_vis_data.len;
        dvui.label(@src(), "total:  {Bi:.1}", .{prealloc_mem}, .{ .expand = .horizontal });
        const prealloc_mem_frame = prealloc_mem / self.backend.renderer.?.frames.len;
        const prealloc_mem_frame_used = stats.indices * @sizeOf(VkRenderer.Indice) + stats.verts * @sizeOf(VkRenderer.Vertex);
        dvui.label(@src(), "current frame:  {Bi:.1} / {Bi:.1}", .{ prealloc_mem_frame_used, prealloc_mem_frame }, .{ .expand = .horizontal });
        prc = @as(f32, @floatFromInt(prealloc_mem_frame_used)) / @as(f32, @floatFromInt(prealloc_mem_frame));
        dvui.currentWindow().theme.highlight.fill = dvui.Color.fromHSLuv(@max(12, (1 - prc * prc) * 155), 99, 50, 100);
        dvui.progress(@src(), .{ .percent = prc }, .{ .expand = .horizontal });
    }
};
pub const AppState = struct {
    backend: *VkBackend,
};

pub fn paint(app: dvui.App, app_state: *AppState, ctx: *WindowContext) !void {
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
        .p_color_attachments = &[_]vk.RenderingAttachmentInfo{.{
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
    b.renderer.?.beginFrame(command_buffer, extent);
    const nstime = ctx.dvui_window.beginWait(ctx.dvui_interrupted);
    try ctx.dvui_window.begin(nstime);
    const res = try app.frameFn();
    const end_micros = try ctx.dvui_window.end(.{});
    if (res != .ok) ctx.received_close = true;
    cmd.endRendering();
    try frame.submitAndPresent(.{});
    if (@hasDecl(dvui.backend, "waitEventTimeout")) {
        const wait_event_micros = ctx.dvui_window.waitTime(end_micros);
        ctx.dvui_interrupted = try dvui.backend.waitEventTimeout(@ptrCast(ctx), wait_event_micros);
    }
}

pub fn openURL(gpa: std.mem.Allocator, url: []const u8) !void {
    // precaution as this runs through shell which can get hairy from security perspective
    if (!std.ascii.startsWithIgnoreCase(url, "http://") and !std.ascii.startsWithIgnoreCase(url, "https://")) {
        return error.BackendError;
    }
    _ = std.Uri.parse(url) catch return error.BackendError; // another security check

    if (builtin.os.tag == .windows) {
        const Win = struct {
            pub extern "shell32" fn ShellExecuteW(
                hwnd: ?std.os.windows.HWND,
                lpOperation: ?[*:0]const u16,
                lpFile: ?[*:0]const u16,
                lpParameters: ?[*:0]const u16,
                lpDirectory: ?[*:0]const u16,
                nShowCmd: i32,
            ) callconv(.winapi) ?std.os.windows.HINSTANCE;
            pub extern "ole32" fn CoInitialize(
                pvReserved: ?*anyopaque,
            ) callconv(.winapi) std.os.windows.LONG;
            pub extern "ole32" fn CoUninitialize() callconv(.winapi) void;
        };

        if (Win.CoInitialize(null) != 0) return error.BackendError;
        defer Win.CoUninitialize();
        const wurl = std.unicode.utf8ToUtf16LeAllocZ(gpa, url) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.BackendError,
        };
        defer gpa.free(wurl);

        const SW_SHOWNORMAL = 1;
        const rc = Win.ShellExecuteW(null, null, @ptrCast(wurl), null, null, SW_SHOWNORMAL);
        if (@intFromPtr(rc) <= 32) {
            slog.err("Failed to open url! rc: {any}; last_err: {}", .{ @intFromPtr(rc), std.os.windows.GetLastError() });
            return error.BackendError;
        }
        return;
    } else if (builtin.os.tag == .linux) {
        const open_cmd = "xdg-open";
        const c_environ = std.c.environ;
        var env_count: usize = 0;
        while (c_environ[env_count] != null) : (env_count += 1) {}
        const environ: [:null]?[*:0]const u8 = @ptrCast(c_environ[0..env_count :null]);
        var io_threaded = std.Io.Threaded.init(gpa, .{
            .environ = .{ .block = .{ .slice = environ } },
        });
        defer io_threaded.deinit();
        const io = io_threaded.io();
        slog.debug("Opening URL with {s}: {s}", .{ open_cmd, url });
        var child = std.process.spawn(io, .{
            .argv = &.{ open_cmd, url },
            .stdin = .ignore,
            .stdout = .ignore,
            .stderr = .ignore,
        }) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => {
                slog.err("Failed to launch {s} for '{s}': {}", .{ open_cmd, url, err });
                return error.BackendError;
            },
        };
        defer child.kill(io);
        const term = child.wait(io) catch |err| {
            slog.err("Failed while waiting for {s} to open '{s}': {}", .{ open_cmd, url, err });
            return error.BackendError;
        };
        switch (term) {
            .exited => |code| if (code != 0) {
                slog.err("{s} failed to open '{s}' with exit code {}", .{ open_cmd, url, code });
                return error.BackendError;
            },
            else => {
                slog.err("{s} failed to open '{s}': {}", .{ open_cmd, url, term });
                return error.BackendError;
            },
        }
        return;
    }
    return error.BackendError;
}
