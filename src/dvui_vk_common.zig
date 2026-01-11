const std = @import("std");
const builtin = @import("builtin");
const vk = @import("vk");
const vkk = @import("vk_kickstart");
const dvui = @import("dvui");
const VkRenderer = @import("dvui_vk_renderer.zig");
const slog = std.log.scoped(.dvui_vk_common);

pub const CreateSurfaceCallback = *const fn (window_context: *WindowContext, instance: vk.InstanceProxy) bool;

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
        window_context: *WindowContext,
        createSurface: CreateSurfaceCallback,
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

        if (!createSurface(window_context, instance)) return error.FailedToCreateSurface;

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
                .sampler_anisotropy = .true,
            },
            .required_features_12 = .{
                .descriptor_indexing = .true,
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
            .flags = .{ .reset_command_buffer_bit = true },
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

pub const VkBackend = struct {
    gpa: std.mem.Allocator,
    contexts: std.ArrayListUnmanaged(*WindowContext) = .{},
    contexts_pool: std.heap.MemoryPool(WindowContext),

    vkc: VkContext,
    renderer: ?VkRenderer = null, // dvui renderer
    prev_frame_stats: VkRenderer.Stats = .{},

    pub fn init(gpa: std.mem.Allocator, vkc: VkContext) VkBackend {
        return .{
            .gpa = gpa,
            .contexts_pool = std.heap.MemoryPool(WindowContext).init(gpa),
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
    pub fn allocContext(self: *@This()) !*WindowContext {
        const v = try self.contexts_pool.create();
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
    received_close: bool = false,
    resized: bool = false,

    last_pixel_size: dvui.Size.Physical = .{ .w = 800, .h = 600 },
    last_window_size: dvui.Size.Natural = .{ .w = 800, .h = 600 },

    surface: vk.SurfaceKHR = vk.SurfaceKHR.null_handle,
    swapchain_state: ?SwapchainState = null,

    hwnd: if (builtin.os.tag != .windows) void else *anyopaque, // win32.HWND

    pub const SwapchainState = struct {
        swapchain: vkk.Swapchain,
        images: []vk.Image,
        image_views: []vk.ImageView,
        framebuffers: []vk.Framebuffer = &.{},
        // options for recreate, cached from init or modified before manually recreating swapchain
        options: Options,

        const Options = struct {
            desired_formats: []const vk.SurfaceFormatKHR = &.{},
            desired_present_modes: []vk.PresentModeKHR = &.{},
        };

        pub fn init(ctx: *WindowContext, options: vkk.Swapchain.CreateOptions) !SwapchainState {
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
                .options = .{
                    .desired_formats = try gpa.dupe(vk.SurfaceFormatKHR, options.desired_formats),
                    .desired_present_modes = try gpa.dupe(vk.PresentModeKHR, options.desired_present_modes),
                },
            };
        }

        pub fn deinit(self: *@This(), ctx: *WindowContext) void {
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

            gpa.free(self.options.desired_present_modes);
            gpa.free(self.options.desired_formats);
        }

        pub fn recreate(self: *@This(), ctx: *WindowContext) !void {
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
                    .desired_formats = self.options.desired_formats,
                    .desired_present_modes = self.options.desired_present_modes,
                },
                null,
            );
            slog.debug("recreated swapchain: {}X {}x{} {}", .{ new_swapchain.image_count, new_swapchain.extent.width, new_swapchain.extent.height, new_swapchain.image_format });
            ctx.last_pixel_size.w = @floatFromInt(new_swapchain.extent.width);
            ctx.last_pixel_size.h = @floatFromInt(new_swapchain.extent.height);
            vkc.device.destroySwapchainKHR(old_swapchain.handle, vkc.alloc);

            const options = self.options;
            self.options = .{};
            self.deinit(ctx);
            self.options = options;
            self.images = try gpa.alloc(vk.Image, self.swapchain.image_count);
            self.swapchain = new_swapchain;
            try self.swapchain.getImages(self.images);
            self.image_views = try self.swapchain.getImageViewsAlloc(gpa, self.images, vkc.alloc);
            self.framebuffers = &.{};
        }
    };

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
