const std = @import("std");
const builtin = @import("builtin");
const vk = @import("vk");
const vkk = @import("vk_kickstart");
const dvui = @import("dvui");
const VkRenderer = @import("dvui_vk_renderer.zig");
const slog = std.log.scoped(.dvui_vk_common);

pub const CreateSurfaceCallback = *const fn (window_context: *WindowContext, instance: vk.InstanceProxy) bool;

/// global vulkan context
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

    pub const Options = struct {
        instance_settings: vkk.instance.CreateSettings = .{
            .required_api_version = vk.API_VERSION_1_2,
            // .enable_validation = true,
        },
        device_select_settings: vkk.PhysicalDevice.SelectSettings = .{
            .required_api_version = vk.API_VERSION_1_2,
            .required_extensions = &.{
                vk.extensions.khr_swapchain.name,
            },
        },
        /// use to attach extension feature bits etc. to VkDeviceCreateInfo chain
        device_create_info_p_next: ?*anyopaque = null,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        loader: anytype,
        window_context: *WindowContext,
        createSurface: CreateSurfaceCallback,
        opt: Options,
    ) !VkContext {
        const instance = vkk.instance.create(
            allocator,
            loader,
            opt.instance_settings,
            null,
        ) catch blk: {
            slog.err("Failed to get instance! Retrying without validation", .{});
            var instance_settings = opt.instance_settings;
            instance_settings.enable_validation = false;
            break :blk try vkk.instance.create(
                allocator,
                loader,
                instance_settings,
                null,
            );
        };
        errdefer instance.destroyInstance(null);

        // const debug_messenger = try vkk.instance.createDebugMessenger(instance_handle, .{}, null);
        // errdefer vkk.instance.destroyDebugMessenger(instance_handle, debug_messenger, null);

        if (!createSurface(window_context, instance)) return error.FailedToCreateSurface;

        var select_settings = opt.device_select_settings;
        if (select_settings.surface != .null_handle) unreachable;
        select_settings.surface = window_context.surface;
        const physical_device = try vkk.PhysicalDevice.select(allocator, instance, select_settings);

        std.log.info("selected {s}", .{physical_device.name()});

        const device = try vkk.device.create(allocator, instance, &physical_device, @ptrCast(opt.device_create_info_p_next), null);
        errdefer device.destroyDevice(null);

        const graphics_queue_index = physical_device.graphics_queue_index;
        const present_queue_index = physical_device.present_queue_index;
        const graphics_queue_handle = device.getDeviceQueue(graphics_queue_index, 0);
        const present_queue_handle = if (present_queue_index) |pq| device.getDeviceQueue(pq, 0) else graphics_queue_handle;
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
    dvui_interrupted: bool = true, // interrupt dvui sleep if its used
    received_close: bool = false,
    /// requests swapchain recreate with then latest_pixel_size. executed & flag reset on next image is acquire
    recreate_swapchain_requested: bool = false,

    last_pixel_size: dvui.Size.Physical = .{ .w = 800, .h = 600 },
    last_window_size: dvui.Size.Natural = .{ .w = 800, .h = 600 },

    surface: vk.SurfaceKHR = vk.SurfaceKHR.null_handle,
    swapchain_state: ?SwapchainState = null,

    arena: std.mem.Allocator = undefined,

    hwnd: if (builtin.os.tag != .windows) void else *anyopaque, // win32.HWND
    glfw_win: ?*c_long = null,

    pub const SwapchainState = struct {
        pub const max_images = 4;
        swapchain: vkk.Swapchain,
        images: []vk.Image,
        image_views: []vk.ImageView,
        framebuffers: []vk.Framebuffer = &.{},
        // options for recreate, cached from init or modified before manually recreating swapchain
        options: Options,
        current_layout: [max_images]vk.ImageLayout = [_]vk.ImageLayout{.undefined} ** max_images, // used only with dynamic_rendering

        const Options = struct {
            desired_formats: []const vk.SurfaceFormatKHR = &.{},
            desired_present_modes: []vk.PresentModeKHR = &.{},
            msaa: vk.SampleCountFlags = .{ .@"1_bit" = true },
            dynamic_rendering: bool = false, // should not be changed after init
        };

        pub fn init(ctx: *WindowContext, options: vkk.Swapchain.CreateSettings) !SwapchainState {
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
            errdefer vkc.device.destroySwapchainKHR(swapchain.handle, vkc.alloc);
            slog.debug("created swapchain: {}X {}x{} {}", .{ swapchain.image_count, swapchain.extent.width, swapchain.extent.height, swapchain.image_format });
            if (max_images < swapchain.image_count) unreachable;

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

        pub fn initForDynamicRendering(ctx: *WindowContext, options: vkk.Swapchain.CreateSettings) !SwapchainState {
            var s = try SwapchainState.init(ctx, options);
            s.options.dynamic_rendering = true;
            return s;
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
            vkc.device.destroySwapchainKHR(self.swapchain.handle, vkc.alloc);
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
                    .present_queue_index = vkc.physical_device.present_queue_index orelse vkc.physical_device.graphics_queue_index,
                    .desired_extent = extent,
                    .old_swapchain = old_swapchain.handle,
                    .desired_min_image_count = old_swapchain.image_count,
                    .desired_formats = self.options.desired_formats,
                    .desired_present_modes = self.options.desired_present_modes,
                },
                null,
            );
            // slog.debug("recreated swapchain: {}X {}x{} {}", .{ new_swapchain.image_count, new_swapchain.extent.width, new_swapchain.extent.height, new_swapchain.image_format });
            ctx.last_pixel_size.w = @floatFromInt(new_swapchain.extent.width);
            ctx.last_pixel_size.h = @floatFromInt(new_swapchain.extent.height);

            const options = self.options;
            self.options = .{};
            self.deinit(ctx);
            self.options = options;
            self.images = try gpa.alloc(vk.Image, self.swapchain.image_count);
            self.swapchain = new_swapchain;
            try self.swapchain.getImages(self.images);
            self.image_views = try self.swapchain.getImageViewsAlloc(gpa, self.images, vkc.alloc);
            self.framebuffers = &.{};
            self.current_layout = [_]vk.ImageLayout{.undefined} ** max_images;
        }

        pub fn maybeRecreate(swapchain_state: *@This(), ctx: *WindowContext) !bool {
            if (ctx.recreate_swapchain_requested) {
                try swapchain_state.recreate(ctx);
                ctx.recreate_swapchain_requested = false;
                return true;
            }
            return false;
        }

        pub fn maybeCreateFramebuffer(
            swapchain_state: *@This(),
            gpa: std.mem.Allocator,
            ctx: *WindowContext,
            shared_attachments: []const vk.ImageView,
            render_pass: vk.RenderPass,
        ) !bool {
            if (swapchain_state.options.dynamic_rendering) unreachable;
            const vkc = ctx.backend.vkc;
            if (swapchain_state.framebuffers.len == 0) {
                try swapchain_state.createFramebuffers(
                    gpa,
                    vkc,
                    shared_attachments,
                    render_pass,
                );
                return true;
            }
            return false;
        }

        pub fn acquireImageMaybeRecreate(
            swapchain_state: *@This(),
            gpa: std.mem.Allocator,
            ctx: *WindowContext,
            sync: FrameSync,
            shared_attachments: []const vk.ImageView, // for framebuffer recreate
            render_pass: vk.RenderPass, // for framebuffer recreate
        ) !u32 {
            const vkc = ctx.backend.vkc;
            const device = vkc.device;
            const image_index = blk: while (true) {
                _ = try swapchain_state.maybeRecreate(ctx);
                if (!swapchain_state.options.dynamic_rendering) _ = try swapchain_state.maybeCreateFramebuffer(gpa, ctx, shared_attachments, render_pass);

                const next_image_result = device.acquireNextImageKHR(
                    ctx.swapchain_state.?.swapchain.handle,
                    std.math.maxInt(u64),
                    sync.imageAvailableSemaphore(),
                    .null_handle,
                ) catch |err| {
                    if (err == error.OutOfDateKHR) {
                        ctx.recreate_swapchain_requested = true;
                        continue;
                    }
                    return err;
                };
                switch (next_image_result.result) {
                    .success => {},
                    .suboptimal_khr => {
                        ctx.recreate_swapchain_requested = true;
                        // its stil valid to render, lets try to run with it for 1 frame
                        // continue;
                    },
                    else => |err| std.debug.panic("Failed to acquire next frame: {}", .{err}),
                }
                break :blk next_image_result.image_index;
            };
            // slog.debug("acq {}", .{image_index});
            return image_index; // autofix
        }

        fn createFramebuffers(
            swapchain_state: *@This(),
            allocator: std.mem.Allocator,
            vkc: VkContext,
            shared_attachments: []const vk.ImageView,
            render_pass: vk.RenderPass,
        ) !void {
            if (swapchain_state.options.dynamic_rendering) unreachable;
            const device = vkc.device;
            const vk_alloc = vkc.alloc;
            for (swapchain_state.framebuffers) |fb| device.destroyFramebuffer(fb, vk_alloc);
            const extent: vk.Extent2D = swapchain_state.swapchain.extent;
            const image_count: u32 = swapchain_state.swapchain.image_count;
            const image_views: []const vk.ImageView = swapchain_state.image_views;
            var framebuffers = try std.ArrayList(vk.Framebuffer).initCapacity(allocator, image_count);
            errdefer {
                for (framebuffers.items) |framebuffer| {
                    device.destroyFramebuffer(framebuffer, vk_alloc);
                }
                framebuffers.deinit(allocator);
            }

            for (0..image_count) |i| {
                var attachments = [1]vk.ImageView{.null_handle} ** 8;
                attachments[0] = image_views[i];
                std.debug.assert(shared_attachments.len < 7);
                for (shared_attachments, 1..) |sa, si| attachments[si] = sa;
                const framebuffer_info = vk.FramebufferCreateInfo{
                    .render_pass = render_pass,
                    .attachment_count = @intCast(1 + shared_attachments.len),
                    .p_attachments = &attachments,
                    .width = extent.width,
                    .height = extent.height,
                    .layers = 1,
                };

                const framebuffer = try device.createFramebuffer(&framebuffer_info, null);
                framebuffers.appendAssumeCapacity(framebuffer);
            }

            swapchain_state.framebuffers = framebuffers.items;
        }

        pub fn transitionLayout(self: *@This(), cmd: vk.CommandBufferProxy, image_index: usize, layout: vk.ImageLayout) void {
            if (!self.options.dynamic_rendering) unreachable;
            if (layout != .present_src_khr and layout != .color_attachment_optimal) unreachable;
            const current_layout = self.current_layout[image_index];
            if (current_layout != layout) {
                var img_barrier = vk.ImageMemoryBarrier2{
                    .old_layout = current_layout,
                    .new_layout = layout,
                    .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                    .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                    .image = self.images[image_index],
                    .subresource_range = .{
                        .aspect_mask = .{ .color_bit = true },
                        .base_mip_level = 0,
                        .level_count = 1,
                        .base_array_layer = 0,
                        .layer_count = 1,
                    },
                };
                if (current_layout != .undefined) {
                    if (layout == .color_attachment_optimal) {}
                    if (layout == .present_src_khr) {
                        img_barrier.src_access_mask.color_attachment_write_bit = true;
                        img_barrier.src_stage_mask.color_attachment_output_bit = true;
                    }
                }
                cmd.pipelineBarrier2(&.{ .p_image_memory_barriers = @ptrCast(&img_barrier), .image_memory_barrier_count = 1 });
                self.current_layout[image_index] = layout;
            }
        }
    };

    pub fn deinit(self: *@This()) void {
        self.dvui_window.deinit();
        if (self.swapchain_state) |*s| s.deinit(self);
        self.backend.vkc.instance.destroySurfaceKHR(self.surface, self.backend.vkc.alloc);
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

pub fn destroyFramebuffers(allocator: std.mem.Allocator, device: vk.DeviceProxy, framebuffers: []const vk.Framebuffer) void {
    for (framebuffers) |framebuffer| {
        device.destroyFramebuffer(framebuffer, null);
    }
    allocator.free(framebuffers);
}

pub const FrameSync = struct {
    pub const Frame = struct {
        /// Signaled by vkAcquireNextImageKHR
        /// GPU waits on this before starting rendering commands
        /// Ensures the swapchain image is ready for rendering
        image_available: vk.Semaphore = .null_handle,
        /// Signaled when rendering commands finish
        /// Presentation engine waits on this before presenting the image
        /// GPU → GPU synchronization (graphics → present)
        render_finished: vk.Semaphore = .null_handle,
        /// CPU waits for GPU to be able to reuse frame resources
        in_flight_fence: vk.Fence = .null_handle,

        pub fn init(vkc: VkContext) !Frame {
            const device = vkc.device;
            const vk_alloc = vkc.alloc;
            const semaphore_info = vk.SemaphoreCreateInfo{};
            const fence_info = vk.FenceCreateInfo{ .flags = .{ .signaled_bit = true } };
            const image_available = try device.createSemaphore(&semaphore_info, vk_alloc);
            errdefer device.destroySemaphore(image_available, vk_alloc);
            const render_finished = try device.createSemaphore(&semaphore_info, vk_alloc);
            errdefer device.destroySemaphore(render_finished, vk_alloc);
            const in_flight_fence = try device.createFence(&fence_info, vk_alloc);
            errdefer device.destroySemaphore(in_flight_fence, vk_alloc);
            return .{
                .image_available = image_available,
                .render_finished = render_finished,
                .in_flight_fence = in_flight_fence,
            };
        }

        pub fn deinit(it: Frame, device: vk.DeviceProxy) void {
            if (it.image_available != .null_handle) device.destroySemaphore(it.image_available, null);
            if (it.render_finished != .null_handle) device.destroySemaphore(it.render_finished, null);
            if (it.in_flight_fence != .null_handle) device.destroyFence(it.in_flight_fence, null);
        }
    };

    items: []Frame,
    current_frame: u8 = 0,
    frames_done: u56 = 0,

    pub fn init(
        alloc: std.mem.Allocator,
        max_frames: u8,
        vkc: VkContext,
    ) !@This() {
        const items = try alloc.alloc(Frame, max_frames);
        for (items) |*it| it.* = .{};
        errdefer for (items) |it| it.deinit(vkc.device);
        for (items) |*it| it.* = try Frame.init(vkc);
        return .{ .items = items };
    }

    pub fn deinit(sync: @This(), alloc: std.mem.Allocator, device: vk.DeviceProxy) void {
        for (sync.items) |it| it.deinit(device);
        alloc.free(sync.items);
    }

    pub fn inFlightFence(self: @This()) vk.Fence {
        return self.items[self.current_frame].in_flight_fence;
    }

    pub fn imageAvailableSemaphore(self: @This()) vk.Semaphore {
        return self.items[self.current_frame].image_available;
    }

    pub fn renderFinished(self: @This()) vk.Semaphore {
        return self.items[self.current_frame].render_finished;
    }

    pub fn begin(sync: *@This(), device: vk.DeviceProxy) vk.DeviceProxy.WaitForFencesError!void {
        sync.current_frame = @intCast(sync.frames_done % sync.items.len);
        // check/wait for previous frame to finish
        const result = try device.waitForFences(1, @ptrCast(&sync.items[sync.current_frame].in_flight_fence), .true, std.math.maxInt(u64));
        std.debug.assert(result == .success); // no timeout is used, so if result is returned should be successes
    }

    pub fn end(sync: *@This()) void {
        sync.frames_done += 1;
    }
};

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

pub fn present(
    ctx: *WindowContext,
    command_buffer: vk.CommandBuffer,
    sync: FrameSync.Frame,
    swapchain: vk.SwapchainKHR,
    image_index: u32,
) !bool {
    // slog.debug("present {}", .{image_index});
    const vkc = ctx.backend.vkc;
    const wait_semaphores = [_]vk.Semaphore{sync.image_available};
    const wait_stages = [_]vk.PipelineStageFlags{.{ .color_attachment_output_bit = true }};
    const signal_semaphores = [_]vk.Semaphore{sync.render_finished};
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
    switch (present_result) {
        .success => {},
        .suboptimal_khr => ctx.recreate_swapchain_requested = true,
        else => |other| slog.debug("presentKHR result: {}", .{other}),
    }
    return true;
}

pub const AppState = struct {
    backend: *VkBackend,
    render_pass: vk.RenderPass,
    sync: FrameSync,
    command_buffers: []vk.CommandBuffer,
};

pub fn paint(app: dvui.App, app_state: *AppState, ctx: *WindowContext) !void {
    const b = ctx.backend;
    const gpa = b.gpa;
    const render_pass = app_state.render_pass;
    const device = b.vkc.device;

    if (ctx.last_pixel_size.w < 1 or ctx.last_pixel_size.h < 1) return;

    // wait for previous frame to finish
    try app_state.sync.begin(device);
    defer app_state.sync.end();
    const image_index = try ctx.swapchain_state.?.acquireImageMaybeRecreate(gpa, ctx, app_state.sync, &.{}, render_pass);

    const command_buffer = app_state.command_buffers[app_state.sync.current_frame];
    const framebuffer = ctx.swapchain_state.?.framebuffers[image_index];
    try device.beginCommandBuffer(command_buffer, &.{ .flags = .{} });
    const cmd = vk.CommandBufferProxy.init(command_buffer, device.wrapper);

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
    const nstime = ctx.dvui_window.beginWait(ctx.dvui_interrupted);
    // marks the beginning of a frame for dvui, can call dvui functions after thisz
    try ctx.dvui_window.begin(nstime);
    const res = try app.frameFn();
    // marks end of dvui frame, don't call dvui functions after this
    // - sends all dvui stuff to backend for rendering, must be called before renderPresent()
    const end_micros = try ctx.dvui_window.end(.{});

    if (res != .ok) ctx.received_close = true;

    // cursor management
    // TODO: reenable
    // b.setCursor(win.cursorRequested());

    cmd.endRenderPass();
    cmd.endCommandBuffer() catch |err| std.debug.panic("Failed to end vulkan cmd buffer: {}", .{err});

    if (!try present(
        ctx,
        app_state.command_buffers[app_state.sync.current_frame],
        app_state.sync.items[app_state.sync.current_frame],
        ctx.swapchain_state.?.swapchain.handle,
        image_index,
    )) {
        // should_recreate_swapchain = true;
    }

    // sleep when nothing to do
    if (@hasDecl(dvui.backend, "waitEventTimeout")) {
        const wait_event_micros = ctx.dvui_window.waitTime(end_micros);
        ctx.dvui_interrupted = try dvui.backend.waitEventTimeout(@ptrCast(ctx), wait_event_micros);
    }
}

pub fn openURL(arena: std.mem.Allocator, url: []const u8) !void {
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
            ) callconv(.winapi) std.os.windows.HRESULT;
            pub extern "ole32" fn CoUninitialize() callconv(.winapi) void;
        };

        if (Win.CoInitialize(null) != 0) return error.BackendError;
        defer Win.CoUninitialize();
        const wurl = std.unicode.utf8ToUtf16LeAllocZ(arena, url) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.BackendError,
        };
        defer arena.free(wurl);

        const SW_SHOWNORMAL = 1;
        const rc = Win.ShellExecuteW(null, null, @ptrCast(wurl), null, null, SW_SHOWNORMAL);
        if (@intFromPtr(rc) <= 32) {
            slog.err("Failed to open url! rc: {any}; last_err: {}", .{ @intFromPtr(rc), std.os.windows.GetLastError() });
            return error.BackendError;
        }
        return;
    } else if (builtin.os.tag == .linux) {
        // TODO: review, this can block
        const open_cmd = "xdg-open";
        var cmd = std.process.Child.init(&[_][]const u8{ open_cmd, url }, arena);
        const term = cmd.spawnAndWait() catch |err| {
            slog.warn("openURL: failed: {}", .{err});
            return error.BackendError;
        };
        if (term == .Exited and term.Exited == 0) return; // success
    }
    return error.BackendError;
}
