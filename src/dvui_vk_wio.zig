//! DVUI Vulkan backend using ypsvlq/wio for the native window and input layer.
//!
//! The low backend remains available in dvui_vk_low.zig. This backend uses
//! wio's platform-neutral Vulkan surface API and owns the swapchain bridge.

const std = @import("std");
const dvui = @import("dvui");
pub const wio = @import("wio");
pub const vk = @import("vk");
pub const dvui_vk_common = @import("dvui_vk_common.zig");
pub const VkRenderer = @import("dvui_vk_renderer.zig");
const slog = std.log.scoped(.dvu_vk_wio);
comptime {
    _ = @import("dvui_c.zig");
}

pub const kind: dvui.enums.Backend = .custom;
pub const VkContext = dvui_vk_common.VkContext;
pub const ExternalDevice = VkContext.ExternalDevice;
pub const VulkanSource = union(enum) {
    create: vk.PfnGetInstanceProcAddr,
    external: ExternalDevice,
};
pub const VkBackend = dvui_vk_common.VkBackend;
pub const AppState = dvui_vk_common.AppState;
pub const GenericError = dvui.Backend.GenericError;
pub const TextureError = dvui.Backend.TextureError;
pub const max_frames_in_flight = 2;

pub const InitOptions = struct {
    dvui_gpa: std.mem.Allocator,
    dvui_window_init_options: dvui.Window.InitOptions = .{},
    gpa: std.mem.Allocator,
    io: std.Io,
    size: ?dvui.Size = null,
    title: [:0]const u8,
    vsync: bool,
    vk_alloc: ?*vk.AllocationCallbacks = null,
    vulkan: VulkanSource,
    max_frames_in_flight: u8 = max_frames_in_flight,
    color_formats: []const vk.Format = &.{
        .a2b10g10r10_unorm_pack32,
        .a2r10g10b10_unorm_pack32,
        .b8g8r8a8_unorm,
    },
};

pub const WindowContext = struct {
    backend: *VkBackend,
    dvui_window: dvui.Window,
    wio_window: wio.Window,
    events: wio.EventQueue = .empty,
    swapchain: WioSwapchain,
    arena: std.mem.Allocator = undefined,
    io: std.Io,
    size_natural: dvui.Size.Natural,
    size_physical: dvui.Size.Physical,
    scale: f32 = 1,
    modifiers: dvui.enums.Mod = .none,
    received_close: bool = false,
    dvui_interrupted: bool = true,

    pub fn deinit(self: *@This()) void {
        self.backend.vkc.device.deviceWaitIdle() catch {};
        self.swapchain.deinit();
        self.dvui_window.deinit();
        self.wio_window.destroy();
        self.events.deinit();
        self.* = undefined;
    }

    pub fn drawStats(_: *@This()) void {}
};

pub const InitWindowResult = struct {
    backend: *VkBackend,
    window: *WindowContext,

    pub fn deinit(self: @This()) void {
        const gpa = self.backend.gpa;
        self.window.deinit();
        gpa.destroy(self.window);
        self.backend.deinit();
        gpa.destroy(self.backend);
        wio.deinit();
    }
};

pub const ContextHandle = *@This();
pub inline fn get(ch: ContextHandle) *WindowContext {
    return @as(*WindowContext, @ptrCast(@alignCast(ch)));
}
pub inline fn backend(ch: ContextHandle) *VkBackend {
    return get(ch).backend;
}
pub inline fn renderer(ch: ContextHandle) *VkRenderer {
    return &get(ch).backend.renderer.?;
}
pub fn dvuiBackend(context: *WindowContext) dvui.Backend {
    return dvui.Backend.init(@ptrCast(@alignCast(context)));
}

pub const WioSwapchain = struct {
    const Self = @This();
    const Slot = struct {
        command_buffer: vk.CommandBuffer,
        image_available: vk.Semaphore,
        fence: vk.Fence,
    };

    allocator: std.mem.Allocator,
    command_pool: vk.CommandPool,
    device: vk.DeviceProxy,
    instance: vk.InstanceProxy,
    physical_device: vk.PhysicalDevice,
    graphics_queue: vk.QueueProxy,
    present_queue: vk.QueueProxy,
    graphics_queue_family: u32,
    present_queue_family: u32,
    surface: vk.SurfaceKHR,
    vk_alloc: ?*vk.AllocationCallbacks,
    color_formats: []vk.Format,
    vsync: bool,
    extent: vk.Extent2D = .{ .width = 1, .height = 1 },
    color_format: vk.Format = .b8g8r8a8_unorm,
    color_space: vk.ColorSpaceKHR = .srgb_nonlinear_khr,
    swapchain: vk.SwapchainKHR = .null_handle,
    images: []vk.Image = &.{},
    image_views: []vk.ImageView = &.{},
    image_fences: []vk.Fence = &.{},
    image_layouts: []vk.ImageLayout = &.{},
    render_finished: []vk.Semaphore = &.{},
    slots: []Slot = &.{},
    slot_index: usize = 0,
    frames_in_flight: u8 = 2,
    needs_recreate: bool = false,

    pub const Frame = struct {
        target: *Self,
        slot_index: usize,
        image_index: u32,
        command_buffer: vk.CommandBuffer,
        image_view: vk.ImageView,
        extent: vk.Extent2D,
        finished: bool = false,

        pub fn submitAndPresent(self: *@This()) !void {
            if (self.finished) return error.FrameAlreadyFinished;
            const target = self.target;
            const slot = &target.slots[self.slot_index];
            target.transitionForPresent(self.command_buffer, self.image_index);
            try target.device.endCommandBuffer(self.command_buffer);
            try target.device.resetFences(&.{slot.fence});
            try target.device.queueSubmit(target.graphics_queue.handle, &.{.{
                .wait_semaphore_count = 1,
                .p_wait_semaphores = &.{slot.image_available},
                .p_wait_dst_stage_mask = &.{.{ .color_attachment_output_bit = true }},
                .command_buffer_count = 1,
                .p_command_buffers = &.{self.command_buffer},
                .signal_semaphore_count = 1,
                .p_signal_semaphores = &.{target.render_finished[self.image_index]},
            }}, slot.fence);
            const present_result = try target.device.queuePresentKHR(target.present_queue.handle, &.{
                .wait_semaphore_count = 1,
                .p_wait_semaphores = &.{target.render_finished[self.image_index]},
                .swapchain_count = 1,
                .p_swapchains = &.{target.swapchain},
                .p_image_indices = &.{self.image_index},
            });
            target.image_fences[self.image_index] = slot.fence;
            target.slot_index = (self.slot_index + 1) % target.slots.len;
            self.finished = true;
            if (present_result == .suboptimal_khr) {
                target.needs_recreate = true;
                return error.FrameOutOfDate;
            }
        }

        pub fn abort(self: *@This()) void {
            self.finished = true;
        }
    };

    pub fn init(
        allocator: std.mem.Allocator,
        vkc: *const VkContext,
        surface: vk.SurfaceKHR,
        size: vk.Extent2D,
        color_formats: []const vk.Format,
        vsync: bool,
        frames_in_flight: u8,
        vk_alloc: ?*vk.AllocationCallbacks,
    ) !Self {
        var self = Self{
            .allocator = allocator,
            .command_pool = vkc.cmd_pool,
            .device = vkc.device,
            .instance = vkc.instance,
            .physical_device = vkc.physical_device.handle,
            .graphics_queue = vkc.graphics_queue,
            .present_queue = vkc.present_queue,
            .graphics_queue_family = vkc.physical_device.graphics_queue_index,
            .present_queue_family = vkc.physical_device.present_queue_index orelse vkc.physical_device.graphics_queue_index,
            .surface = surface,
            .vk_alloc = vk_alloc,
            .color_formats = &.{},
            .vsync = vsync,
            .frames_in_flight = frames_in_flight,
        };
        errdefer self.deinit();
        self.color_formats = try allocator.dupe(vk.Format, color_formats);
        try self.create(size, frames_in_flight);
        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.swapchain != .null_handle) self.device.deviceWaitIdle() catch {};
        self.destroySlots();
        self.destroySwapchain();
        if (self.color_formats.len != 0) self.allocator.free(self.color_formats);
        self.instance.destroySurfaceKHR(self.surface, self.vk_alloc);
        self.* = undefined;
    }

    fn destroySlots(self: *Self) void {
        for (self.slots) |slot| {
            if (slot.image_available != .null_handle) self.device.destroySemaphore(slot.image_available, self.vk_alloc);
            if (slot.fence != .null_handle) self.device.destroyFence(slot.fence, self.vk_alloc);
            if (slot.command_buffer != .null_handle) self.device.freeCommandBuffers(self.command_pool, &.{slot.command_buffer});
        }
        if (self.slots.len != 0) self.allocator.free(self.slots);
        self.slots = &.{};
    }

    fn destroySwapchain(self: *Self) void {
        for (self.image_views) |view| if (view != .null_handle) self.device.destroyImageView(view, self.vk_alloc);
        for (self.render_finished) |semaphore| if (semaphore != .null_handle) self.device.destroySemaphore(semaphore, self.vk_alloc);
        if (self.image_views.len != 0) self.allocator.free(self.image_views);
        if (self.images.len != 0) self.allocator.free(self.images);
        if (self.image_fences.len != 0) self.allocator.free(self.image_fences);
        if (self.image_layouts.len != 0) self.allocator.free(self.image_layouts);
        if (self.render_finished.len != 0) self.allocator.free(self.render_finished);
        if (self.swapchain != .null_handle) self.device.destroySwapchainKHR(self.swapchain, self.vk_alloc);
        self.image_views = &.{};
        self.images = &.{};
        self.image_fences = &.{};
        self.image_layouts = &.{};
        self.render_finished = &.{};
        self.swapchain = .null_handle;
    }

    fn create(self: *Self, requested_extent: vk.Extent2D, frames_in_flight: u8) !void {
        const capabilities = try self.instance.getPhysicalDeviceSurfaceCapabilitiesKHR(self.physical_device, self.surface);
        self.extent = if (capabilities.current_extent.width != std.math.maxInt(u32)) capabilities.current_extent else .{
            .width = std.math.clamp(requested_extent.width, capabilities.min_image_extent.width, capabilities.max_image_extent.width),
            .height = std.math.clamp(requested_extent.height, capabilities.min_image_extent.height, capabilities.max_image_extent.height),
        };
        const formats = try self.instance.getPhysicalDeviceSurfaceFormatsAllocKHR(self.physical_device, self.surface, self.allocator);
        defer self.allocator.free(formats);
        const selected_format = chooseColorFormat(formats, self.color_formats);
        self.color_format = selected_format.format;
        self.color_space = selected_format.color_space;
        const present_modes = try self.instance.getPhysicalDeviceSurfacePresentModesAllocKHR(self.physical_device, self.surface, self.allocator);
        defer self.allocator.free(present_modes);
        const present_mode = choosePresentMode(present_modes, self.vsync);
        const image_count = if (capabilities.max_image_count != 0) @min(capabilities.min_image_count + 1, capabilities.max_image_count) else capabilities.min_image_count + 1;
        self.swapchain = try self.device.createSwapchainKHR(&.{
            .surface = self.surface,
            .min_image_count = image_count,
            .image_format = self.color_format,
            .image_color_space = self.color_space,
            .image_extent = self.extent,
            .image_array_layers = 1,
            .image_usage = .{ .color_attachment_bit = true },
            .image_sharing_mode = if (self.graphics_queue_family == self.present_queue_family) .exclusive else .concurrent,
            .queue_family_index_count = if (self.graphics_queue_family == self.present_queue_family) 0 else 2,
            .p_queue_family_indices = if (self.graphics_queue_family == self.present_queue_family) null else &.{ self.graphics_queue_family, self.present_queue_family },
            .pre_transform = if (capabilities.supported_transforms.identity_bit_khr) .{ .identity_bit_khr = true } else .{ .inherit_bit_khr = true },
            .composite_alpha = if (capabilities.supported_composite_alpha.opaque_bit_khr) .{ .opaque_bit_khr = true } else .{ .inherit_bit_khr = true },
            .present_mode = present_mode,
            .clipped = .true,
        }, self.vk_alloc);
        errdefer self.destroySwapchain();
        self.images = try self.device.getSwapchainImagesAllocKHR(self.swapchain, self.allocator);
        self.image_views = try self.allocator.alloc(vk.ImageView, self.images.len);
        @memset(self.image_views, .null_handle);
        for (self.images, self.image_views) |image, *view| {
            view.* = try self.device.createImageView(&.{
                .image = image,
                .view_type = .@"2d",
                .format = self.color_format,
                .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
                .subresource_range = .{ .aspect_mask = .{ .color_bit = true }, .base_mip_level = 0, .level_count = 1, .base_array_layer = 0, .layer_count = 1 },
            }, self.vk_alloc);
        }
        self.image_fences = try self.allocator.alloc(vk.Fence, self.images.len);
        @memset(self.image_fences, .null_handle);
        self.image_layouts = try self.allocator.alloc(vk.ImageLayout, self.images.len);
        @memset(self.image_layouts, .undefined);
        self.render_finished = try self.allocator.alloc(vk.Semaphore, self.images.len);
        @memset(self.render_finished, .null_handle);
        for (self.render_finished) |*semaphore| semaphore.* = try self.device.createSemaphore(&.{}, self.vk_alloc);
        self.slots = try self.allocator.alloc(Slot, @max(@as(usize, 1), @as(usize, frames_in_flight)));
        for (self.slots) |*slot| {
            slot.* = .{
                .command_buffer = .null_handle,
                .image_available = .null_handle,
                .fence = .null_handle,
            };
            slot.image_available = try self.device.createSemaphore(&.{}, self.vk_alloc);
            slot.fence = try self.device.createFence(&.{ .flags = .{ .signaled_bit = true } }, self.vk_alloc);
            try self.device.allocateCommandBuffers(&.{ .command_pool = self.command_pool, .level = .primary, .command_buffer_count = 1 }, @ptrCast(&slot.command_buffer));
        }
    }

    pub fn recreate(self: *Self, requested_extent: vk.Extent2D) !void {
        if (requested_extent.width == 0 or requested_extent.height == 0) return;
        try self.device.deviceWaitIdle();
        self.destroySlots();
        self.destroySwapchain();
        try self.create(requested_extent, self.frames_in_flight);
        self.needs_recreate = false;
    }

    pub fn acquire(self: *Self) !Frame {
        const slot_index = self.slot_index;
        const slot = &self.slots[slot_index];
        _ = try self.device.waitForFences(&.{slot.fence}, .true, std.math.maxInt(u64));
        const acquired = try self.device.acquireNextImageKHR(self.swapchain, std.math.maxInt(u64), slot.image_available, .null_handle);
        const image_index = acquired.image_index;
        if (self.image_fences[image_index] != .null_handle) _ = try self.device.waitForFences(&.{self.image_fences[image_index]}, .true, std.math.maxInt(u64));
        try self.device.resetCommandBuffer(slot.command_buffer, .{});
        try self.device.beginCommandBuffer(slot.command_buffer, &.{ .flags = .{ .one_time_submit_bit = true } });
        self.transitionForRendering(slot.command_buffer, image_index);
        return .{ .target = self, .slot_index = slot_index, .image_index = image_index, .command_buffer = slot.command_buffer, .image_view = self.image_views[image_index], .extent = self.extent };
    }

    fn transitionForRendering(self: *Self, command_buffer: vk.CommandBuffer, image_index: u32) void {
        const barrier = vk.ImageMemoryBarrier{
            .src_access_mask = .{},
            .dst_access_mask = .{ .color_attachment_read_bit = true, .color_attachment_write_bit = true },
            .old_layout = self.image_layouts[image_index],
            .new_layout = .color_attachment_optimal,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = self.images[image_index],
            .subresource_range = .{ .aspect_mask = .{ .color_bit = true }, .base_mip_level = 0, .level_count = 1, .base_array_layer = 0, .layer_count = 1 },
        };
        self.device.cmdPipelineBarrier(command_buffer, .{ .color_attachment_output_bit = true }, .{ .color_attachment_output_bit = true }, .{}, null, null, &.{barrier});
        self.image_layouts[image_index] = .color_attachment_optimal;
    }

    fn transitionForPresent(self: *Self, command_buffer: vk.CommandBuffer, image_index: u32) void {
        const barrier = vk.ImageMemoryBarrier{
            .src_access_mask = .{ .color_attachment_write_bit = true },
            .dst_access_mask = .{},
            .old_layout = .color_attachment_optimal,
            .new_layout = .present_src_khr,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = self.images[image_index],
            .subresource_range = .{ .aspect_mask = .{ .color_bit = true }, .base_mip_level = 0, .level_count = 1, .base_array_layer = 0, .layer_count = 1 },
        };
        self.device.cmdPipelineBarrier(command_buffer, .{ .color_attachment_output_bit = true }, .{ .bottom_of_pipe_bit = true }, .{}, null, null, &.{barrier});
        self.image_layouts[image_index] = .present_src_khr;
    }
};

fn chooseColorFormat(available: []const vk.SurfaceFormatKHR, preferred: []const vk.Format) vk.SurfaceFormatKHR {
    for (preferred) |format| for (available) |item| if (item.format == format) return item;
    return available[0];
}

fn choosePresentMode(available: []const vk.PresentModeKHR, vsync: bool) vk.PresentModeKHR {
    if (!vsync) for (available) |mode| if (mode == .mailbox_khr) return mode;
    if (!vsync) for (available) |mode| if (mode == .immediate_khr) return mode;
    return .fifo_khr;
}

pub fn initWindow(options: InitOptions) !InitWindowResult {
    const b = try options.dvui_gpa.create(VkBackend);
    errdefer options.dvui_gpa.destroy(b);
    b.* = VkBackend.init(options.dvui_gpa, undefined, options.vk_alloc);
    try wio.init(.{ .allocator = options.gpa, .io = options.io, .eventFn = wio.EventQueue.eventFn });
    errdefer wio.deinit();

    const initial: dvui.Size = options.size orelse dvui.Size{ .w = 1280, .h = 720 };
    const context = try options.dvui_gpa.create(WindowContext);
    errdefer options.dvui_gpa.destroy(context);
    context.* = .{
        .backend = undefined,
        .dvui_window = undefined,
        .wio_window = undefined,
        .swapchain = undefined,
        .io = options.io,
        .size_natural = .{ .w = initial.w, .h = initial.h },
        .size_physical = .{ .w = initial.w, .h = initial.h },
    };
    context.wio_window = try wio.Window.create(.{
        .event_fn_data = &context.events,
        .title = options.title,
        .size = .{ .width = @max(1, @as(u16, @intFromFloat(@round(initial.w)))), .height = @max(1, @as(u16, @intFromFloat(@round(initial.h)))) },
        .scale = 1,
    });
    errdefer context.wio_window.destroy();

    var instance_resources: ?VkContext.Instance = null;
    errdefer if (instance_resources) |*resources| resources.deinit(options.vk_alloc);
    var vkc_initialized = false;
    errdefer if (vkc_initialized) b.vkc.deinit(options.gpa);
    var surface_instance: vk.InstanceProxy = undefined;
    var ext_dynamic_state_features = vk.PhysicalDeviceExtendedDynamicStateFeaturesEXT{ .extended_dynamic_state = .true };
    const surface: vk.SurfaceKHR = switch (options.vulkan) {
        .create => |loader| blk: {
            instance_resources = try VkContext.createInstance(options.gpa, loader, .{
                .vk_alloc = options.vk_alloc,
                .instance_extensions = wio.getRequiredVulkanInstanceExtensions(),
                .device_extensions = &.{ vk.extensions.khr_swapchain.name, vk.extensions.ext_extended_dynamic_state.name },
                .device_create_info_p_next = @ptrCast(&ext_dynamic_state_features),
            });
            surface_instance = instance_resources.?.proxy;
            var handle: u64 = 0;
            try context.wio_window.vkCreateSurface(@intFromEnum(instance_resources.?.proxy.handle), options.vk_alloc, &handle);
            break :blk @enumFromInt(handle);
        },
        .external => |resources| blk: {
            b.vkc = try VkContext.initExternal(resources, options.vk_alloc);
            vkc_initialized = true;
            surface_instance = b.vkc.instance;
            var handle: u64 = 0;
            try context.wio_window.vkCreateSurface(@intFromEnum(b.vkc.instance.handle), options.vk_alloc, &handle);
            break :blk @enumFromInt(handle);
        },
    };
    var surface_transferred = false;
    errdefer if (!surface_transferred and surface != .null_handle) surface_instance.destroySurfaceKHR(surface, options.vk_alloc);
    if (options.vulkan == .create) {
        const resources = instance_resources.?;
        b.vkc = try VkContext.initDeviceForSurface(options.gpa, resources.proxy, surface, .{
            .vk_alloc = options.vk_alloc,
            .device_extensions = &.{ vk.extensions.khr_swapchain.name, vk.extensions.ext_extended_dynamic_state.name },
            .device_create_info_p_next = @ptrCast(&ext_dynamic_state_features),
        });
        b.vkc.instance_ownership = .{ .owned = resources.wrapper };
        instance_resources = null;
        vkc_initialized = true;
    }

    context.backend = b;
    // WioSwapchain takes ownership of the surface even when its initialization fails.
    surface_transferred = true;
    context.swapchain = try WioSwapchain.init(options.gpa, &b.vkc, surface, .{ .width = @intFromFloat(@round(initial.w)), .height = @intFromFloat(@round(initial.h)) }, options.color_formats, options.vsync, options.max_frames_in_flight, options.vk_alloc);
    errdefer context.swapchain.deinit();
    context.dvui_window = try dvui.Window.init(@src(), options.dvui_gpa, dvuiBackend(context), options.dvui_window_init_options);
    errdefer context.dvui_window.deinit();
    b.renderer = try VkRenderer.init(options.gpa, .{
        .dev = b.vkc.device,
        .graphics_queue_family_index = b.vkc.physical_device.graphics_queue_index,
        .submit_prepass = .{ .userdata = b, .submit = VkBackend.submitRendererPrepass },
        .gpu_allocator = .{ .builtin = VkRenderer.VkMemory.init(b.vkc.physical_device.memory_properties, b.vkc.physical_device.properties.limits.non_coherent_atom_size) orelse return error.OutOfMemory },
        .render_pass = .{ .dynamic = .{ .info = .{ .view_mask = 0, .color_attachment_count = 1, .p_color_attachment_formats = &.{context.swapchain.color_format}, .depth_attachment_format = .undefined, .stencil_attachment_format = .undefined } } },
        .max_frames_in_flight = options.max_frames_in_flight,
        .vk_alloc = options.vk_alloc,
    });
    return .{ .backend = b, .window = context };
}

pub fn begin(ch: ContextHandle, arena: std.mem.Allocator) !void {
    get(ch).arena = arena;
    if (renderer(ch).frame_active) renderer(ch).begin(arena, pixelSize(ch));
}
pub fn end(ch: ContextHandle) !void {
    if (renderer(ch).frame_active) renderer(ch).end();
    get(ch).arena = undefined;
}
pub fn nanoTime(ch: ContextHandle) i128 {
    return std.Io.Clock.awake.now(get(ch).io).nanoseconds;
}
pub fn sleep(ch: ContextHandle, ns: u64) void {
    std.Io.Clock.Duration.sleep(.{ .clock = .awake, .raw = .fromNanoseconds(ns) }, get(ch).io) catch {};
}
pub fn pixelSize(ch: ContextHandle) dvui.Size.Physical {
    return get(ch).size_physical;
}
pub fn windowSize(ch: ContextHandle) dvui.Size.Natural {
    return get(ch).size_natural;
}
pub fn contentScale(ch: ContextHandle) f32 {
    const context = get(ch);
    return if (context.size_natural.w == context.size_physical.w and context.size_natural.h == context.size_physical.h) context.scale else 1;
}
pub fn drawClippedTriangles(ch: ContextHandle, texture: ?dvui.Texture, vtx: []const dvui.Vertex, idx: []const u16, clipr: ?dvui.Rect.Physical) GenericError!void {
    return renderer(ch).drawClippedTriangles(texture, vtx, idx, clipr);
}
pub fn textureCreate(ch: ContextHandle, pixels: [*]const u8, options: dvui.Texture.CreateOptions) TextureError!dvui.Texture {
    return renderer(ch).textureCreate(pixels, options);
}
pub fn textureDestroy(ch: ContextHandle, texture: dvui.Texture) void {
    renderer(ch).textureDestroy(texture);
}
pub fn textureCreateTarget(ch: ContextHandle, options: dvui.Texture.CreateOptions) TextureError!dvui.TextureTarget {
    return renderer(ch).textureCreateTarget(options);
}
pub fn textureReadTarget(ch: ContextHandle, texture: dvui.TextureTarget, pixels: [*]u8) TextureError!void {
    return renderer(ch).textureReadTarget(texture, pixels);
}
pub fn textureClearTarget(ch: ContextHandle, texture: dvui.Texture.Target) void {
    renderer(ch).renderTarget(texture) catch {};
    renderer(ch).renderTarget(null) catch {};
}
pub fn textureDestroyTarget(ch: ContextHandle, texture: dvui.Texture.Target) void {
    renderer(ch).textureDestroy(dvui.Texture.cast(texture));
}
pub fn textureFromTarget(ch: ContextHandle, texture: dvui.TextureTarget) TextureError!dvui.Texture {
    return renderer(ch).textureFromTarget(texture);
}
pub fn textureFromTargetTemp(ch: ContextHandle, texture: dvui.TextureTarget) TextureError!dvui.Texture {
    return renderer(ch).textureFromTarget(texture);
}
pub fn renderTarget(ch: ContextHandle, texture: ?dvui.TextureTarget) GenericError!void {
    return renderer(ch).renderTarget(texture);
}
pub fn clipboardText(ch: ContextHandle) GenericError![]const u8 {
    const context = get(ch);
    return context.wio_window.getClipboardText(context.arena) orelse "";
}
pub fn clipboardTextSet(ch: ContextHandle, text: []const u8) GenericError!void {
    get(ch).wio_window.setClipboardText(text);
}
pub fn openURL(_: ContextHandle, url: []const u8, _: bool) GenericError!void {
    wio.openUri(url);
}
pub fn preferredColorScheme(_: ContextHandle) ?dvui.enums.ColorScheme {
    return null;
}
pub fn prefersReducedMotion(_: ContextHandle) bool {
    return false;
}
pub fn cursorShow(_: ContextHandle, _: ?bool) GenericError!bool {
    return true;
}
pub fn title(ch: ContextHandle, _: *dvui.Window, new_title: []const u8) void {
    get(ch).wio_window.setTitle(new_title);
}
pub fn windowStateSet(ch: ContextHandle, _: *dvui.Window, state: dvui.enums.WindowState) void {
    get(ch).wio_window.setMode(switch (state) {
        .normal => .normal,
        .maximize => .maximized,
        .fullscreen => .fullscreen,
    });
}
pub fn setCursor(ch: ContextHandle, cursor: dvui.enums.Cursor) void {
    get(ch).wio_window.setCursor(switch (cursor) {
        .arrow => .default,
        .ibeam => .text,
        .wait => .wait,
        .wait_arrow => .progress,
        .crosshair => .crosshair,
        .arrow_nw_se => .nwse_resize,
        .arrow_ne_sw => .nesw_resize,
        .arrow_w_e => .ew_resize,
        .arrow_n_s => .ns_resize,
        .arrow_all => .all_scroll,
        .bad => .not_allowed,
        .hand => .pointer,
        .hidden => .none,
    });
}
pub fn textInputRect(ch: ContextHandle, rect: ?dvui.Rect.Natural) void {
    if (rect) |r| get(ch).wio_window.enableTextInput(.{ .cursor = .{ .x = std.math.lossyCast(u16, r.x), .y = std.math.lossyCast(u16, r.y) } }) else get(ch).wio_window.disableTextInput();
}
pub fn renderPresent(_: ContextHandle) void {}
pub fn deinit(_: ContextHandle) void {}
pub fn refresh(_: ContextHandle) void {
    wio.cancelWait();
}

pub fn waitEventTimeout(ch: ContextHandle, timeout_us: u32) !bool {
    if (timeout_us == 0) return false;
    if (timeout_us == std.math.maxInt(u32)) wio.wait(.{}) else wio.wait(.{ .timeout_ns = @as(u64, timeout_us) * std.time.ns_per_us });
    pollEvents(get(ch));
    return true;
}

pub fn pollEvents(ch: *WindowContext) void {
    wio.update();
    while (ch.events.pop()) |event| processEvent(ch, event) catch |err| slog.err("wio event failed: {}", .{err});
}

fn processEvent(ch: *WindowContext, event: wio.Event) !void {
    switch (event) {
        .close => {
            ch.received_close = true;
            _ = try ch.dvui_window.addEventWindow(.{ .action = .close });
        },
        .size_logical => |size| ch.size_natural = .{ .w = size.width, .h = size.height },
        .size_physical => |size| {
            ch.size_physical = .{ .w = size.width, .h = size.height };
            ch.swapchain.needs_recreate = true;
        },
        .scale => |scale| ch.scale = scale,
        .modifiers => |mods| {
            ch.modifiers = .none;
            if (mods.shift) ch.modifiers.combine(.lshift);
            if (mods.control) ch.modifiers.combine(.lcontrol);
            if (mods.alt) ch.modifiers.combine(.lalt);
            if (mods.gui) ch.modifiers.combine(.lcommand);
        },
        .char => |char| {
            var bytes: [4]u8 = undefined;
            const len = try std.unicode.utf8Encode(char, &bytes);
            _ = try ch.dvui_window.addEventText(.{ .text = bytes[0..len] });
        },
        .button_press => |button| try addButtonEvent(ch, button, .down),
        .button_release => |button| try addButtonEvent(ch, button, .up),
        .button_repeat => |button| {
            _ = try ch.dvui_window.addEventKey(.{ .code = buttonToDvuiKey(button), .action = .repeat, .mod = ch.modifiers });
        },
        .mouse => |mouse| {
            const sx = if (ch.size_natural.w > 0) ch.size_physical.w / ch.size_natural.w else 1;
            const sy = if (ch.size_natural.h > 0) ch.size_physical.h / ch.size_natural.h else sx;
            _ = try ch.dvui_window.addEventMouseMotion(.{ .pt = .{ .x = mouse.x * sx, .y = mouse.y * sy } });
        },
        .scroll_vertical => |value| _ = try ch.dvui_window.addEventMouseWheel(-value * dvui.scroll_speed, .vertical, null),
        .scroll_horizontal => |value| _ = try ch.dvui_window.addEventMouseWheel(-value * dvui.scroll_speed, .horizontal, null),
        else => {},
    }
}

fn addButtonEvent(ch: *WindowContext, button: wio.Button, action: @FieldType(dvui.Event.Key, "action")) !void {
    switch (button) {
        .mouse_left, .mouse_right, .mouse_middle, .mouse_back, .mouse_forward => |mouse| {
            const mapped: dvui.enums.Button = switch (mouse) {
                .mouse_left => .left,
                .mouse_right => .right,
                .mouse_middle => .middle,
                .mouse_back => .four,
                .mouse_forward => .five,
                else => unreachable,
            };
            _ = try ch.dvui_window.addEventMouseButton(mapped, if (action == .down) .press else .release);
        },
        else => _ = try ch.dvui_window.addEventKey(.{ .code = buttonToDvuiKey(button), .action = action, .mod = ch.modifiers }),
    }
}

fn buttonToDvuiKey(button: wio.Button) dvui.enums.Key {
    return switch (button) {
        .mouse_left, .mouse_right, .mouse_middle, .mouse_back, .mouse_forward => .unknown,
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
        .dot => .period,
        .left_gui => .left_command,
        .right_gui => .right_command,
        else => std.meta.stringToEnum(dvui.enums.Key, @tagName(button)) orelse .unknown,
    };
}

pub fn paint(app: dvui.App, app_state: *AppState, ch: *WindowContext) !void {
    if (ch.size_physical.w < 1 or ch.size_physical.h < 1) return;
    if (ch.swapchain.needs_recreate) try ch.swapchain.recreate(.{ .width = @intFromFloat(@round(ch.size_physical.w)), .height = @intFromFloat(@round(ch.size_physical.h)) });
    var frame = ch.swapchain.acquire() catch {
        ch.swapchain.needs_recreate = true;
        return;
    };
    defer frame.abort();
    const cmd = vk.CommandBufferProxy.init(frame.command_buffer, ch.backend.vkc.device.wrapper);
    cmd.beginRendering(&.{ .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = frame.extent }, .view_mask = 0, .layer_count = 1, .color_attachment_count = 1, .p_color_attachments = &[_]vk.RenderingAttachmentInfo{.{ .image_view = frame.image_view, .image_layout = .color_attachment_optimal, .resolve_mode = .{}, .resolve_image_view = .null_handle, .resolve_image_layout = .undefined, .load_op = .clear, .store_op = .store, .clear_value = .{ .color = .{ .float_32 = .{ 0.1, 0.1, 0.1, 1 } } } }} });
    try ch.backend.renderer.?.beginFrame(frame.command_buffer, frame.extent);
    const nstime = ch.dvui_window.beginWait(ch.dvui_interrupted);
    try ch.dvui_window.begin(nstime);
    const result = try app.frameFn();
    const end_micros = try ch.dvui_window.end(.{});
    _ = try ch.backend.renderer.?.endFrame();
    cmd.endRendering();
    frame.submitAndPresent() catch {
        ch.swapchain.needs_recreate = true;
        return;
    };
    if (result != .ok) ch.received_close = true;
    if (@hasDecl(dvui.backend, "waitEventTimeout")) ch.dvui_interrupted = try waitEventTimeout(@ptrCast(@alignCast(ch)), ch.dvui_window.waitTime(end_micros));
    _ = app_state;
}

pub fn main() !void {
    const app = dvui.App.get() orelse return error.DvuiAppNotDefined;
    var gpa_instance = std.heap.DebugAllocator(.{}){};
    const gpa = gpa_instance.allocator();
    defer _ = gpa_instance.deinit();
    var threaded: std.Io.Threaded = .init(gpa, .{});
    const io = threaded.io();
    const loader: vk.PfnGetInstanceProcAddr = &wioGetInstanceProcAddr;
    const config = app.config.get();
    const iw = try initWindow(.{ .dvui_gpa = gpa, .gpa = gpa, .io = io, .title = config.title, .size = config.size, .vsync = config.vsync, .vulkan = .{ .create = loader }, .dvui_window_init_options = config.window_init_options });
    defer iw.deinit();
    var app_state = AppState{ .backend = iw.backend };
    if (app.initFn) |init_fn| {
        try iw.window.dvui_window.begin(iw.window.dvui_window.frame_time_ns);
        try init_fn(&iw.window.dvui_window);
        _ = try iw.window.dvui_window.end(.{});
    }
    defer if (app.deinitFn) |deinit_fn| deinit_fn(&iw.window.dvui_window);
    while (!iw.window.received_close) {
        pollEvents(iw.window);
        try paint(app, &app_state, iw.window);
    }
}

fn wioGetInstanceProcAddr(instance: vk.Instance, name: [*:0]const u8) callconv(vk.vulkan_call_conv) vk.PfnVoidFunction {
    // wio exposes a Zig-ABI wrapper taking usize; Vulkan requires this exact ABI.
    // Calling wio's wrapper through a direct pointer cast corrupts the instance argument.
    return @ptrCast(wio.vkGetInstanceProcAddr(@intFromEnum(instance), name));
}

test {
    std.testing.refAllDecls(@This());
}
