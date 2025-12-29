//! Experimental Vulkan dvui renderer
//! * Follows `max_frames_in_flight` convention which needs to be specified during init:
//!     For resource management/synchronization its expected that no more than fixed number of begin/end frames will be in flight (alive, submitted, not yet finished by gpu).
//!     It is used to avoid any unnecessary host<->gpu synchronization when just delaying such operation is possible.
//!     If `max_frames_in_flight` is misconfigured or not enforced by app or user of renderer then data races and undefined behavior will happen. (for example texture might get deleted while its still in use on gpu).
//! * Memory: all space for vertex & index buffers is preallocated at start requiring setting appropriate limits in options.
//!     Requests to render over limit is safe but will lead to draw commands being ignored. Note that these settings are per frame and total memory allocated will depend on max frames in flight.
//!     Texture bookkeeping is similarly preallocated. But actual image data on gpu are allocated individually at runtime. Currently 1 image = 1 allocation which is ok for large or few images,
//!     but not great for many smaller images that can eat in max gpu allocation limit. TODO: implement hooks for general purpose allocator
//! * Queue submissions: renderer expects itself to be given vk.CommandBuffer where it will record all render commands between begin/end calls.
//!     However to support runtime texture upload direct queue access is also required. Multi-threaded scenarios have not been tested but optional queue lock/release callbacks can be specified if needed.
const std = @import("std");
const builtin = @import("builtin");
const slog = std.log.scoped(.dvui_vulkan);
const dvui = @import("dvui");
const vk = @import("vk");
const Size = dvui.Size;

const vs_spv align(64) = @embedFile("dvui.vert.spv").*;
const fs_spv align(64) = @embedFile("dvui.frag.spv").*;

const Self = @This();

pub const DeviceProxy = vk.DeviceProxy;
pub const Vertex = dvui.Vertex;
pub const Indice = u16;
pub const invalid_texture: *anyopaque = @ptrFromInt(0xBAD0BAD0); //@ptrFromInt(0xFFFF_FFFF);
/// following image applications, dvui does math in srgb color space, so interpret all textures as unorm so no gamma correction occurs
pub const img_format = vk.Format.r8g8b8a8_unorm; // dvui works in srgb color space, so we don't linearize srgb -> linear colorspace
pub const TextureIdx = u16;

// debug flags
const enable_breakpoints = false;
const texture_tracing = false; // tace leaks and usage

/// initialization options, caller still owns all passed in resources
pub const InitOptions = struct {
    dev: vk.DeviceProxy,
    pdev: vk.PhysicalDevice,
    mem_props: vk.PhysicalDeviceMemoryProperties,
    /// render pass from which renderer will be used
    render_pass: vk.RenderPass,
    /// optional vulkan host side allocator
    vk_alloc: ?*vk.AllocationCallbacks = null,

    /// queue - used only for texture upload
    /// used here only once during initialization, afterwards texture upload queue must be provided with beginFrame()
    queue: vk.Queue,
    /// command pool - used only for texture upload
    comamnd_pool: vk.CommandPool,

    /// How many frames can be in flight in worst case (usually equals swapchain image count)
    max_frames_in_flight: u32,

    /// Maximum number of indices that can be submitted in single frame
    /// Draw requests above this limit will get discarded
    max_indices_per_frame: u32 = 1024 * 128,
    max_vertices_per_frame: u32 = 1024 * 64,

    /// Maximum number of alive textures supported including render targets. global - across all frames in flight
    /// Overflow should be safe but will lead to heavy visual artifacts as font etc. textures can get evicted
    /// Note: as this is only book keeping limit it can be set quite high. Real texture memory usage could be more concerning, as well as large allocation count.
    max_textures: TextureIdx = 256,

    /// error and invalid texture handle color
    /// if by any chance renderer runs out of textures or due to other reason fails to create a texture then this color will be used as texture
    error_texture_color: [4]u8 = [4]u8{ 255, 0, 255, 255 }, // default bright pink so its noticeable for debug, can be set to alpha 0 for invisible etc.

    /// if uv coords go out of bounds, how should the sampling behave
    texture_wrap: vk.SamplerAddressMode = .repeat,

    /// bytes - total host visible memory allocated ahead of time
    pub inline fn hostVisibleMemSize(s: @This()) u32 {
        const vtx_space = std.mem.alignForward(u32, s.max_vertices_per_frame * @sizeOf(dvui.Vertex), vk_alignment);
        const idx_space = std.mem.alignForward(u32, s.max_indices_per_frame * @sizeOf(Indice), vk_alignment);
        return s.max_frames_in_flight * (vtx_space + idx_space);
    }
};

/// TODO:
/// allocation strategy for device (gpu) memory
// const ImageAllocStrategy = union(enum) {
//     /// user provides proper allocator
//     allocator: struct {},
//     /// most basic implementation, ok for few images created with backend.createTexture
//     /// WARNING: can consume much of or hit vk.maxMemoryAllocationCount limit too many resources are used, see:
//     /// https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxMemoryAllocationCount&platform=all
//     allocate_each: void,
// };

/// just simple debug and informative metrics
pub const Stats = struct {
    // per frame
    draw_calls: u32 = 0,
    verts: u32 = 0,
    indices: u32 = 0,
    // global
    textures_alive: u16 = 0, // including render targets
    textures_mem: usize = 0,
};

// not owned by us:
dev: DeviceProxy,
pdev: vk.PhysicalDevice,
vk_alloc: ?*vk.AllocationCallbacks,
cmdbuf: vk.CommandBuffer = .null_handle,
dpool: vk.DescriptorPool,
queue: vk.Queue = .null_handle,
queue_lock: ?LockCallbacks = null,
cpool: vk.CommandPool = .null_handle,
cpool_lock: ?LockCallbacks = null,

// owned by us
samplers: [2]vk.Sampler,
frames: []FrameData,
textures: []Texture,
destroy_textures_offset: TextureIdx = 0,
destroy_textures: []TextureIdx,
pipeline: vk.Pipeline,
pipeline_layout: vk.PipelineLayout,
dset_layout: vk.DescriptorSetLayout,
current_frame: *FrameData, // points somewhere in frames

/// if set render to render texture instead of default cmdbuf
render_target: ?vk.CommandBuffer = null,
render_target_pass: vk.RenderPass,
render_target_pipeline: vk.Pipeline,

dummy_texture: Texture = undefined, // dummy/null white texture
error_texture: Texture = undefined,

host_vis_mem_idx: u32,
host_vis_mem: vk.DeviceMemory,
host_vis_coherent: bool,
host_vis_data: []u8, // mapped host_vis_mem
device_local_mem_idx: u32,

framebuffer_size: vk.Extent2D = .{ .width = 0, .height = 0 },
vtx_overflow_logged: bool = false,
idx_overflow_logged: bool = false,
// just for info / dbg
stats: Stats = .{},

/// for potentially multi threaded shared resources, lock callbacks can be set that will be called
const LockCallbacks = struct {
    lockCB: *const fn (userdata: ?*anyopaque) void = undefined,
    unlockCB: *const fn (userdata: ?*anyopaque) void = undefined,
    lock_userdata: ?*anyopaque = null, // user defined data that will be returned in callbacks
};

const FrameData = struct {
    // buffers to host_vis memory
    vtx_buff: vk.Buffer = .null_handle,
    vtx_data: []u8 = &.{},
    vtx_offset: u32 = 0,
    idx_buff: vk.Buffer = .null_handle,
    idx_data: []u8 = &.{},
    idx_offset: u32 = 0,
    /// textures to be destroyed after frames cycle through
    /// offset & len points to backend.destroy_textures[]
    destroy_textures_offset: u16 = 0,
    destroy_textures_len: u16 = 0,

    fn deinit(f: *@This(), b: *Backend) void {
        f.freeTextures(b);
        b.dev.destroyBuffer(f.vtx_buff, b.vk_alloc);
        b.dev.destroyBuffer(f.idx_buff, b.vk_alloc);
    }

    fn reset(f: *@This(), b: *Backend) void {
        f.vtx_offset = 0;
        f.idx_offset = 0;
        f.destroy_textures_offset = b.destroy_textures_offset;
        f.destroy_textures_len = 0;
    }

    fn freeTextures(f: *@This(), b: *Backend) void {
        // free textures
        for (f.destroy_textures_offset..(f.destroy_textures_offset + f.destroy_textures_len)) |i| {
            const tidx = b.destroy_textures[i % b.destroy_textures.len]; // wrap around on overflow
            // just for debug and monitoring
            b.stats.textures_alive -= 1;
            b.stats.textures_mem -= b.dev.getImageMemoryRequirements(b.textures[tidx].img).size;

            //slog.debug("destroy texture {}({x}) | {}", .{ tidx, @intFromPtr(&b.textures[tidx]), b.stats.textures_alive });
            b.textures[tidx].deinit(b);
            b.textures[tidx].img = .null_handle;
            b.textures[tidx].dset = .null_handle;
            b.textures[tidx].img_view = .null_handle;
            b.textures[tidx].mem = .null_handle;
            b.textures[tidx].trace.addAddr(@returnAddress(), "destroy"); // keep tracing
        }
        f.destroy_textures_len = 0;
    }
};

pub fn init(alloc: std.mem.Allocator, opt: InitOptions) !Self {
    // TODO: FIXME: in multiple places here in this function we will leak if error gets thrown
    const dev = opt.dev;
    // Memory
    // host visible
    var host_coherent: bool = false;
    const host_vis_mem_type_index: u32 = blk: {
        // device local, host visible
        for (opt.mem_props.memory_types[0..opt.mem_props.memory_type_count], 0..) |mem_type, i|
            if (mem_type.property_flags.device_local_bit and mem_type.property_flags.host_visible_bit) {
                host_coherent = mem_type.property_flags.host_coherent_bit;
                slog.debug("chosen host_visible_mem: {} {}", .{ i, mem_type });
                break :blk @truncate(i);
            };
        // not device local
        for (opt.mem_props.memory_types[0..opt.mem_props.memory_type_count], 0..) |mem_type, i|
            if (mem_type.property_flags.host_visible_bit) {
                host_coherent = mem_type.property_flags.host_coherent_bit;
                slog.info("chosen host_visible_mem is NOT device local - Are we running on integrated graphics?", .{});
                slog.debug("chosen host_visible_mem: {} {}", .{ i, mem_type });
                break :blk @truncate(i);
            };
        return error.NoSuitableMemoryType;
    };
    slog.debug("host_vis allocation size: {Bi}", .{opt.hostVisibleMemSize()});
    const host_visible_mem = try dev.allocateMemory(&.{
        .allocation_size = opt.hostVisibleMemSize(),
        .memory_type_index = host_vis_mem_type_index,
    }, opt.vk_alloc);
    errdefer dev.freeMemory(host_visible_mem, opt.vk_alloc);
    const host_vis_data = @as([*]u8, @ptrCast((try dev.mapMemory(host_visible_mem, 0, vk.WHOLE_SIZE, .{})).?))[0..opt.hostVisibleMemSize()];
    // device local mem
    const device_local_mem_idx: u32 = blk: {
        for (opt.mem_props.memory_types[0..opt.mem_props.memory_type_count], 0..) |mem_type, i|
            if (mem_type.property_flags.device_local_bit and !mem_type.property_flags.host_visible_bit) {
                slog.debug("chosen device local mem: {} {}", .{ i, mem_type });
                break :blk @truncate(i);
            };
        break :blk host_vis_mem_type_index;
    };
    // Memory sub-allocation into FrameData
    const frames = try alloc.alloc(FrameData, opt.max_frames_in_flight);
    errdefer alloc.free(frames);
    {
        var mem_offset: usize = 0;
        for (frames) |*f| {
            f.* = .{};
            // TODO: on error here cleanup will leak previous initialized frames
            { // vertex buf
                const buf = try dev.createBuffer(&.{
                    .size = @sizeOf(Vertex) * opt.max_vertices_per_frame,
                    .usage = .{ .vertex_buffer_bit = true },
                    .sharing_mode = .exclusive,
                }, opt.vk_alloc);
                errdefer dev.destroyBuffer(buf, opt.vk_alloc);
                const mreq = dev.getBufferMemoryRequirements(buf);
                mem_offset = std.mem.alignForward(usize, mem_offset, mreq.alignment);
                try dev.bindBufferMemory(buf, host_visible_mem, mem_offset);
                f.vtx_data = host_vis_data[mem_offset..][0..mreq.size];
                f.vtx_buff = buf;
                mem_offset += mreq.size;
            }
            { // index buf
                const buf = try dev.createBuffer(&.{
                    .size = @sizeOf(Indice) * opt.max_indices_per_frame,
                    .usage = .{ .index_buffer_bit = true },
                    .sharing_mode = .exclusive,
                }, opt.vk_alloc);
                errdefer dev.destroyBuffer(buf, opt.vk_alloc);
                const mreq = dev.getBufferMemoryRequirements(buf);
                mem_offset = std.mem.alignForward(usize, mem_offset, mreq.alignment);
                try dev.bindBufferMemory(buf, host_visible_mem, mem_offset);
                f.idx_data = host_vis_data[mem_offset..][0..mreq.size];
                f.idx_buff = buf;
                mem_offset += mreq.size;
            }
        }
    }

    // Descriptors
    const extra: u32 = 8; // idk, exact pool sizes returns OutOfPoolMemory slightly too soon, add extra margin
    const dpool_sizes = [_]vk.DescriptorPoolSize{
        .{ .type = .combined_image_sampler, .descriptor_count = opt.max_textures + extra },
        //.{ .type = .uniform_buffer, .descriptor_count = opt.max_frames_in_flight },
    };
    const dpool = try dev.createDescriptorPool(&.{
        .max_sets = opt.max_textures + extra,
        .pool_size_count = dpool_sizes.len,
        .p_pool_sizes = &dpool_sizes,
        .flags = .{ .free_descriptor_set_bit = true },
    }, opt.vk_alloc);
    const dsl = try dev.createDescriptorSetLayout(
        &vk.DescriptorSetLayoutCreateInfo{
            .binding_count = 1,
            .p_bindings = &.{
                // vk.DescriptorSetLayoutBinding{
                //     .binding = ubo_binding,
                //     .descriptor_count = 1,
                //     .descriptor_type = .uniform_buffer,
                //     .stage_flags = .{ .vertex_bit = true },
                // },
                vk.DescriptorSetLayoutBinding{
                    .binding = tex_binding,
                    .descriptor_count = 1,
                    .descriptor_type = .combined_image_sampler,
                    .stage_flags = .{ .fragment_bit = true },
                },
            },
        },
        opt.vk_alloc,
    );
    const pipeline_layout = try dev.createPipelineLayout(&.{
        .flags = .{},
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast(&dsl),
        .push_constant_range_count = 1,
        .p_push_constant_ranges = &.{.{
            .stage_flags = .{ .vertex_bit = true },
            .offset = 0,
            .size = @sizeOf(f32) * 4,
        }},
    }, opt.vk_alloc);
    const pipeline = try createPipeline(dev, pipeline_layout, opt.render_pass, opt.vk_alloc);

    const samplers = [_]vk.SamplerCreateInfo{
        .{ // dvui.TextureInterpolation.nearest
            .mag_filter = .nearest,
            .min_filter = .nearest,
            .mipmap_mode = .nearest,
            .address_mode_u = opt.texture_wrap,
            .address_mode_v = opt.texture_wrap,
            .address_mode_w = opt.texture_wrap,
            .mip_lod_bias = 0,
            .anisotropy_enable = .false,
            .max_anisotropy = 0,
            .compare_enable = .false,
            .compare_op = .always,
            .min_lod = 0,
            .max_lod = vk.LOD_CLAMP_NONE,
            .border_color = .int_opaque_white,
            .unnormalized_coordinates = .false,
        },
        .{ // dvui.TextureInterpolation.linear
            .mag_filter = .linear,
            .min_filter = .linear,
            .mipmap_mode = .linear,
            .address_mode_u = opt.texture_wrap,
            .address_mode_v = opt.texture_wrap,
            .address_mode_w = opt.texture_wrap,
            .mip_lod_bias = 0,
            .anisotropy_enable = .false,
            .max_anisotropy = 0,
            .compare_enable = .false,
            .compare_op = .always,
            .min_lod = 0,
            .max_lod = vk.LOD_CLAMP_NONE,
            .border_color = .int_opaque_white,
            .unnormalized_coordinates = .false,
        },
    };

    const render_target_pass = try createOffscreenRenderPass(dev, img_format);
    var res: Self = .{
        .dev = dev,
        .dpool = dpool,
        .pdev = opt.pdev,
        .vk_alloc = opt.vk_alloc,

        .dset_layout = dsl,
        .samplers = .{
            try dev.createSampler(&samplers[0], opt.vk_alloc),
            try dev.createSampler(&samplers[1], opt.vk_alloc),
        },
        .textures = try alloc.alloc(Texture, opt.max_textures),
        .destroy_textures = try alloc.alloc(u16, opt.max_textures),
        .render_target_pass = render_target_pass,
        .render_target_pipeline = try createPipeline(dev, pipeline_layout, render_target_pass, opt.vk_alloc),
        .pipeline = pipeline,
        .pipeline_layout = pipeline_layout,
        .host_vis_mem_idx = host_vis_mem_type_index,
        .host_vis_mem = host_visible_mem,
        .host_vis_data = host_vis_data,
        .host_vis_coherent = host_coherent,
        .device_local_mem_idx = device_local_mem_idx,
        .queue = opt.queue,
        .cpool = opt.comamnd_pool,
        .frames = frames,
        .current_frame = &frames[0],
    };
    @memset(res.textures, Texture{});

    res.dummy_texture = try res.createAndUplaodTexture(&[4]u8{ 255, 255, 255, 255 }, 1, 1, .nearest);
    res.error_texture = try res.createAndUplaodTexture(&opt.error_texture_color, 1, 1, .nearest);
    return res;
}

/// for sync safety, better call queueWaitIdle before destruction
pub fn deinit(self: *Self, alloc: std.mem.Allocator) void {
    for (self.frames) |*f| f.deinit(self);
    alloc.free(self.frames);
    for (self.textures, 0..) |tex, i| if (!tex.isNull()) {
        slog.debug("TEXTURE LEAKED {}:\n", .{i});
        tex.trace.dump();
        tex.deinit(self);
    };
    alloc.free(self.textures);
    alloc.free(self.destroy_textures);

    self.dummy_texture.deinit(self);
    self.error_texture.deinit(self);
    for (self.samplers) |s| self.dev.destroySampler(s, self.vk_alloc);

    self.dev.destroyDescriptorPool(self.dpool, self.vk_alloc);
    self.dev.destroyDescriptorSetLayout(self.dset_layout, self.vk_alloc);
    self.dev.destroyPipelineLayout(self.pipeline_layout, self.vk_alloc);
    self.dev.destroyPipeline(self.pipeline, self.vk_alloc);
    self.dev.unmapMemory(self.host_vis_mem);
    self.dev.freeMemory(self.host_vis_mem, self.vk_alloc);
    // self.dev.destroyRenderPass(self.render_pass_texture_target, self.vk_alloc);
}

pub const RenderPassInfo = struct {
    framebuffer: vk.Framebuffer,
    render_area: vk.Rect2D,
};

/// Begins new frame
/// Command buffer has to be in a render pass
pub fn beginFrame(self: *Self, cmdbuf: vk.CommandBuffer, framebuffer_size: vk.Extent2D) void {
    self.cmdbuf = cmdbuf;
    self.framebuffer_size = framebuffer_size;

    // advance frame pointer,
    const current_frame_idx = (@intFromPtr(self.current_frame) - @intFromPtr(self.frames.ptr) + @sizeOf(FrameData)) / @sizeOf(FrameData) % self.frames.len;
    const cf = &self.frames[current_frame_idx];
    self.current_frame = cf;

    // clean up old frame data
    cf.freeTextures(self);

    // reset frame data
    self.current_frame.reset(self);
    self.stats.draw_calls = 0;
    self.stats.indices = 0;
    self.stats.verts = 0;
    self.vtx_overflow_logged = false;
    self.idx_overflow_logged = false;
}

/// Ends current frame
/// returns command buffer (same one given at init)
pub fn endFrame(self: *Self) vk.CommandBuffer {
    const cmdbuf = self.cmdbuf;
    self.dev.cmdEndRenderPass(cmdbuf);
    self.cmdbuf = .null_handle;
    return cmdbuf;
}

//
// Dvui backend interface matching functions
//  see: dvui/Backend.zig
//
const Backend = Self;

pub fn begin(self: *Self, arena: std.mem.Allocator, framebuffer_size: dvui.Size.Physical) void {
    _ = arena; // autofix
    self.render_target = null;
    if (self.cmdbuf == .null_handle) @panic("dvui_vulkan_renderer: Command bufer not set! (missing beginFrame()?)");

    const dev = self.dev;
    const cmdbuf = self.cmdbuf;
    dev.cmdBindPipeline(cmdbuf, .graphics, self.pipeline);

    const viewport = vk.Viewport{
        .x = 0,
        .y = 0,
        .width = framebuffer_size.w,
        .height = framebuffer_size.h,
        .min_depth = 0,
        .max_depth = 1,
    };
    dev.cmdSetViewport(cmdbuf, 0, 1, @ptrCast(&viewport));

    const PushConstants = struct {
        view_scale: @Vector(2, f32),
        view_translate: @Vector(2, f32),
    };
    const push_constants = PushConstants{
        .view_scale = .{ 2.0 / framebuffer_size.w, 2.0 / framebuffer_size.h },
        .view_translate = .{ -1.0, -1.0 },
    };
    dev.cmdPushConstants(cmdbuf, self.pipeline_layout, .{ .vertex_bit = true }, 0, @sizeOf(PushConstants), &push_constants);
}

pub fn end(self: *Backend) void {
    _ = self; // autofix
}

pub fn pixelSize(self: *Backend) Size {
    return .{ .w = @floatFromInt(self.framebuffer_size.width), .h = @floatFromInt(self.framebuffer_size.height) };
}

pub fn drawClippedTriangles(self: *Backend, texture_: ?dvui.Texture, vtx: []const Vertex, idx: []const Indice, clipr: ?dvui.Rect.Physical) void {
    const texture: ?*anyopaque = if (texture_) |t| @as(*anyopaque, @ptrCast(@alignCast(t.ptr))) else null;
    const dev = self.dev;
    const cmdbuf = if (self.render_target) |t| t else self.cmdbuf;
    const cf = self.current_frame;
    const vtx_bytes = vtx.len * @sizeOf(Vertex);
    const idx_bytes = idx.len * @sizeOf(Indice);

    { // updates stats even if draw is skipped due to overflow
        self.stats.draw_calls += 1;
        self.stats.verts += @intCast(vtx.len);
        self.stats.indices += @intCast(idx.len);
    }

    if (cf.vtx_data[cf.vtx_offset..].len < vtx_bytes) {
        if (!self.vtx_overflow_logged) slog.warn("vertex buffer out of space", .{});
        self.vtx_overflow_logged = true;
        if (enable_breakpoints) @breakpoint();
        return;
    }
    if (cf.idx_data[cf.idx_offset..].len < idx_bytes) {
        // if only index buffer alone is out of bounds, we could just shrinking it... but meh
        if (!self.idx_overflow_logged) slog.warn("index buffer out of space", .{});
        self.idx_overflow_logged = true;
        if (enable_breakpoints) @breakpoint();
        return;
    }

    { // clip / scissor
        const scissor = if (clipr) |c| vk.Rect2D{
            .offset = .{ .x = @intFromFloat(@max(0, c.x)), .y = @intFromFloat(@max(0, c.y)) },
            .extent = .{ .width = @intFromFloat(c.w), .height = @intFromFloat(c.h) },
        } else vk.Rect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = self.framebuffer_size,
        };
        dev.cmdSetScissor(cmdbuf, 0, 1, @ptrCast(&scissor));
    }

    const idx_offset: u32 = cf.idx_offset;
    const vtx_offset: u32 = cf.vtx_offset;
    { // upload indices & vertices
        var modified_ranges: [2]vk.MappedMemoryRange = undefined;
        { // indices
            const dst = cf.idx_data[cf.idx_offset..][0..idx_bytes];
            cf.idx_offset += @intCast(dst.len);
            modified_ranges[0] = .{ .memory = self.host_vis_mem, .offset = @intFromPtr(dst.ptr) - @intFromPtr(self.host_vis_data.ptr), .size = dst.len };
            @memcpy(dst, std.mem.sliceAsBytes(idx));
        }
        { // vertices
            const dst = cf.vtx_data[cf.vtx_offset..][0..vtx_bytes];
            cf.vtx_offset += @intCast(dst.len);
            modified_ranges[1] = .{ .memory = self.host_vis_mem, .offset = @intFromPtr(dst.ptr) - @intFromPtr(self.host_vis_data.ptr), .size = dst.len };
            @memcpy(dst, std.mem.sliceAsBytes(vtx));
        }
        if (!self.host_vis_coherent)
            dev.flushMappedMemoryRanges(modified_ranges.len, &modified_ranges) catch |err|
                slog.err("flushMappedMemoryRanges: {}", .{err});
    }

    if (@sizeOf(Indice) != 2) unreachable;
    dev.cmdBindIndexBuffer(cmdbuf, cf.idx_buff, idx_offset, .uint16);
    dev.cmdBindVertexBuffers(cmdbuf, 0, 1, @ptrCast(&cf.vtx_buff), &[_]vk.DeviceSize{vtx_offset});
    var dset: vk.DescriptorSet = if (texture == null) self.dummy_texture.dset else blk: {
        if (texture.? == invalid_texture) break :blk self.error_texture.dset;
        const tex = @as(*Texture, @ptrCast(@alignCast(texture)));
        if (tex.trace.index < tex.trace.addrs.len / 2 + 1) tex.trace.addAddr(@returnAddress(), "render"); // if trace has some free room, trace this
        break :blk tex.dset;
    };
    dev.cmdBindDescriptorSets(
        cmdbuf,
        .graphics,
        self.pipeline_layout,
        0,
        1,
        @ptrCast(&dset),
        0,
        null,
    );
    dev.cmdDrawIndexed(cmdbuf, @intCast(idx.len), 1, 0, 0, 0);
}

fn findEmptyTextureSlot(self: *Backend) ?TextureIdx {
    for (self.textures, 0..) |*out_tex, s| {
        if (out_tex.isNull()) return @intCast(s);
    }
    slog.err("textureCreate: Out of texture slots!", .{});
    return null;
}

pub fn textureCreate(self: *Backend, pixels: [*]const u8, width: u32, height: u32, interpolation: dvui.enums.TextureInterpolation) dvui.Texture {
    const slot = self.findEmptyTextureSlot() orelse return .{ .ptr = invalid_texture, .width = 1, .height = 1 };
    const out_tex: *Texture = &self.textures[slot];
    const tex = self.createAndUplaodTexture(pixels, width, height, interpolation) catch |err| {
        if (enable_breakpoints) @breakpoint();
        slog.err("Can't create texture: {}", .{err});
        return .{ .ptr = invalid_texture, .width = 1, .height = 1 };
    };
    out_tex.* = tex;
    out_tex.trace.addAddr(@returnAddress(), "create");

    self.stats.textures_alive += 1;
    self.stats.textures_mem += self.dev.getImageMemoryRequirements(out_tex.img).size;
    //slog.debug("textureCreate {} ({x}) | {}", .{ slot, @intFromPtr(out_tex), self.stats.textures_alive });
    return .{ .ptr = @ptrCast(out_tex), .width = width, .height = height };
}

pub fn textureCreateTarget(self: *Backend, width: u32, height: u32, interpolation: dvui.enums.TextureInterpolation) GenericError!dvui.TextureTarget {
    const tex_slot = self.findEmptyTextureSlot() orelse return error.OutOfMemory;

    const dev = self.dev;
    var tex = self.createTextureWithMem(.{
        .image_type = .@"2d",
        .format = img_format,
        .extent = .{ .width = width, .height = height, .depth = 1 },
        .mip_levels = 1,
        .array_layers = 1,
        .samples = .{ .@"1_bit" = true },
        .tiling = .optimal,
        .usage = .{
            .color_attachment_bit = true,
            .sampled_bit = true,
        },
        .sharing_mode = .exclusive,
        .initial_layout = .undefined,
    }, interpolation) catch |err| {
        if (enable_breakpoints) @breakpoint();
        slog.err("textureCreateTarget failed to create framebuffer: {}", .{err});
        return error.BackendError;
    };
    errdefer tex.deinit(self);

    tex.framebuffer = dev.createFramebuffer(&.{
        .flags = .{},
        .render_pass = self.render_target_pass,
        .attachment_count = 1,
        .p_attachments = @ptrCast(&tex.img_view),
        .width = width,
        .height = height,
        .layers = 1,
    }, self.vk_alloc) catch |err| {
        if (enable_breakpoints) @breakpoint();
        slog.err("textureCreateTarget failed to create framebuffer: {}", .{err});
        return error.BackendError;
    };
    errdefer dev.destroyFramebuffer(tex.framebuffer, self.vk_alloc);

    self.textures[tex_slot] = tex;
    self.stats.textures_alive += 1;
    self.stats.textures_mem += self.dev.getImageMemoryRequirements(tex.img).size;
    return .{ .ptr = &self.textures[tex_slot], .width = width, .height = height };
}
pub fn textureRead(self: *Backend, texture: dvui.Texture, pixels_out: [*]u8, width: u32, height: u32) !void {
    slog.debug("textureRead({}, {*}, {}x{}) Not implemented!", .{ texture, pixels_out, width, height });
    _ = self; // autofix
    return error.NotImplemented;
}
pub fn textureDestroy(self: *Backend, texture: dvui.Texture) void {
    if (texture.ptr == invalid_texture) return;
    const dslot = self.destroy_textures_offset;
    self.destroy_textures_offset = (dslot + 1) % @as(u16, @intCast(self.destroy_textures.len));
    if (self.destroy_textures[dslot] != 0xFFFF) {
        self.destroy_textures[dslot] = @intCast((@intFromPtr(texture.ptr) - @intFromPtr(self.textures.ptr)) / @sizeOf(Texture));
    }
    self.current_frame.destroy_textures_len += 1;
    // slog.debug("schedule destroy texture: {} ({x})", .{ self.destroy_textures[dslot], @intFromPtr(texture) });
}

/// Read pixel data (RGBA) from `texture` into `pixels_out`.
pub fn textureReadTarget(self: *Backend, texture: dvui.TextureTarget, pixels_out: [*]u8) TextureError!void {
    slog.info("textureReadTarget", .{});
    _ = pixels_out;
    _ = self;
    _ = texture;
    return error.NotImplemented;
}

/// Convert texture target made with `textureCreateTarget` into return texture
/// as if made by `textureCreate`.  After this call, texture target will not be
/// used by dvui.
pub fn textureFromTarget(self: *Backend, texture_target: dvui.TextureTarget) dvui.Texture {
    _ = self; // autofix
    return .{ .ptr = texture_target.ptr, .width = texture_target.width, .height = texture_target.height };
}

pub fn renderTarget(self: *Backend, dvui_texture_target: ?dvui.TextureTarget) GenericError!void {
    // TODO: all errors are set to unreachable, add handling?
    const dev = self.dev;
    const srr = vk.ImageSubresourceRange{
        .aspect_mask = .{ .color_bit = true },
        .base_mip_level = 0,
        .level_count = 1,
        .base_array_layer = 0,
        .layer_count = 1,
    };
    _ = srr; // autofix

    if (self.render_target) |cmdbuf| { // finalize previous render target
        self.render_target = null;
        dev.cmdEndRenderPass(cmdbuf);
        // TODO: transition to shader_src_optimal
        self.endSingleTimeCommands(cmdbuf) catch unreachable;
        return;
    }

    const texture: *Texture = if (dvui_texture_target) |t| @ptrCast(@alignCast(t.ptr)) else return;
    const cmdbuf = self.beginSingleTimeCommands() catch unreachable;

    const w: f32 = @floatFromInt(self.framebuffer_size.width); // @floatFromInt(tt.fb_size.width)
    const h: f32 = @floatFromInt(self.framebuffer_size.height); // @floatFromInt(tt.fb_size.height)
    { // begin render-pass & reset viewport
        dev.cmdBindPipeline(cmdbuf, .graphics, self.render_target_pipeline);
        const clear = vk.ClearValue{
            .color = .{ .float_32 = .{ 0, 0, 0, 0 } },
        };
        const viewport = vk.Viewport{
            .x = 0,
            .y = 0,
            .width = w,
            .height = h,
            .min_depth = 0,
            .max_depth = 1,
        };

        dev.cmdBeginRenderPass(cmdbuf, &.{
            .render_pass = self.render_target_pass,
            .framebuffer = texture.framebuffer,
            .render_area = vk.Rect2D{
                .offset = .{ .x = 0, .y = 0 },
                .extent = .{ .width = dvui_texture_target.?.width, .height = dvui_texture_target.?.height },
            },
            .clear_value_count = 1,
            .p_clear_values = @ptrCast(&clear),
        }, .@"inline");
        dev.cmdSetViewport(cmdbuf, 0, 1, @ptrCast(&viewport));
    }

    const PushConstants = struct {
        view_scale: @Vector(2, f32),
        view_translate: @Vector(2, f32),
    };
    const push_constants = PushConstants{
        .view_scale = .{ 2.0 / w, 2.0 / h },
        .view_translate = .{ -1.0, -1.0 },
    };
    dev.cmdPushConstants(cmdbuf, self.pipeline_layout, .{ .vertex_bit = true }, 0, @sizeOf(PushConstants), &push_constants);

    self.render_target = cmdbuf; // activate render-texture on success
}

//
// Private functions
// Some can be pub just to allow using them as utils
//
const Texture = struct {
    img: vk.Image = .null_handle,
    img_view: vk.ImageView = .null_handle,
    mem: vk.DeviceMemory = .null_handle,
    dset: vk.DescriptorSet = .null_handle,
    /// for render-textures only
    framebuffer: vk.Framebuffer = .null_handle,

    trace: Trace = Trace.init,
    const Trace = std.debug.ConfigurableTrace(6, 5, texture_tracing);

    pub fn isNull(self: @This()) bool {
        return self.dset == .null_handle;
    }

    pub fn deinit(tex: Texture, b: *Backend) void {
        const dev = b.dev;
        const vk_alloc = b.vk_alloc;
        dev.freeDescriptorSets(b.dpool, 1, &[_]vk.DescriptorSet{tex.dset}) catch |err| slog.err("Failed to free descriptor set: {}", .{err});
        dev.destroyImageView(tex.img_view, vk_alloc);
        dev.destroyImage(tex.img, vk_alloc);
        dev.freeMemory(tex.mem, vk_alloc);
        dev.destroyFramebuffer(tex.framebuffer, vk_alloc);
    }
};

fn createPipeline(
    dev: DeviceProxy,
    layout: vk.PipelineLayout,
    render_pass: vk.RenderPass,
    vk_alloc: ?*vk.AllocationCallbacks,
) DeviceProxy.CreateGraphicsPipelinesError!vk.Pipeline {
    //  NOTE: VK_KHR_maintenance5 (which was promoted to vulkan 1.4) deprecates ShaderModules.
    // todo: check for extension and then enable
    const ext_m5 = false; // VK_KHR_maintenance5
    const vert_shdd = vk.ShaderModuleCreateInfo{
        .code_size = vs_spv.len,
        .p_code = @ptrCast(&vs_spv),
    };
    const frag_shdd = vk.ShaderModuleCreateInfo{
        .code_size = fs_spv.len,
        .p_code = @ptrCast(&fs_spv),
    };
    var pssci = [_]vk.PipelineShaderStageCreateInfo{
        .{
            .stage = .{ .vertex_bit = true },
            .p_name = "main",
            .module = if (ext_m5) null else try dev.createShaderModule(&vert_shdd, vk_alloc),
            .p_next = if (ext_m5) &vert_shdd else null,
        },
        .{
            .stage = .{ .fragment_bit = true },
            //.module = frag,
            .p_name = "main",
            .module = if (ext_m5) null else try dev.createShaderModule(&frag_shdd, vk_alloc),
            .p_next = if (ext_m5) &frag_shdd else null,
        },
    };
    defer if (!ext_m5) for (pssci) |p| if (p.module != .null_handle) dev.destroyShaderModule(p.module, vk_alloc);

    const pvisci = vk.PipelineVertexInputStateCreateInfo{
        .vertex_binding_description_count = VertexBindings.binding_description.len,
        .p_vertex_binding_descriptions = &VertexBindings.binding_description,
        .vertex_attribute_description_count = VertexBindings.attribute_description.len,
        .p_vertex_attribute_descriptions = &VertexBindings.attribute_description,
    };

    const piasci = vk.PipelineInputAssemblyStateCreateInfo{
        .topology = .triangle_list,
        .primitive_restart_enable = .false,
    };

    var viewport: vk.Viewport = undefined;
    var scissor: vk.Rect2D = undefined;
    const pvsci = vk.PipelineViewportStateCreateInfo{
        .viewport_count = 1,
        .p_viewports = @ptrCast(&viewport), // set in createCommandBuffers with cmdSetViewport
        .scissor_count = 1,
        .p_scissors = @ptrCast(&scissor), // set in createCommandBuffers with cmdSetScissor
    };

    const prsci = vk.PipelineRasterizationStateCreateInfo{
        .depth_clamp_enable = .false,
        .rasterizer_discard_enable = .false,
        .polygon_mode = .fill,
        .cull_mode = .{ .back_bit = false },
        .front_face = .clockwise,
        .depth_bias_enable = .false,
        .depth_bias_constant_factor = 0,
        .depth_bias_clamp = 0,
        .depth_bias_slope_factor = 0,
        .line_width = 1,
    };

    const pmsci = vk.PipelineMultisampleStateCreateInfo{
        .rasterization_samples = .{ .@"1_bit" = true },
        .sample_shading_enable = .false,
        .min_sample_shading = 1,
        .alpha_to_coverage_enable = .false,
        .alpha_to_one_enable = .false,
    };

    // do premultiplied alpha blending:
    const pcbas = vk.PipelineColorBlendAttachmentState{
        .blend_enable = .true,
        .src_color_blend_factor = .one,
        .dst_color_blend_factor = .one_minus_src_alpha,
        .color_blend_op = .add,
        .src_alpha_blend_factor = .one,
        .dst_alpha_blend_factor = .one_minus_src_alpha,
        .alpha_blend_op = .add,
        .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
    };

    const pcbsci = vk.PipelineColorBlendStateCreateInfo{
        .logic_op_enable = .false,
        .logic_op = .copy,
        .attachment_count = 1,
        .p_attachments = @ptrCast(&pcbas),
        .blend_constants = [_]f32{ 0, 0, 0, 0 },
    };

    const dynstate = [_]vk.DynamicState{ .viewport, .scissor };
    const pdsci = vk.PipelineDynamicStateCreateInfo{
        .flags = .{},
        .dynamic_state_count = dynstate.len,
        .p_dynamic_states = &dynstate,
    };

    const gpci = vk.GraphicsPipelineCreateInfo{
        .flags = .{},
        .stage_count = pssci.len,
        .p_stages = &pssci,
        .p_vertex_input_state = &pvisci,
        .p_input_assembly_state = &piasci,
        .p_tessellation_state = null,
        .p_viewport_state = &pvsci,
        .p_rasterization_state = &prsci,
        .p_multisample_state = &pmsci,
        .p_depth_stencil_state = null,
        .p_color_blend_state = &pcbsci,
        .p_dynamic_state = &pdsci,
        .layout = layout,
        .render_pass = render_pass,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    };

    var pipeline: vk.Pipeline = undefined;
    _ = try dev.createGraphicsPipelines(
        .null_handle,
        1,
        @ptrCast(&gpci),
        vk_alloc,
        @ptrCast(&pipeline),
    );
    return pipeline;
}

const AllocatedBuffer = struct {
    buf: vk.Buffer,
    mem: vk.DeviceMemory,
};

/// allocates space for staging, creates buffer, and copies content to it
fn stageToBuffer(
    self: *@This(),
    buf_info: vk.BufferCreateInfo,
    contents: []const u8,
) !AllocatedBuffer {
    const buf = self.dev.createBuffer(&buf_info, self.vk_alloc) catch |err| {
        slog.err("createBuffer: {}", .{err});
        return err;
    };
    errdefer self.dev.destroyBuffer(buf, self.vk_alloc);
    const mreq = self.dev.getBufferMemoryRequirements(buf);
    const mem = try self.dev.allocateMemory(&.{ .allocation_size = mreq.size, .memory_type_index = self.host_vis_mem_idx }, self.vk_alloc);
    errdefer self.dev.freeMemory(mem, self.vk_alloc);
    const mem_offset = 0;
    try self.dev.bindBufferMemory(buf, mem, mem_offset);
    const data = @as([*]u8, @ptrCast((try self.dev.mapMemory(mem, mem_offset, vk.WHOLE_SIZE, .{})).?))[0..mreq.size];
    @memcpy(data[0..contents.len], contents);
    if (!self.host_vis_coherent)
        try self.dev.flushMappedMemoryRanges(1, &.{.{ .memory = mem, .offset = mem_offset, .size = mreq.size }});
    return .{ .buf = buf, .mem = mem };
}

pub fn beginSingleTimeCommands(self: *Self) !vk.CommandBuffer {
    if (self.cpool_lock) |l| l.lockCB(l.lock_userdata);
    defer if (self.cpool_lock) |l| l.unlockCB(l.lock_userdata);

    var cmdbuf: vk.CommandBuffer = undefined;
    self.dev.allocateCommandBuffers(&.{
        .command_pool = self.cpool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&cmdbuf)) catch |err| {
        if (enable_breakpoints) @breakpoint();
        return err;
    };
    try self.dev.beginCommandBuffer(cmdbuf, &.{
        .flags = .{ .one_time_submit_bit = true },
    });
    return cmdbuf;
}

pub fn endSingleTimeCommands(self: *Self, cmdbuf: vk.CommandBuffer) !void {
    try self.dev.endCommandBuffer(cmdbuf);
    defer self.dev.freeCommandBuffers(self.cpool, 1, @ptrCast(&cmdbuf));

    if (self.queue_lock) |l| l.lockCB(l.lock_userdata);
    defer if (self.queue_lock) |l| l.unlockCB(l.lock_userdata);
    const qs = [_]vk.SubmitInfo{.{
        .wait_semaphore_count = 0,
        .p_wait_semaphores = null,
        .p_wait_dst_stage_mask = null,
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&cmdbuf),
        .signal_semaphore_count = 0,
        .p_signal_semaphores = null,
    }};

    const fence = try self.dev.createFence(&.{}, self.vk_alloc);
    defer self.dev.destroyFence(fence, self.vk_alloc);
    try self.dev.queueSubmit(self.queue, 1, &qs, fence);
    if (try self.dev.waitForFences(1, &.{fence}, .true, std.math.maxInt(u64)) != .success) unreachable; // VK_TIMEOUT should be unlikely
}

pub fn createTextureWithMem(self: Backend, img_info: vk.ImageCreateInfo, interpolation: dvui.enums.TextureInterpolation) !Texture {
    const dev = self.dev;

    const img: vk.Image = try dev.createImage(&img_info, self.vk_alloc);
    errdefer dev.destroyImage(img, self.vk_alloc);
    const mreq = dev.getImageMemoryRequirements(img);

    const mem = dev.allocateMemory(&.{
        .allocation_size = mreq.size,
        .memory_type_index = self.device_local_mem_idx,
    }, self.vk_alloc) catch |err| {
        slog.err("Failed to alloc texture mem: {}", .{err});
        return err;
    };
    errdefer dev.freeMemory(mem, self.vk_alloc);
    try dev.bindImageMemory(img, mem, 0);

    const srr = vk.ImageSubresourceRange{
        .aspect_mask = .{ .color_bit = true },
        .base_mip_level = 0,
        .level_count = 1,
        .base_array_layer = 0,
        .layer_count = 1,
    };
    const img_view = try dev.createImageView(&.{
        .flags = .{},
        .image = img,
        .view_type = .@"2d",
        .format = img_format,
        .components = .{
            .r = .identity,
            .g = .identity,
            .b = .identity,
            .a = .identity,
        },
        .subresource_range = srr,
    }, self.vk_alloc);
    errdefer dev.destroyImageView(img_view, self.vk_alloc);

    var dset: [1]vk.DescriptorSet = undefined;
    dev.allocateDescriptorSets(&.{
        .descriptor_pool = self.dpool,
        .descriptor_set_count = 1,
        .p_set_layouts = @ptrCast(&self.dset_layout),
    }, &dset) catch |err| {
        if (enable_breakpoints) @breakpoint();
        slog.err("Failed to allocate descriptor set: {}", .{err});
        return err;
    };
    const dii = [1]vk.DescriptorImageInfo{.{
        .sampler = self.samplers[@intFromEnum(interpolation)],
        .image_view = img_view,
        .image_layout = .shader_read_only_optimal,
    }};
    const write_dss = [_]vk.WriteDescriptorSet{.{
        .dst_set = dset[0],
        .dst_binding = tex_binding,
        .dst_array_element = 0,
        .descriptor_count = 1,
        .descriptor_type = .combined_image_sampler,
        .p_image_info = &dii,
        .p_buffer_info = undefined,
        .p_texel_buffer_view = undefined,
    }};
    dev.updateDescriptorSets(write_dss.len, &write_dss, 0, null);

    return Texture{ .img = img, .img_view = img_view, .mem = mem, .dset = dset[0] };
}

pub fn createAndUplaodTexture(self: *Backend, pixels: [*]const u8, width: u32, height: u32, interpolation: dvui.enums.TextureInterpolation) !Texture {
    //slog.debug("img {}x{}; req size {}", .{ width, height, mreq.size });
    const tex = try self.createTextureWithMem(.{
        .image_type = .@"2d",
        .format = img_format,
        .extent = .{ .width = width, .height = height, .depth = 1 },
        .mip_levels = 1,
        .array_layers = 1,
        .samples = .{ .@"1_bit" = true },
        .tiling = .optimal,
        .usage = .{
            .transfer_dst_bit = true,
            .sampled_bit = true,
        },
        .sharing_mode = .exclusive,
        .initial_layout = .undefined,
    }, interpolation);
    errdefer tex.deinit(self);
    const dev = self.dev;

    // prep host side staging buffer for transfer
    const mreq = dev.getImageMemoryRequirements(tex.img);
    const img_staging = try self.stageToBuffer(.{
        .flags = .{},
        .size = mreq.size,
        .usage = .{ .transfer_src_bit = true },
        .sharing_mode = .exclusive,
    }, pixels[0 .. width * height * 4]);
    defer dev.destroyBuffer(img_staging.buf, self.vk_alloc);
    defer dev.freeMemory(img_staging.mem, self.vk_alloc);

    const cmdbuf = try self.beginSingleTimeCommands();
    errdefer dev.resetCommandBuffer(cmdbuf, .{});

    const srr = vk.ImageSubresourceRange{
        .aspect_mask = .{ .color_bit = true },
        .base_mip_level = 0,
        .level_count = 1,
        .base_array_layer = 0,
        .layer_count = 1,
    };
    { // prep image to receive
        const img_barrier = vk.ImageMemoryBarrier{
            .src_access_mask = .{},
            .dst_access_mask = .{ .transfer_write_bit = true },
            .old_layout = .undefined,
            .new_layout = .transfer_dst_optimal,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = tex.img,
            .subresource_range = srr,
        };
        dev.cmdPipelineBarrier(cmdbuf, .{ .host_bit = true, .top_of_pipe_bit = true }, .{ .transfer_bit = true }, .{}, 0, null, 0, null, 1, @ptrCast(&img_barrier));
    }
    { // copy host staging -> device mem
        const buff_img_copy = vk.BufferImageCopy{
            .buffer_offset = 0,
            .buffer_row_length = 0,
            .buffer_image_height = 0,
            .image_subresource = .{
                .aspect_mask = .{ .color_bit = true },
                .mip_level = 0,
                .base_array_layer = 0,
                .layer_count = 1,
            },
            .image_offset = .{ .x = 0, .y = 0, .z = 0 },
            .image_extent = .{ .width = width, .height = height, .depth = 1 },
        };
        dev.cmdCopyBufferToImage(cmdbuf, img_staging.buf, tex.img, .transfer_dst_optimal, 1, @ptrCast(&buff_img_copy));
    }
    { // transition to read only optimal
        const img_barrier = vk.ImageMemoryBarrier{
            .src_access_mask = .{ .transfer_write_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true },
            .old_layout = .transfer_dst_optimal,
            .new_layout = .shader_read_only_optimal,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = tex.img,
            .subresource_range = srr,
        };
        dev.cmdPipelineBarrier(cmdbuf, .{ .transfer_bit = true }, .{ .fragment_shader_bit = true }, .{}, 0, null, 0, null, 1, @ptrCast(&img_barrier));
    }

    self.endSingleTimeCommands(cmdbuf) catch unreachable; // submits transfer, waits for it to finish
    return tex;
}

pub fn createOffscreenRenderPass(dev: DeviceProxy, format: vk.Format) !vk.RenderPass {
    var subpasses: [1]vk.SubpassDescription = undefined;
    var color_attachments: [1]vk.AttachmentDescription = undefined;

    { // Render to framebuffer
        color_attachments[0] = vk.AttachmentDescription{
            .format = format, // swapchain / framebuffer image format
            .samples = .{ .@"1_bit" = true },
            .load_op = .clear,
            .store_op = .store,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .undefined,
            .final_layout = .present_src_khr,
        };
        const color_attachment_ref = vk.AttachmentReference{
            .attachment = 0,
            .layout = .color_attachment_optimal,
        };
        subpasses[0] = vk.SubpassDescription{
            .pipeline_bind_point = .graphics,
            .color_attachment_count = 1,
            .p_color_attachments = @ptrCast(&color_attachment_ref),
        };
    }

    const deps = [2]vk.SubpassDependency{
        .{
            .src_subpass = vk.SUBPASS_EXTERNAL,
            .dst_subpass = 0,
            .src_stage_mask = .{ .fragment_shader_bit = true },
            .dst_stage_mask = .{ .color_attachment_output_bit = true },
            .src_access_mask = .{},
            .dst_access_mask = .{ .color_attachment_read_bit = true, .color_attachment_write_bit = true },
            .dependency_flags = .{ .by_region_bit = true },
        },
        .{
            .src_subpass = 0,
            .dst_subpass = vk.SUBPASS_EXTERNAL,
            .src_stage_mask = .{ .color_attachment_output_bit = true },
            .dst_stage_mask = .{ .fragment_shader_bit = true },
            .src_access_mask = .{ .color_attachment_read_bit = true, .color_attachment_write_bit = true },
            .dst_access_mask = .{},
            .dependency_flags = .{ .by_region_bit = true },
        },
    };

    return try dev.createRenderPass(&.{
        .attachment_count = @intCast(color_attachments.len),
        .p_attachments = &color_attachments,
        .subpass_count = @intCast(subpasses.len),
        .p_subpasses = &subpasses,
        .dependency_count = 0,
        .p_dependencies = @ptrCast(&deps),
    }, null);
}

const VertexBindings = struct {
    const binding_description = [_]vk.VertexInputBindingDescription{.{
        .binding = 0,
        .stride = @sizeOf(Vertex),
        .input_rate = .vertex,
    }};

    const attribute_description = [_]vk.VertexInputAttributeDescription{
        .{
            .binding = 0,
            .location = 0,
            .format = .r32g32_sfloat,
            .offset = @offsetOf(Vertex, "pos"),
        },
        .{
            .binding = 0,
            .location = 1,
            .format = .r8g8b8a8_unorm,
            .offset = @offsetOf(Vertex, "col"),
        },
        .{
            .binding = 0,
            .location = 2,
            .format = .r32g32_sfloat,
            .offset = @offsetOf(Vertex, "uv"),
        },
    };
};

pub const tex_binding = 1; // shader binding slot must match shader

/// device memory min alignment
/// we could query it at runtime, but this is reasonable safe number. We don't use this for anything critical.
/// https://vulkan.gpuinfo.org/displaydevicelimit.php?name=minMemoryMapAlignment&platform=all
const vk_alignment = if (builtin.target.os.tag.isDarwin()) 16384 else 4096;

pub const GenericError = std.mem.Allocator.Error || error{BackendError};
pub const TextureError = GenericError || error{ TextureCreate, TextureRead, TextureUpdate, NotImplemented };
