const std = @import("std");
const builtin = @import("builtin");
const dvui = @import("dvui");
const vk = @import("vulkan");
// const zm = @import("zig_math");
const zm = @import("zmath");

const DvuiVkBackend = dvui.backend;
const uses_win32 = builtin.target.os.tag == .windows and @hasDecl(DvuiVkBackend, "win32");
const win32 = if (uses_win32) DvuiVkBackend.win32 else void;
const win = if (uses_win32) DvuiVkBackend.win else void;
const uses_vk_dll = @hasDecl(DvuiVkBackend, "vk_dll");
const vk_dll = if (uses_vk_dll) DvuiVkBackend.vk_dll else void;
const uses_glfw = @hasDecl(DvuiVkBackend, "glfw");
const glfw = if (uses_glfw) DvuiVkBackend.glfw else void;
const FrameSync = DvuiVkBackend.FrameSync;
const slog = std.log.scoped(.main);

const vs_spv align(64) = @embedFile("3d.vert.spv").*;
const fs_spv align(64) = @embedFile("3d.frag.spv").*;

const Mat4 = zm.Matrix(4, 4, f32, .{});

pub const DepthBuffer = struct {
    image: vk.Image,
    view: vk.ImageView,
    memory: vk.DeviceMemory,
    size: vk.Extent2D,
    const vk_alloc = null;
    const format = vk.Format.d32_sfloat;

    pub fn init(dev: vk.DeviceProxy, size: vk.Extent2D, device_local_mem_idx: u32) !DepthBuffer {
        // createImage( vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage, depthImageMemory
        const image = try dev.createImage(&.{
            .image_type = .@"2d",
            .format = format,
            .extent = .{ .width = size.width, .height = size.height, .depth = 1 },
            .mip_levels = 1,
            .array_layers = 1,
            .samples = .{ .@"1_bit" = true },
            .tiling = .optimal,
            .usage = .{
                // .sampled_bit = true,
                .depth_stencil_attachment_bit = true,
            },
            .sharing_mode = .exclusive,
            .initial_layout = .undefined,
        }, vk_alloc);
        const mem = try dev.allocateMemory(&.{
            .allocation_size = dev.getImageMemoryRequirements(image).size,
            .memory_type_index = device_local_mem_idx,
        }, vk_alloc);
        try dev.bindImageMemory(image, mem, 0);
        const view = try dev.createImageView(&.{
            .flags = .{},
            .image = image,
            .view_type = .@"2d",
            .format = format,
            .components = .{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            },
            .subresource_range = .{
                .aspect_mask = .{ .depth_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }, vk_alloc);

        return .{
            .image = image,
            .view = view,
            .memory = mem,
            .size = size,
        };
    }
    pub fn deinit(self: @This(), dev: vk.DeviceProxy) void {
        dev.destroyImageView(self.view, vk_alloc);
        dev.destroyImage(self.image, vk_alloc);
        dev.freeMemory(self.memory, vk_alloc);
    }
};

pub const AppState = struct {
    backend: *DvuiVkBackend.VkBackend,
    render_pass: vk.RenderPass,
    sync: FrameSync,
    command_buffers: []vk.CommandBuffer,
    depth_buffer: DepthBuffer,

    pub fn init(gpa: std.mem.Allocator) !AppState {
        const window_class = if (uses_win32) win32.L("DvuiWindow") else void;
        if (uses_win32) {
            win.RegisterClass(window_class, .{}) catch win32.panicWin32(
                "RegisterClass",
                win32.GetLastError(),
            );
        }

        if (uses_vk_dll) vk_dll.init() catch |err| std.debug.panic("Failed to init Vulkan: {}", .{err});
        errdefer if (uses_vk_dll) vk_dll.deinit();

        const loader = if (uses_vk_dll) vk_dll.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse @panic("Failed to lookup Vulkan VkGetInstanceProcAddr: {}") else DvuiVkBackend.getInstanceProcAddress;

        var b = try gpa.create(DvuiVkBackend.VkBackend);
        b.* = DvuiVkBackend.VkBackend.init(gpa, undefined);
        errdefer b.deinit();

        // init backend (creates and owns OS window)
        var window_context: *DvuiVkBackend.WindowContext = try b.allocContext();
        window_context.* = .{
            .backend = b,
            .dvui_window = try dvui.Window.init(@src(), gpa, DvuiVkBackend.dvuiBackend(window_context), .{}),
            .hwnd = undefined,
        };
        if (uses_win32) {
            win.initWindow(window_context, window_class, .{
                .dvui_gpa = gpa,
                .gpa = gpa,
                .title = "3d example",
                .vsync = false,
            }) catch |err| {
                slog.err("initWindow failed: {}", .{err});
                return err;
            };
        } else if (uses_glfw) {
            DvuiVkBackend.initWindow(window_context, .{
                .dvui_gpa = gpa,
                .gpa = gpa,
                .title = "3d example",
            }) catch |err| {
                slog.err("initWindow failed: {}", .{err});
                return err;
            };
        } else @compileError("Platform not implemented!");

        var ext_dynamic_state_features = vk.PhysicalDeviceExtendedDynamicStateFeaturesEXT{
            .extended_dynamic_state = .true,
        };
        b.vkc = try DvuiVkBackend.VkContext.init(gpa, loader, window_context, &DvuiVkBackend.createVkSurface, .{
            .device_select_settings = .{
                .required_extensions = &.{
                    vk.extensions.khr_swapchain.name,
                    vk.extensions.ext_extended_dynamic_state.name, // for dynamic depth on/off
                },
            },
            .device_create_info_p_next = @ptrCast(&ext_dynamic_state_features),
        });

        window_context.swapchain_state = try DvuiVkBackend.WindowContext.SwapchainState.init(window_context, .{
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
            },
            .desired_present_modes = if (!vsync) &.{.immediate_khr} else &.{.fifo_khr},
        });

        // const render_pass = try DvuiVkBackend.createRenderPass(b.vkc.device, window_context.swapchain_state.?.swapchain.image_format);
        const render_pass = try createRenderPass(b.vkc.device, window_context.swapchain_state.?.swapchain.image_format);
        errdefer b.vkc.device.destroyRenderPass(render_pass, null);

        const command_buffers = try DvuiVkBackend.createCommandBuffers(gpa, b.vkc.device, b.vkc.cmd_pool, max_frames_in_flight);
        errdefer gpa.free(command_buffers);

        b.renderer = try DvuiVkBackend.VkRenderer.init(b.gpa, .{
            .dev = b.vkc.device,
            .comamnd_pool = b.vkc.cmd_pool,
            .queue = b.vkc.graphics_queue.handle,
            .pdev = b.vkc.physical_device.handle,
            .mem_props = b.vkc.physical_device.memory_properties,
            .render_pass = render_pass,
            .max_frames_in_flight = max_frames_in_flight,
        });

        const sync = try FrameSync.init(gpa, max_frames_in_flight, b.vkc.device);
        errdefer sync.deinit(gpa, b.vkc.device);

        const depth_buffer = try DepthBuffer.init(
            b.vkc.device,
            .{ .width = @intFromFloat(window_context.last_pixel_size.w), .height = @intFromFloat(window_context.last_pixel_size.h) },
            b.renderer.?.device_local_mem_idx,
        );
        errdefer depth_buffer.deinit(b.vkc.device);

        return .{
            .backend = b,
            .command_buffers = command_buffers,
            .render_pass = render_pass,
            .sync = sync,
            .depth_buffer = depth_buffer,
        };
    }

    pub fn recreateSwapchain(self: *@This(), ctx: *DvuiVkBackend.WindowContext) !void {
        try ctx.swapchain_state.?.recreate(ctx);
        self.depth_buffer.deinit(g_app_state.device());
        self.depth_buffer = try DepthBuffer.init(
            self.backend.vkc.device,
            .{ .width = @intFromFloat(ctx.last_pixel_size.w), .height = @intFromFloat(ctx.last_pixel_size.h) },
            self.backend.renderer.?.device_local_mem_idx,
        );
    }

    pub fn deinit(self: AppState, gpa: std.mem.Allocator) void {
        defer if (uses_vk_dll) vk_dll.deinit();
        defer gpa.destroy(self.backend);
        defer self.backend.deinit();
        defer self.backend.vkc.device.destroyRenderPass(self.render_pass, null);
        defer self.sync.deinit(gpa, self.backend.vkc.device);
        defer gpa.free(self.command_buffers);
    }

    pub fn device(self: @This()) vk.DeviceProxy {
        return self.backend.vkc.device;
    }
};
pub var g_app_state: AppState = undefined;
pub var g_scene: Scene = undefined;

pub const Scene = struct {
    host_visible_mem: vk.DeviceMemory,
    host_vis_data: []u8,
    index_buffer: vk.Buffer,
    vertex_buffer: vk.Buffer,
    pipeline_layout: vk.PipelineLayout,
    pipeline: vk.Pipeline,
    timer: std.time.Timer,

    pub fn init() !Scene {
        const dev: vk.DeviceProxy = g_app_state.backend.vkc.device;
        const vk_alloc = null;
        const indices_bytes = std.mem.sliceAsBytes(&Cube.indices);
        const vertices_bytes = std.mem.sliceAsBytes(&Cube.vertices);
        var index_buffer = vk.Buffer.null_handle;
        var vertex_buffer = vk.Buffer.null_handle;

        // TODO: here we are lazy, we just allocate some host visible mem that we can directly write mesh to. we should transfer mem to gpu normally
        const size: usize = vertices_bytes.len + indices_bytes.len + 64; // +64 for alignment space in between
        const host_visible_mem = try dev.allocateMemory(&.{
            .allocation_size = size,
            .memory_type_index = g_app_state.backend.renderer.?.host_vis_mem_idx,
        }, vk_alloc);
        errdefer dev.freeMemory(host_visible_mem, vk_alloc);
        const host_vis_data = @as([*]u8, @ptrCast((try dev.mapMemory(host_visible_mem, 0, vk.WHOLE_SIZE, .{})).?))[0..size];
        @memcpy(host_vis_data[0..vertices_bytes.len], vertices_bytes);
        const dst: []u8 = std.mem.alignInSlice(host_vis_data[vertices_bytes.len..], @alignOf(@TypeOf(Cube.indices[0]))).?[0..indices_bytes.len];
        @memcpy(dst, indices_bytes);

        var mem_offset: usize = 0;
        { // vertex buf
            const buf = try dev.createBuffer(&.{
                .size = @sizeOf(@TypeOf(Cube.vertices)),
                .usage = .{ .vertex_buffer_bit = true },
                .sharing_mode = .exclusive,
            }, vk_alloc);
            errdefer dev.destroyBuffer(buf, vk_alloc);
            const mreq = dev.getBufferMemoryRequirements(buf);
            mem_offset = std.mem.alignForward(usize, mem_offset, mreq.alignment);
            try dev.bindBufferMemory(buf, host_visible_mem, mem_offset);
            mem_offset += mreq.size;
            vertex_buffer = buf;
        }
        { // index buf
            const buf = try dev.createBuffer(&.{
                .size = @sizeOf(@TypeOf(Cube.indices)),
                .usage = .{ .index_buffer_bit = true },
                .sharing_mode = .exclusive,
            }, vk_alloc);
            errdefer dev.destroyBuffer(buf, vk_alloc);
            const mreq = dev.getBufferMemoryRequirements(buf);
            mem_offset = std.mem.alignForward(usize, mem_offset, mreq.alignment);
            try dev.bindBufferMemory(buf, host_visible_mem, mem_offset);
            mem_offset += mreq.size;
            index_buffer = buf;
        }

        const layout = try dev.createPipelineLayout(&.{
            .flags = .{},
            // .set_layout_count = 1,
            // .p_set_layouts = @ptrCast(&dsl),
            .push_constant_range_count = 1,
            .p_push_constant_ranges = &.{.{
                .stage_flags = .{ .vertex_bit = true },
                .offset = 0,
                .size = @sizeOf(f32) * 4 * 4 * 3,
            }},
        }, vk_alloc);
        return .{
            .host_visible_mem = host_visible_mem,
            .host_vis_data = host_vis_data,
            .index_buffer = index_buffer,
            .vertex_buffer = vertex_buffer,
            .pipeline = try createScenePipeline(dev, layout, g_app_state.render_pass, vk_alloc),
            .pipeline_layout = layout,
            .timer = try std.time.Timer.start(),
        };
    }

    pub fn deinit(self: Scene) void {
        const vk_alloc = null;
        const dev = g_app_state.backend.vkc.device;
        dev.destroyBuffer(self.vertex_buffer, vk_alloc);
        dev.destroyBuffer(self.index_buffer, vk_alloc);
        dev.freeMemory(self.host_visible_mem, vk_alloc);
        dev.destroyPipeline(self.pipeline, vk_alloc);
        dev.destroyPipelineLayout(self.pipeline_layout, vk_alloc);
    }

    pub fn draw(self: *Scene, cmdbuf: vk.CommandBuffer) void {
        const dev = g_app_state.backend.vkc.device;
        dev.cmdBindPipeline(cmdbuf, .graphics, self.pipeline);

        var framebuffer_size = g_app_state.backend.contexts.items[0].last_pixel_size;
        if (framebuffer_size.w < 1) framebuffer_size.w = 1;
        if (framebuffer_size.h < 1) framebuffer_size.h = 1;
        const viewport = vk.Viewport{
            .x = 0,
            .y = 0,
            .width = framebuffer_size.w,
            .height = framebuffer_size.h,
            .min_depth = 0,
            .max_depth = 1,
        };
        dev.cmdSetViewport(cmdbuf, 0, 1, @ptrCast(&viewport));

        const scissor = vk.Rect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = .{ .width = @intFromFloat(framebuffer_size.w), .height = @intFromFloat(framebuffer_size.h) },
        };
        dev.cmdSetScissor(cmdbuf, 0, 1, @ptrCast(&scissor));

        // const PushConstants = extern struct {
        //     model: zm.Mat4f = zm.Mat4f.identity,
        //     view: zm.Mat4f = zm.Mat4f.identity,
        //     projection: zm.Mat4f,
        // };
        // var push_constants = PushConstants{
        //     .projection = zm.Mat4f.perspectiveGl(std.math.degreesToRadians(60), framebuffer_size.w / framebuffer_size.h, 0.001, 9999),
        //     .model = zm.Mat4f.fromTranslation(zm.FastVec3f.initXYZ(0, 0, 2)),
        // };
        // push_constants.projection = push_constants.projection.mulMatrix(4, push_constants.model);
        const PushConstants = extern struct {
            // model: zm.Mat,
            // view: zm.Mat,
            // projection: zm.Mat,
            mvp: zm.Mat,
        };
        const t: f32 = @as(f32, @floatFromInt(self.timer.read() / std.time.ns_per_ms)) / 1000; // sec f32
        const rotation = zm.mul(zm.rotationX(t), zm.rotationY(t));
        const model = zm.mul(rotation, zm.translation(0, 0, -10));
        const view = zm.identity();
        const projection = zm.perspectiveFovRh(std.math.degreesToRadians(60.0), framebuffer_size.w / framebuffer_size.h, 0.01, 100);
        const push_constants = PushConstants{
            .mvp = zm.mul(model, zm.mul(view, projection)),
        };
        if (@sizeOf(f32) * 4 * 4 != @sizeOf(@TypeOf(push_constants))) unreachable;
        dev.cmdPushConstants(cmdbuf, self.pipeline_layout, .{ .vertex_bit = true }, 0, @sizeOf(PushConstants), &push_constants);

        dev.cmdBindIndexBuffer(cmdbuf, self.index_buffer, 0, .uint16);
        dev.cmdBindVertexBuffers(cmdbuf, 0, 1, @ptrCast(&self.vertex_buffer), &[_]vk.DeviceSize{0});
        dev.cmdDrawIndexed(cmdbuf, @intCast(Cube.indices.len), 1, 0, 0, 0);
    }
};

// 3d renderpass
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

    const depth_attachment = vk.AttachmentDescription{
        .format = DepthBuffer.format,
        .samples = .{ .@"1_bit" = true },
        .load_op = .clear,
        .store_op = .dont_care,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .depth_stencil_attachment_optimal,
    };

    const subpasses = [_]vk.SubpassDescription{.{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = color_attachment_refs.len,
        .p_color_attachments = &color_attachment_refs,
        .p_depth_stencil_attachment = &.{
            .attachment = 1,
            .layout = .depth_stencil_attachment_optimal,
        },
    }};

    const dependencies = [_]vk.SubpassDependency{.{
        .src_subpass = vk.SUBPASS_EXTERNAL,
        .dst_subpass = 0,
        .src_stage_mask = .{ .color_attachment_output_bit = true, .late_fragment_tests_bit = true },
        .src_access_mask = .{ .depth_stencil_attachment_write_bit = true },
        .dst_stage_mask = .{ .color_attachment_output_bit = true, .early_fragment_tests_bit = true },
        .dst_access_mask = .{ .color_attachment_write_bit = true, .depth_stencil_attachment_write_bit = true },
    }};

    const attachments = [_]vk.AttachmentDescription{ color_attachment, depth_attachment };
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

pub const max_frames_in_flight = 3;
pub const vsync = false;

pub fn main() !void {
    if (builtin.target.os.tag == .windows) dvui.Backend.Common.windowsAttachConsole() catch {};
    dvui.Examples.show_demo_window = false;

    var gpa_instance = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = gpa_instance.allocator();
    defer _ = gpa_instance.deinit();

    if (uses_glfw) {
        try glfw.init();
        var major: i32 = 0;
        var minor: i32 = 0;
        var rev: i32 = 0;
        glfw.getVersion(&major, &minor, &rev);
        slog.info("GLFW {}.{}.{} vk_support: {}", .{ major, minor, rev, glfw.vulkanSupported() });
    }
    defer if (uses_glfw) glfw.terminate();

    g_app_state = try AppState.init(gpa);
    defer g_app_state.deinit(gpa);

    g_scene = try Scene.init();
    defer g_scene.deinit();

    // hijack the dvui pipeline, so that it accepts depth buffer
    g_app_state.device().destroyPipeline(g_app_state.backend.renderer.?.pipeline, g_app_state.backend.renderer.?.vk_alloc);
    g_app_state.backend.renderer.?.pipeline = try createDvuiPipeline(g_app_state.device(), g_app_state.backend.renderer.?.pipeline_layout, g_app_state.render_pass, g_app_state.backend.renderer.?.vk_alloc);

    defer g_app_state.backend.vkc.device.queueWaitIdle(g_app_state.backend.vkc.graphics_queue.handle) catch {}; // let gpu finish its work on exit, otherwise we will get validation errors

    if (uses_win32) {
        DvuiVkBackend.windowDamageRefreshCallback = &win32DamageRefresh;
        main_loop: while (g_app_state.backend.contexts.items.len > 0) {
            // slog.info("frame: {}", .{current_frame_in_flight});
            switch (win.serviceMessageQueue()) {
                .queue_empty => {
                    for (g_app_state.backend.contexts.items) |ctx| {
                        try paint(&g_app_state, ctx);
                        g_app_state.backend.prev_frame_stats = g_app_state.backend.renderer.?.stats;
                        if (ctx.received_close) {
                            _ = win32.PostMessageA(@ptrCast(ctx.hwnd), win32.WM_CLOSE, 0, 0);
                            continue;
                        }
                    }
                },
                .quit => break :main_loop,
            }
        }
    } else if (uses_glfw) {
        const b = g_app_state.backend;
        const window = b.contexts.items[0].glfw_win;
        DvuiVkBackend.registerDvuiIO(window.?); // registers glfw callbacks
        if (builtin.os.tag == .windows) { // windows blocks event loop while resizing, use separate callback to keep rendering
            _ = glfw.setWindowRefreshCallback(window, &refreshCB);
        }
        while (!glfw.windowShouldClose(window)) {
            if (glfw.getKey(window, glfw.KeyEscape) == glfw.Press) {
                glfw.setWindowShouldClose(window, true);
            }

            // TODO: fixme: implement actual multi-window implementation, this won't work right
            for (b.contexts.items, 0..) |ctx, ctx_i| {
                _ = ctx_i; // autofix
                try paint(&g_app_state, ctx);
                b.prev_frame_stats = b.renderer.?.stats;
            }

            glfw.pollEvents();
        }
    } else @compileError("Platform not implemented!");
}

/// Window damage and refresh
pub fn refreshCB(window: *glfw.Window) callconv(.c) void {
    if (uses_glfw) {
        const ctx: *DvuiVkBackend.WindowContext = @ptrCast(@alignCast(glfw.getWindowUserPointer(window)));
        paint(&g_app_state, ctx) catch {};
    }
}
pub fn win32DamageRefresh(ctx: *DvuiVkBackend.WindowContext) void {
    paint(&g_app_state, ctx) catch {};
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

    dvui.Examples.demo();
}

pub fn paint(app_state: *AppState, ctx: *DvuiVkBackend.WindowContext) !void {
    const b = ctx.backend;
    const gpa = b.gpa;
    const render_pass = app_state.render_pass;
    const sync = &app_state.sync;
    const device = b.vkc.device;

    if (ctx.last_pixel_size.w < 1 or ctx.last_pixel_size.h < 1) return;

    // wait for previous frame to finish
    try sync.begin(device);
    defer sync.end();
    const image_index = try acquireImageMaybeRecreate(
        &ctx.swapchain_state.?,
        gpa,
        ctx,
        sync.*,

        render_pass,
        b.renderer.?.device_local_mem_idx,
    );

    const framebuffer = ctx.swapchain_state.?.framebuffers[image_index];
    const command_buffer = app_state.command_buffers[sync.current_frame];
    try device.beginCommandBuffer(command_buffer, &.{ .flags = .{} });
    const cmd = vk.CommandBufferProxy.init(command_buffer, device.wrapper);

    const clear_values = [_]vk.ClearValue{
        .{ .color = .{ .float_32 = .{ 0.1, 0.1, 0.1, 1 } } },
        .{ .depth_stencil = .{ .depth = 1.0, .stencil = 0 } },
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

    cmd.setDepthTestEnableEXT(.true);
    cmd.setDepthWriteEnableEXT(.true);
    g_scene.draw(command_buffer);
    cmd.setDepthTestEnableEXT(.false);
    cmd.setDepthWriteEnableEXT(.false);

    if (true) { // draw dvui
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

    }

    cmd.endRenderPass();
    cmd.endCommandBuffer() catch |err| std.debug.panic("Failed to end vulkan cmd buffer: {}", .{err});

    if (!try DvuiVkBackend.present(
        ctx,
        command_buffer,
        sync.items[sync.current_frame],
        ctx.swapchain_state.?.swapchain.handle,
        image_index,
    )) {
        // ctx.recreate_swapchain_requested = true;
        slog.err("present failed!", .{});
    }
    // slog.debug("frame done", .{});
}

fn createScenePipeline(
    dev: vk.DeviceProxy,
    layout: vk.PipelineLayout,
    render_pass: vk.RenderPass,
    vk_alloc: ?*vk.AllocationCallbacks,
) vk.DeviceProxy.CreateGraphicsPipelinesError!vk.Pipeline {
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
    const pssci = [_]vk.PipelineShaderStageCreateInfo{
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

    return createPipeline(dev, layout, render_pass, &pssci, pvisci, vk_alloc);
}

fn createDvuiPipeline(
    dev: vk.DeviceProxy,
    layout: vk.PipelineLayout,
    render_pass: vk.RenderPass,
    vk_alloc: ?*vk.AllocationCallbacks,
) vk.DeviceProxy.CreateGraphicsPipelinesError!vk.Pipeline {
    //  NOTE: VK_KHR_maintenance5 (which was promoted to vulkan 1.4) deprecates ShaderModules.
    // todo: check for extension and then enable
    const ext_m5 = false; // VK_KHR_maintenance5
    const vert_shdd = vk.ShaderModuleCreateInfo{
        .code_size = DvuiVkBackend.VkRenderer.vs_spv.len,
        .p_code = @ptrCast(&DvuiVkBackend.VkRenderer.vs_spv),
    };
    const frag_shdd = vk.ShaderModuleCreateInfo{
        .code_size = DvuiVkBackend.VkRenderer.fs_spv.len,
        .p_code = @ptrCast(&DvuiVkBackend.VkRenderer.fs_spv),
    };
    const pssci = [_]vk.PipelineShaderStageCreateInfo{
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
        .vertex_binding_description_count = DvuiVkBackend.VkRenderer.VertexBindings.binding_description.len,
        .p_vertex_binding_descriptions = &DvuiVkBackend.VkRenderer.VertexBindings.binding_description,
        .vertex_attribute_description_count = DvuiVkBackend.VkRenderer.VertexBindings.attribute_description.len,
        .p_vertex_attribute_descriptions = &DvuiVkBackend.VkRenderer.VertexBindings.attribute_description,
    };

    return createPipeline(dev, layout, render_pass, &pssci, pvisci, vk_alloc);
}

fn createPipeline(
    dev: vk.DeviceProxy,
    layout: vk.PipelineLayout,
    render_pass: vk.RenderPass,
    pssci: []const vk.PipelineShaderStageCreateInfo,
    pvisci: vk.PipelineVertexInputStateCreateInfo,
    vk_alloc: ?*vk.AllocationCallbacks,
) vk.DeviceProxy.CreateGraphicsPipelinesError!vk.Pipeline {
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

    const dynstate = [_]vk.DynamicState{
        .viewport,
        .scissor,
        .depth_test_enable,
        .depth_write_enable,
    };
    const pdsci = vk.PipelineDynamicStateCreateInfo{
        .flags = .{},
        .dynamic_state_count = dynstate.len,
        .p_dynamic_states = &dynstate,
    };

    const sops = std.mem.zeroes(vk.StencilOpState);
    const ds = vk.PipelineDepthStencilStateCreateInfo{
        .depth_test_enable = .true,
        .depth_write_enable = .true,
        .depth_compare_op = .less,
        .depth_bounds_test_enable = .false,
        .stencil_test_enable = .false,
        .front = sops,
        .back = sops,
        .min_depth_bounds = 0.0,
        .max_depth_bounds = 1.0,
    };

    const gpci = vk.GraphicsPipelineCreateInfo{
        .flags = .{},
        .stage_count = @intCast(pssci.len),
        .p_stages = pssci.ptr,
        .p_vertex_input_state = &pvisci,
        .p_input_assembly_state = &piasci,
        .p_tessellation_state = null,
        .p_viewport_state = &pvsci,
        .p_rasterization_state = &prsci,
        .p_multisample_state = &pmsci,
        .p_depth_stencil_state = &ds,
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

// similar dvui_vk_common.acquire.. because of depth buffer
pub fn acquireImageMaybeRecreate(
    swapchain_state: *DvuiVkBackend.WindowContext.SwapchainState,
    gpa: std.mem.Allocator,
    ctx: *DvuiVkBackend.WindowContext,
    sync: FrameSync,
    render_pass: vk.RenderPass, // for framebuffer recreate
    device_local_mem_idx: u32, // mem where to create depth buffer
) !u32 {
    const vkc = ctx.backend.vkc;
    const device = vkc.device;
    const image_index = blk: while (true) {
        if (try swapchain_state.maybeResize(ctx)) {
            // recreate depth
            g_app_state.depth_buffer.deinit(device);
            g_app_state.depth_buffer = try DepthBuffer.init(
                vkc.device,
                ctx.swapchain_state.?.swapchain.extent,
                device_local_mem_idx,
            );
        }
        const shared_attachments = &[1]vk.ImageView{g_app_state.depth_buffer.view};
        _ = try swapchain_state.maybeCreateFramebuffer(gpa, ctx, shared_attachments, render_pass);

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
    return image_index; // autofix
}

const VertexBindings = struct {
    const binding_description = [_]vk.VertexInputBindingDescription{.{
        .binding = 0,
        .stride = @sizeOf(f32) * 4,
        .input_rate = .vertex,
    }};

    const attribute_description = [_]vk.VertexInputAttributeDescription{
        .{
            .binding = 0,
            .location = 0,
            .format = .r32g32b32a32_sfloat,
            .offset = 0, //@offsetOf(Vertex, "pos"),
        },
        // .{
        //     .binding = 0,
        //     .location = 1,
        //     .format = .r8g8b8a8_unorm,
        //     .offset = @offsetOf(Vertex, "col"),
        // },
        // .{
        //     .binding = 0,
        //     .location = 2,
        //     .format = .r32g32_sfloat,
        //     .offset = @offsetOf(Vertex, "uv"),
        // },
    };
};

pub const Cube = struct {
    pub const vertices = [_][4]f32{
        // Front face
        .{ -0.5, -0.5, 0.5, 0 },
        .{ 0.5, -0.5, 0.5, 1 },
        .{ 0.5, 0.5, 0.5, 2 },
        .{ -0.5, 0.5, 0.5, 3 },
        // Back face
        .{ -0.5, -0.5, -0.5, 4 },
        .{ 0.5, -0.5, -0.5, 5 },
        .{ 0.5, 0.5, -0.5, 6 },
        .{ -0.5, 0.5, -0.5, 7 },
    };
    pub const indices = [_]u16{
        // Front
        0, 1, 2,
        2, 3, 0,
        // Right
        1, 5, 6,
        6, 2, 1,
        // Back
        5, 4, 7,
        7, 6, 5,
        // Left
        4, 0, 3,
        3, 7, 4,
        // Top
        3, 2, 6,
        6, 7, 3,
        // Bottom
        4, 5, 1,
        1, 0, 4,
    };
};
