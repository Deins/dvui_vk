const std = @import("std");
const builtin = @import("builtin");
const dvui = @import("dvui");
const vk = @import("vulkan");
// const zm = @import("zig_math");
const zm = @import("zmath");

const DvuiVkBackend = dvui.backend;
const win32 = if (builtin.target.os.tag == .windows) DvuiVkBackend.win32 else void;
const win = if (builtin.target.os.tag == .windows) DvuiVkBackend.win else void;
const vk_dll = DvuiVkBackend.vk_dll;
const SyncObjects = DvuiVkBackend.SyncObjects;
const slog = std.log.scoped(.main);

const vs_spv align(64) = @embedFile("3d.vert.spv").*;
const fs_spv align(64) = @embedFile("3d.frag.spv").*;

const Mat4 = zm.Matrix(4, 4, f32, .{});

pub const AppState = struct {
    backend: *DvuiVkBackend.VkBackend,
    render_pass: vk.RenderPass,
    sync: SyncObjects,
    command_buffers: []vk.CommandBuffer,

    pub fn init(gpa: std.mem.Allocator) !AppState {
        const window_class = win32.L("DvuiWindow");
        win.RegisterClass(window_class, .{}) catch win32.panicWin32(
            "RegisterClass",
            win32.GetLastError(),
        );

        vk_dll.init() catch |err| std.debug.panic("Failed to init Vulkan: {}", .{err});
        errdefer vk_dll.deinit();
        const loader = vk_dll.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse @panic("Failed to lookup Vulkan VkGetInstanceProcAddr: {}");

        var b = try gpa.create(DvuiVkBackend.VkBackend);
        b.* = DvuiVkBackend.VkBackend.init(gpa, undefined);
        errdefer b.deinit();

        // init backend (creates and owns OS window)
        var window_context: *DvuiVkBackend.Context = try b.allocContext();
        window_context.* = .{
            .backend = b,
            .dvui_window = try dvui.Window.init(@src(), gpa, DvuiVkBackend.dvuiBackend(window_context), .{}),
            .hwnd = undefined,
        };
        try win.initWindow(window_context, .{
            .registered_class = window_class,
            .dvui_gpa = gpa,
            .gpa = gpa,
            .title = "3d example",
        });

        b.vkc = try DvuiVkBackend.VkContext.init(gpa, loader, window_context);

        window_context.swapchain_state = try DvuiVkBackend.Context.SwapchainState.init(window_context, .{
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

        const sync = try DvuiVkBackend.SyncObjects.init(b.vkc.device);
        errdefer sync.deinit(b.vkc.device);

        return .{
            .backend = b,
            .command_buffers = command_buffers,
            .render_pass = render_pass,
            .sync = sync,
        };
    }

    pub fn deinit(self: AppState, gpa: std.mem.Allocator) void {
        defer vk_dll.deinit();
        defer gpa.destroy(self.backend);
        defer self.backend.deinit();
        defer self.backend.vkc.device.destroyRenderPass(self.render_pass, null);
        defer self.sync.deinit(self.backend.vkc.device);
        defer gpa.free(self.command_buffers);
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
            .pipeline = try createPipeline(dev, layout, g_app_state.render_pass, vk_alloc),
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
            .min_depth = -1,
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
            model: zm.Mat,
            view: zm.Mat,
            projection: zm.Mat,
        };
        const t: f32 = @as(f32, @floatFromInt(self.timer.read() / std.time.ns_per_ms)) / 1000; // sec f32
        const rotation = zm.mul(zm.rotationX(t), zm.rotationY(t));
        const push_constants = PushConstants{
            .model = zm.mul(rotation, zm.translation(0, 0, -10)),
            .view = zm.identity(),
            .projection = zm.perspectiveFovRh(std.math.degreesToRadians(60.0), framebuffer_size.w / framebuffer_size.h, 0.01, 100),
        };
        if (@sizeOf(f32) * 4 * 4 * 3 != @sizeOf(@TypeOf(push_constants))) unreachable;
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

pub const max_frames_in_flight = 3;
pub const vsync = false;

pub fn main() !void {
    dvui.Backend.Common.windowsAttachConsole() catch {};
    dvui.Examples.show_demo_window = false;

    var gpa_instance = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = gpa_instance.allocator();
    defer _ = gpa_instance.deinit();

    g_app_state = try AppState.init(gpa);
    defer g_app_state.deinit(gpa);
    defer g_app_state.backend.vkc.device.queueWaitIdle(g_app_state.backend.vkc.graphics_queue.handle) catch {}; // let gpu finish its work on exit, otherwise we will get validation errors

    g_scene = try Scene.init();
    defer g_scene.deinit();

    var current_frame_in_flight: u32 = 0;
    main_loop: while (g_app_state.backend.contexts.items.len > 0) {
        defer current_frame_in_flight = (current_frame_in_flight + 1) % max_frames_in_flight;
        // slog.info("frame: {}", .{current_frame_in_flight});
        switch (win.serviceMessageQueue()) {
            .queue_empty => {
                for (g_app_state.backend.contexts.items) |ctx| {
                    try paint(&g_app_state, ctx, current_frame_in_flight);
                    g_app_state.backend.prev_frame_stats = g_app_state.backend.renderer.?.stats;
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

pub fn drawGUI(ctx: *DvuiVkBackend.Context) void {
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

pub fn paint(app_state: *AppState, ctx: *DvuiVkBackend.Context, current_frame_in_flight: usize) !void {
    const b = ctx.backend;
    const gpa = b.gpa;
    const render_pass = app_state.render_pass;
    const sync = &app_state.sync;

    if (ctx.last_pixel_size.w < 1 or ctx.last_pixel_size.h < 1) return;

    { // check/wait for previous frame to finish
        const result = try b.vkc.device.waitForFences(1, @ptrCast(&sync.in_flight_fences[current_frame_in_flight]), .true, std.math.maxInt(u64));
        std.debug.assert(result == .success);
    }

    const image_index = blk: while (true) {
        if (ctx.swapchain_state.?.framebuffers.len == 0) {
            ctx.swapchain_state.?.framebuffers = try DvuiVkBackend.createFramebuffers(
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
            .success => break :blk next_image_result.image_index,
            .suboptimal_khr => {
                try ctx.swapchain_state.?.recreate(ctx);
                continue; // need framebuffer
            },
            else => |err| std.debug.panic("Failed to acquire next frame: {}", .{err}),
        }
    };
    // slog.debug("paint current frame {} image_index {}", .{ current_frame_in_flight, image_index });

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

    g_scene.draw(command_buffer);
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
    const frame_sync_objects = DvuiVkBackend.FrameSyncObjects{
        .image_available_semaphore = sync.image_available_semaphores[current_frame_in_flight],
        .render_finished_semaphore = sync.render_finished_semaphores[current_frame_in_flight],
        .in_flight_fence = sync.in_flight_fences[current_frame_in_flight],
    };
    cmd.endRenderPass();
    cmd.endCommandBuffer() catch |err| std.debug.panic("Failed to end vulkan cmd buffer: {}", .{err});

    if (!try DvuiVkBackend.present(
        ctx,
        app_state.command_buffers[current_frame_in_flight],
        frame_sync_objects,
        ctx.swapchain_state.?.swapchain.handle,
        image_index,
    )) {
        // should_recreate_swapchain = true;
        slog.err("present failed!", .{});
    }
    // slog.debug("frame done", .{});
}

fn createPipeline(
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
        .polygon_mode = .line,
        .cull_mode = .{ .back_bit = false },
        .front_face = .clockwise,
        .depth_bias_enable = .false,
        .depth_bias_constant_factor = 0,
        .depth_bias_clamp = 0,
        .depth_bias_slope_factor = 0,
        .line_width = 4,
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
