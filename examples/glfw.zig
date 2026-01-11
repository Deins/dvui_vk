const std = @import("std");
const glfw = @import("glfw");
const dvui = @import("dvui");
const vk = @import("vulkan");

const slog = std.log.scoped(.main);
const DvuiVkBackend = dvui.backend;

pub const VkContext = struct {
const vk = @import("vulkan");
const vkk = @import("vk-kickstart");
const std = @import("std");
const builtin = @import("builtin");
const c = @import("c.zig");
const GraphicsContext = @This();

pub const Instance = vk.InstanceProxy;
pub const Device = vk.DeviceProxy;
pub const Queue = vk.QueueProxy;
pub const CommandBuffer = vk.CommandBufferProxy;

allocator: std.mem.Allocator,
instance: Instance,
debug_messenger: ?vk.DebugUtilsMessengerEXT,
device: Device,
physical_device: vkk.PhysicalDevice,
surface: vk.SurfaceKHR,
graphics_queue_index: u32,
present_queue_index: u32,
graphics_queue: Queue,
present_queue: Queue,

pub fn init(allocator: std.mem.Allocator, window: *glfw.Window) !GraphicsContext {
    const is_debug = builtin.mode == .Debug;

    const instance = try vkk.instance.create(
        allocator,
        c.glfwGetInstanceProcAddress,
        .{
            .required_api_version = vk.API_VERSION_1_3,
            .enable_validation = true,
            .debug_messenger = .{ .enable = true },
            .enabled_validation_features = &.{.best_practices_ext},
        },
        null,
    );
    errdefer allocator.destroy(instance.wrapper);
    errdefer instance.destroyInstance(null);

    const debug_messenger = switch (is_debug) {
        true => try vkk.instance.createDebugMessenger(instance, .{}, null),
        false => .null_handle,
    };
    errdefer if (is_debug) vkk.instance.destroyDebugMessenger(instance, debug_messenger, null);

    var surface: vk.SurfaceKHR = .null_handle;
    if (c.glfwCreateWindowSurface(instance.handle, window, null, &surface) != .success)
        return error.SurfaceInitFailed;
    errdefer instance.destroySurfaceKHR(surface, null);

    const physical_device = try vkk.PhysicalDevice.select(allocator, instance, .{
        .surface = surface,
        .required_api_version = vk.API_VERSION_1_2,
        .required_extensions = &.{
            vk.extensions.khr_swapchain.name,
            vk.extensions.khr_ray_tracing_pipeline.name,
            vk.extensions.khr_acceleration_structure.name,
            vk.extensions.khr_deferred_host_operations.name,
        },
        .required_features = .{
            .sampler_anisotropy = .true,
        },
        .required_features_12 = .{
            .descriptor_indexing = .true,
        },
    });
    errdefer physical_device.deinit();

    std.log.info("selected {s}", .{physical_device.name()});

    var rt_features = vk.PhysicalDeviceRayTracingPipelineFeaturesKHR{
        .ray_tracing_pipeline = .true,
    };

    const device = try vkk.device.create(allocator, instance, &physical_device, @ptrCast(&rt_features), null);
    errdefer allocator.destroy(device.wrapper);
    errdefer device.destroyDevice(null);

    const graphics_queue_index = physical_device.graphics_queue_index;
    const present_queue_index = physical_device.present_queue_index orelse return error.NoPresentQueue;
    const graphics_queue_handle = device.getDeviceQueue(graphics_queue_index, 0);
    const present_queue_handle = device.getDeviceQueue(present_queue_index, 0);
    const graphics_queue = Queue.init(graphics_queue_handle, device.wrapper);
    const present_queue = Queue.init(present_queue_handle, device.wrapper);

    return .{
        .allocator = allocator,
        .instance = instance,
        .debug_messenger = debug_messenger,
        .device = device,
        .physical_device = physical_device,
        .surface = surface,
        .graphics_queue_index = graphics_queue_index,
        .present_queue_index = present_queue_index,
        .graphics_queue = graphics_queue,
        .present_queue = present_queue,
    };
}

pub fn deinit(self: *GraphicsContext) void {
    self.device.destroyDevice(null);
    self.instance.destroySurfaceKHR(self.surface, null);
    vkk.instance.destroyDebugMessenger(self.instance, self.debug_messenger, null);
    self.instance.destroyInstance(null);
    self.physical_device.deinit();
    self.allocator.destroy(self.instance.wrapper);
    self.allocator.destroy(self.device.wrapper);
}

};

pub const AppState = struct {
    const SyncObjects = DvuiVkBackend.SyncObjects;

    backend: *DvuiVkBackend.VkBackend,
    render_pass: vk.RenderPass,
    sync: SyncObjects,
    command_buffers: []vk.CommandBuffer,

    pub fn init(gpa: std.mem.Allocator) !AppState {
        _ = gpa; // autofix

        {
            try glfw.init();
            var major: i32 = 0;
            var minor: i32 = 0;
            var rev: i32 = 0;
            glfw.getVersion(&major, &minor, &rev);
            slog.info("GLFW {}.{}.{} vk_support: {}", .{ major, minor, rev, glfw.vulkanSupported() });
            // initVulkanLoader(loader);
        }

        // vk_dll.init() catch |err| std.debug.panic("Failed to init Vulkan: {}", .{err});
        // errdefer vk_dll.deinit();
        // const loader = vk_dll.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse @panic("Failed to lookup Vulkan VkGetInstanceProcAddr: {}");
        const loader = glfw.getInstanceProcAddress(0, "vkCreateInstance");
        
        GraphicsContext.init(gpa, )

        // var b = try gpa.create(DvuiVkBackend.VkBackend);
        // b.* = DvuiVkBackend.VkBackend.init(gpa, undefined);
        // errdefer b.deinit();

        return .{
            .backend = undefined,
            .command_buffers = undefined,
            .render_pass = undefined,
            .sync = undefined,
        };
    }

    pub fn deinit(self: AppState, gpa: std.mem.Allocator) void {
        // defer vk_dll.deinit();
        defer glfw.terminate();
        defer gpa.destroy(self.backend);
        defer self.backend.deinit();
        defer self.backend.vkc.device.destroyRenderPass(self.render_pass, null);
        defer self.sync.deinit(self.backend.vkc.device);
        defer gpa.free(self.command_buffers);
    }
};
pub var g_app_state: AppState = undefined;

pub fn main() !void {
    dvui.Backend.Common.windowsAttachConsole() catch {};
    dvui.Examples.show_demo_window = true;

    var gpa_instance = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = gpa_instance.allocator();
    defer _ = gpa_instance.deinit();

    g_app_state = try AppState.init(gpa);
    defer g_app_state.deinit(gpa);
    defer g_app_state.backend.vkc.device.queueWaitIdle(g_app_state.backend.vkc.graphics_queue.handle) catch {}; // let gpu finish its work on exit, otherwise we will get validation errors

    //Example of something that fails with GLFW_NOT_INITIALIZED - but will continue with execution
    //var monitor: ?*glfw.Monitor = glfw.getPrimaryMonitor();

    const window: *glfw.Window = try glfw.createWindow(800, 640, "glfw", null, null);
    defer glfw.destroyWindow(window);

    while (!glfw.windowShouldClose(window)) {
        if (glfw.getKey(window, glfw.KeyEscape) == glfw.Press) {
            glfw.setWindowShouldClose(window, true);
        }

        glfw.pollEvents();
    }
}
