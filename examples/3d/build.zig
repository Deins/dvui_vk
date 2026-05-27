const std = @import("std");
const Build = std.Build;
const dvui_vk = @import("dvui_vk");

pub fn build(b: *Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const glfw_on = b.option(bool, "glfw", "Use glfw for input and windowing") orelse (target.result.os.tag != .windows);
    const dvui_vk_dep = b.dependency("dvui_vk", .{
        .target = target,
        .optimize = optimize,
        .glfw = glfw_on,
    });

    // Vulkan
    const vk_registry_opt = b.option([]const u8, "vk_registry", "Path to vulkan registry vk.xml");
    const vk_registry: Build.LazyPath = if (vk_registry_opt) |ps| Build.LazyPath{ .cwd_relative = ps } else blk: {
        if (b.graph.environ_map.get("VULKAN_SDK")) |vk_path| {
            break :blk Build.LazyPath{ .cwd_relative = b.pathJoin(&.{ vk_path, "share", "vulkan", "registry", "vk.xml" }) };
        }

        // fallback to registry from lazy dependency
        const vk_headers = dvui_vk_dep.builder.lazyDependency("vulkan_headers", .{});
        if (vk_headers) |h| {
            std.log.info("VulkanSDK not found - falling back to vulkan_headers dependency", .{});
            break :blk h.path("registry/vk.xml");
        } else return;

        // std.log.err("VULKAN_SDK not found. Pass in -Dvk_registry=/path/to/vk.xml or install vulkan SDK.", .{});
        // break :blk "/usr/share/vulkan/registry/vk.xml"; // best guess
    };
    const vkzig_dep = dvui_vk_dep.builder.dependency("vulkan", .{
        .registry = vk_registry,
    });

    const dvui_dep = dvui_vk_dep.builder.dependency("dvui", .{ .target = target, .optimize = optimize, .backend = .custom, .@"tree-sitter" = false });
    const dvui_module = dvui_dep.module("dvui");

    //
    //   Examples
    const exe_standalone_mod = b.addModule("3d", .{
        .root_source_file = b.path("3d.zig"),
        .target = target,
        .optimize = optimize,
    });
    const exe_standalone = b.addExecutable(.{
        .name = "3d",
        .root_module = exe_standalone_mod,
    });
    if (patchedGlibcLibcFile(b)) |lf| exe_standalone.setLibCFile(lf);
    exe_standalone.root_module.addImport("dvui", dvui_module);
    exe_standalone.root_module.addImport("vulkan", vkzig_dep.module("vulkan-zig"));
    // exe_standalone.root_module.addImport("zig_math", b.dependency("zig_math", .{ .target = target }).module("zig_math"));
    exe_standalone.root_module.addImport("zmath", b.dependency("zmath", .{ .target = target, .optimize = .ReleaseFast }).module("root"));
    exe_standalone.root_module.addImport("zgltf", b.dependency("zgltf", .{ .target = target, .optimize = optimize }).module("zgltf"));

    // if (target.result.os.tag == .windows) {
    //     exe_standalone.win32_manifest = dvui_dep.path("./src/main.manifest");
    //     exe_standalone.subsystem = .Windows;
    // }
    b.installArtifact(exe_standalone);
    const run = b.addRunArtifact(exe_standalone);
    run.cwd = .{ .src_path = .{ .owner = b, .sub_path = "." } };
    b.step("run", "Run demo").dependOn(&run.step);

    { // Shaders
        const slangc = b.option(bool, "slangc", "Compile slang shaders") orelse false;
        const shaders = dvui_vk.compileShaders(b, ".", .{ .slang = if (slangc) .{} else null, .optimize = optimize != .Debug });
        for (shaders.items) |shader| {
            exe_standalone.root_module.addAnonymousImport(shader.name, .{
                .root_source_file = shader.path,
            });
            if (shader.step) |step| exe_standalone.step.dependOn(step);
        }
    }
}

fn patchedGlibcLibcFile(b: *Build) ?Build.LazyPath {
    if (b.graph.host.result.os.tag != .linux) return null;

    const patch_dir = b.cache_root.join(b.allocator, &.{"crt-patched"}) catch return null;

    const patch = b.addSystemCommand(&[_][]const u8{ "sh", "-c", b.fmt(
        "CRT1=$(cc -print-file-name=crt1.o 2>/dev/null || echo /usr/lib/crt1.o) && " ++
        "CRTDIR=$(dirname \"$CRT1\") && mkdir -p {0s} && " ++
        "find \"$CRTDIR\" -maxdepth 1 \\( -name \"*.o\" -o \\( -name \"lib*.a\" -size -200k \\) -o \\( -name \"lib*.so\" -size -10k \\) \\) -exec cp -P {{}} {0s}/ \";\" && " ++
        "objcopy --remove-section .sframe {0s}/crt1.o 2>/dev/null; true",
        .{patch_dir},
    ) });

    const wf = b.addWriteFiles();
    wf.step.dependOn(&patch.step);
    return wf.add("patched-libc.txt", b.fmt(
        "include_dir=/usr/include\nsys_include_dir=/usr/include\ncrt_dir={s}\nmsvc_lib_dir=\nkernel32_lib_dir=\ngcc_dir=\n",
        .{patch_dir},
    ));
}
