const std = @import("std");
const Build = std.Build;
const OptimizeMode = std.builtin.OptimizeMode;
const ResolvedTarget = Build.ResolvedTarget;
const Dependency = Build.Dependency;
const sokol = @import("sokol");

pub fn build(b: *Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Vulkan
    const vk_registry_opt = b.option([]const u8, "vk_registry", "Path to vulkan registry vk.xml");
    const vk_registry = vk_registry_opt orelse blk: {
        const env = std.process.getEnvMap(b.allocator) catch unreachable;
        if (env.get("VULKAN_SDK")) |vk_path| {
            break :blk b.pathJoin(&.{ vk_path, "share", "vulkan", "registry", "vk.xml" });
        }
        std.log.err("VULKAN_SDK not found. Pass in -Dvk_registry=/path/to/vk.xml or install vulkan SDK.", .{});
        break :blk "/usr/share/vulkan/registry/vk.xml"; // best guess
    };
    const vkzig_dep = b.dependency("vulkan_zig", .{
        .registry = @as([]const u8, vk_registry),
    });
    const vkzig_bindings = vkzig_dep.module("vulkan-zig");
    // Add vk-kickstart
    const kickstart_dep = b.dependency("vk_kickstart", .{
        .registry = vk_registry,
        .enable_validation = if (optimize == .Debug) true else false,
        // .verbose = true,
    });
    const kickstart_mod = kickstart_dep.module("vk-kickstart");
    kickstart_mod.import_table.put(b.allocator, "vulkan", vkzig_bindings) catch @panic("OOM"); // replace with same version

    // DVUI
    const dvui = @import("dvui");

    const dvui_dep = b.dependency("dvui", .{ .target = target, .optimize = optimize, .backend = .custom });
    const dvui_module = dvui_dep.module("dvui");

    const dvui_vk_backend = b.addModule("dvui_vk_backend", .{ .target = target, .optimize = optimize, .root_source_file = b.path("src/dvui_vulkan.zig") });
    dvui_vk_backend.addImport("vk", vkzig_bindings);
    dvui_vk_backend.addImport("vk_kickstart", kickstart_mod);
    dvui.linkBackend(dvui_module, dvui_vk_backend);
    if (target.result.os.tag == .windows) {
        // const dvui_win = b.createModule(.{ .target = target, .optimize = optimize, .root_source_file = dvui_dep.path("src/backends/dx11.zig") });
        if (b.lazyDependency("win32", .{})) |zigwin32| {
            // dvui_win.addImport("win32", zigwin32.module("win32"));
            dvui_vk_backend.addImport("win32", zigwin32.module("win32"));
        }
        // dvui_win.addImport("dvui", dvui_module);
        // dvui_vk_backend.addImport("dvui_win", dvui_win);
    } else @panic("TODO");

    //
    //   Examples
    const exe = b.addExecutable(.{
        .name = "app_demo",
        .root_source_file = b.path("examples/app.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("dvui", dvui_module);
    exe.root_module.addImport("vulkan", vkzig_bindings);
    if (target.result.os.tag == .windows) {
        exe.win32_manifest = dvui_dep.path("./src/main.manifest");
        exe.subsystem = .Windows;
    }
    b.installArtifact(exe);
    b.step("run-app", "Run demo").dependOn(&b.addRunArtifact(exe).step);

    const exe_standalone = b.addExecutable(.{
        .name = "standalone_demo",
        .root_source_file = b.path("examples/standalone.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe_standalone.root_module.addImport("dvui", dvui_module);
    exe_standalone.root_module.addImport("vulkan", vkzig_bindings);
    if (target.result.os.tag == .windows) {
        exe_standalone.win32_manifest = dvui_dep.path("./src/main.manifest");
        exe_standalone.subsystem = .Windows;
    }
    b.installArtifact(exe_standalone);
    b.step("run", "Run demo").dependOn(&b.addRunArtifact(exe_standalone).step);

    { // Shaders
        const Shader = struct {
            name: []const u8,
            path: std.Build.LazyPath,
            step: ?*std.Build.Step = null,
        };
        var shaders = std.ArrayList(Shader).init(b.allocator);
        const glslc = b.option(bool, "glslc", "Compile glsl shaders") orelse false;
        const slangc = b.option(bool, "slangc", "Compile slang shaders") orelse false;

        const shader_subpath = "src";
        const dir = std.fs.cwd().openDir(shader_subpath, .{ .iterate = true }) catch unreachable;
        var it = dir.iterate();
        while (it.next() catch unreachable) |f| {
            if (f.kind == .file) {
                const is_glsl =
                    std.mem.endsWith(u8, f.name, ".vert") or
                    std.mem.endsWith(u8, f.name, ".frag") or
                    std.mem.endsWith(u8, f.name, ".tesc") or
                    std.mem.endsWith(u8, f.name, ".tese") or
                    std.mem.endsWith(u8, f.name, ".geom") or
                    std.mem.endsWith(u8, f.name, ".comp");
                const is_slang = std.mem.endsWith(u8, f.name, ".slang");

                if (is_slang and !glslc) {
                    const shader_path = b.pathJoin(&.{ shader_subpath, f.name });
                    const ShaderType = enum { vertex, fragment, compute };
                    _ = ShaderType; // autofix
                    const file_contents = std.fs.cwd().readFileAlloc(b.allocator, shader_path, 10 * 1024 * 1024) catch unreachable;
                    defer b.allocator.free(file_contents);
                    const shader_types: []const []const u8 = &.{ "[shader(\"vertex\")]", "[shader(\"fragment\")]" };
                    for (shader_types, 0..) |_, shader_type_idx| {
                        if (std.mem.indexOfAnyPos(u8, file_contents, 0, shader_types[shader_type_idx])) |_| {
                            const out_name = std.mem.join(b.allocator, "", &.{
                                f.name[0 .. f.name.len - ".slang".len],
                                switch (shader_type_idx) {
                                    0 => ".vert",
                                    1 => ".frag",
                                    else => unreachable,
                                },
                                ".spv",
                            }) catch unreachable;
                            const out_path = b.pathJoin(&.{ shader_subpath, out_name });
                            if (!slangc) {
                                shaders.append(.{ .name = out_name, .path = b.path(out_path) }) catch @panic("OOM"); // compilation not requested, just point to output
                            } else {
                                const compile = b.addSystemCommand(&.{
                                    "slangc",
                                    "-target",
                                    "spirv",
                                    "-entry",
                                    switch (shader_type_idx) {
                                        0 => "vertexMain",
                                        1 => "fragmentMain",
                                        else => unreachable,
                                    },
                                    "-o",
                                });
                                compile.addArg(out_name); // output file
                                if (optimize == .Debug) compile.addArg("-minimum-slang-optimization") else compile.addArg("-O3");
                                compile.addArg(f.name); // input file

                                compile.setCwd(b.path(shader_subpath));

                                const gf = b.allocator.create(std.Build.GeneratedFile) catch unreachable;
                                gf.* = std.Build.GeneratedFile{ .step = &compile.step, .path = out_path };
                                shaders.append(Shader{ .name = out_name, .path = std.Build.LazyPath{ .generated = .{ .file = gf } } }) catch @panic("OOM");
                            }
                        }
                    }
                }

                if (is_glsl and !slangc) {
                    const out_name = std.mem.join(b.allocator, "", &.{ f.name, ".spv" }) catch unreachable;
                    const out_path = b.pathJoin(&.{ shader_subpath, out_name });
                    shaders.append(blk: {
                        if (!glslc) break :blk Shader{ .name = out_name, .path = b.path(out_path) }; // compilation not requested, just point to output

                        const compile = b.addSystemCommand(&.{
                            "glslc",
                            "--target-env=vulkan1.2",
                            "-o",
                        });
                        compile.setCwd(b.path(shader_subpath));
                        compile.addArg(out_name); // output file
                        compile.addArg(f.name); // input file

                        const gf = b.allocator.create(std.Build.GeneratedFile) catch unreachable;
                        gf.* = std.Build.GeneratedFile{ .step = &compile.step, .path = out_path };
                        break :blk Shader{ .name = out_name, .path = std.Build.LazyPath{ .generated = .{ .file = gf } } };
                    }) catch @panic("OOM");
                }
            }
        }

        // add shader modules
        for (shaders.items) |shader| {
            exe.root_module.addAnonymousImport(shader.name, .{
                .root_source_file = shader.path,
            });
            exe_standalone.root_module.addAnonymousImport(shader.name, .{
                .root_source_file = shader.path,
            });
            // exe_unit_tests.root_module.addAnonymousImport(shader.name, .{
            //     .root_source_file = shader.path,
            // });
            if (shader.step) |step| {
                exe.step.dependOn(step);
                exe_standalone.step.dependOn(step);
            }
            // if (shader.step) |step| exe_unit_tests.step.dependOn(step);
        }
    }
}
