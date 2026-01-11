const std = @import("std");
const Build = std.Build;
const OptimizeMode = std.builtin.OptimizeMode;
const ResolvedTarget = Build.ResolvedTarget;
const Dependency = Build.Dependency;
const sokol = @import("sokol");

const TracyMode = enum {
    off,
    on,
    on_demand, // starts profiling only once tracy connects
};

pub fn build(b: *Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const options = .{
        .ztracy = b.option(
            TracyMode,
            "ztracy",
            "Enable Tracy profiling",
        ) orelse .off,
        .ztracy_fibers = b.option(
            bool,
            "ztracy_fibers",
            "Enable Tracy fiber support",
        ) orelse false,
    };

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
    const vkzig_dep = b.dependency("vulkan", .{
        .registry = @as([]const u8, vk_registry),
    });
    const vkzig_bindings = vkzig_dep.module("vulkan-zig");
    // Add vk-kickstart
    const kickstart_dep = b.lazyDependency("vk_kickstart", .{
        .registry = vk_registry,
        // .verbose = true,
    });
    const kickstart_mod = if (kickstart_dep) |d| d.module("vk-kickstart") else null;
    if (kickstart_mod) |m| m.import_table.put(b.allocator, "vulkan", vkzig_bindings) catch @panic("OOM"); // replace with same version

    const glfw_on = b.option(bool, "glfw", "Use glfw for input and windowing") orelse false;
    const glfw = if (glfw_on) b.lazyDependency("glfw", .{}) else null;
    const glfw_build = if (glfw_on) b.lazyDependency("glfw_build", .{}) else null;

    // ZTracy
    const ztracy =
        //if (options.ztracy != .off)
        if (true)
            b.lazyDependency("ztracy", .{
                .enable_ztracy = options.ztracy != .off,
                .enable_fibers = options.ztracy_fibers,
                .on_demand = options.ztracy == .on_demand,
            })
        else
            null;

    // DVUI
    const dvui = @import("dvui");
    const dvui_dep = b.dependency("dvui", .{ .target = target, .optimize = optimize, .backend = .custom });
    const dvui_module = dvui_dep.module("dvui");
    const stb_source = "vendor/stb/";
    // dvui deps
    dvui_module.link_libc = true;
    dvui_module.addCMacro("INCLUDE_CUSTOM_LIBC_FUNCS", "0");
    dvui_module.addCSourceFiles(.{
        .files = &.{
            stb_source ++ "stb_image_impl.c",
                // stb_source ++ "stb_image_write_impl.c",
                // stb_source ++ "stb_image_libc.c",
                // stb_source ++ "stb_truetype_libc.c",
                // stb_source ++ "stb_truetype_impl.c",
        },
        .flags = &.{ "-DINCLUDE_CUSTOM_LIBC_FUNCS=1", "-DSTBI_NO_STDLIB=1", "-DSTBIW_NO_STDLIB=1" },
    });

    const dvui_vk_backend = b.addModule("dvui_vk_backend", .{ .target = target, .optimize = optimize, .root_source_file = if (glfw_on) b.path("src/dvui_vk_glfw.zig") else b.path("src/dvui_vk_win32.zig") });
    if (glfw_on) {
        dvui_vk_backend.addImport("glfw", glfw.?.module("glfw"));
        dvui_vk_backend.linkLibrary(glfw_build.?.artifact("glfw"));
    }
    dvui_vk_backend.addImport("vk", vkzig_bindings);
    if (kickstart_mod) |m| dvui_vk_backend.addImport("vk_kickstart", m);
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
    if (ztracy) |zt| {
        dvui_vk_backend.addImport("ztracy", zt.module("root"));
        dvui_vk_backend.linkLibrary(zt.artifact("tracy"));
    }

    //
    //   Examples
    const exe_mod = b.addModule("app_mod", .{
        .root_source_file = b.path("examples/app.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const exe = b.addExecutable(.{
        .name = "app_demo",
        .root_module = exe_mod,
    });
    exe.root_module.addImport("dvui", dvui_module);
    exe.root_module.addImport("vulkan", vkzig_bindings);
    if (target.result.os.tag == .windows) {
        exe.win32_manifest = dvui_dep.path("./src/main.manifest");
        exe.subsystem = .Windows;
    }
    b.installArtifact(exe);
    b.step("run-app", "Run demo").dependOn(&b.addRunArtifact(exe).step);

    const exe_standalone_mod = b.addModule("standalone_mod", .{
        .root_source_file = b.path("examples/standalone.zig"),
        .target = target,
        .optimize = optimize,
    });
    const exe_standalone = b.addExecutable(.{
        .name = "standalone_demo",
        .root_module = exe_standalone_mod,
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
        const glslc = b.option(bool, "glslc", "Compile glsl shaders") orelse false;
        const slangc = b.option(bool, "slangc", "Compile slang shaders") orelse false;
        const shaders = compileShaders(b, "src", .{ .slang = if (slangc) .{} else null, .glsl = glslc, .optimize = optimize != .Debug });

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

pub const Shader = struct {
    name: []const u8,
    path: std.Build.LazyPath,
    step: ?*std.Build.Step = null,
};

pub const ShaderCompileOptions = struct {
    pub const SLangOptions = struct {
        row_major: bool = true, // instruct slang to use row major matrices
    };
    glsl: bool = false,
    slang: ?SLangOptions = null,
    optimize: bool = true,
};

pub fn compileShaders(b: *Build, shader_subpath: []const u8, options: ShaderCompileOptions) std.ArrayList(Shader) {
    var shaders: std.ArrayList(Shader) = .{};
    const cwd = b.build_root.handle;
    const dir = cwd.openDir(shader_subpath, .{ .iterate = true }) catch std.debug.panic("can't open dir: '{s}'' from '{s}'", .{ shader_subpath, cwd.realpathAlloc(b.allocator, "") catch unreachable });
    const dbg_print = false;
    if (dbg_print) { // debug print dir
        var dir_path_buf: [256]u8 = undefined;
        const dir_path = dir.realpath(".", &dir_path_buf) catch unreachable;
        std.debug.print("compileShaders({s})\n", .{dir_path});
    }
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
            if (is_slang and !options.glsl) {
                if (dbg_print) std.debug.print("slang: {s}\n", .{f.name});
                const ShaderType = enum { vertex, fragment, compute };
                _ = ShaderType; // autofix
                const file_contents = dir.readFileAlloc(b.allocator, f.name, 10 * 1024 * 1024) catch unreachable;
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
                        if (options.slang == null) {
                            shaders.append(b.allocator, .{ .name = out_name, .path = b.path(out_path) }) catch @panic("OOM"); // compilation not requested, just point to output
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
                            compile.setName("compile slang shaders");
                            compile.addArg(out_name); // output file
                            if (!options.optimize) {
                                compile.addArg("-minimum-slang-optimization");
                                compile.addArg("-g");
                            } else compile.addArg("-O3");
                            if (options.slang.?.row_major) compile.addArg("-matrix-layout-row-major") else compile.addArg("-matrix-layout-column-major");
                            compile.addArg(f.name); // input file

                            compile.setCwd(b.path(shader_subpath));
                            compile.addFileInput(.{ .src_path = .{ .owner = b, .sub_path = b.pathJoin(&.{ shader_subpath, f.name }) } });

                            const gf = b.allocator.create(std.Build.GeneratedFile) catch unreachable;
                            gf.* = std.Build.GeneratedFile{ .step = &compile.step, .path = out_path };
                            shaders.append(b.allocator, Shader{ .name = out_name, .path = std.Build.LazyPath{ .generated = .{ .file = gf } }, .step = &compile.step }) catch @panic("OOM");
                        }
                    }
                }
            }

            if (is_glsl and options.slang == null) { // NOTE: might be broken by now
                const out_name = std.mem.join(b.allocator, "", &.{ f.name, ".spv" }) catch unreachable;
                const out_path = b.pathJoin(&.{ shader_subpath, out_name });
                shaders.append(b.allocator, blk: {
                    if (!options.glsl) break :blk Shader{ .name = out_name, .path = b.path(out_path) }; // compilation not requested, just point to output

                    const compile = b.addSystemCommand(&.{
                        "glslc",
                        "--target-env=vulkan1.2",
                        "-o",
                    });
                    compile.setName("compile glsl shaders");
                    compile.setCwd(b.path(shader_subpath));
                    compile.addFileInput(.{ .src_path = .{ .owner = b, .sub_path = b.pathJoin(&.{ shader_subpath, f.name }) } });
                    compile.addArg(out_name); // output file
                    compile.addArg(f.name); // input file

                    const gf = b.allocator.create(std.Build.GeneratedFile) catch unreachable;
                    gf.* = std.Build.GeneratedFile{ .step = &compile.step, .path = out_path };
                    break :blk Shader{ .name = out_name, .path = std.Build.LazyPath{ .generated = .{ .file = gf } } };
                }) catch @panic("OOM");
            }
        }
    }
    return shaders;
}
