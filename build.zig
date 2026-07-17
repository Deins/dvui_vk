const std = @import("std");
const Build = std.Build;

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
    var link_shaders_to = std.ArrayList(*std.Build.Step.Compile).initCapacity(b.allocator, 16) catch unreachable;

    // Vulkan
    const vk_registry_opt = b.option([]const u8, "vk_registry", "Path to vulkan registry vk.xml");
    const vk_registry: Build.LazyPath = if (vk_registry_opt) |ps| Build.LazyPath{ .cwd_relative = ps } else blk: {
        if (b.graph.environ_map.get("VULKAN_SDK")) |vk_path| {
            break :blk Build.LazyPath{ .cwd_relative = b.pathJoin(&.{ vk_path, "share", "vulkan", "registry", "vk.xml" }) };
        }

        // fallback to registry from lazy dependency
        const vk_headers = b.lazyDependency("vulkan_headers", .{});
        if (vk_headers) |h| {
            std.log.info("VulkanSDK not found - falling back to vulkan_headers dependency", .{});
            break :blk h.path("registry/vk.xml");
        } else return;

        // std.log.err("VULKAN_SDK not found. Pass in -Dvk_registry=/path/to/vk.xml or install vulkan SDK.", .{});
        // break :blk "/usr/share/vulkan/registry/vk.xml"; // best guess
    };
    const vkzig_dep = b.dependency("vulkan", .{
        .registry = vk_registry,
    });
    const vkzig_bindings = vkzig_dep.module("vulkan-zig");
    const low_x11 = b.option(bool, "low_x11", "Enable X11 in the low backend") orelse (target.result.os.tag == .linux);
    const low_wayland = b.option(bool, "low_wayland", "Enable Wayland in the low backend") orelse (target.result.os.tag == .linux);
    const low = b.dependency("low", .{
        .target = target,
        .optimize = optimize,
        .x11 = low_x11,
        .wayland = low_wayland,
        .vk_extras = true,
    });

    // ZTracy
    const ztracy =
        if (options.ztracy != .off)
            b.lazyDependency("ztracy", .{
                .enable_ztracy = options.ztracy != .off,
                .enable_fibers = options.ztracy_fibers,
                .on_demand = options.ztracy == .on_demand,
                .target = target,
                .optimize = optimize,
            })
        else
            null;

    // DVUI
    const dvui = @import("dvui");
    const dvui_dep = b.dependency("dvui", .{ .target = target, .optimize = optimize, .backend = .custom, .@"tree-sitter" = false });
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

    const dvui_vk_backend = b.addModule("dvui_vk_backend", .{
        .target = target,
        .optimize = optimize,
        .root_source_file = b.path("src/dvui_vk_low.zig"),
    });
    dvui_vk_backend.addImport("low", low.module("low"));
    dvui_vk_backend.addImport("vk", vkzig_bindings);
    dvui.linkBackend(dvui_module, dvui_vk_backend);
    if (target.result.os.tag == .windows) {
        // const dvui_win = b.createModule(.{ .target = target, .optimize = optimize, .root_source_file = dvui_dep.path("src/backends/dx11.zig") });
        if (b.lazyDependency("win32", .{})) |zigwin32| {
            // dvui_win.addImport("win32", zigwin32.module("win32"));
            dvui_vk_backend.addImport("win32", zigwin32.module("win32"));
        }
        // dvui_win.addImport("dvui", dvui_module);
        // dvui_vk_backend.addImport("dvui_win", dvui_win);
    }
    if (ztracy) |zt| {
        dvui_vk_backend.addImport("ztracy", zt.module("root"));
        dvui_vk_backend.linkLibrary(zt.artifact("tracy"));
    }

    //
    //   Examples
    {
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
        if (target.query.isNative() and target.result.os.tag == .linux and target.result.abi == .gnu) {
            if (patchedGlibcLibcFile(b)) |lf| exe.setLibCFile(lf);
        }
        exe.root_module.addImport("dvui", dvui_module);
        exe.root_module.addImport("vulkan", vkzig_bindings);
        if (target.result.os.tag == .windows) {
            exe.win32_manifest = dvui_dep.path("./src/main.manifest");
            exe.subsystem = .Windows;
        }
        b.installArtifact(exe);
        b.step("run-app", "Run demo").dependOn(&b.addRunArtifact(exe).step);
        link_shaders_to.append(b.allocator, exe) catch unreachable;
    }

    {
        const exe_standalone_mod = b.addModule("standalone_mod", .{
            .root_source_file = b.path("examples/standalone.zig"),
            .target = target,
            .optimize = optimize,
        });
        const exe_standalone = b.addExecutable(.{
            .name = "standalone_demo",
            .root_module = exe_standalone_mod,
        });
        if (target.query.isNative() and target.result.os.tag == .linux and target.result.abi == .gnu) {
            if (patchedGlibcLibcFile(b)) |lf| exe_standalone.setLibCFile(lf);
        }
        exe_standalone.root_module.addImport("dvui", dvui_module);
        exe_standalone.root_module.addImport("vulkan", vkzig_bindings);
        if (target.result.os.tag == .windows) {
            exe_standalone.win32_manifest = dvui_dep.path("./src/main.manifest");
            exe_standalone.subsystem = .Windows;
        }
        b.installArtifact(exe_standalone);
        b.step("run", "Run demo").dependOn(&b.addRunArtifact(exe_standalone).step);
        link_shaders_to.append(b.allocator, exe_standalone) catch unreachable;
    }

    { // Shaders
        const glslc = b.option(bool, "glslc", "Compile glsl shaders") orelse false;
        const slangc = b.option(bool, "slangc", "Compile slang shaders") orelse false;
        const shaders = compileShaders(b, "src", .{ .slang = if (slangc) .{} else null, .glsl = glslc, .optimize = optimize != .Debug });

        // add shader modules
        for (shaders.items) |shader| {
            for (link_shaders_to.items) |mod| mod.root_module.addAnonymousImport(shader.name, .{
                .root_source_file = shader.path,
            });
            if (shader.step) |step| {
                for (link_shaders_to.items) |mod| mod.step.dependOn(step);
            }
        }
    }
}

// TODO: get rid of this in future
// Workaround for glibc 2.43+ (GCC 16) adding .sframe sections with R_X86_64_PC64
// relocations to crt1.o, which Zig's self-hosted ELF linker can't handle.
// Strips .sframe from a local copy of crt1.o and produces a libc file pointing at it.
// Returns null on non-Linux hosts where the issue doesn't apply.
fn patchedGlibcLibcFile(b: *Build) ?Build.LazyPath {
    if (b.graph.host.result.os.tag != .linux) return null;

    const patch_dir = b.cache_root.join(b.allocator, &.{"crt-patched"}) catch return null;

    // Copy system CRT files + glibc stubs to our dir, then strip .sframe from crt1.o.
    // The libc file's crt_dir is also the library search path for glibc stubs (libdl.a etc.),
    // so we need all small files (*.o, stub *.a < 200KB, linker-script *.so < 10KB) not just CRT.
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
    var shaders: std.ArrayList(Shader) = .empty;
    const cwd = b.build_root.handle;
    const dir = cwd.openDir(b.graph.io, shader_subpath, .{ .iterate = true }) catch std.debug.panic("can't open dir: '{s}'", .{shader_subpath});
    const dbg_print = false;
    if (dbg_print) { // debug print dir
        var dir_path_buf: [256]u8 = undefined;
        const dir_path = dir.realpath(".", &dir_path_buf) catch unreachable;
        std.debug.print("compileShaders({s})\n", .{dir_path});
    }
    var it = dir.iterate();
    while (it.next(b.graph.io) catch unreachable) |f| {
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
                const file_contents = dir.readFileAlloc(b.graph.io, f.name, b.allocator, .limited(10 * 1024 * 1024)) catch unreachable;
                defer b.allocator.free(file_contents);
                const shader_types: []const []const u8 = &.{ "[shader(\"vertex\")]", "[shader(\"fragment\")]" };
                const base_name = blk: {
                    const main_suffix = ".main.slang";
                    if (std.mem.endsWith(u8, f.name, main_suffix)) {
                        break :blk f.name[0 .. f.name.len - main_suffix.len];
                    }
                    break :blk f.name[0 .. f.name.len - ".slang".len];
                };
                for (shader_types, 0..) |_, shader_type_idx| {
                    if (std.mem.indexOfAnyPos(u8, file_contents, 0, shader_types[shader_type_idx])) |_| {
                        const out_name = std.mem.join(b.allocator, "", &.{
                            base_name,
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
                                "-matrix-layout-row-major",
                                "-fvk-use-gl-layout",
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
