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
    const dvui_dep = b.dependency("dvui", .{
        .target = target,
        .optimize = optimize,
        .backend = .custom,
        .renderer = .vulkan_external,
        .@"tree-sitter" = false,
    });
    const dvui_module = dvui_dep.module("dvui");
    const dvui_vulkan_renderer = dvui_dep.module("dvui_vulkan_renderer");
    dvui_vulkan_renderer.addImport("vk", vkzig_bindings);
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
    dvui_vk_backend.addImport("dvui_vulkan_renderer", dvui_vulkan_renderer);
    dvui.linkBackend(dvui_module, dvui_vk_backend);
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
            exe.subsystem = .windows;
        }
        b.installArtifact(exe);
        b.step("run-app", "Run demo").dependOn(&b.addRunArtifact(exe).step);
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
            exe_standalone.subsystem = .windows;
        }
        b.installArtifact(exe_standalone);
        b.step("run", "Run demo").dependOn(&b.addRunArtifact(exe_standalone).step);
    }
}

// TODO: get rid of this in future
// Workaround for glibc 2.43+ (GCC 16) adding .sframe sections with R_X86_64_PC64
// relocations to crt1.o, which Zig's self-hosted ELF linker can't handle.
// Strips .sframe from a local copy of crt1.o and produces a libc file pointing at it.
// Returns null on non-Linux hosts where the issue doesn't apply.
fn patchedGlibcLibcFile(b: *Build) ?Build.LazyPath {
    if (b.graph.host.result.os.tag != .linux) return null;

    const patch = b.addSystemCommand(&.{
        "sh",        "-ec",
        \\crt1=$(cc -print-file-name=crt1.o)
        \\crt_dir=$(dirname "$crt1")
        \\cc_dir=$(dirname "$(cc -print-file-name=crtbegin.o)")
        \\find "$crt_dir" -maxdepth 1 \( -name '*.o' -o \( -name 'lib*.a' -size -200k \) -o \( -name 'lib*.so' -size -10k \) \) -exec cp -P {} "$1/" \;
        \\objcopy --remove-section .sframe "$1/crt1.o"
        \\printf 'include_dir=/usr/include\nsys_include_dir=/usr/include\ncrt_dir=%s\nmsvc_lib_dir=\nkernel32_lib_dir=\ncc_dir=%s\ndarwin_sdk_dir=\n' "$1" "$cc_dir" > "$1/patched-libc.txt"
        ,
        "patch-crt",
    });
    return patch.addOutputDirectoryArg("crt-patched").path(b, "patched-libc.txt");
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
    b.graph.poisonCache();
    var shaders: std.ArrayList(Shader) = .empty;
    const cwd = b.root;
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
                            const output = compile.addOutputFileArg(out_name);
                            if (!options.optimize) {
                                compile.addArg("-minimum-slang-optimization");
                                compile.addArg("-g");
                            } else compile.addArg("-O3");
                            if (options.slang.?.row_major) compile.addArg("-matrix-layout-row-major") else compile.addArg("-matrix-layout-column-major");
                            compile.setCwd(b.path(shader_subpath));
                            compile.addFileArg(b.path(b.pathJoin(&.{ shader_subpath, f.name })));
                            shaders.append(b.allocator, Shader{ .name = out_name, .path = output, .step = &compile.step }) catch @panic("OOM");
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
                    const output = compile.addOutputFileArg(out_name);
                    compile.addFileArg(b.path(b.pathJoin(&.{ shader_subpath, f.name })));
                    break :blk Shader{ .name = out_name, .path = output, .step = &compile.step };
                }) catch @panic("OOM");
            }
        }
    }
    return shaders;
}
