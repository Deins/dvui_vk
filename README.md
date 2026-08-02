# Vulkan Backend for dvui 
[![Build](https://github.com/Deins/dvui_vk/actions/workflows/build.yml/badge.svg)](https://github.com/Deins/dvui_vk/actions/workflows/build.yml)

This project provides a vulkan backend for [dvui](https://github.com/david-vanderson/dvui).  
Targeting `vulkan 1.3` and newer, and `zig v0.16` (see tags for older)

![screenshot](examples/3d/screenshot.png)

### Current platform support

* **Windows** and **Linux** through the cross-platform [`low`](https://github.com/Deins/low) platform layer.
  Linux builds support X11 and Wayland.

The backend is split into these parts:
* [dvui_vk_renderer.zig](./src/dvui_vk_renderer.zig) - platform-independent Vulkan rendering for applications that provide their own windowing and input.
* [dvui_vk_low.zig](./src/dvui_vk_low.zig) - the main DVUI backend, using `low` for cross-platform windowing, input, and Vulkan presentation.
* [dvui_vk_common.zig](./src/dvui_vk_common.zig) - shared Vulkan context and rendering helpers used by the `low` backend.

### todo, known issues
* Rendering:
    * textureRead() not implemented
    * option to pass in general purpose gpu memory allocator for textures
    * linear color space frame-buffers. (easily switchable with source modifications, but tricky to expose).
* Platform functionality:
    * should try to get free-type running for dvui text rendering instead of default stb-truetype. As text rendering quality is not as good. 
    * Touch events
    * Some platform-specific functionality is still incomplete; see [TODO.md](./TODO.md).

## Build & Run
```sh
zig build run-app -Doptimize=ReleaseFast
```
Vulkan SDK is optional but recommended for development (to be able to run vulkan validation layers etc). To use vulkan sdk make sure `VULKAN_SDK` environment variable is correctly defined. 

Shaders when modified need be recompiled by adding `-Dslangc` to build command. ([slangc](https://github.com/shader-slang/slang/releases) comes packaged with vulkan SDK or needs to be installed separately).

### Standalone with Vulkan 3D rendering
`zig build run --build-file ./examples/3d/build.zig`
Or alternatively `cd examples/3d` and `zig build run`

### Standalone example
Similarly as others, use `run` instead of `run-app`.

### Cross compilation

Build portable binaries by specifying the target explicitly:

```sh
# Windows (GNU ABI)
zig build -Dtarget=x86_64-windows-gnu

# Linux with glibc 2.17 as the minimum supported runtime version
zig build -Dtarget=x86_64-linux-gnu.2.17
```

The Linux version suffix controls the minimum glibc version required on the
target system. Adjust it to match the oldest Linux distribution you support.
