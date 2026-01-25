# Vulkan Backend for dvui
This project provides a vulkan backend for [dvui](https://github.com/david-vanderson/dvui).  
Targeting `vulkan 1.2` an newer and `zig v0.15.2` (see tags for older ver)

### Current platform support
* Windows native [![Windows Build](https://github.com/Deins/dvui_vk/actions/workflows/build.yml/badge.svg?job=Build%20windows%20native)](https://github.com/Deins/dvui_vk/actions/workflows/build.yml)
* Through use of GLFW library for windowing. Uses build flag `-Dglfw`.  
Only app example implemented at the moment.
    * Windows GLFW [![Windows GLFW Build](https://github.com/Deins/dvui_vk/actions/workflows/build.yml/badge.svg?job=Build%20windows-latest%20glfw)](https://github.com/Deins/dvui_vk/actions/workflows/build.yml)
    * Linux GLFW [![Ubuntu GLFW Build](https://github.com/Deins/dvui_vk/actions/workflows/build.yml/badge.svg?job=Build%20ubuntu-latest%20glfw)](https://github.com/Deins/dvui_vk/actions/workflows/build.yml) 

Backend is separated in parts:
* [dvui_vk_renderer.zig](./src/dvui_vk_renderer.zig) - implements platform independent renderer, suitable for already existing vulkan apps or apps that want to do their own windowing, input etc. Only dependency: [vulkan_zig](https://github.com/Snektron/vulkan-zig)
* DVUI backend implementations, has additional dependencies (see build.zig.zon):
    * [dvui_vk_win32.zig](./src/dvui_vk_win32.zig) - backend based on native win32 api for windowing and input. (Windows only)
    * [dvui_vk_glfw.zig](./src/dvui_vk_glfw.zig) - backend based on glfw for windowing and input. Use `-Dglfw` build flag to enable.
    * [dvui_vk_common.zig](./src/dvui_vk_common.zig) - additional common stuff used by all backends.

### todo - not yet implemented 🚧
* Rendering:
    * textureRead()
    * option to pass in general purpose gpu memory allocator for textures
    * linear color space frame-buffers. (easily switchable with source modifications, but tricky to expose).
* App/Platform functionality:
    * Variable frame rate (sleeping when inactive)
    * Touch events
    * Other misc platform functions such as openURL, clipboard etc.
* Misc/known issues:
    * windows: app swapchain resize is not synced to window resize causing small visual jerks. (no easy fix - general issue with vulkan on windows, maybe some hackery is possible to use dx11/12 swapchain instead of vulkan one). standalone example: doesn't have realtime resize implemented.

## Build & Run
### With vulkan sdk (recommended)
With Vulkan SDK installed and sourced:
```sh
zig build run -Doptimize=ReleaseFast
```

Shaders when modified can be recompiled by passing `-Dslangc` or `-Dglslc` depending on what shader language is used.

### Without vulkan sdk
Get vk.xml form somewhere such as [vulkan-headers](https://github.com/KhronosGroup/Vulkan-Headers/blob/main/registry/vk.xml). Pass it in arguments:
```sh
zig build run -Dvk_registry=/path/to/vk.xml -Doptimize=ReleaseFast
```

### App example
Similarly only target `run-app` instead of `run`
For glfw backend add flag `-Dglfw`

### Standalone with vulkan 3D rendering
`zig build run --build-file ./examples/3d/build.zig`  
Or alternatively `cd examples/3d` and `zig build run`.

![screenshot](examples/3d/screenshot.png)