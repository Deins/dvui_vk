# Vulkan Backend for dvui

This project provides a vulkan backend for [dvui](https://github.com/david-vanderson/dvui).  
Targeting `vulkan 1.2` and `zig v0.15.1` (see tags for older ver)

Backend is separated in parts:
* [dvui_vk_renderer.zig](./src/dvui_vk_renderer.zig) - implements platform independent renderer, suitable for already existing vulkan apps or apps that want to do their own windowing, input etc. Only dependency: [vulkan_zig](https://github.com/Snektron/vulkan-zig)
* DVUI backend implementations:
    * [dvui_vk_win32.zig](./src/dvui_vk_win32.zig) - backend based on native win32 api for windowing and input. (Windows only)
        Has additional dependencies:
        * [zigwin32](https://github.com/marlersoft/zigwin32#be58d3816810c1e4c20781cc7223a60906467d3c) (on Windows) 
    * [dvui_vk_win32.zig](./src/dvui_vk_glfw.zig) - backend based on glfw for windowing and input.
    * [dvui_vk_common.zig](./src/dvui_vk_common.zig) - common stuff for all backends.


### Current platform support
Renderer alone should be cross-platform. Full 'batteries included' integration:
* ✔️ Windows
* Linux - work in progress, using GLFW library. Use flag `-Dglfw` for app example.

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