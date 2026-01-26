# Vulkan Backend for dvui 
[![Build](https://github.com/Deins/dvui_vk/actions/workflows/build.yml/badge.svg)](https://github.com/Deins/dvui_vk/actions/workflows/build.yml)

This project provides a vulkan backend for [dvui](https://github.com/david-vanderson/dvui).  
Targeting `vulkan 1.2` an newer and `zig v0.15.2` (see tags for older ver)

### Current platform support

* **Windows** native
* Using GLFW library. Build with flag `-Dglfw`.  
Standalone example not implemented at the moment. 
    * **Windows** GLFW
    * **Linux** GLFW 

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
Vulkan SDK is recommended for development to be able to run validation layers etc. However it is not required and will still compile with compilation message `VulkanSDK not found`.
If its unexpected check that `VULKAN_SDK` is correctly defined in your environment.

```sh
zig build run-app -Doptimize=ReleaseFast -Dglfw
```

Shaders when modified can be recompiled by passing `-Dslangc` or `-Dglslc` depending on what shader language is used.

### Standalone example
Similarly only `run` instead of `run-app`. And skip `-Dglfw`. Windows only, glfw not implemented. See 3d example (idea the same).

### Standalone with vulkan 3D rendering
`zig build run --build-file ./examples/3d/build.zig -Dglfw`  
Or alternatively `cd examples/3d` and `zig build run -Dglfw`.

![screenshot](examples/3d/screenshot.png)