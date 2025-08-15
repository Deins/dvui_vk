# Vulkan Backend for dvui

This project provides a vulkan backend for [dvui](https://github.com/david-vanderson/dvui). 

Backend is separated in two main parts:
* [dvui_vulkan_renderer.zig](./src/dvui_vulkan_renderer.zig) - implements platform independent renderer, suitable for already existing vulkan apps or apps that want to do their own windowing, input etc.
    * Depends on [vulkan_zig](https://github.com/Snektron/vulkan-zig)
* [dvui_vulkan.zig](./src/dvui_vulkan.zig)  - handles setup and platform specific functionality such as input&windowing. Also implements dvui app interface.  
    Additionally depends on:
    * [vk_kickstart](https://github.com/mikastiv/vk-kickstart.git)
    * [zigwin32](https://github.com/marlersoft/zigwin32#be58d3816810c1e4c20781cc7223a60906467d3c) (on Windows) 

### Platform support
* Windows ✔️
* Linux X11 ❌ (planned in future)
* Linux Wayland ❌

### todo - not yet implemented 🚧
* Rendering:
    * textureRead()
    * option to pass in general purpose gpu memory allocator for textures
    * linear color-space framebuffers. (easly switchable with source modifications, but tricky to expose).
* App/Platform functionality:
    * Variable frame rate (sleeping when inactive)
    * Touch events
    * Other misc platform functions such as openURL, clipboard etc.

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
Similar as rest only `zig build run-app`
