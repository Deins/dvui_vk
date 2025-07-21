# Sokol Backend for dvui

This project provides a vulkan backend for [dvui](https://github.com/david-vanderson/dvui). 

### Platform support
* Windows ✔️
* Linux X11 ❌ (planned in future)
* Linux Wayland ❌

### 🚧 Not yet implemented / TODO 🚧
* Render textures
* Touch events
* On top example
* variable frame rate (sleeping when inactive)

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
