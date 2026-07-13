# TODO
## windows backend

port covers the normal dvui/Vulkan path, but it does not preserve several Win32-specific capabilities from [dvui_vk_win32.zig](/home/dee/dev/dvui_vk/src/dvui_vk_win32.zig).

- Custom Win32 integration is absent: registering/configuring window classes, supplying a custom `wndProc`, accessing `HWND`, and manually dispatching/handling `WM_*` messages. The low compatibility `win` only exposes event servicing.
- Window-class options are unsupported: custom class cursor/icon(s), background brush, menu name, instance, and extra class/window bytes ([win32 backend lines 493–539](/home/dee/dev/dvui_vk/src/dvui_vk_win32.zig:493)).
- Windows dark title-bar configuration via `DwmSetWindowAttribute` is not implemented by low ([line 587](/home/dee/dev/dvui_vk/src/dvui_vk_win32.zig:587)). Low’s color scheme is only the `COLORSCHEME` environment override.
- Precise Win32 initial-window sizing/centering—DPI conversion, `AdjustWindowRectEx`, and `SetWindowPos`—is absent ([lines 593–621](/home/dee/dev/dvui_vk/src/dvui_vk_win32.zig:593)). Low creates at the default OS position and its Windows backend treats the requested size as the outer window size.
- On Windows, low’s fullscreen is currently a no-op, and its `setResizable`, min-size, and max-size operations are also no-ops ([low Windows backend](/home/dee/dev/low/src/windows/backend.zig:194)).
- Low’s Windows cursor implementation only toggles visibility; it does not actually install the requested cursor shape. So `setCursor` degrades versus the `LoadCursorW`/`SetClassLongPtrW` implementation ([win32 lines 309–345](/home/dee/dev/dvui_vk/src/dvui_vk_win32.zig:309)).
- Some input fidelity is lower: no side-specific modifiers in the callback contract, no repeat-count expansion, and its Windows key map omits many keys that the Win32 backend maps (keypad keys, F13–F24, Pause/Print/Menu, NumLock/CapsLock, etc.).
- Low clipboard is currently an in-process fallback, not the system clipboard. This is not a regression from this file—its Win32 clipboard functions are stubs—but it is not native Windows clipboard support.

Notably, `InitOptions.icon` appears unused in both implementations, and `textInputRect` was already a no-op in the Win32 file.
