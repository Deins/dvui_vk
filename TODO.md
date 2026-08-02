# TODO

## Windows backend

The normal DVUI/Vulkan path now uses `low`. The functionality needed by a
regular desktop application is covered:

- requested sizes are drawable-content sizes and are converted with the
  window's DPI;
- explicitly sized DVUI windows are centered by default, with configurable
  platform/default placement;
- fullscreen, maximize/restore, resizability, and min/max content sizes work;
- all DVUI cursor shapes are installed, including wait/progress cursors;
- the expanded key map, repeat counts, and side-specific modifiers reach DVUI;
- Windows uses the system UTF-16 clipboard, with the in-process clipboard as a
  fallback when the system clipboard is temporarily unavailable;
- Wayland uses the standard data-device clipboard with UTF-8 MIME types, with
  a warning when it must use the in-process UTF-8 fallback;
- X11 uses the standard `CLIPBOARD` selection with UTF-8 and legacy Latin-1
  interoperability, with the same warned in-process fallback;
- the Windows theme preference comes from the user registry and the native
  title bar follows dark/light setting changes.

The `dvui_vk` app path also forwards visibility, vsync, min/max size, and
`dvui.Window.InitOptions` instead of silently replacing them with defaults.

### Deferred platform integration

Custom Win32 integration remains intentionally out of scope for now:

- custom window-class registration and class options (cursor/icons,
  background brush, menu, module instance, extra bytes);
- supplying or chaining a custom `wndProc` and manually dispatching `WM_*`
  messages.

Applications can still access the backend-neutral native handle through
`low.Window.nativeSurface()` when a narrowly scoped platform call is needed.

`InitOptions.icon` is still unused, as it was in the removed Win32 backend.
Native IME candidate placement is also still pending; `textInputRect` was
already a no-op in that backend.
