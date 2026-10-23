"""macOS platform layer: the counterpart of ``app/win32/``.

Every module here mirrors the behaviour of the Windows module with the same name, so
``app.core.platform`` can pick one package or the other and the rest of the app does not
care which host it runs on. Nothing in this package imports Qt, core, audio or engine code.

Modules
-------
``_mac_api``      ``RealMacApi``: the policy-free ctypes wrapper over the macOS calls the app
                  uses (CoreGraphics events, Accessibility, secure input, Carbon hot keys,
                  NSPasteboard, AVFoundation authorisation). The ``WinApi`` analogue.
``hotkey_parse``  ``parse_hotkey`` / ``format_hotkey`` / ``MacHotkeyStringParser``: the same
                  hotkey grammar as Windows, with ``Hotkey.vk`` holding a Carbon ``kVK_*`` code.
``hotkey``        ``MacHotkeyManager``: global hotkey via Carbon ``RegisterEventHotKey``.
``clipboard``     ``MacClipboard``: save / set text / restore over ``NSPasteboard``.
``injector``      ``MacTextInjector``: types text with ``CGEventKeyboardSetUnicodeString``,
                  or pastes it with Cmd+V.
``instance``      ``FileLockSingleInstance``: ``flock`` + a Unix socket for "show window".
``autostart``     ``LaunchAgentAutostart``: "Open at login" via a LaunchAgent plist.
``permissions``   Accessibility and microphone permission checks, prompts and settings panes.

Every module imports on any platform (the frameworks are bound lazily, exactly like
``app/win32/_winapi.py``); calls that need macOS raise or no-op off macOS, as documented
per module.
"""
