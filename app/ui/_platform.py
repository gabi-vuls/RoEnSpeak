"""The native helpers the UI uses, bound to the host platform.

``app.ui.app`` and ``app.ui.overlay`` need a handful of things Qt does not expose: a
click-through always-on-top overlay window, bringing our own window to the front, and (on
Windows) the registered window message a second instance broadcasts. Those live in
``app.ui._win`` (ctypes/user32) and ``app.ui._mac`` (ctypes/Objective-C). This module picks
one of them, so no UI module needs a platform branch for them.

Both back ends define every name, and both are safe to import - and to call - on the wrong
platform, where every function is a no-op. On a host that is neither Windows nor macOS the
Windows back end is used, which means "do nothing at all".

The three macOS-only additions (activation policy, NSSound) get an explicit no-op Windows
implementation here rather than in ``_win``, which stays exactly as it was.
"""

from __future__ import annotations

from app.contracts import IS_MACOS

if IS_MACOS:
    from app.ui import _mac as _impl

    set_activation_policy_accessory = _impl.set_activation_policy_accessory
    set_activation_policy_regular = _impl.set_activation_policy_regular
    play_sound_async = _impl.play_sound_async
else:
    from app.ui import _win as _impl  # type: ignore[no-redef]

    def set_activation_policy_accessory() -> bool:
        """No-op: only macOS has an app activation policy (Dock icon / app switcher).
        A Windows tray app hides its taskbar button with WS_EX_TOOLWINDOW instead."""
        return False

    def set_activation_policy_regular() -> bool:
        """No-op; see set_activation_policy_accessory()."""
        return False

    def play_sound_async(path: str) -> bool:
        """No-op: Windows plays the beeps with winsound (see app.ui.sounds)."""
        return False


#: The ex-style / window-level helper for the REC overlay.
make_click_through_noactivate = _impl.make_click_through_noactivate
#: Bring our own window (and app) to the front.
set_foreground_window = _impl.set_foreground_window
#: Taskbar identity (Windows) / nothing (macOS).
set_app_user_model_id = _impl.set_app_user_model_id
#: Process-wide message id for the "show window" broadcast; 0 on macOS, which keeps the
#: native-filter branch in app.ui.app switched off.
register_window_message = _impl.register_window_message
#: Let a lower-integrity second instance's broadcast reach our window (Windows only).
allow_message_through_uipi = _impl.allow_message_through_uipi
#: Copy the MSG handed to a native event filter; None on macOS.
read_msg = _impl.read_msg
#: The Win32 MSG structure (an empty placeholder on macOS).
MSG = _impl.MSG

__all__ = [
    "make_click_through_noactivate",
    "set_foreground_window",
    "set_app_user_model_id",
    "register_window_message",
    "allow_message_through_uipi",
    "read_msg",
    "MSG",
    "set_activation_policy_accessory",
    "set_activation_policy_regular",
    "play_sound_async",
]
