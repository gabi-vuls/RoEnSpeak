"""Small ctypes helpers the UI needs on Windows (no pywin32, no app.win32 import).

Every function is a safe no-op off Windows so the UI modules stay importable everywhere.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes
from typing import Optional

log = logging.getLogger("roenspeak.ui.win")

IS_WINDOWS = sys.platform == "win32"

GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000
OVERLAY_EXSTYLE = WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE

HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020

WM_SYSKEYDOWN = 0x0104
WM_SYSCOMMAND = 0x0112
WM_ENDSESSION = 0x0016
SC_KEYMENU = 0xF100
VK_SPACE = 0x20
MSGFLT_ALLOW = 1

if IS_WINDOWS:
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _shell32 = ctypes.WinDLL("shell32", use_last_error=True)

    _user32.RegisterWindowMessageW.argtypes = [wintypes.LPCWSTR]
    _user32.RegisterWindowMessageW.restype = wintypes.UINT
    _user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    _user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    _user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    _user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
    _user32.SetWindowPos.argtypes = [
        wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT,
    ]
    _user32.SetWindowPos.restype = wintypes.BOOL
    _user32.GetForegroundWindow.argtypes = []
    _user32.GetForegroundWindow.restype = wintypes.HWND
    _user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    _user32.SetForegroundWindow.restype = wintypes.BOOL
    _user32.ChangeWindowMessageFilterEx.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.DWORD, ctypes.c_void_p]
    _user32.ChangeWindowMessageFilterEx.restype = wintypes.BOOL
    _shell32.SetCurrentProcessExplicitAppUserModelID.argtypes = [wintypes.LPCWSTR]
    _shell32.SetCurrentProcessExplicitAppUserModelID.restype = ctypes.HRESULT


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", wintypes.POINT),
    ]


def read_msg(address: int) -> Optional[MSG]:
    """Interpret a native-event-filter message pointer as a Win32 MSG (copy)."""
    if not IS_WINDOWS or not address:
        return None
    return MSG.from_buffer_copy(ctypes.string_at(address, ctypes.sizeof(MSG)))


def register_window_message(name: str) -> int:
    if not IS_WINDOWS:
        return 0
    msg = int(_user32.RegisterWindowMessageW(name))
    if not msg:
        log.warning("RegisterWindowMessageW(%r) failed, error=%d", name, ctypes.get_last_error())
    return msg


def allow_message_through_uipi(hwnd: int, message: int) -> None:
    """Let a (possibly lower-integrity) second instance's broadcast reach this window."""
    if not IS_WINDOWS or not hwnd or not message:
        return
    if not _user32.ChangeWindowMessageFilterEx(hwnd, message, MSGFLT_ALLOW, None):
        log.debug("ChangeWindowMessageFilterEx failed, error=%d", ctypes.get_last_error())


def get_exstyle(hwnd: int) -> int:
    if not IS_WINDOWS or not hwnd:
        return 0
    return int(_user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)) & 0xFFFFFFFF


def make_click_through_noactivate(hwnd: int) -> int:
    """Add WS_EX_TRANSPARENT|LAYERED|TOOLWINDOW|NOACTIVATE (+ topmost) without activating.
    Returns the resulting extended style."""
    if not IS_WINDOWS or not hwnd:
        return 0
    style = get_exstyle(hwnd)
    wanted = style | OVERLAY_EXSTYLE | WS_EX_TOPMOST
    if wanted != style:
        _user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, wanted)
    _user32.SetWindowPos(
        hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED
    )
    return get_exstyle(hwnd)


def get_foreground_window() -> int:
    if not IS_WINDOWS:
        return 0
    return int(_user32.GetForegroundWindow() or 0)


def set_foreground_window(hwnd: int) -> bool:
    if not IS_WINDOWS or not hwnd:
        return False
    return bool(_user32.SetForegroundWindow(hwnd))


def set_app_user_model_id(app_id: str) -> None:
    """Give the process its own taskbar identity (our icon instead of python.exe's)."""
    if not IS_WINDOWS:
        return
    try:
        _shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except OSError as exc:
        log.debug("SetCurrentProcessExplicitAppUserModelID failed: %s", exc)
