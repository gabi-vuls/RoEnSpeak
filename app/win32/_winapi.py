"""ctypes implementation of ``contracts.WinApi`` (no pywin32).

Every method maps 1:1 to the named Win32 call and adds no policy. The DLLs are bound
lazily on first use, so this module imports on any platform (tests use fakes off Windows).
Calling a method off Windows raises OSError.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time
from ctypes import wintypes
from typing import Optional, Sequence

from app.contracts import KeyEvent, WinMessage

log = logging.getLogger("roenspeak.win32.api")

# --- constants ----------------------------------------------------------------------------

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
MAPVK_VK_TO_VSC = 0

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TOKEN_QUERY = 0x0008
TOKEN_INTEGRITY_LEVEL_CLASS = 25  # TOKEN_INFORMATION_CLASS.TokenIntegrityLevel

ERROR_ACCESS_DENIED = 5
ERROR_INSUFFICIENT_BUFFER = 122
ERROR_ALREADY_EXISTS = 183

GMEM_MOVEABLE = 0x0002
HWND_BROADCAST = 0xFFFF

# Clipboard formats whose handle is NOT an HGLOBAL with plain bytes (GDI objects, handles
# wrapped in structs, owner-display). Reading them through GlobalLock is meaningless or unsafe.
CF_BITMAP = 2
CF_METAFILEPICT = 3
CF_PALETTE = 9
CF_ENHMETAFILE = 14
CF_OWNERDISPLAY = 0x0080
CF_DSPBITMAP = 0x0082
CF_DSPMETAFILEPICT = 0x0083
CF_DSPENHMETAFILE = 0x008E
CF_PRIVATEFIRST, CF_PRIVATELAST = 0x0200, 0x02FF  # app-private handles, not freed/owned as HGLOBAL
CF_GDIOBJFIRST, CF_GDIOBJLAST = 0x0300, 0x03FF  # GDI object handles
_NON_HGLOBAL_FORMATS = frozenset(
    {CF_BITMAP, CF_METAFILEPICT, CF_PALETTE, CF_ENHMETAFILE, CF_OWNERDISPLAY, CF_DSPBITMAP,
     CF_DSPMETAFILEPICT, CF_DSPENHMETAFILE}
)


def is_hglobal_clipboard_format(fmt: int) -> bool:
    """True if ``fmt``'s clipboard data is an HGLOBAL holding plain bytes."""
    if fmt in _NON_HGLOBAL_FORMATS:
        return False
    if CF_PRIVATEFIRST <= fmt <= CF_PRIVATELAST or CF_GDIOBJFIRST <= fmt <= CF_GDIOBJLAST:
        return False
    return fmt > 0


# --- structures (64-bit safe) -------------------------------------------------------------

ULONG_PTR = ctypes.c_size_t


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", wintypes.POINT),
        ("lPrivate", wintypes.DWORD),
    ]


class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]


class TOKEN_MANDATORY_LABEL(ctypes.Structure):
    _fields_ = [("Label", SID_AND_ATTRIBUTES)]


# --- lazy DLL binding ---------------------------------------------------------------------


class _Lib:
    """Prototyped Win32 functions. Built once, on first use, on Windows only."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("The Win32 API is only available on Windows")
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        u32 = ctypes.WinDLL("user32", use_last_error=True)
        adv = ctypes.WinDLL("advapi32", use_last_error=True)
        W = wintypes

        def bind(dll, name, restype, *argtypes):
            fn = getattr(dll, name)
            fn.restype = restype
            fn.argtypes = list(argtypes)
            setattr(self, name, fn)

        # threads / message loop
        bind(k32, "GetCurrentThreadId", W.DWORD)
        bind(u32, "GetMessageW", W.BOOL, ctypes.POINTER(MSG), W.HWND, W.UINT, W.UINT)
        bind(u32, "PostThreadMessageW", W.BOOL, W.DWORD, W.UINT, W.WPARAM, W.LPARAM)
        bind(u32, "PostQuitMessage", None, ctypes.c_int)
        # hotkeys
        bind(u32, "RegisterHotKey", W.BOOL, W.HWND, ctypes.c_int, W.UINT, W.UINT)
        bind(u32, "UnregisterHotKey", W.BOOL, W.HWND, ctypes.c_int)
        # keyboard
        bind(u32, "GetAsyncKeyState", W.SHORT, ctypes.c_int)
        bind(u32, "SendInput", W.UINT, W.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
        bind(u32, "MapVirtualKeyW", W.UINT, W.UINT, W.UINT)
        # windows / processes
        bind(u32, "GetForegroundWindow", W.HWND)
        bind(u32, "GetWindowThreadProcessId", W.DWORD, W.HWND, ctypes.POINTER(W.DWORD))
        bind(k32, "GetCurrentProcess", W.HANDLE)
        bind(k32, "OpenProcess", W.HANDLE, W.DWORD, W.BOOL, W.DWORD)
        bind(k32, "CloseHandle", W.BOOL, W.HANDLE)
        bind(adv, "OpenProcessToken", W.BOOL, W.HANDLE, W.DWORD, ctypes.POINTER(W.HANDLE))
        bind(adv, "GetTokenInformation", W.BOOL, W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD,
             ctypes.POINTER(W.DWORD))
        bind(adv, "GetSidSubAuthorityCount", ctypes.POINTER(ctypes.c_ubyte), ctypes.c_void_p)
        bind(adv, "GetSidSubAuthority", ctypes.POINTER(W.DWORD), ctypes.c_void_p, W.DWORD)
        # clipboard
        bind(u32, "OpenClipboard", W.BOOL, W.HWND)
        bind(u32, "CloseClipboard", W.BOOL)
        bind(u32, "EmptyClipboard", W.BOOL)
        bind(u32, "EnumClipboardFormats", W.UINT, W.UINT)
        bind(u32, "GetClipboardData", W.HANDLE, W.UINT)
        bind(u32, "SetClipboardData", W.HANDLE, W.UINT, W.HANDLE)
        bind(k32, "GlobalAlloc", W.HGLOBAL, W.UINT, ctypes.c_size_t)
        bind(k32, "GlobalFree", W.HGLOBAL, W.HGLOBAL)
        bind(k32, "GlobalLock", ctypes.c_void_p, W.HGLOBAL)
        bind(k32, "GlobalUnlock", W.BOOL, W.HGLOBAL)
        bind(k32, "GlobalSize", ctypes.c_size_t, W.HGLOBAL)
        bind(u32, "GetClipboardSequenceNumber", W.DWORD)
        # single instance / misc
        bind(k32, "CreateMutexW", W.HANDLE, ctypes.c_void_p, W.BOOL, W.LPCWSTR)
        bind(u32, "RegisterWindowMessageW", W.UINT, W.LPCWSTR)
        bind(u32, "AllowSetForegroundWindow", W.BOOL, W.DWORD)
        bind(u32, "RegisterClipboardFormatW", W.UINT, W.LPCWSTR)
        bind(u32, "PostMessageW", W.BOOL, W.HWND, W.UINT, W.WPARAM, W.LPARAM)


_lib: Optional[_Lib] = None
_lib_lock = threading.Lock()


def _get_lib() -> _Lib:
    global _lib
    if _lib is None:
        with _lib_lock:
            if _lib is None:
                _lib = _Lib()
    return _lib


def _token_integrity_level(lib: _Lib, process_handle: int) -> Optional[int]:
    """Mandatory-label RID of the process's primary token, or None on any failure."""
    token = wintypes.HANDLE()
    if not lib.OpenProcessToken(process_handle, TOKEN_QUERY, ctypes.byref(token)):
        return None
    try:
        needed = wintypes.DWORD(0)
        lib.GetTokenInformation(token, TOKEN_INTEGRITY_LEVEL_CLASS, None, 0, ctypes.byref(needed))
        if needed.value == 0:
            return None
        buf = ctypes.create_string_buffer(needed.value)
        if not lib.GetTokenInformation(token, TOKEN_INTEGRITY_LEVEL_CLASS, buf, needed, ctypes.byref(needed)):
            return None
        label = TOKEN_MANDATORY_LABEL.from_buffer(buf)
        sid = label.Label.Sid
        if not sid:
            return None
        count = lib.GetSidSubAuthorityCount(sid)[0]
        if count == 0:
            return None
        return int(lib.GetSidSubAuthority(sid, count - 1)[0])
    finally:
        lib.CloseHandle(token)


class RealWinApi:
    """``contracts.WinApi`` over ctypes. Thread-safe (stateless apart from the bound DLLs);
    per-thread Win32 semantics (hotkeys, message queue, last error) follow the calling thread."""

    @property
    def _lib(self) -> _Lib:
        return _get_lib()

    # threads / message loop --------------------------------------------------------------

    def get_current_thread_id(self) -> int:
        return int(self._lib.GetCurrentThreadId())

    def get_message(self) -> Optional[WinMessage]:
        msg = MSG()
        r = self._lib.GetMessageW(ctypes.byref(msg), None, 0, 0)
        if r == -1:
            raise ctypes.WinError(ctypes.get_last_error())
        if r == 0:
            return None
        return WinMessage(int(msg.message), int(msg.wParam or 0), int(msg.lParam or 0))

    def post_thread_message(self, thread_id: int, message: int, wparam: int = 0, lparam: int = 0) -> bool:
        return bool(self._lib.PostThreadMessageW(thread_id, message, wparam, lparam))

    def post_quit_message(self) -> None:
        self._lib.PostQuitMessage(0)

    # hotkeys -----------------------------------------------------------------------------

    def register_hotkey(self, hotkey_id: int, modifiers: int, vk: int) -> bool:
        return bool(self._lib.RegisterHotKey(None, hotkey_id, modifiers, vk))

    def unregister_hotkey(self, hotkey_id: int) -> bool:
        return bool(self._lib.UnregisterHotKey(None, hotkey_id))

    def get_last_error(self) -> int:
        return int(ctypes.get_last_error())

    # keyboard ----------------------------------------------------------------------------

    def get_async_key_state(self, vk: int) -> int:
        # Raw SHORT reinterpreted as unsigned 16 bits so that "& 0x8000" works as documented.
        return int(self._lib.GetAsyncKeyState(vk)) & 0xFFFF

    def send_input(self, events: Sequence[KeyEvent]) -> int:
        n = len(events)
        if n == 0:
            return 0
        lib = self._lib
        arr = (INPUT * n)()
        for i, ev in enumerate(events):
            rec = arr[i]
            rec.type = INPUT_KEYBOARD
            flags = KEYEVENTF_KEYUP if ev.key_up else 0
            if ev.unicode_char is not None:
                code = ord(ev.unicode_char)
                if code > 0xFFFF:
                    raise ValueError("KeyEvent.unicode_char must be a single UTF-16 code unit")
                rec.ki.wVk = 0
                rec.ki.wScan = code
                flags |= KEYEVENTF_UNICODE
            else:
                rec.ki.wVk = ev.vk
                rec.ki.wScan = lib.MapVirtualKeyW(ev.vk, MAPVK_VK_TO_VSC) & 0xFFFF
            rec.ki.dwFlags = flags
            rec.ki.time = 0
            rec.ki.dwExtraInfo = 0
        return int(lib.SendInput(n, arr, ctypes.sizeof(INPUT)))

    # windows / processes -----------------------------------------------------------------

    def get_foreground_window(self) -> int:
        return int(self._lib.GetForegroundWindow() or 0)

    def get_window_process_id(self, hwnd: int) -> int:
        if not hwnd:
            return 0
        pid = wintypes.DWORD(0)
        self._lib.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value)

    def get_process_integrity_level(self, pid: int) -> Optional[int]:
        if not pid:
            return None
        lib = self._lib
        handle = lib.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            return _token_integrity_level(lib, handle)
        finally:
            lib.CloseHandle(handle)

    def get_current_process_integrity_level(self) -> int:
        lib = self._lib
        il = _token_integrity_level(lib, lib.GetCurrentProcess())
        if il is None:
            raise ctypes.WinError(ctypes.get_last_error())
        return il

    # clipboard ---------------------------------------------------------------------------

    def open_clipboard(self) -> bool:
        return bool(self._lib.OpenClipboard(None))

    def close_clipboard(self) -> None:
        self._lib.CloseClipboard()

    def empty_clipboard(self) -> bool:
        return bool(self._lib.EmptyClipboard())

    def enum_clipboard_formats(self) -> list[int]:
        formats: list[int] = []
        fmt = 0
        while True:
            fmt = int(self._lib.EnumClipboardFormats(fmt))
            if fmt == 0:
                break
            formats.append(fmt)
        return formats

    def get_clipboard_bytes(self, fmt: int) -> Optional[bytes]:
        if not is_hglobal_clipboard_format(fmt):
            return None
        lib = self._lib
        handle = lib.GetClipboardData(fmt)
        if not handle:
            return None
        ptr = lib.GlobalLock(handle)
        if not ptr:
            return None
        try:
            size = int(lib.GlobalSize(handle))
            return ctypes.string_at(ptr, size)
        finally:
            lib.GlobalUnlock(handle)

    def get_clipboard_sequence_number(self) -> int:
        return int(self._lib.GetClipboardSequenceNumber())

    def set_clipboard_bytes(self, fmt: int, data: bytes) -> bool:
        if not is_hglobal_clipboard_format(fmt):
            return False
        lib = self._lib
        handle = lib.GlobalAlloc(GMEM_MOVEABLE, max(len(data), 1))
        if not handle:
            return False
        ptr = lib.GlobalLock(handle)
        if not ptr:
            lib.GlobalFree(handle)
            return False
        try:
            if data:
                ctypes.memmove(ptr, data, len(data))
            else:
                ctypes.memset(ptr, 0, 1)
        finally:
            lib.GlobalUnlock(handle)
        if not lib.SetClipboardData(fmt, handle):
            lib.GlobalFree(handle)  # ownership passes to the system only on success
            return False
        return True

    # single instance / misc --------------------------------------------------------------

    def create_mutex(self, name: str) -> tuple[int, bool]:
        handle = self._lib.CreateMutexW(None, False, name)
        err = ctypes.get_last_error()
        if handle:
            return int(handle), err == ERROR_ALREADY_EXISTS
        # NULL + ERROR_ACCESS_DENIED: the mutex exists but was created with a DACL we can't open
        # (e.g. by an elevated instance), so it does exist.
        return 0, err == ERROR_ACCESS_DENIED

    def close_handle(self, handle: int) -> None:
        if handle:
            self._lib.CloseHandle(handle)

    def register_window_message(self, name: str) -> int:
        return int(self._lib.RegisterWindowMessageW(name))

    def allow_set_foreground_window(self, pid: int) -> bool:
        return bool(self._lib.AllowSetForegroundWindow(pid & 0xFFFFFFFF))

    def register_clipboard_format(self, name: str) -> int:
        return int(self._lib.RegisterClipboardFormatW(name))

    def broadcast_message(self, message: int) -> None:
        if not self._lib.PostMessageW(HWND_BROADCAST, message, 0, 0):
            log.warning("PostMessageW(HWND_BROADCAST) failed, GetLastError=%d", ctypes.get_last_error())

    # time --------------------------------------------------------------------------------

    def monotonic(self) -> float:
        return time.perf_counter()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)
