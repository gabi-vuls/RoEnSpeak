"""ctypes implementation of the macOS system calls the app needs (no PyObjC).

This is the macOS counterpart of ``app.win32._winapi.RealWinApi``: every method maps 1:1 to
the named system call and adds no policy. ``contracts.WinApi`` itself is a Windows-shaped
protocol, so this class deliberately implements only what macOS needs (a different set of
calls with the same spirit): CoreGraphics keyboard events instead of ``SendInput``, Carbon
hot keys instead of ``RegisterHotKey``, ``NSPasteboard`` instead of the Win32 clipboard,
Accessibility + secure event input instead of UIPI integrity levels.

The frameworks are bound lazily on first use, so this module imports on any platform (tests
use fakes off macOS). Calling a method off macOS raises OSError.

Objective-C is driven through ``libobjc`` (``objc_getClass`` / ``sel_registerName`` /
``objc_msgSend``). ``objc_msgSend`` is never called untyped: every call casts its address to
a ``CFUNCTYPE`` with the exact argument and return types for that selector, which is also
what makes the calls thread-safe (no shared, mutated ``argtypes``). Objects returned by
methods that do not start with ``alloc``/``new``/``copy`` are autoreleased, so every method
that touches Objective-C objects runs inside its own ``NSAutoreleasePool``.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

log = logging.getLogger("roenspeak.macos.api")

# --- constants ----------------------------------------------------------------------------

#: CGEventSourceStateID
KCG_EVENT_SOURCE_STATE_COMBINED_SESSION = 0
KCG_EVENT_SOURCE_STATE_HID_SYSTEM = 1
#: CGEventTapLocation
KCG_HID_EVENT_TAP = 0
KCG_SESSION_EVENT_TAP = 1

#: CGEventFlags. ALPHA_SHIFT (Caps Lock) and NON_COALESCED are *not* "held keys": Caps Lock
#: is a latch that would make the injector wait forever, and NON_COALESCED is always set.
KCG_FLAG_ALPHA_SHIFT = 0x00010000
KCG_FLAG_SHIFT = 0x00020000
KCG_FLAG_CONTROL = 0x00040000
KCG_FLAG_ALTERNATE = 0x00080000  # Option
KCG_FLAG_COMMAND = 0x00100000
KCG_FLAG_NUMERIC_PAD = 0x00200000
KCG_FLAG_HELP = 0x00400000
KCG_FLAG_SECONDARY_FN = 0x00800000
#: The modifiers the injector waits to be released (the Ctrl/Shift/Alt/Win analogue).
MODIFIER_FLAG_MASK = KCG_FLAG_SHIFT | KCG_FLAG_CONTROL | KCG_FLAG_ALTERNATE | KCG_FLAG_COMMAND

#: Carbon virtual key codes used outside the hotkey table.
KVK_ANSI_V = 0x09

#: Carbon event constants (HIToolbox/Events.h, CarbonEvents.h).
K_EVENT_CLASS_KEYBOARD = 0x6B657962  # 'keyb'
K_EVENT_HOT_KEY_PRESSED = 5
K_EVENT_PARAM_DIRECT_OBJECT = 0x2D2D2D2D  # '----'
K_TYPE_EVENT_HOT_KEY_ID = 0x686B6964  # 'hkid'
#: Signature stored in every EventHotKeyID we register ('RoEn').
HOTKEY_SIGNATURE = 0x526F456E
NO_ERR = 0
EVENT_NOT_HANDLED_ERR = -9874
EVENT_HOT_KEY_EXISTS_ERR = -9878

#: CFStringEncoding
K_CF_STRING_ENCODING_UTF8 = 0x08000100

#: NSPasteboardTypeString. Hard-coded rather than read from the AppKit global because the UTI
#: is part of the public, stable pasteboard contract.
NS_PASTEBOARD_TYPE_STRING = "public.utf8-plain-text"

#: AVMediaTypeAudio ('soun'), and AVAuthorizationStatus.
AV_MEDIA_TYPE_AUDIO = "soun"
AV_AUTH_NOT_DETERMINED = 0
AV_AUTH_RESTRICTED = 1
AV_AUTH_DENIED = 2
AV_AUTH_AUTHORIZED = 3

#: Option key for AXIsProcessTrustedWithOptions; the global is dlsym'd when available.
AX_TRUSTED_CHECK_OPTION_PROMPT = "AXTrustedCheckOptionPrompt"

_FRAMEWORK = "/System/Library/Frameworks/{0}.framework/{0}"


@dataclass(frozen=True)
class MacAppInfo:
    """The frontmost application, as NSWorkspace reports it."""

    pid: int
    bundle_id: str
    name: str


# --- structures ---------------------------------------------------------------------------


class EventHotKeyID(ctypes.Structure):
    _fields_ = [("signature", ctypes.c_uint32), ("id", ctypes.c_uint32)]


class EventTypeSpec(ctypes.Structure):
    _fields_ = [("eventClass", ctypes.c_uint32), ("eventKind", ctypes.c_uint32)]


class CFRunLoopSourceContext(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_long),
        ("info", ctypes.c_void_p),
        ("retain", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("copyDescription", ctypes.c_void_p),
        ("equal", ctypes.c_void_p),
        ("hash", ctypes.c_void_p),
        ("schedule", ctypes.c_void_p),
        ("cancel", ctypes.c_void_p),
        ("perform", ctypes.c_void_p),
    ]


#: OSStatus (*EventHandlerProcPtr)(EventHandlerCallRef, EventRef, void*)
EVENT_HANDLER_PROC = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
#: void (*perform)(void* info)
RUN_LOOP_PERFORM_PROC = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
#: void (^handler)(BOOL granted) -- the block's invoke function (block pointer, then args).
BLOCK_INVOKE_PROC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_bool)


class _BlockDescriptor(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ulong), ("size", ctypes.c_ulong)]


class _BlockLiteral(ctypes.Structure):
    """A global (never copied, no captured state) Objective-C block literal."""

    _fields_ = [
        ("isa", ctypes.c_void_p),
        ("flags", ctypes.c_int),
        ("reserved", ctypes.c_int),
        ("invoke", ctypes.c_void_p),
        ("descriptor", ctypes.POINTER(_BlockDescriptor)),
    ]


BLOCK_IS_GLOBAL = 1 << 28


# --- lazy framework binding ---------------------------------------------------------------


class _Lib:
    """Prototyped macOS functions. Built once, on first use, on macOS only."""

    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise OSError("The macOS system API is only available on macOS")
        V, I32, U32, U64, L, B = (
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint64,
            ctypes.c_long, ctypes.c_bool,
        )
        cg = ctypes.cdll.LoadLibrary(_FRAMEWORK.format("CoreGraphics"))
        appsvc = ctypes.cdll.LoadLibrary(_FRAMEWORK.format("ApplicationServices"))
        carbon = ctypes.cdll.LoadLibrary(_FRAMEWORK.format("Carbon"))
        cf = ctypes.cdll.LoadLibrary(_FRAMEWORK.format("CoreFoundation"))
        foundation = ctypes.cdll.LoadLibrary(_FRAMEWORK.format("Foundation"))
        appkit = ctypes.cdll.LoadLibrary(_FRAMEWORK.format("AppKit"))
        objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.dylib")
        self._cf = cf
        self._appkit = appkit  # kept so AppKit classes stay registered
        self._foundation = foundation

        def bind(dll, name, restype, *argtypes):
            fn = getattr(dll, name)
            fn.restype = restype
            fn.argtypes = list(argtypes)
            setattr(self, name, fn)

        # CoreGraphics events
        bind(cg, "CGEventSourceFlagsState", U64, I32)
        bind(cg, "CGEventCreateKeyboardEvent", V, V, ctypes.c_uint16, B)
        bind(cg, "CGEventKeyboardSetUnicodeString", None, V, ctypes.c_ulong,
             ctypes.POINTER(ctypes.c_uint16))
        bind(cg, "CGEventSetFlags", None, V, U64)
        bind(cg, "CGEventPost", None, U32, V)
        # Accessibility / secure input
        bind(appsvc, "AXIsProcessTrusted", B)
        bind(appsvc, "AXIsProcessTrustedWithOptions", B, V)
        bind(carbon, "IsSecureEventInputEnabled", B)
        # Carbon hot keys
        bind(carbon, "GetApplicationEventTarget", V)
        bind(carbon, "RegisterEventHotKey", I32, U32, U32, EventHotKeyID, V, U32,
             ctypes.POINTER(V))
        bind(carbon, "UnregisterEventHotKey", I32, V)
        bind(carbon, "InstallEventHandler", I32, V, EVENT_HANDLER_PROC, L,
             ctypes.POINTER(EventTypeSpec), V, ctypes.POINTER(V))
        bind(carbon, "RemoveEventHandler", I32, V)
        bind(carbon, "GetEventParameter", I32, V, U32, U32, ctypes.POINTER(U32), U32,
             ctypes.POINTER(U32), V)
        # CoreFoundation: strings, data, arrays, dictionaries, run loop
        bind(cf, "CFRelease", None, V)
        bind(cf, "CFRetain", V, V)
        bind(cf, "CFStringCreateWithBytes", V, V, ctypes.c_char_p, L, U32, B)
        bind(cf, "CFStringGetLength", L, V)
        bind(cf, "CFStringGetCString", B, V, ctypes.c_char_p, L, U32)
        bind(cf, "CFDataCreate", V, V, ctypes.c_char_p, L)
        bind(cf, "CFDataGetLength", L, V)
        bind(cf, "CFDataGetBytePtr", V, V)
        bind(cf, "CFArrayGetCount", L, V)
        bind(cf, "CFArrayGetValueAtIndex", V, V, L)
        bind(cf, "CFDictionaryCreate", V, V, ctypes.POINTER(V), ctypes.POINTER(V), L, V, V)
        bind(cf, "CFRunLoopGetMain", V)
        bind(cf, "CFRunLoopGetCurrent", V)
        bind(cf, "CFRunLoopSourceCreate", V, V, L, ctypes.POINTER(CFRunLoopSourceContext))
        bind(cf, "CFRunLoopSourceSignal", None, V)
        bind(cf, "CFRunLoopSourceInvalidate", None, V)
        bind(cf, "CFRunLoopAddSource", None, V, V, V)
        bind(cf, "CFRunLoopRemoveSource", None, V, V, V)
        bind(cf, "CFRunLoopWakeUp", None, V)
        bind(cf, "CFRunLoopRunInMode", I32, V, ctypes.c_double, B)
        self.kCFRunLoopCommonModes = ctypes.c_void_p.in_dll(cf, "kCFRunLoopCommonModes")
        self.kCFRunLoopDefaultMode = ctypes.c_void_p.in_dll(cf, "kCFRunLoopDefaultMode")
        self.kCFBooleanTrue = ctypes.c_void_p.in_dll(cf, "kCFBooleanTrue")
        self.kCFTypeDictionaryKeyCallBacks = ctypes.c_void_p.in_dll(
            cf, "kCFTypeDictionaryKeyCallBacks")
        self.kCFTypeDictionaryValueCallBacks = ctypes.c_void_p.in_dll(
            cf, "kCFTypeDictionaryValueCallBacks")
        # Objective-C runtime
        bind(objc, "objc_getClass", V, ctypes.c_char_p)
        bind(objc, "sel_registerName", V, ctypes.c_char_p)
        self._msg_send = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        self._msg_cache: dict[tuple, Callable] = {}
        self._cls_cache: dict[bytes, Optional[int]] = {}
        self._sel_cache: dict[bytes, int] = {}
        self._cache_lock = threading.Lock()
        # AVFoundation is loaded on demand: it is a heavy framework and only the microphone
        # permission helpers need it.
        self._av: Optional[ctypes.CDLL] = None
        try:
            self.kAXTrustedCheckOptionPrompt: Optional[ctypes.c_void_p] = ctypes.c_void_p.in_dll(
                appsvc, "kAXTrustedCheckOptionPrompt")
        except ValueError:  # pragma: no cover - the symbol has existed since 10.9
            self.kAXTrustedCheckOptionPrompt = None
        try:
            libsystem = ctypes.cdll.LoadLibrary("/usr/lib/libSystem.dylib")
            self._global_block_isa: Optional[int] = ctypes.addressof(
                ctypes.c_void_p.in_dll(libsystem, "_NSConcreteGlobalBlock"))
        except (OSError, ValueError):  # pragma: no cover
            self._global_block_isa = None

    # --- Objective-C helpers -------------------------------------------------------------

    def av(self) -> ctypes.CDLL:
        if self._av is None:
            self._av = ctypes.cdll.LoadLibrary(_FRAMEWORK.format("AVFoundation"))
        return self._av

    def msg(self, restype, *argtypes) -> Callable:
        """``objc_msgSend`` cast to the exact signature of one selector (cached)."""
        key = (restype, argtypes)
        fn = self._msg_cache.get(key)
        if fn is None:
            with self._cache_lock:
                fn = self._msg_cache.get(key)
                if fn is None:
                    fn = ctypes.CFUNCTYPE(restype, *argtypes)(self._msg_send)
                    self._msg_cache[key] = fn
        return fn

    def cls(self, name: str) -> Optional[int]:
        raw = name.encode("ascii")
        if raw not in self._cls_cache:
            with self._cache_lock:
                self._cls_cache[raw] = self.objc_getClass(raw)
        return self._cls_cache[raw]

    def sel(self, name: str) -> int:
        raw = name.encode("ascii")
        sel = self._sel_cache.get(raw)
        if sel is None:
            with self._cache_lock:
                sel = self._sel_cache.get(raw)
                if sel is None:
                    sel = self.sel_registerName(raw)
                    self._sel_cache[raw] = sel
        return sel

    def _send(self, restype, obj: Optional[int], selector: str, *args: Optional[int]):
        if not obj:
            raise OSError(f"objc_msgSend({selector}) on nil")
        argtypes = (ctypes.c_void_p, ctypes.c_void_p) + (ctypes.c_void_p,) * len(args)
        return self.msg(restype, *argtypes)(obj, self.sel(selector), *args)

    def send_obj(self, obj: Optional[int], selector: str, *args: Optional[int]) -> Optional[int]:
        return self._send(ctypes.c_void_p, obj, selector, *args)

    def send_bool(self, obj: Optional[int], selector: str, *args: Optional[int]) -> bool:
        return bool(self._send(ctypes.c_bool, obj, selector, *args))

    def send_long(self, obj: Optional[int], selector: str, *args: Optional[int]) -> int:
        return int(self._send(ctypes.c_long, obj, selector, *args))

    def send_int32(self, obj: Optional[int], selector: str, *args: Optional[int]) -> int:
        return int(self._send(ctypes.c_int32, obj, selector, *args))

    def send_void(self, obj: Optional[int], selector: str, *args: Optional[int]) -> None:
        self._send(None, obj, selector, *args)

    # --- CoreFoundation helpers ----------------------------------------------------------

    def cfstr(self, text: str) -> Optional[int]:
        """New CFStringRef (caller releases) or None."""
        raw = text.encode("utf-8")
        return self.CFStringCreateWithBytes(None, raw, len(raw), K_CF_STRING_ENCODING_UTF8, False)

    def pystr(self, ref: Optional[int]) -> Optional[str]:
        if not ref:
            return None
        size = int(self.CFStringGetLength(ref)) * 4 + 8
        buf = ctypes.create_string_buffer(size)
        if not self.CFStringGetCString(ref, buf, size, K_CF_STRING_ENCODING_UTF8):
            return None
        return buf.value.decode("utf-8", "replace")

    def pybytes(self, data_ref: Optional[int]) -> Optional[bytes]:
        if not data_ref:
            return None
        length = int(self.CFDataGetLength(data_ref))
        if length == 0:
            return b""
        ptr = self.CFDataGetBytePtr(data_ref)
        if not ptr:
            return None
        return ctypes.string_at(ptr, length)


_lib: Optional[_Lib] = None
_lib_lock = threading.Lock()


def _get_lib() -> _Lib:
    global _lib
    if _lib is None:
        with _lib_lock:
            if _lib is None:
                _lib = _Lib()
    return _lib


def utf16_units(text: str) -> list[int]:
    """``text`` as a list of UTF-16 code units (astral chars become a surrogate pair)."""
    raw = text.encode("utf-16-le", "surrogatepass")
    return [int.from_bytes(raw[i:i + 2], "little") for i in range(0, len(raw), 2)]


class RealMacApi:
    """The macOS system calls the app uses, one method per call, no policy.

    Thread-safe: the only mutable state is the lazily built framework table and the
    callbacks kept alive for the run loop / Carbon handlers (both behind locks). Thread
    affinity follows the platform rules: Carbon hot key registration and delivery belong to
    the main thread, which is why ``MacHotkeyManager`` marshals its work there.
    """

    def __init__(self) -> None:
        self._keepalive_lock = threading.Lock()
        #: CFRunLoopSourceRef / EventHandlerRef -> the ctypes callback object ObjC calls.
        #: Without this the trampoline would be garbage collected and the callback would
        #: crash the process.
        self._keepalive: dict[int, object] = {}

    @property
    def _lib(self) -> _Lib:
        return _get_lib()

    # --- Objective-C autorelease pool ----------------------------------------------------

    @contextlib.contextmanager
    def _pool(self) -> Iterator[None]:
        """An NSAutoreleasePool around a block of Objective-C calls.

        Every method here may be called from a plain Python thread, which has no pool of its
        own, and the pasteboard/workspace getters return autoreleased objects.
        """
        lib = self._lib
        pool = None
        cls = lib.cls("NSAutoreleasePool")
        if cls:
            pool = lib.send_obj(lib.send_obj(cls, "alloc"), "init")
        try:
            yield
        finally:
            if pool:
                try:
                    lib.send_void(pool, "drain")
                except OSError:  # pragma: no cover
                    log.debug("NSAutoreleasePool drain failed")

    # --- keyboard state ------------------------------------------------------------------

    def modifier_flags(self) -> int:
        """``CGEventSourceFlagsState(kCGEventSourceStateCombinedSessionState)``.

        The physical modifier state right now, as CGEventFlags bits. Test it against
        MODIFIER_FLAG_MASK; the Caps Lock and non-coalesced bits are not held keys.
        """
        return int(self._lib.CGEventSourceFlagsState(KCG_EVENT_SOURCE_STATE_COMBINED_SESSION))

    # --- keyboard output -----------------------------------------------------------------

    def post_unicode_string(self, text: str) -> int:
        """Post one key-down + key-up pair carrying ``text`` as its unicode payload.

        ``CGEventCreateKeyboardEvent(NULL, 0, down)`` +
        ``CGEventKeyboardSetUnicodeString`` + ``CGEventPost(kCGHIDEventTap, ...)``.
        Returns the number of events posted (2 on success, fewer if an event could not be
        created). The flags are cleared explicitly so no real modifier state leaks in.
        """
        if not text:
            return 0
        units = utf16_units(text)
        buf = (ctypes.c_uint16 * len(units))(*units)
        lib = self._lib
        posted = 0
        for key_down in (True, False):
            event = lib.CGEventCreateKeyboardEvent(None, 0, key_down)
            if not event:
                log.warning("CGEventCreateKeyboardEvent returned NULL (unicode, down=%s)", key_down)
                return posted
            try:
                lib.CGEventSetFlags(event, 0)
                lib.CGEventKeyboardSetUnicodeString(event, len(units), buf)
                lib.CGEventPost(KCG_HID_EVENT_TAP, event)
            finally:
                lib.CFRelease(event)
            posted += 1
        return posted

    def post_key_chord(self, keycode: int, flags: int = 0) -> int:
        """Post key-down + key-up for one Carbon key code with ``flags`` (CGEventFlags) held.

        Returns the number of events posted (2 on success). This is how Cmd+V is sent: the V
        key code with ``KCG_FLAG_COMMAND``.
        """
        lib = self._lib
        posted = 0
        for key_down in (True, False):
            event = lib.CGEventCreateKeyboardEvent(None, keycode & 0xFFFF, key_down)
            if not event:
                log.warning("CGEventCreateKeyboardEvent returned NULL (keycode=%#x, down=%s)",
                            keycode, key_down)
                return posted
            try:
                lib.CGEventSetFlags(event, flags)
                lib.CGEventPost(KCG_HID_EVENT_TAP, event)
            finally:
                lib.CFRelease(event)
            posted += 1
        return posted

    # --- permissions ---------------------------------------------------------------------

    def is_process_trusted(self) -> bool:
        """``AXIsProcessTrusted()``: Accessibility permission, required for CGEventPost."""
        return bool(self._lib.AXIsProcessTrusted())

    def request_process_trust(self) -> bool:
        """``AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: true})``.

        Returns the current trust state and, when untrusted, asks the system to show the
        "allow Accessibility" prompt once.
        """
        lib = self._lib
        global_key = lib.kAXTrustedCheckOptionPrompt
        key = global_key.value if global_key is not None else None
        own_key = None if key else lib.cfstr(AX_TRUSTED_CHECK_OPTION_PROMPT)
        key = key or own_key
        if not key:
            # No option dictionary: still answers "am I trusted?", just without the prompt.
            return bool(lib.AXIsProcessTrustedWithOptions(None))
        keys = (ctypes.c_void_p * 1)(key)
        values = (ctypes.c_void_p * 1)(lib.kCFBooleanTrue.value)
        options = lib.CFDictionaryCreate(
            None, keys, values, 1,
            ctypes.addressof(lib.kCFTypeDictionaryKeyCallBacks),
            ctypes.addressof(lib.kCFTypeDictionaryValueCallBacks),
        )
        try:
            return bool(lib.AXIsProcessTrustedWithOptions(options))
        finally:
            if options:
                lib.CFRelease(options)
            if own_key:
                lib.CFRelease(own_key)

    def is_secure_event_input_enabled(self) -> bool:
        """``IsSecureEventInputEnabled()``: a password field owns the keyboard, so synthetic
        keystrokes are dropped. The macOS analogue of the Windows UIPI check."""
        return bool(self._lib.IsSecureEventInputEnabled())

    def microphone_authorization_status(self) -> int:
        """``[AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio]``
        (AV_AUTH_* above). Raises OSError if AVFoundation is unavailable."""
        lib = self._lib
        lib.av()
        with self._pool():
            cls = lib.cls("AVCaptureDevice")
            if not cls:
                raise OSError("AVCaptureDevice is not available")
            media = lib.cfstr(AV_MEDIA_TYPE_AUDIO)
            try:
                return lib.send_long(cls, "authorizationStatusForMediaType:", media)
            finally:
                if media:
                    lib.CFRelease(media)

    def request_microphone_access(self) -> bool:
        """``[AVCaptureDevice requestAccessForMediaType:completionHandler:]``.

        Returns True if the request was issued. The completion handler is a global
        Objective-C block that only logs: the answer is read back with
        ``microphone_authorization_status()``, because the handler runs on an arbitrary
        queue long after this returns.
        """
        lib = self._lib
        lib.av()
        if lib._global_block_isa is None:
            raise OSError("_NSConcreteGlobalBlock is unavailable; cannot build the ObjC block")
        with self._pool():
            cls = lib.cls("AVCaptureDevice")
            if not cls:
                raise OSError("AVCaptureDevice is not available")
            media = lib.cfstr(AV_MEDIA_TYPE_AUDIO)
            try:
                block = _microphone_block(lib)
                lib.send_void(cls, "requestAccessForMediaType:completionHandler:", media,
                              ctypes.addressof(block))
            finally:
                if media:
                    lib.CFRelease(media)
        return True

    # --- pasteboard ----------------------------------------------------------------------

    def _general_pasteboard(self) -> int:
        lib = self._lib
        cls = lib.cls("NSPasteboard")
        pasteboard = lib.send_obj(cls, "generalPasteboard") if cls else None
        if not pasteboard:
            raise OSError("[NSPasteboard generalPasteboard] returned nil")
        return pasteboard

    def pasteboard_change_count(self) -> int:
        with self._pool():
            return self._lib.send_long(self._general_pasteboard(), "changeCount")

    def pasteboard_types(self) -> list[str]:
        """``[[NSPasteboard generalPasteboard] types]`` as UTI strings, in order."""
        lib = self._lib
        with self._pool():
            types = lib.send_obj(self._general_pasteboard(), "types")
            if not types:
                return []
            out: list[str] = []
            for i in range(int(lib.CFArrayGetCount(types))):
                name = lib.pystr(lib.CFArrayGetValueAtIndex(types, i))
                if name is not None:
                    out.append(name)
            return out

    def pasteboard_data(self, type_id: str) -> Optional[bytes]:
        """``[pasteboard dataForType:type_id]`` as bytes, or None when absent/unreadable."""
        lib = self._lib
        with self._pool():
            pasteboard = self._general_pasteboard()
            ref = lib.cfstr(type_id)
            try:
                if not ref:
                    return None
                data = lib.send_obj(pasteboard, "dataForType:", ref)
                return lib.pybytes(data)
            finally:
                if ref:
                    lib.CFRelease(ref)

    def pasteboard_clear(self) -> int:
        """``[pasteboard clearContents]``; returns the new change count."""
        with self._pool():
            return self._lib.send_long(self._general_pasteboard(), "clearContents")

    def pasteboard_set_data(self, type_id: str, data: bytes) -> bool:
        """``[pasteboard setData:forType:]``. Must follow a clearContents."""
        lib = self._lib
        with self._pool():
            pasteboard = self._general_pasteboard()
            type_ref = lib.cfstr(type_id)
            data_ref = lib.CFDataCreate(None, data, len(data))
            try:
                if not type_ref or not data_ref:
                    return False
                return lib.send_bool(pasteboard, "setData:forType:", data_ref, type_ref)
            finally:
                if type_ref:
                    lib.CFRelease(type_ref)
                if data_ref:
                    lib.CFRelease(data_ref)

    def pasteboard_set_string(self, text: str) -> bool:
        """``[pasteboard setString:forType:NSPasteboardTypeString]``. Must follow a clearContents."""
        lib = self._lib
        with self._pool():
            pasteboard = self._general_pasteboard()
            value = lib.cfstr(text)
            type_ref = lib.cfstr(NS_PASTEBOARD_TYPE_STRING)
            try:
                if not value or not type_ref:
                    return False
                return lib.send_bool(pasteboard, "setString:forType:", value, type_ref)
            finally:
                if value:
                    lib.CFRelease(value)
                if type_ref:
                    lib.CFRelease(type_ref)

    def pasteboard_string(self) -> Optional[str]:
        """``[pasteboard stringForType:NSPasteboardTypeString]`` or None."""
        lib = self._lib
        with self._pool():
            pasteboard = self._general_pasteboard()
            type_ref = lib.cfstr(NS_PASTEBOARD_TYPE_STRING)
            try:
                if not type_ref:
                    return None
                return lib.pystr(lib.send_obj(pasteboard, "stringForType:", type_ref))
            finally:
                if type_ref:
                    lib.CFRelease(type_ref)

    # --- frontmost application -----------------------------------------------------------

    def frontmost_app(self) -> Optional[MacAppInfo]:
        """``[[NSWorkspace sharedWorkspace] frontmostApplication]``, for log lines.

        The Windows injector reads the foreground window's integrity level; macOS has no
        equivalent, so this is diagnostic only.
        """
        lib = self._lib
        with self._pool():
            cls = lib.cls("NSWorkspace")
            workspace = lib.send_obj(cls, "sharedWorkspace") if cls else None
            app = lib.send_obj(workspace, "frontmostApplication") if workspace else None
            if not app:
                return None
            return MacAppInfo(
                pid=lib.send_int32(app, "processIdentifier"),
                bundle_id=lib.pystr(lib.send_obj(app, "bundleIdentifier")) or "",
                name=lib.pystr(lib.send_obj(app, "localizedName")) or "",
            )

    # --- Carbon hot keys -----------------------------------------------------------------

    def install_hotkey_handler(self, on_hotkey: Callable[[int], None]) -> tuple[int, int]:
        """``InstallEventHandler`` for kEventClassKeyboard/kEventHotKeyPressed.

        ``on_hotkey(hotkey_id)`` is called with the EventHotKeyID.id of each press, on the
        thread that runs the handler (the main thread). Returns (handler_ref, OSStatus);
        handler_ref is 0 when the status is not noErr. Call on the main thread.
        """
        lib = self._lib

        def _trampoline(_call_ref, event, _user_data) -> int:
            try:
                hotkey_id = EventHotKeyID()
                status = lib.GetEventParameter(
                    event, K_EVENT_PARAM_DIRECT_OBJECT, K_TYPE_EVENT_HOT_KEY_ID, None,
                    ctypes.sizeof(hotkey_id), None, ctypes.byref(hotkey_id))
                if status != NO_ERR:
                    log.warning("GetEventParameter(hkid) failed, OSStatus=%d", status)
                    return EVENT_NOT_HANDLED_ERR
                on_hotkey(int(hotkey_id.id))
            except BaseException:
                # Never let an exception unwind into Carbon's dispatcher.
                log.exception("hotkey event handler raised")
            return NO_ERR

        callback = EVENT_HANDLER_PROC(_trampoline)
        spec = EventTypeSpec(K_EVENT_CLASS_KEYBOARD, K_EVENT_HOT_KEY_PRESSED)
        handler = ctypes.c_void_p()
        status = int(lib.InstallEventHandler(
            lib.GetApplicationEventTarget(), callback, 1, ctypes.byref(spec), None,
            ctypes.byref(handler)))
        if status != NO_ERR or not handler.value:
            return 0, status
        with self._keepalive_lock:
            self._keepalive[handler.value] = callback
        return int(handler.value), status

    def remove_event_handler(self, handler_ref: int) -> int:
        """``RemoveEventHandler``; returns the OSStatus. Call on the main thread."""
        if not handler_ref:
            return NO_ERR
        status = int(self._lib.RemoveEventHandler(handler_ref))
        with self._keepalive_lock:
            self._keepalive.pop(handler_ref, None)
        return status

    def register_event_hotkey(self, keycode: int, modifiers: int, hotkey_id: int) -> tuple[int, int]:
        """``RegisterEventHotKey`` on the application event target.

        Returns (EventHotKeyRef, OSStatus); the ref is 0 unless the status is noErr. A combo
        already owned by another app or by macOS gives EVENT_HOT_KEY_EXISTS_ERR (-9878).
        Call on the main thread.
        """
        lib = self._lib
        ref = ctypes.c_void_p()
        status = int(lib.RegisterEventHotKey(
            keycode & 0xFFFFFFFF, modifiers & 0xFFFFFFFF,
            EventHotKeyID(HOTKEY_SIGNATURE, hotkey_id & 0xFFFFFFFF),
            lib.GetApplicationEventTarget(), 0, ctypes.byref(ref)))
        if status != NO_ERR:
            return 0, status
        return int(ref.value or 0), status

    def unregister_event_hotkey(self, hotkey_ref: int) -> int:
        """``UnregisterEventHotKey``; returns the OSStatus. Call on the main thread."""
        if not hotkey_ref:
            return NO_ERR
        return int(self._lib.UnregisterEventHotKey(hotkey_ref))

    # --- main run loop -------------------------------------------------------------------

    def is_main_thread(self) -> bool:
        return threading.current_thread() is threading.main_thread()

    def create_main_run_loop_source(self, perform: Callable[[], None]) -> int:
        """A version-0 CFRunLoopSource added to the main run loop in the common modes.

        ``perform()`` runs on the main thread whenever the source has been signalled and the
        main run loop gets a turn. Returns the CFRunLoopSourceRef (0 on failure). Safe to
        call from any thread: CFRunLoop itself is thread-safe.
        """
        lib = self._lib

        def _trampoline(_info) -> None:
            try:
                perform()
            except BaseException:
                log.exception("run loop source callback raised")

        callback = RUN_LOOP_PERFORM_PROC(_trampoline)
        context = CFRunLoopSourceContext()
        context.version = 0
        context.perform = ctypes.cast(callback, ctypes.c_void_p)
        source = lib.CFRunLoopSourceCreate(None, 0, ctypes.byref(context))
        if not source:
            return 0
        with self._keepalive_lock:
            self._keepalive[int(source)] = (callback, context)
        lib.CFRunLoopAddSource(lib.CFRunLoopGetMain(), source, lib.kCFRunLoopCommonModes)
        return int(source)

    def signal_main_run_loop_source(self, source: int) -> None:
        """Mark the source ready and wake the main run loop. Any thread."""
        if not source:
            return
        lib = self._lib
        lib.CFRunLoopSourceSignal(source)
        lib.CFRunLoopWakeUp(lib.CFRunLoopGetMain())

    def destroy_main_run_loop_source(self, source: int) -> None:
        """Remove, invalidate and release the source. Any thread."""
        if not source:
            return
        lib = self._lib
        lib.CFRunLoopRemoveSource(lib.CFRunLoopGetMain(), source, lib.kCFRunLoopCommonModes)
        lib.CFRunLoopSourceInvalidate(source)
        lib.CFRelease(source)
        with self._keepalive_lock:
            self._keepalive.pop(source, None)

    def run_main_run_loop_once(self, timeout_s: float) -> int:
        """``CFRunLoopRunInMode(kCFRunLoopDefaultMode, timeout_s, true)``.

        Only useful where nobody else drives the main run loop (tests, CLI). In the app Qt
        owns the main run loop and this is never called. Main thread only.
        """
        lib = self._lib
        return int(lib.CFRunLoopRunInMode(lib.kCFRunLoopDefaultMode, float(timeout_s), True))

    # --- time (injectable for tests) -----------------------------------------------------

    def monotonic(self) -> float:
        return time.perf_counter()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def _microphone_block(lib: _Lib) -> _BlockLiteral:
    """A process-wide global block for requestAccessForMediaType:completionHandler:.

    It is cached in a module global because the block and its invoke trampoline must outlive
    this call: AVFoundation invokes the handler on its own queue after the user answers.
    """
    global _MIC_BLOCK, _MIC_BLOCK_INVOKE, _MIC_BLOCK_DESCRIPTOR
    if _MIC_BLOCK is not None:
        return _MIC_BLOCK

    def _invoke(_block, granted: bool) -> None:
        log.info("microphone permission answered granted=%s", bool(granted))

    _MIC_BLOCK_INVOKE = BLOCK_INVOKE_PROC(_invoke)
    _MIC_BLOCK_DESCRIPTOR = _BlockDescriptor(0, ctypes.sizeof(_BlockLiteral))
    block = _BlockLiteral()
    # _NSConcreteGlobalBlock is itself the Class struct, so its *address* is the isa pointer.
    block.isa = lib._global_block_isa
    block.flags = BLOCK_IS_GLOBAL
    block.reserved = 0
    block.invoke = ctypes.cast(_MIC_BLOCK_INVOKE, ctypes.c_void_p)
    block.descriptor = ctypes.pointer(_MIC_BLOCK_DESCRIPTOR)
    _MIC_BLOCK = block
    return block


_MIC_BLOCK: Optional[_BlockLiteral] = None
_MIC_BLOCK_INVOKE: Optional[Callable] = None
_MIC_BLOCK_DESCRIPTOR: Optional[_BlockDescriptor] = None
