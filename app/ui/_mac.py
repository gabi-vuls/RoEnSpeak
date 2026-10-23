"""Small ctypes/Objective-C helpers the UI needs on macOS (no app.macos import, no pyobjc).

The macOS counterpart of ``app/ui/_win.py``: self-contained, dependency-free, and every
function is a safe no-op off macOS so the UI modules stay importable everywhere. The UI
imports it through ``app.ui._platform``, never directly (except ``sounds``, which asks for
the NSSound backend by name).

Why this exists at all
----------------------
Qt alone cannot keep the "● REC" overlay usable on macOS:

* ``Qt.Tool`` maps onto an ``NSPanel`` whose ``hidesOnDeactivate`` is YES, so the overlay
  would vanish the moment the user's target app takes focus - which is exactly when we are
  dictating into it.
* ``Qt.WindowStaysOnTopHint`` only gives ``NSFloatingWindowLevel``, which is above our own
  windows but not above another application's.
* ``Qt.WindowTransparentForInput`` is honoured per view, not per window; the window still
  swallows clicks in the areas Qt considers opaque.
* A window that belongs to one Space disappears when the user switches Space.

So we reach the ``NSWindow`` behind the Qt window id and set ``ignoresMouseEvents``, a window
level above every normal application window, ``hidesOnDeactivate = NO`` and a collection
behaviour that joins all Spaces.

``objc_msgSend`` is **always** called through a fully typed function pointer (one
``CFUNCTYPE`` per signature, cached). An untyped call is undefined behaviour on arm64, where
the ABI passes small integers and floats differently from pointers.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from functools import lru_cache
from typing import Optional, Tuple

log = logging.getLogger("roenspeak.ui.mac")

IS_MACOS = sys.platform == "darwin"

# -- AppKit constants (AppKit/NSWindow.h, AppKit/NSApplication.h) -------------
NSWindowCollectionBehaviorCanJoinAllSpaces = 1 << 0
NSWindowCollectionBehaviorStationary = 1 << 4
NSWindowCollectionBehaviorIgnoresCycle = 1 << 6
NSWindowStyleMaskNonactivatingPanel = 1 << 7

NSNormalWindowLevel = 0
NSFloatingWindowLevel = 3
NSMainMenuWindowLevel = 24
NSStatusWindowLevel = 25
NSScreenSaverWindowLevel = 1000

NSApplicationActivationPolicyRegular = 0
NSApplicationActivationPolicyAccessory = 1

#: Level for the REC overlay. Screen-saver level is the only one that also stays above
#: another app's full-screen window; the overlay is placed inside availableGeometry(), so it
#: never covers the menu bar even though its level is above it.
OVERLAY_WINDOW_LEVEL = NSScreenSaverWindowLevel
#: Visible on every Space (canJoinAllSpaces), does not move with Space switches (stationary),
#: and is skipped by Cmd+` window cycling (ignoresCycle).
OVERLAY_COLLECTION_BEHAVIOR = (
    NSWindowCollectionBehaviorCanJoinAllSpaces
    | NSWindowCollectionBehaviorStationary
    | NSWindowCollectionBehaviorIgnoresCycle
)

_objc: Optional[ctypes.CDLL] = None
_libc: Optional[ctypes.CDLL] = None
_msg_send_addr = 0

if IS_MACOS:  # pragma: no cover - exercised on macOS only
    try:
        _objc = ctypes.CDLL("/usr/lib/libobjc.dylib", use_errno=True)
        _libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
        # AppKit may not be linked yet when this module is imported before QApplication;
        # loading it here makes NSWindow / NSApplication / NSSound resolvable either way.
        ctypes.CDLL("/System/Library/Frameworks/AppKit.framework/AppKit")
        _objc.objc_getClass.argtypes = [ctypes.c_char_p]
        _objc.objc_getClass.restype = ctypes.c_void_p
        _objc.sel_registerName.argtypes = [ctypes.c_char_p]
        _objc.sel_registerName.restype = ctypes.c_void_p
        _objc.object_getClass.argtypes = [ctypes.c_void_p]
        _objc.object_getClass.restype = ctypes.c_void_p
        _objc.class_getName.argtypes = [ctypes.c_void_p]
        _objc.class_getName.restype = ctypes.c_char_p
        _libc.malloc_zone_from_ptr.argtypes = [ctypes.c_void_p]
        _libc.malloc_zone_from_ptr.restype = ctypes.c_void_p
        _msg_send_addr = ctypes.cast(_objc.objc_msgSend, ctypes.c_void_p).value or 0
    except (OSError, AttributeError) as exc:
        log.warning("Objective-C runtime unavailable: %s", exc)
        _objc = None
        _libc = None
        _msg_send_addr = 0


@lru_cache(maxsize=None)
def _cls(name: str) -> Optional[ctypes.c_void_p]:
    """The Objective-C class object, or None when the runtime/class is unavailable."""
    if _objc is None:
        return None
    ptr = _objc.objc_getClass(name.encode("utf-8"))
    if not ptr:
        log.warning("objc class %s not found", name)
        return None
    return ctypes.c_void_p(ptr)


@lru_cache(maxsize=None)
def _sel(name: str) -> Optional[ctypes.c_void_p]:
    if _objc is None:
        return None
    return ctypes.c_void_p(_objc.sel_registerName(name.encode("utf-8")))


@lru_cache(maxsize=None)
def _msg_send(restype, argtypes: Tuple[type, ...]):
    """A typed objc_msgSend for exactly this (restype, argtypes) signature."""
    proto = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)
    return proto(_msg_send_addr)


def _send(receiver, selector: str, restype=ctypes.c_void_p, argtypes: Tuple[type, ...] = (), *args):
    """``[receiver selector:args...]`` through a properly typed objc_msgSend.

    Returns None when the runtime is unavailable or the receiver is nil.
    """
    if not _msg_send_addr or not receiver:
        return None
    sel = _sel(selector)
    if sel is None:
        return None
    return _msg_send(restype, tuple(argtypes))(receiver, sel, *args)


def _responds(obj, selector: str) -> bool:
    return bool(_send(obj, "respondsToSelector:", ctypes.c_bool, (ctypes.c_void_p,), _sel(selector)))


def _is_kind_of(obj, class_name: str) -> bool:
    cls = _cls(class_name)
    if cls is None:
        return False
    return bool(_send(obj, "isKindOfClass:", ctypes.c_bool, (ctypes.c_void_p,), cls))


def _ns_string(text: str):
    cls = _cls("NSString")
    if cls is None:
        return None
    return _send(cls, "stringWithUTF8String:", ctypes.c_void_p, (ctypes.c_char_p,), text.encode("utf-8"))


def _ns_app():
    """``NSApp``. Only ever called after QApplication exists: ``[NSApplication
    sharedApplication]`` creates the shared instance, and Qt must be the one to create it
    (it installs its own NSApplication subclass)."""
    cls = _cls("NSApplication")
    if cls is None:
        return None
    return _send(cls, "sharedApplication", ctypes.c_void_p, ())


def _looks_like_objc_object(winid: int) -> bool:
    """Cheap sanity check before messaging a pointer that came from Qt.

    ``QWidget.winId()`` is only an Objective-C object pointer under the *cocoa* platform
    plugin. Under ``QT_QPA_PLATFORM=offscreen`` (tests, icon generation, CI) it is a plain
    counter, and sending a message to 1 or 2 crashes the process. A valid object lives in a
    malloc zone and has a readable class, so check both.
    """
    if _objc is None or _libc is None:
        return False
    try:
        if not _libc.malloc_zone_from_ptr(ctypes.c_void_p(winid)):
            return False
        cls = _objc.object_getClass(ctypes.c_void_p(winid))
        if not cls or not _libc.malloc_zone_from_ptr(ctypes.c_void_p(cls)):
            return False
        return bool(_objc.class_getName(ctypes.c_void_p(cls)))
    except (OSError, ValueError):
        return False


def _ns_window(winid: int):
    """The NSWindow behind a Qt ``winId()``.

    Qt returns the ``NSView*`` of the widget on macOS, so we ask the view for its window.
    A raw NSWindow is accepted too, in case a future Qt changes that.
    """
    if not IS_MACOS or not winid or not _msg_send_addr:
        return None
    if not _looks_like_objc_object(int(winid)):
        log.debug("winid %#x is not an Objective-C object (non-cocoa Qt platform?)", int(winid))
        return None
    obj = ctypes.c_void_p(int(winid))
    if _is_kind_of(obj, "NSWindow"):
        return obj
    if not _responds(obj, "window"):
        log.debug("winid %#x is neither an NSWindow nor a view with a window", int(winid))
        return None
    win = _send(obj, "window", ctypes.c_void_p, ())
    return ctypes.c_void_p(win) if win else None


# ---------------------------------------------------------------------------
# Overlay window: click-through, above everything, visible while unfocused
# ---------------------------------------------------------------------------


def make_click_through_noactivate(winid: int) -> int:
    """Make the NSWindow behind ``winid`` a click-through, always-visible overlay.

    Sets ``ignoresMouseEvents = YES``, ``level = OVERLAY_WINDOW_LEVEL`` (above every normal
    application window), ``hidesOnDeactivate = NO`` (the Qt.Tool default would hide it as
    soon as the user's target app takes focus), ``canHide = NO`` and
    ``collectionBehavior = OVERLAY_COLLECTION_BEHAVIOR``. For the ``NSPanel`` Qt creates for
    a ``Qt.Tool`` window it also sets the non-activating panel style, so clicking near it
    never makes RoEnSpeak the active app.

    Returns the resulting window level (the Windows twin returns the resulting ex-style);
    0 when there is no window to touch.
    """
    win = _ns_window(winid)
    if win is None:
        return 0
    try:
        _send(win, "setIgnoresMouseEvents:", None, (ctypes.c_bool,), True)
        _send(win, "setHidesOnDeactivate:", None, (ctypes.c_bool,), False)
        _send(win, "setCanHide:", None, (ctypes.c_bool,), False)
        _send(win, "setCollectionBehavior:", None, (ctypes.c_ulong,), OVERLAY_COLLECTION_BEHAVIOR)
        if _responds(win, "setBecomesKeyOnlyIfNeeded:"):  # NSPanel only (Qt.Tool)
            _send(win, "setBecomesKeyOnlyIfNeeded:", None, (ctypes.c_bool,), True)
            _send(win, "setFloatingPanel:", None, (ctypes.c_bool,), True)
            mask = _send(win, "styleMask", ctypes.c_ulong, ())
            if mask is not None:
                _send(
                    win, "setStyleMask:", None, (ctypes.c_ulong,),
                    int(mask) | NSWindowStyleMaskNonactivatingPanel,
                )
        # setLevel: LAST: -setFloatingPanel:YES resets the level to NSFloatingWindowLevel
        # (3), which is above other apps' normal windows but below their panels and below a
        # full-screen window. Qt also re-applies its own level when it orders the window
        # front, which is why RecOverlay re-asserts this after show().
        _send(win, "setLevel:", None, (ctypes.c_long,), OVERLAY_WINDOW_LEVEL)
        level = _send(win, "level", ctypes.c_long, ())
    except (OSError, ValueError) as exc:  # never let overlay styling break dictation
        log.warning("overlay window styling failed: %s", exc)
        return 0
    return int(level or 0)


def set_foreground_window(winid: int) -> bool:
    """Bring our window and our app to the front (the SetForegroundWindow twin).

    ``activateIgnoringOtherApps:YES`` is needed because a menu-bar (accessory) app is not
    activated by Qt's ``activateWindow()`` alone, and ``orderFrontRegardless`` puts the
    window above the previously active app's windows.
    """
    if not IS_MACOS:
        return False
    app = _ns_app()
    if app is not None:
        _send(app, "activateIgnoringOtherApps:", None, (ctypes.c_bool,), True)
    win = _ns_window(winid)
    if win is None:
        return app is not None
    _send(win, "orderFrontRegardless", None, ())
    return True


# ---------------------------------------------------------------------------
# Activation policy (Dock icon)
# ---------------------------------------------------------------------------


def _set_activation_policy(policy: int) -> bool:
    if not IS_MACOS:
        return False
    app = _ns_app()
    if app is None:
        return False
    ok = _send(app, "setActivationPolicy:", ctypes.c_bool, (ctypes.c_long,), policy)
    log.debug("setActivationPolicy(%d) -> %s", policy, bool(ok))
    return bool(ok)


def set_activation_policy_accessory() -> bool:
    """Menu-bar app: no Dock icon, no app switcher entry, windows still usable.

    The shipped bundle declares ``LSUIElement`` in its Info.plist, but a dev run
    (``python -m app.main``) has no Info.plist, so it would otherwise bounce into the Dock
    as "Python". create_qapplication() calls this on macOS to match the bundle.
    """
    return _set_activation_policy(NSApplicationActivationPolicyAccessory)


def set_activation_policy_regular() -> bool:
    """Back to a normal Dock app (not used by the app; here for debugging and symmetry)."""
    return _set_activation_policy(NSApplicationActivationPolicyRegular)


# ---------------------------------------------------------------------------
# Sounds (NSSound; the winsound twin)
# ---------------------------------------------------------------------------

#: One retained NSSound per file, so repeats do not allocate and nothing is deallocated
#: while it is still playing.
_sounds: dict[str, ctypes.c_void_p] = {}


def _ns_sound(path: str):
    snd = _sounds.get(path)
    if snd is not None:
        return snd
    cls = _cls("NSSound")
    nsstr = _ns_string(path)
    if cls is None or not nsstr:
        return None
    obj = _send(cls, "alloc", ctypes.c_void_p, ())
    if not obj:
        return None
    snd_ptr = _send(
        ctypes.c_void_p(obj), "initWithContentsOfFile:byReference:",
        ctypes.c_void_p, (ctypes.c_void_p, ctypes.c_bool), ctypes.c_void_p(nsstr), True,
    )
    if not snd_ptr:
        log.debug("NSSound could not load %s", path)
        return None
    snd = ctypes.c_void_p(snd_ptr)
    _sounds[path] = snd  # kept alive for the process lifetime (two files at most)
    return snd


def play_sound_async(path: str) -> bool:
    """Start playing a sound file and return immediately. True when it started.

    NSSound.play is asynchronous, so the Qt thread never blocks. A sound that is still
    playing is stopped first, so a quick start/stop pair cannot be dropped.
    """
    if not IS_MACOS:
        return False
    try:
        snd = _ns_sound(path)
        if snd is None:
            return False
        if _send(snd, "isPlaying", ctypes.c_bool, ()):
            _send(snd, "stop", ctypes.c_bool, ())
        return bool(_send(snd, "play", ctypes.c_bool, ()))
    except (OSError, ValueError) as exc:
        log.debug("NSSound play(%s) failed: %s", path, exc)
        return False


# ---------------------------------------------------------------------------
# Windows-only concepts: present as no-ops so the UI needs no platform branches
# ---------------------------------------------------------------------------


class MSG(ctypes.Structure):
    """Placeholder for the Win32 MSG struct. macOS has no window messages; it exists only
    so ``app.ui._platform`` can re-export the same names on both hosts."""

    _fields_ = []  # type: ignore[var-annotated]


def read_msg(address: int) -> Optional[MSG]:
    """Always None. On Windows this copies the MSG a native event filter was handed;
    on macOS no native event filter is installed (see app.ui.app.UiRoot)."""
    return None


def register_window_message(name: str) -> int:
    """Always 0. RegisterWindowMessageW gives Windows a process-wide message id a second
    instance can broadcast. On macOS a second launch connects to the first instance's Unix
    socket instead (app.macos.instance.FileLockSingleInstance), and returning 0 keeps the
    native-filter branch in app.ui.app switched off."""
    return 0


def allow_message_through_uipi(hwnd: int, message: int) -> None:
    """No-op. ChangeWindowMessageFilterEx lets a lower-integrity second instance's broadcast
    reach our window; macOS has no UIPI and the show-window ping arrives on a socket."""
    return None


def set_app_user_model_id(app_id: str) -> None:
    """No-op. The AppUserModelID gives a Windows process its own taskbar identity (our icon
    instead of python.exe's). On macOS the bundle's Info.plist (CFBundleIdentifier,
    CFBundleIconFile) is the equivalent, and a dev run simply shows up as Python."""
    return None
