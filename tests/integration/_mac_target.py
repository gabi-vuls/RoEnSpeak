"""Test target window for the macOS desktop injection test (run as a separate process).

    python _mac_target.py OUT_DIR

The macOS analogue of _tk_target.py. Opens an always-on-top tkinter window with one Text
widget that has keyboard focus, and writes:

  OUT_DIR/pid.txt    this process's pid (decimal), once the window is mapped and focused
  OUT_DIR/text.txt   the widget's current content, rewritten atomically every 100 ms

It watches for two sentinels, polled on the same 100 ms tick:

  OUT_DIR/raise      "bring yourself to the front and take the keyboard"; deleted once done
  OUT_DIR/quit       exit

Why a pid and not a window handle: there is no HWND on macOS, and nothing in the public API
takes "a window" the way SetForegroundWindow does -- activation is per **application**, so the
pid is the handle and ``NSWorkspace.frontmostApplication.processIdentifier`` is the check.

Why the ``raise`` sentinel and not just ``NSRunningApplication.activateWithOptions:`` from the
test: macOS only honours a *cross*-app activation request (even with
``activateIgnoringOtherApps``) when the requesting process is itself an active app. pytest is a
background CLI tool, so the call returns YES and nothing happens -- measured here: with Finder
frontmost, 20 cross-app attempts over 6 s never took the front, while a self-activation took it
on the first try. An app activating *itself* always works and needs no permission, so the test
asks and this process does it.
"""

from __future__ import annotations

import ctypes
import os
import sys
import tkinter as tk
from pathlib import Path

#: NSApplicationActivationOptions: activateAllWindows | activateIgnoringOtherApps.
NS_ACTIVATE_ALL_WINDOWS = 1 << 0
NS_ACTIVATE_IGNORING_OTHER_APPS = 1 << 1


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _self_activate() -> None:
    """``NSRunningApplication.currentApplication().activateWithOptions:`` through ctypes.

    An app activating *itself* never needs a permission, which is why this is here and not
    only in the test process. Failure is non-fatal: the test re-activates by pid anyway.
    """
    try:
        objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AppKit.framework/AppKit")
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        send = objc.objc_msgSend
        send.restype = ctypes.c_void_p
        send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cls = objc.objc_getClass(b"NSRunningApplication")
        app = send(cls, objc.sel_registerName(b"currentApplication"))
        if not app:
            return
        activate = objc.objc_msgSend
        activate.restype = ctypes.c_bool
        activate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        activate(app, objc.sel_registerName(b"activateWithOptions:"),
                 NS_ACTIVATE_ALL_WINDOWS | NS_ACTIVATE_IGNORING_OTHER_APPS)
    except Exception as e:  # pragma: no cover - diagnostics only
        print(f"_mac_target: self-activation failed: {e}", file=sys.stderr, flush=True)


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    root.title("RoEnSpeak desktop test target")
    root.geometry("700x240+80+80")
    root.attributes("-topmost", True)
    text = tk.Text(root, wrap="word", font=("Helvetica", 14), undo=False)
    text.pack(fill="both", expand=True)

    last: list[str | None] = [None]

    def come_forward() -> None:
        _self_activate()
        root.lift()
        root.attributes("-topmost", True)
        root.focus_force()
        text.focus_set()

    def tick() -> None:
        if (out / "quit").exists():
            root.destroy()
            return
        raise_flag = out / "raise"
        if raise_flag.exists():
            come_forward()
            try:
                raise_flag.unlink()  # the test waits for this to disappear
            except OSError:
                pass
        content = text.get("1.0", "end-1c")
        if content != last[0]:
            _atomic_write(out / "text.txt", content)
            last[0] = content
        root.after(100, tick)

    def ready() -> None:
        come_forward()
        # Last, so the test only sees the pid once the window really has the keyboard.
        _atomic_write(out / "pid.txt", str(os.getpid()))

    root.after(400, ready)
    root.after(100, tick)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
