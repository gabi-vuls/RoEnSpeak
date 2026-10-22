"""Test target window for the desktop injection test (run as a separate process).

    python _tk_target.py OUT_DIR

Opens a topmost tkinter window with one Text widget that has keyboard focus. It writes:
  OUT_DIR/hwnd.txt   the top-level window handle (decimal), once the window is mapped
  OUT_DIR/text.txt   the widget's current content, rewritten atomically every 100 ms
It exits when OUT_DIR/quit exists, or when the window is closed.
"""

from __future__ import annotations

import ctypes
import os
import sys
import tkinter as tk
from pathlib import Path


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    root.title("RoEnSpeak desktop test target")
    root.geometry("700x240+80+80")
    root.attributes("-topmost", True)
    text = tk.Text(root, wrap="word", font=("Segoe UI", 12))
    text.pack(fill="both", expand=True)

    last = [None]

    def tick() -> None:
        if (out / "quit").exists():
            root.destroy()
            return
        content = text.get("1.0", "end-1c")
        if content != last[0]:
            _atomic_write(out / "text.txt", content)
            last[0] = content
        root.after(100, tick)

    def ready() -> None:
        root.focus_force()
        text.focus_set()
        ga_root = 2
        hwnd = ctypes.windll.user32.GetAncestor(root.winfo_id(), ga_root)
        _atomic_write(out / "hwnd.txt", str(hwnd))

    root.after(200, ready)
    root.after(100, tick)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
