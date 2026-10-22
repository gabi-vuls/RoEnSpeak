"""Unit tests for hotkey parsing and Win32HotkeyManager (TEST-UNIT-WIN32).

Written from SPEC.md and app/contracts.py only. All Win32 calls go through FakeWinApi,
which is also imported by test_injection.py.

Implementation modules are imported lazily inside helpers so that a missing module fails
only the tests that need it (and so test_injection.py can import FakeWinApi without
pulling in app.win32.hotkey).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Optional, Sequence

import pytest

from app.contracts import (
    DEFAULT_HOTKEY,
    IL_MEDIUM,
    MOD_ALT,
    MOD_CONTROL,
    MOD_NOREPEAT,
    MOD_SHIFT,
    MOD_WIN,
    VK_MENU_MASK,
    AppError,
    Hotkey,
    HotkeyConflictError,
    HotkeyParseError,
    KeyEvent,
    Modifier,
    WinMessage,
)

# ---------------------------------------------------------------------------
# Win32 constants used by the fake
# ---------------------------------------------------------------------------

WM_QUIT = 0x0012
WM_TIMER = 0x0113
WM_HOTKEY = 0x0312
ERROR_HOTKEY_ALREADY_REGISTERED = 1409
ERROR_HOTKEY_NOT_REGISTERED = 1419
_MOD_MASK = MOD_ALT | MOD_CONTROL | MOD_SHIFT | MOD_WIN

#: Generic modifier VK -> the L/R VKs it reflects (GetAsyncKeyState(VK_MENU) is down when
#: either Alt is down, etc.).
_GENERIC_VKS = {0x10: (0xA0, 0xA1), 0x11: (0xA2, 0xA3), 0x12: (0xA4, 0xA5)}

_QUIT = object()


def run_with_timeout(fn: Callable, *args, timeout: float = 2.0, **kwargs):
    """Run fn on a helper thread; fail (instead of hanging) if it doesn't return in time.
    Re-raises the function's exception on the calling thread."""
    box: dict = {}

    def target() -> None:
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True, name=f"timeout-{getattr(fn, '__name__', 'fn')}")
    t.start()
    t.join(timeout)
    assert not t.is_alive(), f"{getattr(fn, '__qualname__', fn)} did not return within {timeout} s"
    if "error" in box:
        raise box["error"]
    return box.get("value")


class _ThreadQueue:
    def __init__(self) -> None:
        self.items: deque = deque()
        self.waiting = False  # True while the owner thread is blocked in get_message on an empty queue


class FakeWinApi:
    """In-memory implementation of contracts.WinApi.

    * Thread message queues keyed by the native thread id (get_current_thread_id returns
      threading.get_native_id(), so an implementation using either works). get_message
      blocks on the caller's queue; WM_QUIT (posted either way) or shutdown() -> None.
    * Hotkeys are per-thread registrations. A combo already registered (by anyone) or listed
      in ``conflicts`` fails with GetLastError = 1409. Every call records threading.get_ident().
    * Keyboard: a down-set with optional scheduled release at a fake time; send_input is
      recorded (with the fake time and a copy of the clipboard) and its return value is
      configurable (None = all events, int, callable(events) -> int, or an exception to raise).
    * Foreground hwnd -> pid -> integrity tables.
    * Clipboard: ordered list of (format, bytes | None) (None = not HGLOBAL-backed), open
      ownership tracking, configurable open failures; misuse is recorded in clipboard_violations.
    * Clock: monotonic() returns ``now``; sleep() advances ``now`` (and runs scheduled actions)
      without real waiting.
    * Mutexes, registered window messages and broadcasts are recorded.
    * ``log`` is an ordered list of (kind, payload, fake_time) tuples shared with test doubles.
    """

    def __init__(self, *, start_time: float = 1000.0, own_integrity: int = IL_MEDIUM) -> None:
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._closed = False
        self._queues: dict[int, _ThreadQueue] = {}
        self.get_message_idents: set[int] = set()
        self.posted: list[tuple[int, int, int, int]] = []

        # hotkeys
        self.registrations: list[dict] = []
        self.conflicts: set[tuple[int, int]] = set()  # (modifier bits without NOREPEAT, vk)
        self.register_calls: list[dict] = []
        self.unregister_calls: list[dict] = []
        self._last_error: dict[int, int] = {}

        # keyboard
        self._down: set[int] = set()
        self.key_queries: list[int] = []
        self.send_input_calls: list[dict] = []
        self.send_input_result = None
        self.send_input_cost_s = 0.0

        # windows / processes
        self.foreground_hwnd = 0x1000
        self.window_pids: dict[int, int] = {0x1000: 4242}
        self.integrity: dict[int, Optional[int]] = {4242: IL_MEDIUM}
        self.own_integrity = own_integrity

        # clipboard
        self.clipboard: list[list] = []  # [fmt, bytes | None]
        self.clipboard_owner: Optional[int] = None
        self.open_fail_remaining = 0
        self.open_always_fail = False
        self.open_attempts = 0
        self.clipboard_violations: list[str] = []

        # mutex / messages
        self._next_handle = 0x100
        self.mutexes: dict[str, int] = {}
        self.handles: dict[int, str] = {}
        self.closed_handles: list[int] = []
        # RegisterWindowMessage and RegisterClipboardFormat share one atom table in Windows.
        self._registered_names: dict[str, int] = {}
        self.broadcasts: list[int] = []
        self.allow_set_foreground_calls: list[int] = []

        # time
        self.now = float(start_time)
        self.sleeps: list[float] = []
        self._scheduled: list[tuple[float, Callable[[], None]]] = []
        self.max_sleep_calls = 200_000

        self.log: list[tuple] = []

    # ----------------------------------------------------------------- helpers
    def record(self, kind: str, payload=None) -> None:
        with self._lock:
            self.log.append((kind, payload, self.now))

    def kinds(self) -> list[str]:
        with self._lock:
            return [entry[0] for entry in self.log]

    def _queue(self, tid: int) -> _ThreadQueue:
        q = self._queues.get(tid)
        if q is None:
            q = self._queues[tid] = _ThreadQueue()
        return q

    def _run_due(self) -> None:
        due = [a for a in self._scheduled if a[0] <= self.now]
        self._scheduled = [a for a in self._scheduled if a[0] > self.now]
        for _, action in sorted(due, key=lambda a: a[0]):
            action()

    def at(self, t: float, action: Callable[[], None]) -> None:
        """Run action when the fake clock reaches t (checked on sleep/advance)."""
        with self._lock:
            self._scheduled.append((t, action))
            self._run_due()

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds
            self._run_due()

    def press(self, vk: int, *, until: Optional[float] = None) -> None:
        """Hold vk down; release it when the fake clock reaches ``until`` (None = forever)."""
        with self._lock:
            self._down.add(vk)
        if until is not None:
            self.at(until, lambda: self.release(vk))

    def release(self, vk: int) -> None:
        with self._lock:
            self._down.discard(vk)

    def sent_events(self) -> list[KeyEvent]:
        with self._lock:
            return [ev for call in self.send_input_calls for ev in call["events"]]

    def active_combos(self) -> set[tuple[int, int]]:
        with self._lock:
            return {(r["modifiers"] & _MOD_MASK, r["vk"]) for r in self.registrations}

    def thread_ids_with_registrations(self) -> set[int]:
        with self._lock:
            return {r["ident"] for r in self.registrations}

    @property
    def loop_ident(self) -> int:
        """threading ident of the (single) thread that runs the GetMessage loop."""
        with self._lock:
            assert len(self.get_message_idents) == 1, f"expected one message-loop thread, got {self.get_message_idents}"
            return next(iter(self.get_message_idents))

    def fire_hotkey(self, modifiers: Optional[int] = None, vk: Optional[int] = None) -> int:
        """Post WM_HOTKEY to the thread that registered the matching hotkey (as Windows does).
        With no args, the single active registration is used. Returns the target thread id."""
        with self._cond:
            regs = self.registrations
            if modifiers is not None:
                regs = [r for r in regs if (r["modifiers"] & _MOD_MASK) == (modifiers & _MOD_MASK) and r["vk"] == vk]
            assert len(regs) >= 1, f"no registered hotkey matches mods={modifiers} vk={vk}: {self.registrations}"
            if modifiers is None:
                assert len(regs) == 1, f"expected exactly one registration, got {regs}"
            reg = regs[0]
            lparam = ((reg["vk"] & 0xFFFF) << 16) | (reg["modifiers"] & _MOD_MASK)
            self._queue(reg["tid"]).items.append(WinMessage(WM_HOTKEY, reg["id"], lparam))
            self._cond.notify_all()
            return reg["tid"]

    def wait_idle(self, tid: int, timeout: float = 2.0) -> bool:
        """Wait until thread tid has drained its queue and is blocked in get_message again."""
        with self._cond:
            q = self._queue(tid)
            return self._cond.wait_for(lambda: q.waiting and not q.items, timeout)

    def shutdown(self) -> None:
        """Make every blocked/future get_message return None (test teardown safety net)."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    # --------------------------------------------------- threads / message loop
    def get_current_thread_id(self) -> int:
        tid = threading.get_native_id()
        with self._lock:
            self._queue(tid)
        return tid

    def get_message(self) -> Optional[WinMessage]:
        tid = threading.get_native_id()
        with self._cond:
            self.get_message_idents.add(threading.get_ident())
            q = self._queue(tid)
            while not q.items and not self._closed:
                q.waiting = True
                self._cond.notify_all()
                self._cond.wait()
            q.waiting = False
            if self._closed:
                self._cond.notify_all()
                return None
            item = q.items.popleft()
            self._cond.notify_all()
        if item is _QUIT or (isinstance(item, WinMessage) and item.message == WM_QUIT):
            return None
        return item

    def post_thread_message(self, thread_id: int, message: int, wparam: int = 0, lparam: int = 0) -> bool:
        with self._cond:
            self.posted.append((thread_id, message, wparam, lparam))
            self._queue(thread_id).items.append(WinMessage(message, wparam, lparam))
            self._cond.notify_all()
        return True

    def post_quit_message(self) -> None:
        with self._cond:
            self._queue(threading.get_native_id()).items.append(_QUIT)
            self._cond.notify_all()

    # ------------------------------------------------------------------ hotkeys
    def register_hotkey(self, hotkey_id: int, modifiers: int, vk: int) -> bool:
        tid = threading.get_native_id()
        with self._lock:
            combo = (modifiers & _MOD_MASK, vk)
            ok = combo not in self.conflicts and combo not in self.active_combos()
            self.register_calls.append(
                {"ident": threading.get_ident(), "id": hotkey_id, "modifiers": modifiers, "vk": vk, "ok": ok}
            )
            self.log.append(("register_hotkey", (hotkey_id, modifiers, vk, ok), self.now))
            if not ok:
                self._last_error[tid] = ERROR_HOTKEY_ALREADY_REGISTERED
                return False
            self.registrations.append(
                {"tid": tid, "ident": threading.get_ident(), "id": hotkey_id, "modifiers": modifiers, "vk": vk}
            )
            return True

    def unregister_hotkey(self, hotkey_id: int) -> bool:
        tid = threading.get_native_id()
        with self._lock:
            before = len(self.registrations)
            self.registrations = [r for r in self.registrations if not (r["tid"] == tid and r["id"] == hotkey_id)]
            ok = len(self.registrations) < before
            self.unregister_calls.append({"ident": threading.get_ident(), "id": hotkey_id, "ok": ok})
            self.log.append(("unregister_hotkey", (hotkey_id, ok), self.now))
            if not ok:
                self._last_error[tid] = ERROR_HOTKEY_NOT_REGISTERED
            return ok

    def get_last_error(self) -> int:
        with self._lock:
            return self._last_error.get(threading.get_native_id(), 0)

    # ----------------------------------------------------------------- keyboard
    def get_async_key_state(self, vk: int) -> int:
        with self._lock:
            self.key_queries.append(vk)
            down = vk in self._down or any(v in self._down for v in _GENERIC_VKS.get(vk, ()))
            return -0x8000 if down else 0  # SHORT with the high bit set

    def send_input(self, events: Sequence[KeyEvent]) -> int:
        events = list(events)
        with self._lock:
            call = {
                "events": events,
                "time": self.now,
                "ident": threading.get_ident(),
                "clipboard": [(f, d) for f, d in self.clipboard],
            }
            self.send_input_calls.append(call)
            self.log.append(("send_input", events, self.now))
            result = self.send_input_result
            self.now += self.send_input_cost_s
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            return int(result(events))
        if result is None:
            return len(events)
        return min(int(result), len(events))

    # ------------------------------------------------------- windows / processes
    def get_foreground_window(self) -> int:
        with self._lock:
            return self.foreground_hwnd

    def get_window_process_id(self, hwnd: int) -> int:
        with self._lock:
            return self.window_pids.get(hwnd, 0)

    def get_process_integrity_level(self, pid: int) -> Optional[int]:
        with self._lock:
            self.log.append(("integrity_query", pid, self.now))
            return self.integrity.get(pid)

    def get_current_process_integrity_level(self) -> int:
        return self.own_integrity

    # ---------------------------------------------------------------- clipboard
    def _require_open(self, what: str) -> bool:
        if self.clipboard_owner is None:
            self.clipboard_violations.append(f"{what} while clipboard closed")
            return False
        return True

    def open_clipboard(self) -> bool:
        with self._lock:
            self.open_attempts += 1
            if self.open_always_fail:
                return False
            if self.open_fail_remaining > 0:
                self.open_fail_remaining -= 1
                return False
            if self.clipboard_owner is not None:
                self.clipboard_violations.append("open_clipboard while already open")
            self.clipboard_owner = threading.get_ident()
            self.log.append(("open_clipboard", None, self.now))
            return True

    def close_clipboard(self) -> None:
        with self._lock:
            self._require_open("close_clipboard")
            self.clipboard_owner = None
            self.log.append(("close_clipboard", None, self.now))

    def empty_clipboard(self) -> bool:
        with self._lock:
            if not self._require_open("empty_clipboard"):
                return False
            self.clipboard = []
            self.log.append(("empty_clipboard", None, self.now))
            return True

    def enum_clipboard_formats(self) -> list[int]:
        with self._lock:
            if not self._require_open("enum_clipboard_formats"):
                return []
            return [f for f, _ in self.clipboard]

    def get_clipboard_bytes(self, fmt: int) -> Optional[bytes]:
        with self._lock:
            if not self._require_open("get_clipboard_bytes"):
                return None
            for f, data in self.clipboard:
                if f == fmt:
                    return data
            return None

    def set_clipboard_bytes(self, fmt: int, data: bytes) -> bool:
        with self._lock:
            if not self._require_open("set_clipboard_bytes"):
                return False
            data = bytes(data)
            for entry in self.clipboard:
                if entry[0] == fmt:
                    entry[1] = data
                    break
            else:
                self.clipboard.append([fmt, data])
            self.log.append(("set_clipboard_bytes", (fmt, data), self.now))
            return True

    def clipboard_items(self) -> list[tuple[int, Optional[bytes]]]:
        with self._lock:
            return [(f, d) for f, d in self.clipboard]

    # --------------------------------------------------- single instance / misc
    def create_mutex(self, name: str) -> tuple[int, bool]:
        with self._lock:
            existed = name in self.mutexes
            self._next_handle += 4
            handle = self._next_handle
            self.handles[handle] = name
            if not existed:
                self.mutexes[name] = handle
            return handle, existed

    def close_handle(self, handle: int) -> None:
        with self._lock:
            self.closed_handles.append(handle)
            name = self.handles.pop(handle, None)
            if name is not None and name not in self.handles.values():
                self.mutexes.pop(name, None)

    def _register_name(self, name: str) -> int:
        with self._lock:
            for known, value in self._registered_names.items():
                if known.casefold() == name.casefold():  # atoms are case-insensitive
                    return value
            value = self._registered_names[name] = 0xC000 + len(self._registered_names)
            return value

    def register_window_message(self, name: str) -> int:
        return self._register_name(name)

    def register_clipboard_format(self, name: str) -> int:
        return self._register_name(name)

    def allow_set_foreground_window(self, pid: int) -> bool:
        with self._lock:
            self.allow_set_foreground_calls.append(pid)
            return True

    def broadcast_message(self, message: int) -> None:
        with self._lock:
            self.broadcasts.append(message)

    # --------------------------------------------------------------------- time
    def monotonic(self) -> float:
        with self._lock:
            return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            assert len(self.sleeps) < self.max_sleep_calls, "runaway sleep loop (fake clock not used for deadline?)"
            self.now += max(0.0, seconds)
            self.log.append(("sleep", seconds, self.now))
            self._run_due()
        time.sleep(0)  # yield the GIL, never really wait


# ===========================================================================
# Hotkey parsing
# ===========================================================================


def _parse(text: str) -> Hotkey:
    from app.win32.hotkey_parse import parse_hotkey

    return parse_hotkey(text)


def _format(h: Hotkey) -> str:
    from app.win32.hotkey_parse import format_hotkey

    return format_hotkey(h)


class TestParseBasics:
    @pytest.mark.parametrize("text", ["alt+q", "Alt+Q", " ALT + q ", "aLt+Q", "Alt + Q", "\tAlt+q\n"])
    def test_alt_q_variants_normalize(self, text):
        h = _parse(text)
        assert h.display == "Alt+Q"
        assert h.key == "Q"
        assert h.vk == 0x51
        assert h.modifiers == frozenset({Modifier.ALT})
        assert h == _parse("Alt+Q")

    def test_default_hotkey_parses(self):
        assert _parse(DEFAULT_HOTKEY).display == "Alt+Q"

    def test_returns_contract_hotkey(self):
        assert isinstance(_parse("ctrl+alt+f9"), Hotkey)

    @pytest.mark.parametrize(
        "text, display",
        [
            ("shift+ctrl+alt+f9", "Ctrl+Alt+Shift+F9"),
            ("win+shift+alt+ctrl+a", "Ctrl+Alt+Shift+Win+A"),
            ("alt+ctrl+z", "Ctrl+Alt+Z"),
            ("win+alt+d", "Alt+Win+D"),
            ("shift+win+q", "Shift+Win+Q"),
        ],
    )
    def test_canonical_modifier_order(self, text, display):
        assert _parse(text).display == display

    @pytest.mark.parametrize(
        "text, display, mods",
        [
            ("control+a", "Ctrl+A", {Modifier.CTRL}),
            ("ctl+a", "Ctrl+A", {Modifier.CTRL}),
            ("CONTROL+a", "Ctrl+A", {Modifier.CTRL}),
            ("windows+e", "Win+E", {Modifier.WIN}),
            ("meta+e", "Win+E", {Modifier.WIN}),
            ("super+e", "Win+E", {Modifier.WIN}),
            ("Super+Shift+E", "Shift+Win+E", {Modifier.WIN, Modifier.SHIFT}),
        ],
    )
    def test_modifier_aliases(self, text, display, mods):
        h = _parse(text)
        assert h.display == display
        assert h.modifiers == frozenset(mods)

    @pytest.mark.parametrize(
        "text, display, vk",
        [
            ("ctrl+return", "Ctrl+Enter", 0x0D),
            ("ctrl+enter", "Ctrl+Enter", 0x0D),
            ("alt+escape", "Alt+Esc", 0x1B),
            ("alt+esc", "Alt+Esc", 0x1B),
            ("ctrl+ins", "Ctrl+Insert", 0x2D),
            ("ctrl+insert", "Ctrl+Insert", 0x2D),
            ("ctrl+del", "Ctrl+Delete", 0x2E),
            ("ctrl+delete", "Ctrl+Delete", 0x2E),
            ("ctrl+pgup", "Ctrl+PageUp", 0x21),
            ("ctrl+pageup", "Ctrl+PageUp", 0x21),
            ("ctrl+pgdn", "Ctrl+PageDown", 0x22),
            ("ctrl+pagedown", "Ctrl+PageDown", 0x22),
        ],
    )
    def test_key_aliases(self, text, display, vk):
        h = _parse(text)
        assert h.display == display
        assert h.vk == vk


class TestParseKeys:
    @pytest.mark.parametrize("n", range(1, 25))
    def test_function_keys(self, n):
        h = _parse(f"ctrl+f{n}")
        assert h.key == f"F{n}"
        assert h.vk == 0x70 + n - 1
        assert h.display == f"Ctrl+F{n}"

    @pytest.mark.parametrize("letter", list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    def test_letters(self, letter):
        h = _parse(f"alt+{letter.lower()}")
        assert h.key == letter
        assert h.vk == ord(letter)

    @pytest.mark.parametrize("digit", range(10))
    def test_digits(self, digit):
        h = _parse(f"ctrl+{digit}")
        assert h.key == str(digit)
        assert h.vk == 0x30 + digit
        assert h.display == f"Ctrl+{digit}"

    @pytest.mark.parametrize("n", range(10))
    def test_numpad(self, n):
        h = _parse(f"ctrl+numpad{n}")
        assert h.key == f"Numpad{n}"
        assert h.vk == 0x60 + n
        assert h.display == f"Ctrl+Numpad{n}"

    @pytest.mark.parametrize(
        "name, canonical, vk",
        [
            ("space", "Space", 0x20),
            ("tab", "Tab", 0x09),
            ("backspace", "Backspace", 0x08),
            ("home", "Home", 0x24),
            ("end", "End", 0x23),
            ("up", "Up", 0x26),
            ("down", "Down", 0x28),
            ("left", "Left", 0x25),
            ("right", "Right", 0x27),
            ("pause", "Pause", 0x13),
            ("ENTER", "Enter", 0x0D),
            ("Esc", "Esc", 0x1B),
        ],
    )
    def test_named_keys(self, name, canonical, vk):
        h = _parse(f"ctrl+{name}")
        assert h.key == canonical
        assert h.vk == vk
        assert h.display == f"Ctrl+{canonical}"

    @pytest.mark.parametrize(
        "char, vk",
        [
            (";", 0xBA),
            ("=", 0xBB),
            (",", 0xBC),
            ("-", 0xBD),
            (".", 0xBE),
            ("/", 0xBF),
            ("`", 0xC0),
            ("[", 0xDB),
            ("\\", 0xDC),
            ("]", 0xDD),
            ("'", 0xDE),
        ],
    )
    def test_oem_punctuation(self, char, vk):
        h = _parse(f"ctrl+{char}")
        assert h.key == char
        assert h.vk == vk
        assert h.display == f"Ctrl+{char}"
        h2 = _parse(f" Ctrl + {char} ")
        assert h2 == h


class TestWin32Modifiers:
    def test_alt_q_bits(self):
        assert _parse("alt+q").win32_modifiers == MOD_ALT | MOD_NOREPEAT

    def test_all_modifiers_bits(self):
        assert _parse("ctrl+alt+shift+win+f1").win32_modifiers == (
            MOD_CONTROL | MOD_ALT | MOD_SHIFT | MOD_WIN | MOD_NOREPEAT
        )

    @pytest.mark.parametrize("text", ["ctrl+a", "win+e", "shift+f9", "ctrl+shift+1"])
    def test_norepeat_always_set(self, text):
        assert _parse(text).win32_modifiers & MOD_NOREPEAT


ROUND_TRIP = [
    "Alt+Q", "Ctrl+Alt+F9", "Ctrl+Alt+Shift+Win+F24", "Ctrl+Enter", "Alt+Esc", "Ctrl+Insert",
    "Ctrl+Delete", "Ctrl+PageUp", "Ctrl+PageDown", "Ctrl+Numpad7", "Ctrl+Space", "Win+Up",
    "Ctrl+;", "Ctrl+=", "Ctrl+,", "Ctrl+-", "Ctrl+.", "Ctrl+/", "Ctrl+`", "Ctrl+[", "Ctrl+\\",
    "Ctrl+]", "Ctrl+'", "Shift+F9", "Ctrl+0", "Alt+Backspace", "Ctrl+Pause", "Alt+Tab",
]


class TestFormat:
    @pytest.mark.parametrize("display", ROUND_TRIP)
    def test_format_parse_round_trip(self, display):
        h = _parse(display)
        assert _format(h) == display
        assert h.display == display
        assert _parse(_format(h)) == h

    @pytest.mark.parametrize("text", ["shift+ctrl+alt+f9", "control+return", "meta+pgdn", " alt + q "])
    def test_format_is_display(self, text):
        h = _parse(text)
        assert _format(h) == h.display

    def test_parser_class_matches_functions(self):
        from app.win32.hotkey_parse import HotkeyStringParser

        p = HotkeyStringParser()
        for text in ["alt+q", "shift+ctrl+alt+f9", "ctrl+;"]:
            assert p.parse(text) == _parse(text)
            assert p.format(p.parse(text)) == _parse(text).display
        with pytest.raises(HotkeyParseError):
            p.parse("Q")


REJECTED = [
    "",
    "   ",
    "Q",
    "F9",
    "Space",
    "Ctrl",
    "Ctrl+Alt",
    "Ctrl+Alt+Shift+Win",
    "Ctrl+A+B",
    "Ctrl+F9+F10",
    "Alt+Alt+Q",
    "Ctrl+Control+Q",  # duplicate through an alias
    "Win+Meta+E",
    "Ctrl+Foo",
    "Foo+Q",
    "Ctrl+F25",
    "Ctrl+F0",
    "Alt+QQ",
    "Ctrl++",  # documented as unsupported
    "Alt+",
    "+Q",
    "Alt++Q",
    "Ctrl-Q",
    "Shift+Q",
    "Shift+a",
    "Shift+1",
    "Shift+Space",
    "Shift+;",
    "Shift+'",
    "Shift+\\",
]


class TestParseRejections:
    @pytest.mark.parametrize("text", REJECTED)
    def test_rejected(self, text):
        with pytest.raises(HotkeyParseError) as ei:
            _parse(text)
        assert isinstance(ei.value, AppError)
        assert ei.value.user_message

    @pytest.mark.parametrize("text, display", [("shift+f9", "Shift+F9"), ("Shift+Up", "Shift+Up"), ("shift+esc", "Shift+Esc")])
    def test_shift_only_allowed_on_non_printable(self, text, display):
        # Contract: Shift-only is rejected only for printable keys (letters, digits, OEM, Space).
        h = _parse(text)
        assert h.display == display
        assert h.modifiers == frozenset({Modifier.SHIFT})

    @pytest.mark.parametrize("text", ["ctrl+shift+q", "shift+win+1", "alt+shift+space", "ctrl+shift+;"])
    def test_shift_with_another_modifier_on_printable_ok(self, text):
        h = _parse(text)
        assert Modifier.SHIFT in h.modifiers
        assert len(h.modifiers) == 2


# ===========================================================================
# Win32HotkeyManager
# ===========================================================================

ALT_Q = Hotkey(frozenset({Modifier.ALT}), "Q", 0x51)
CTRL_ALT_F9 = Hotkey(frozenset({Modifier.CTRL, Modifier.ALT}), "F9", 0x78)
CTRL_F9 = Hotkey(frozenset({Modifier.CTRL}), "F9", 0x78)
CTRL_SHIFT_D = Hotkey(frozenset({Modifier.CTRL, Modifier.SHIFT}), "D", 0x44)


class FireRecorder:
    """on_fire callback. Records the calling thread and the fake time; logs into api.log."""

    def __init__(self, api: FakeWinApi, raise_on: Sequence[int] = ()) -> None:
        self.api = api
        self.calls: list[tuple[int, float]] = []
        self.raise_on = set(raise_on)
        self._lock = threading.Lock()

    def __call__(self) -> None:
        with self._lock:
            idx = len(self.calls)
            self.calls.append((threading.get_ident(), self.api.monotonic()))
        self.api.record("on_fire")
        if idx in self.raise_on:
            raise RuntimeError("on_fire boom (test)")

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.calls)


@pytest.fixture
def api():
    fake = FakeWinApi()
    yield fake
    fake.shutdown()


@pytest.fixture
def make_manager(api):
    created = []

    def make(debounce_s: float = 0.3, on_fire: Optional[Callable[[], None]] = None, start: bool = True):
        from app.win32.hotkey import Win32HotkeyManager

        rec = on_fire if on_fire is not None else FireRecorder(api)
        mgr = Win32HotkeyManager(api, rec, debounce_s=debounce_s)
        created.append(mgr)
        if start:
            run_with_timeout(mgr.start)
        return mgr, rec

    yield make
    for mgr in created:
        try:
            run_with_timeout(mgr.stop)
        except Exception:  # noqa: BLE001 - teardown best effort; the test itself asserts stop()
            pass


def fire(api: FakeWinApi, modifiers: Optional[int] = None, vk: Optional[int] = None) -> None:
    tid = api.fire_hotkey(modifiers, vk)
    assert api.wait_idle(tid), "hotkey thread did not return to GetMessage within 2 s"


def _loop_thread_alive(ident: int) -> bool:
    return any(t.ident == ident and t.is_alive() for t in threading.enumerate())


class TestManagerRegistration:
    def test_register_calls_registerhotkey_on_manager_thread(self, api, make_manager):
        mgr, _ = make_manager()
        run_with_timeout(mgr.register, ALT_Q)
        assert api.active_combos() == {(MOD_ALT, 0x51)}
        ok_calls = [c for c in api.register_calls if c["ok"]]
        assert len(ok_calls) == 1
        assert ok_calls[0]["modifiers"] == ALT_Q.win32_modifiers  # includes MOD_NOREPEAT
        assert ok_calls[0]["vk"] == 0x51
        loop = api.loop_ident
        assert loop != threading.get_ident()
        assert {c["ident"] for c in api.register_calls} == {loop}
        assert mgr.current == ALT_Q

    def test_current_none_initially(self, api, make_manager):
        mgr, _ = make_manager()
        assert mgr.current is None

    def test_register_before_start_raises_runtime_error(self, api, make_manager):
        mgr, _ = make_manager(start=False)
        with pytest.raises(RuntimeError):
            run_with_timeout(mgr.register, ALT_Q)
        assert api.registrations == []
        assert mgr.current is None

    def test_start_is_idempotent(self, api, make_manager):
        mgr, rec = make_manager()
        run_with_timeout(mgr.start)
        run_with_timeout(mgr.register, ALT_Q)
        fire(api)
        assert rec.count == 1
        assert len(api.get_message_idents) == 1

    def test_register_replaces_previous(self, api, make_manager):
        mgr, rec = make_manager()
        run_with_timeout(mgr.register, ALT_Q)
        run_with_timeout(mgr.register, CTRL_ALT_F9)
        assert api.active_combos() == {(MOD_CONTROL | MOD_ALT, 0x78)}
        assert mgr.current == CTRL_ALT_F9
        assert {c["ident"] for c in api.register_calls + api.unregister_calls} == {api.loop_ident}
        fire(api)
        assert rec.count == 1

    def test_register_same_hotkey_again_succeeds(self, api, make_manager):
        # Windows refuses a combo that is already registered, even by the same thread, so
        # replacing Alt+Q with Alt+Q must unregister first (or be a no-op).
        mgr, _ = make_manager()
        run_with_timeout(mgr.register, ALT_Q)
        run_with_timeout(mgr.register, ALT_Q)
        assert api.active_combos() == {(MOD_ALT, 0x51)}
        assert len(api.registrations) == 1
        assert mgr.current == ALT_Q

    def test_conflict_raises_and_keeps_previous(self, api, make_manager):
        api.conflicts.add((MOD_CONTROL | MOD_ALT, 0x78))
        mgr, rec = make_manager()
        run_with_timeout(mgr.register, ALT_Q)
        with pytest.raises(HotkeyConflictError) as ei:
            run_with_timeout(mgr.register, CTRL_ALT_F9)
        err = ei.value
        assert err.win32_error == ERROR_HOTKEY_ALREADY_REGISTERED
        assert err.hotkey == "Ctrl+Alt+F9"
        assert isinstance(err, AppError)
        # previous hotkey still registered and working
        assert mgr.current == ALT_Q
        assert api.active_combos() == {(MOD_ALT, 0x51)}
        assert {c["ident"] for c in api.register_calls + api.unregister_calls} == {api.loop_ident}
        fire(api)
        assert rec.count == 1

    def test_conflict_without_previous(self, api, make_manager):
        api.conflicts.add((MOD_ALT, 0x51))
        mgr, _ = make_manager()
        with pytest.raises(HotkeyConflictError) as ei:
            run_with_timeout(mgr.register, ALT_Q)
        assert ei.value.win32_error == ERROR_HOTKEY_ALREADY_REGISTERED
        assert mgr.current is None
        assert api.registrations == []

    def test_after_conflict_a_new_register_works(self, api, make_manager):
        api.conflicts.add((MOD_CONTROL | MOD_ALT, 0x78))
        mgr, _ = make_manager()
        run_with_timeout(mgr.register, ALT_Q)
        with pytest.raises(HotkeyConflictError):
            run_with_timeout(mgr.register, CTRL_ALT_F9)
        run_with_timeout(mgr.register, CTRL_SHIFT_D)
        assert api.active_combos() == {(MOD_CONTROL | MOD_SHIFT, 0x44)}
        assert mgr.current == CTRL_SHIFT_D

    def test_unregister_idempotent(self, api, make_manager):
        mgr, _ = make_manager()
        run_with_timeout(mgr.unregister)  # nothing registered: no error
        run_with_timeout(mgr.register, ALT_Q)
        run_with_timeout(mgr.unregister)
        assert api.registrations == []
        assert mgr.current is None
        run_with_timeout(mgr.unregister)
        assert api.registrations == []
        assert {c["ident"] for c in api.unregister_calls} <= {api.loop_ident}

    def test_unregistered_hotkey_cannot_fire(self, api, make_manager):
        mgr, rec = make_manager()
        run_with_timeout(mgr.register, ALT_Q)
        run_with_timeout(mgr.unregister)
        with pytest.raises(AssertionError):
            api.fire_hotkey(MOD_ALT, 0x51)
        assert rec.count == 0

    def test_current_readable_from_other_thread(self, api, make_manager):
        mgr, _ = make_manager()
        run_with_timeout(mgr.register, CTRL_SHIFT_D)
        assert run_with_timeout(lambda: mgr.current) == CTRL_SHIFT_D


class TestManagerStop:
    def test_stop_unregisters_and_joins(self, api, make_manager):
        mgr, _ = make_manager()
        run_with_timeout(mgr.register, ALT_Q)
        loop = api.loop_ident
        assert _loop_thread_alive(loop)
        run_with_timeout(mgr.stop)
        assert api.registrations == []
        assert [c["ident"] for c in api.unregister_calls if c["ok"]] == [loop]
        assert not _loop_thread_alive(loop)
        assert mgr.current is None

    def test_stop_twice_is_safe(self, api, make_manager):
        mgr, _ = make_manager()
        run_with_timeout(mgr.register, ALT_Q)
        run_with_timeout(mgr.stop)
        run_with_timeout(mgr.stop)
        assert api.registrations == []

    def test_stop_without_register(self, api, make_manager):
        mgr, _ = make_manager()
        loop_ident_holder = []

        # make sure the loop is running before stopping
        def wait_loop():
            deadline = time.monotonic() + 2.0
            while not api.get_message_idents and time.monotonic() < deadline:
                time.sleep(0.005)
            loop_ident_holder.extend(api.get_message_idents)

        wait_loop()
        run_with_timeout(mgr.stop)
        for ident in loop_ident_holder:
            assert not _loop_thread_alive(ident)

    def test_stop_before_start_is_safe(self, api, make_manager):
        mgr, _ = make_manager(start=False)
        run_with_timeout(mgr.stop)
        assert api.registrations == []


class TestManagerFire:
    def test_wm_hotkey_calls_on_fire_once_on_manager_thread(self, api, make_manager):
        mgr, rec = make_manager()
        run_with_timeout(mgr.register, CTRL_SHIFT_D)
        fire(api)
        assert rec.count == 1
        assert rec.calls[0][0] == api.loop_ident

    def test_non_hotkey_message_does_not_fire(self, api, make_manager):
        mgr, rec = make_manager()
        run_with_timeout(mgr.register, CTRL_SHIFT_D)
        loop_tid = api.registrations[0]["tid"]
        api.post_thread_message(loop_tid, WM_TIMER, 0, 0)
        assert api.wait_idle(loop_tid)
        assert rec.count == 0
        fire(api)
        assert rec.count == 1

    def test_debounce_drops_second_fire_within_window(self, api, make_manager):
        mgr, rec = make_manager(debounce_s=0.3)
        run_with_timeout(mgr.register, CTRL_SHIFT_D)
        fire(api)
        api.advance(0.1)
        fire(api)
        assert rec.count == 1
        api.advance(0.4)  # 0.5 s after the accepted fire
        fire(api)
        assert rec.count == 2

    def test_debounce_window_measured_from_previous_accepted_fire(self, api, make_manager):
        # Contract: "Fires within debounce_s of the previous ACCEPTED fire are dropped."
        mgr, rec = make_manager(debounce_s=0.3)
        run_with_timeout(mgr.register, CTRL_SHIFT_D)
        fire(api)  # t=0 accepted
        api.advance(0.2)
        fire(api)  # t=0.2 dropped
        api.advance(0.2)
        fire(api)  # t=0.4: 0.4 s after the accepted one -> accepted
        assert rec.count == 2

    def test_auto_repeat_flood_dropped(self, api, make_manager):
        mgr, rec = make_manager(debounce_s=0.3)
        run_with_timeout(mgr.register, CTRL_SHIFT_D)
        for _ in range(25):
            fire(api)
            api.advance(0.01)  # 25 repeats over 0.25 s, all within the window
        assert rec.count == 1

    def test_on_fire_exception_does_not_kill_loop(self, api, make_manager):
        rec = FireRecorder(api, raise_on=[0])
        mgr, _ = make_manager(on_fire=rec)
        run_with_timeout(mgr.register, CTRL_SHIFT_D)
        fire(api)
        assert rec.count == 1
        api.advance(1.0)
        fire(api)
        assert rec.count == 2
        # the loop still services register/unregister requests
        run_with_timeout(mgr.register, CTRL_ALT_F9)
        assert api.active_combos() == {(MOD_CONTROL | MOD_ALT, 0x78)}
        run_with_timeout(mgr.stop)
        assert api.registrations == []


def _mask_events(api: FakeWinApi) -> list[KeyEvent]:
    return [ev for ev in api.sent_events() if ev.vk == VK_MENU_MASK and ev.unicode_char is None]


class TestAltSuppression:
    def test_alt_hotkey_sends_mask_key_before_on_fire(self, api, make_manager):
        mgr, rec = make_manager()
        run_with_timeout(mgr.register, ALT_Q)
        api.press(0xA4)  # the user is holding Left Alt when WM_HOTKEY arrives
        api.press(0x51)
        fire(api)
        assert rec.count == 1
        assert _mask_events(api) == [KeyEvent(vk=VK_MENU_MASK, key_up=False), KeyEvent(vk=VK_MENU_MASK, key_up=True)]
        kinds = api.kinds()
        first_send = next(i for i, e in enumerate(api.log) if e[0] == "send_input" and any(ev.vk == VK_MENU_MASK for ev in e[1]))
        assert first_send < kinds.index("on_fire")
        # no other synthetic keys besides the mask key
        assert all(ev.vk == VK_MENU_MASK for ev in api.sent_events())

    def test_alt_hotkey_sends_mask_even_if_alt_state_reads_up(self, api, make_manager):
        # Contract: "if Alt is part of the hotkey OR physically down".
        mgr, rec = make_manager()
        run_with_timeout(mgr.register, ALT_Q)
        fire(api)
        assert rec.count == 1
        assert _mask_events(api) == [KeyEvent(vk=VK_MENU_MASK, key_up=False), KeyEvent(vk=VK_MENU_MASK, key_up=True)]

    def test_non_alt_hotkey_without_alt_down_sends_nothing(self, api, make_manager):
        mgr, rec = make_manager()
        run_with_timeout(mgr.register, CTRL_SHIFT_D)
        api.press(0xA2)
        api.press(0xA0)
        fire(api)
        assert rec.count == 1
        assert _mask_events(api) == []

    @pytest.mark.parametrize("alt_vk", [0xA4, 0xA5])
    def test_non_alt_hotkey_with_alt_physically_down_sends_mask(self, api, make_manager, alt_vk):
        mgr, rec = make_manager()
        run_with_timeout(mgr.register, CTRL_F9)
        api.press(0xA2)
        api.press(alt_vk)
        fire(api)
        assert rec.count == 1
        assert _mask_events(api) == [KeyEvent(vk=VK_MENU_MASK, key_up=False), KeyEvent(vk=VK_MENU_MASK, key_up=True)]
        kinds = api.kinds()
        assert kinds.index("send_input") < kinds.index("on_fire")

    def test_mask_sent_once_per_accepted_fire(self, api, make_manager):
        mgr, rec = make_manager(debounce_s=0.3)
        run_with_timeout(mgr.register, ALT_Q)
        api.press(0xA4)
        fire(api)
        api.advance(1.0)
        fire(api)
        assert rec.count == 2
        downs = [ev for ev in _mask_events(api) if not ev.key_up]
        ups = [ev for ev in _mask_events(api) if ev.key_up]
        assert len(downs) == 2 and len(ups) == 2
