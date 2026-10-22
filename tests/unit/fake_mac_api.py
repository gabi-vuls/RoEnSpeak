"""In-memory fake of the macOS system calls (TEST-UNIT-MACOS).

The macOS counterpart of ``FakeWinApi`` (tests/unit/test_hotkey.py): it implements the
surface of ``app.macos._mac_api.RealMacApi`` with pure Python state, so every macOS unit
test runs on any platform, with no Accessibility grant, no pasteboard, no Carbon and no
real clock.

The OS constants below (CGEventFlags, Carbon modifier masks / OSStatus values, kVK codes)
are taken from Apple's headers, not from the implementation under test, so the tests
compare the implementation against the platform rather than against itself.

What the fake models
--------------------
clock           ``monotonic()`` returns ``now``; ``sleep()`` advances it and runs actions
                scheduled with ``at()`` -- no test ever waits in real time.
modifiers       ``modifier_flags()`` returns ``flags``; ``hold(bit, until=...)`` schedules
                a release on the fake clock.
keyboard out    ``post_unicode_string`` / ``post_key_chord`` record what was posted (with
                the fake time and a copy of the pasteboard) and their return value is
                settable (None = fully posted, int, callable, or an exception to raise).
permissions     ``is_process_trusted`` / ``is_secure_event_input_enabled`` are settable and
                can be made to raise OSError.
pasteboard      an ordered list of (UTI string, bytes | None); ``None`` models a type whose
                data cannot be read. Setting data without a preceding ``clearContents`` is
                recorded in ``pasteboard_violations``.
Carbon hotkeys  ``install_hotkey_handler`` keeps the callback, ``register_event_hotkey``
                honours ``conflicts`` / ``register_status``, and ``deliver_hotkey()`` calls
                every installed handler the way the Carbon dispatcher would.
main run loop   ``create_main_run_loop_source`` / ``signal_main_run_loop_source`` plus
                ``pump()``: a test on the main thread drives the queued work itself.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Platform constants (Apple headers)
# ---------------------------------------------------------------------------

#: CGEventFlags (CoreGraphics/CGEventTypes.h).
CG_FLAG_ALPHA_SHIFT = 0x00010000  # Caps Lock -- a latch, not a held key
CG_FLAG_SHIFT = 0x00020000
CG_FLAG_CONTROL = 0x00040000
CG_FLAG_ALTERNATE = 0x00080000  # Option
CG_FLAG_COMMAND = 0x00100000
CG_FLAG_NUMERIC_PAD = 0x00200000
CG_FLAG_HELP = 0x00400000
CG_FLAG_SECONDARY_FN = 0x00800000
CG_FLAG_NON_COALESCED = 0x00000100

#: The four real modifier keys the injector must wait for.
HELD_MODIFIER_FLAGS = (CG_FLAG_SHIFT, CG_FLAG_CONTROL, CG_FLAG_ALTERNATE, CG_FLAG_COMMAND)
#: Bits that are set but are not a held modifier key.
NON_HELD_FLAGS = (
    CG_FLAG_ALPHA_SHIFT,
    CG_FLAG_NUMERIC_PAD,
    CG_FLAG_HELP,
    CG_FLAG_SECONDARY_FN,
    CG_FLAG_NON_COALESCED,
)

#: Carbon modifier masks (HIToolbox/Events.h).
CMD_KEY = 0x0100
SHIFT_KEY = 0x0200
OPTION_KEY = 0x0800
CONTROL_KEY = 0x1000

#: OSStatus values (MacTypes.h / CarbonEvents.h).
NO_ERR = 0
EVENT_HOT_KEY_EXISTS_ERR = -9878
EVENT_HOT_KEY_INVALID_ERR = -9879
EVENT_INTERNAL_ERR = -9868
PARAM_ERR = -50

#: Carbon virtual key codes used by the tests (HIToolbox/Events.h).
KVK_ANSI_V = 0x09
KVK_ANSI_Q = 0x0C
KVK_ANSI_D = 0x02
KVK_ANSI_A = 0x00
KVK_F9 = 0x65
KVK_F12 = 0x6F

#: NSPasteboardTypeString.
TYPE_STRING = "public.utf8-plain-text"
TYPE_RTF = "public.rtf"
TYPE_PNG = "public.png"
TYPE_URL = "public.file-url"


@dataclass(frozen=True)
class FakeAppInfo:
    """What ``frontmost_app()`` returns (the fields ``MacAppInfo`` exposes)."""

    pid: int = 501
    bundle_id: str = "com.apple.TextEdit"
    name: str = "TextEdit"


def utf16_units(text: str) -> list[int]:
    raw = text.encode("utf-16-le", "surrogatepass")
    return [int.from_bytes(raw[i:i + 2], "little") for i in range(0, len(raw), 2)]


def utf16_len(text: str) -> int:
    return len(utf16_units(text))


class FakeMacApi:
    """See the module docstring. Thread-safe; every mutation holds ``_lock``."""

    def __init__(
        self,
        *,
        start_time: float = 1000.0,
        trusted: bool = True,
        secure_input: bool = False,
        main_ident: Optional[int] = None,
    ) -> None:
        self._lock = threading.RLock()

        # --- clock
        self.now = float(start_time)
        self.sleeps: list[float] = []
        self._scheduled: list[tuple[float, Callable[[], None]]] = []
        self.max_sleep_calls = 200_000

        # --- keyboard state
        self.flags = 0
        self.flag_queries = 0

        # --- keyboard output
        self.posted_unicode: list[dict] = []
        self.posted_chords: list[dict] = []
        self.post_unicode_result = None  # None | int | callable(text) -> int | exception
        self.post_chord_result = None  # None | int | callable(keycode, flags) -> int | exception
        self.post_cost_s = 0.0

        # --- permissions
        self.trusted = bool(trusted)
        self.secure_input = bool(secure_input)
        self.permissions_error: Optional[BaseException] = None
        self.trust_requests = 0
        self.microphone_status = 3  # AV_AUTH_AUTHORIZED
        self.microphone_requests = 0

        # --- frontmost app
        self.frontmost = FakeAppInfo()
        self.frontmost_error: Optional[BaseException] = None

        # --- pasteboard
        self.pasteboard: list[list] = []  # [type id, bytes | None]
        self.change_count = 7
        self.pasteboard_error: Optional[BaseException] = None
        self.set_string_result: Optional[bool] = None
        self.set_data_result: Optional[bool] = None
        self.pasteboard_violations: list[str] = []
        self._cleared_since_set = False

        # --- Carbon hotkeys
        self.handlers: dict[int, Callable[[int], None]] = {}
        self.install_calls = 0
        self.install_status = NO_ERR
        self.install_handler_fails = False
        self.remove_calls: list[int] = []
        self.remove_status = NO_ERR
        self.registrations: list[dict] = []  # {"ref", "id", "keycode", "modifiers"}
        self.register_calls: list[dict] = []
        self.unregister_calls: list[dict] = []
        self.conflicts: set[tuple[int, int]] = set()  # (modifiers, keycode)
        self.register_status: Optional[int] = None  # forces this OSStatus on every register
        self.unregister_status = NO_ERR

        # --- main run loop
        self.sources: dict[int, Callable[[], None]] = {}
        self.signalled: set[int] = set()
        self.create_source_result: Optional[int] = None  # 0 => CFRunLoopSourceCreate failed
        self.destroyed_sources: list[int] = []
        self.pumps = 0
        self.main_ident = main_ident if main_ident is not None else threading.get_ident()

        self._next_handle = 0x1000
        self.log: list[tuple] = []

    # ------------------------------------------------------------------ helpers
    def record(self, kind: str, payload=None) -> None:
        with self._lock:
            self.log.append((kind, payload, self.now))

    def kinds(self) -> list[str]:
        with self._lock:
            return [entry[0] for entry in self.log]

    def _handle(self) -> int:
        with self._lock:
            self._next_handle += 8
            return self._next_handle

    # clock ------------------------------------------------------------------
    def at(self, t: float, action: Callable[[], None]) -> None:
        """Run ``action`` once the fake clock reaches ``t``."""
        with self._lock:
            self._scheduled.append((t, action))
            self._run_due()

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds
            self._run_due()

    def _run_due(self) -> None:
        due = [a for a in self._scheduled if a[0] <= self.now]
        self._scheduled = [a for a in self._scheduled if a[0] > self.now]
        for _, action in sorted(due, key=lambda a: a[0]):
            action()

    def monotonic(self) -> float:
        with self._lock:
            return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            assert len(self.sleeps) < self.max_sleep_calls, \
                "runaway sleep loop (is the deadline taken from api.monotonic()?)"
            self.now += max(0.0, seconds)
            self.log.append(("sleep", seconds, self.now))
            self._run_due()
        time.sleep(0)  # yield the GIL, never really wait

    # modifiers --------------------------------------------------------------
    def hold(self, bits: int, *, until: Optional[float] = None) -> None:
        """Hold the modifier(s) ``bits``; release them when the clock reaches ``until``."""
        with self._lock:
            self.flags |= bits
        if until is not None:
            self.at(until, lambda: self.release(bits))

    def release(self, bits: int) -> None:
        with self._lock:
            self.flags &= ~bits

    def modifier_flags(self) -> int:
        with self._lock:
            self.flag_queries += 1
            return self.flags

    # keyboard output --------------------------------------------------------
    def _result(self, configured, default: int, *args) -> int:
        if isinstance(configured, BaseException):
            raise configured
        if callable(configured):
            return int(configured(*args))
        if configured is None:
            return default
        return min(int(configured), default)

    def post_unicode_string(self, text: str) -> int:
        with self._lock:
            if not text:
                return 0
            call = {
                "text": text,
                "units": utf16_len(text),
                "time": self.now,
                "ident": threading.get_ident(),
                "pasteboard": [(t, d) for t, d in self.pasteboard],
            }
            self.posted_unicode.append(call)
            self.log.append(("post_unicode", text, self.now))
            configured = self.post_unicode_result
            self.now += self.post_cost_s
        return self._result(configured, 2, text)

    def post_key_chord(self, keycode: int, flags: int = 0) -> int:
        with self._lock:
            call = {
                "keycode": keycode,
                "flags": flags,
                "time": self.now,
                "ident": threading.get_ident(),
                "pasteboard": [(t, d) for t, d in self.pasteboard],
            }
            self.posted_chords.append(call)
            self.log.append(("post_chord", (keycode, flags), self.now))
            configured = self.post_chord_result
            self.now += self.post_cost_s
        return self._result(configured, 2, keycode, flags)

    def typed_text(self) -> str:
        """Everything posted with post_unicode_string, concatenated in order."""
        with self._lock:
            return "".join(call["text"] for call in self.posted_unicode)

    # permissions ------------------------------------------------------------
    def is_process_trusted(self) -> bool:
        with self._lock:
            self.log.append(("is_process_trusted", self.trusted, self.now))
            if self.permissions_error is not None:
                raise self.permissions_error
            return self.trusted

    def request_process_trust(self) -> bool:
        with self._lock:
            self.trust_requests += 1
            return self.trusted

    def is_secure_event_input_enabled(self) -> bool:
        with self._lock:
            self.log.append(("is_secure_event_input", self.secure_input, self.now))
            if self.permissions_error is not None:
                raise self.permissions_error
            return self.secure_input

    def microphone_authorization_status(self) -> int:
        with self._lock:
            return self.microphone_status

    def request_microphone_access(self) -> bool:
        with self._lock:
            self.microphone_requests += 1
            return True

    def frontmost_app(self) -> Optional[FakeAppInfo]:
        with self._lock:
            if self.frontmost_error is not None:
                raise self.frontmost_error
            return self.frontmost

    # pasteboard -------------------------------------------------------------
    def _check_pasteboard(self) -> None:
        if self.pasteboard_error is not None:
            raise self.pasteboard_error

    def pasteboard_change_count(self) -> int:
        with self._lock:
            self._check_pasteboard()
            return self.change_count

    def pasteboard_types(self) -> list[str]:
        with self._lock:
            self._check_pasteboard()
            self.log.append(("pb_types", None, self.now))
            return [t for t, _ in self.pasteboard]

    def pasteboard_data(self, type_id: str) -> Optional[bytes]:
        with self._lock:
            self._check_pasteboard()
            for t, data in self.pasteboard:
                if t == type_id:
                    return data
            return None

    def pasteboard_clear(self) -> int:
        with self._lock:
            self._check_pasteboard()
            self.pasteboard = []
            self.change_count += 1
            self._cleared_since_set = True
            self.log.append(("pb_clear", None, self.now))
            return self.change_count

    def _set(self, type_id: str, data: bytes, what: str) -> bool:
        if not self._cleared_since_set:
            self.pasteboard_violations.append(f"{what} without a preceding clearContents")
        for entry in self.pasteboard:
            if entry[0] == type_id:
                entry[1] = data
                break
        else:
            self.pasteboard.append([type_id, data])
        self.log.append((what, (type_id, data), self.now))
        return True

    def pasteboard_set_data(self, type_id: str, data: bytes) -> bool:
        with self._lock:
            self._check_pasteboard()
            self._set(type_id, bytes(data), "pb_set_data")
            if self.set_data_result is not None:
                return bool(self.set_data_result)
            return True

    def pasteboard_set_string(self, text: str) -> bool:
        with self._lock:
            self._check_pasteboard()
            self._set(TYPE_STRING, text.encode("utf-8"), "pb_set_string")
            if self.set_string_result is not None:
                return bool(self.set_string_result)
            return True

    def pasteboard_string(self) -> Optional[str]:
        with self._lock:
            self._check_pasteboard()
            for t, data in self.pasteboard:
                if t == TYPE_STRING and data is not None:
                    return data.decode("utf-8")
            return None

    def pasteboard_items(self) -> list[tuple[str, Optional[bytes]]]:
        with self._lock:
            return [(t, d) for t, d in self.pasteboard]

    def seed_pasteboard(self, items) -> None:
        with self._lock:
            self.pasteboard = [[t, d] for t, d in items]
            self._cleared_since_set = False

    # Carbon hotkeys ---------------------------------------------------------
    def install_hotkey_handler(self, on_hotkey: Callable[[int], None]) -> tuple[int, int]:
        with self._lock:
            self.install_calls += 1
            self.log.append(("install_handler", None, self.now))
            if self.install_handler_fails or self.install_status != NO_ERR:
                return 0, self.install_status if self.install_status != NO_ERR else EVENT_INTERNAL_ERR
            ref = self._handle()
            self.handlers[ref] = on_hotkey
            return ref, NO_ERR

    def remove_event_handler(self, handler_ref: int) -> int:
        with self._lock:
            if not handler_ref:
                return NO_ERR
            self.remove_calls.append(handler_ref)
            self.handlers.pop(handler_ref, None)
            self.log.append(("remove_handler", handler_ref, self.now))
            return self.remove_status

    def register_event_hotkey(self, keycode: int, modifiers: int, hotkey_id: int) -> tuple[int, int]:
        with self._lock:
            combo = (modifiers, keycode)
            if self.register_status is not None and self.register_status != NO_ERR:
                status = self.register_status
            elif combo in self.conflicts or combo in self.active_combos():
                status = EVENT_HOT_KEY_EXISTS_ERR
            else:
                status = NO_ERR
            self.register_calls.append(
                {"id": hotkey_id, "keycode": keycode, "modifiers": modifiers,
                 "status": status, "ident": threading.get_ident()}
            )
            self.log.append(("register_hotkey", (hotkey_id, modifiers, keycode, status), self.now))
            if status != NO_ERR:
                return 0, status
            ref = self._handle()
            self.registrations.append(
                {"ref": ref, "id": hotkey_id, "keycode": keycode, "modifiers": modifiers}
            )
            return ref, NO_ERR

    def unregister_event_hotkey(self, hotkey_ref: int) -> int:
        with self._lock:
            if not hotkey_ref:
                return NO_ERR
            before = len(self.registrations)
            self.registrations = [r for r in self.registrations if r["ref"] != hotkey_ref]
            found = len(self.registrations) < before
            self.unregister_calls.append(
                {"ref": hotkey_ref, "found": found, "ident": threading.get_ident()}
            )
            self.log.append(("unregister_hotkey", (hotkey_ref, found), self.now))
            if not found:
                return EVENT_HOT_KEY_INVALID_ERR
            return self.unregister_status

    def active_combos(self) -> set[tuple[int, int]]:
        with self._lock:
            return {(r["modifiers"], r["keycode"]) for r in self.registrations}

    def active_ids(self) -> set[int]:
        with self._lock:
            return {r["id"] for r in self.registrations}

    def deliver_hotkey(self, hotkey_id: Optional[int] = None) -> int:
        """Dispatch one kEventHotKeyPressed to every installed handler, as Carbon does.

        With no id the single active registration's id is used. Returns the delivered id.
        """
        with self._lock:
            if hotkey_id is None:
                assert len(self.registrations) == 1, \
                    f"expected exactly one registration, got {self.registrations}"
                hotkey_id = self.registrations[0]["id"]
            handlers = list(self.handlers.values())
            self.log.append(("deliver_hotkey", hotkey_id, self.now))
        assert handlers, "no Carbon event handler installed"
        for handler in handlers:
            handler(hotkey_id)
        return hotkey_id

    # main run loop ----------------------------------------------------------
    def is_main_thread(self) -> bool:
        return threading.get_ident() == self.main_ident

    def create_main_run_loop_source(self, perform: Callable[[], None]) -> int:
        with self._lock:
            self.log.append(("create_source", None, self.now))
            if self.create_source_result is not None:
                return int(self.create_source_result)
            source = self._handle()
            self.sources[source] = perform
            return source

    def signal_main_run_loop_source(self, source: int) -> None:
        with self._lock:
            if not source:
                return
            self.signalled.add(source)
            self.log.append(("signal_source", source, self.now))

    def destroy_main_run_loop_source(self, source: int) -> None:
        with self._lock:
            if not source:
                return
            self.destroyed_sources.append(source)
            self.sources.pop(source, None)
            self.signalled.discard(source)
            self.log.append(("destroy_source", source, self.now))

    def pump(self) -> int:
        """Give the main run loop a turn: run every signalled source's perform callback.

        Returns the number of callbacks run. Call from the thread the fake calls "main".
        """
        with self._lock:
            self.pumps += 1
            ready = [(s, self.sources[s]) for s in sorted(self.signalled) if s in self.sources]
            self.signalled.difference_update(s for s, _ in ready)
        for _, perform in ready:
            perform()
        return len(ready)

    def run_main_run_loop_once(self, timeout_s: float) -> int:
        return 2 if self.pump() else 1  # kCFRunLoopRunHandledSource / RunTimedOut
