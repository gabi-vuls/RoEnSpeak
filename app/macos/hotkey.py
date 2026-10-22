"""Global hotkey via Carbon RegisterEventHotKey (``contracts.HotkeyManager``).

The Windows manager owns a thread with a GetMessage loop because RegisterHotKey and WM_HOTKEY
are per-thread. On macOS the equivalent calls (``InstallEventHandler`` on the application
event target, ``RegisterEventHotKey``, and the resulting hot key events) belong to the *main*
thread's run loop, and in the app that run loop is Qt's. So this manager owns no thread:

* All Carbon work runs on the main thread. A caller already on the main thread executes it
  directly; any other thread appends a request to a queue, signals a CFRunLoopSource that
  lives on the main run loop, and blocks until the main thread has run it (<= 2 s), exactly
  as ``Win32HotkeyManager`` does with PostThreadMessage.
* With no run loop spinning (a plain script, before ``app.exec()``), ``start()`` still
  succeeds and main-thread ``register()`` / ``unregister()`` / ``stop()`` still work, because
  they do not need a run loop -- only *delivery* of the key press does. A request marshalled
  from another thread while nothing drives the main run loop cannot be served, so it fails
  honestly with an AppError after the 2 s timeout instead of hanging.
* ``on_fire`` therefore runs on the main (Qt) thread rather than on a private hotkey thread.
  The contract already requires it not to block.

Everything else matches the Windows manager: ``start()``/``unregister()``/``stop()`` are
idempotent and blocking, ``register()`` replaces the current hotkey and on failure raises
``HotkeyConflictError`` with the previous hotkey still registered (two alternating hot key
ids make that possible), ``current`` is readable from any thread, presses within
``debounce_s`` of the previous accepted press are dropped, and an exception from ``on_fire``
is logged and never fatal.

**No Alt-menu suppression.** The Windows manager injects VK_MENU_MASK so the focused app does
not open its menu bar when Alt is released; macOS has no menu-bar-on-Option behaviour, so
that step does not exist here. It is Windows-only.
"""

from __future__ import annotations

import collections
import itertools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.contracts import AppError, Hotkey, HotkeyConflictError, Modifier

from app.macos._mac_api import EVENT_HOT_KEY_EXISTS_ERR, NO_ERR

log = logging.getLogger("roenspeak.macos.hotkey")

#: Carbon modifier masks (HIToolbox/Events.h).
CMD_KEY, SHIFT_KEY, OPTION_KEY, CONTROL_KEY = 0x0100, 0x0200, 0x0800, 0x1000

_CARBON_MODIFIER_BITS = {
    Modifier.CMD: CMD_KEY,
    Modifier.WIN: CMD_KEY,  # never produced by the macOS parser; kept total
    Modifier.SHIFT: SHIFT_KEY,
    Modifier.ALT: OPTION_KEY,  # Option
    Modifier.CTRL: CONTROL_KEY,
}

#: Request kinds (the analogue of the WM_APP+n messages on Windows).
REQ_REGISTER = 1
REQ_UNREGISTER = 2
REQ_STOP = 3

#: The two RegisterEventHotKey ids used alternately, so a new combo can be registered before
#: the old one is removed (on failure the old one simply stays). On Windows the ids are
#: per-thread and a single pair is enough; here every manager installs a handler on the one
#: application event target and sees *every* hot key event, so each manager takes its own
#: pair from this counter and ignores ids that are not its own. The app has one manager, so
#: it gets (1, 2); a second one (a test, say) cannot shadow it.
_ID_PAIRS = itertools.count(1)

REQUEST_TIMEOUT_S = 2.0
_TAKEN_GRACE_S = 1.0


def _next_hotkey_ids() -> tuple[int, int]:
    """The next unused hot key id pair. The first manager gets (1, 2)."""
    base = next(_ID_PAIRS) * 2 - 1
    return base, base + 1


def carbon_modifiers(hotkey: Hotkey) -> int:
    """``inHotKeyModifiers`` for RegisterEventHotKey (cmdKey/shiftKey/optionKey/controlKey).

    The Carbon analogue of ``Hotkey.win32_modifiers``. There is no NOREPEAT bit: Carbon hot
    keys do not auto-repeat, which is why the Windows manager asks for MOD_NOREPEAT.
    """
    bits = 0
    for modifier in hotkey.modifiers:
        bits |= _CARBON_MODIFIER_BITS[modifier]
    return bits


@dataclass
class _Request:
    kind: int
    hotkey: Optional[Hotkey] = None
    done: threading.Event = field(default_factory=threading.Event)
    error: Optional[BaseException] = None


class MacHotkeyManager:
    """See ``contracts.HotkeyManager`` and this module's docstring."""

    def __init__(self, api, on_fire: Callable[[], None], debounce_s: float = 0.3) -> None:
        self._api = api
        self._on_fire = on_fire
        self._debounce_s = float(debounce_s)

        self._lifecycle = threading.RLock()  # start/stop
        self._lock = threading.Lock()  # the state below
        self._started = False
        self._ready = threading.Event()
        self._source = 0
        self._queue: collections.deque[_Request] = collections.deque()
        self._hotkey_ids = _next_hotkey_ids()

        # Touched only on the main thread, read under _lock from other threads.
        self._handler_ref = 0
        self._current: Optional[Hotkey] = None
        self._active_id: Optional[int] = None
        self._active_ref = 0
        self._last_fire: Optional[float] = None
        self._in_handler = False

    # --- public API (any thread) ---------------------------------------------------------

    @property
    def current(self) -> Optional[Hotkey]:
        with self._lock:
            return self._current

    def start(self) -> None:
        with self._lifecycle:
            if self._started:
                return
            self._queue.clear()
            source = self._api.create_main_run_loop_source(self._drain)
            if not source:
                log.warning("CFRunLoopSourceCreate failed; hotkey requests from other threads "
                            "cannot be served")
            self._source = source
            self._started = True
            self._ready.set()
            # Installing the handler needs the main thread. Doing it here when we already are
            # the main thread turns a broken setup into an error at startup instead of at the
            # first register(); otherwise the first request installs it.
            if self._api.is_main_thread():
                try:
                    self._ensure_handler()
                except AppError:
                    self._started = False
                    self._ready.clear()
                    if self._source:
                        self._api.destroy_main_run_loop_source(self._source)
                        self._source = 0
                    raise
            log.debug("hotkey manager started source=%#x main_thread=%s",
                      self._source, self._api.is_main_thread())

    def register(self, hotkey: Hotkey) -> None:
        if not self._started:
            raise RuntimeError("MacHotkeyManager is not started")
        self._request(_Request(REQ_REGISTER, hotkey))

    def unregister(self) -> None:
        if not self._started:
            return
        self._request(_Request(REQ_UNREGISTER))

    def stop(self) -> None:
        with self._lifecycle:
            if not self._started:
                return
            try:
                self._request(_Request(REQ_STOP))
            except (AppError, RuntimeError) as exc:
                log.warning("hotkey stop request failed: %s; unregistering directly",
                            getattr(exc, "details", "") or exc)
                # Nothing is driving the main run loop. Release the registration anyway: a
                # leaked global hotkey would keep the combo taken for the whole session.
                try:
                    self._do_unregister()
                    self._remove_handler()
                except Exception:
                    log.exception("direct hotkey unregister on stop failed")
            self._started = False
            self._ready.clear()
            source, self._source = self._source, 0
            if source:
                self._api.destroy_main_run_loop_source(source)
            with self._lock:
                self._current = None
                self._active_id = None
                self._active_ref = 0
            self._fail_pending("The hotkey service stopped.", "hotkey manager stopped")
            log.debug("hotkey manager stopped")

    # --- request plumbing ----------------------------------------------------------------

    def _request(self, req: _Request) -> None:
        if self._api.is_main_thread():
            self._execute(req)  # we *are* the run loop thread
            if req.error is not None:
                raise req.error
            return
        with self._lock:
            self._queue.append(req)
        self._api.signal_main_run_loop_source(self._source)
        deadline = time.perf_counter() + REQUEST_TIMEOUT_S
        req.done.wait(REQUEST_TIMEOUT_S)
        if not req.done.is_set():
            with self._lock:
                try:
                    self._queue.remove(req)
                    taken = False
                except ValueError:
                    taken = True
            if taken:
                # The main thread picked it up just now; let it finish.
                req.done.wait(_TAKEN_GRACE_S)
            if not req.done.is_set():
                raise AppError(
                    "The hotkey service is not responding. Restart RoEnSpeak.",
                    details=(f"hotkey request {req.kind} timed out after "
                             f"{time.perf_counter() - deadline + REQUEST_TIMEOUT_S:.2f} s "
                             f"(source={self._source:#x}; is the main run loop running?)"),
                )
        if req.error is not None:
            raise req.error

    def _drain(self) -> None:
        """CFRunLoopSource perform callback: run every queued request. Main thread."""
        while True:
            with self._lock:
                if not self._queue:
                    return
                req = self._queue.popleft()
            self._execute(req)

    def _execute(self, req: _Request) -> None:
        try:
            if req.kind == REQ_REGISTER:
                assert req.hotkey is not None
                self._do_register(req.hotkey)
            elif req.kind == REQ_UNREGISTER:
                self._do_unregister()
            elif req.kind == REQ_STOP:
                self._do_unregister()
                self._remove_handler()
        except BaseException as exc:  # handed back to the caller
            req.error = exc
        finally:
            req.done.set()

    def _fail_pending(self, message: str, details: str) -> None:
        with self._lock:
            pending = list(self._queue)
            self._queue.clear()
        for req in pending:
            req.error = AppError(message, details=details)
            req.done.set()

    # --- main thread ---------------------------------------------------------------------

    def _ensure_handler(self) -> None:
        if self._handler_ref:
            return
        handler_ref, status = self._api.install_hotkey_handler(self._on_hotkey)
        if not handler_ref:
            raise AppError(
                "The hotkey service could not be started.",
                details=f"InstallEventHandler failed, OSStatus={status}",
            )
        self._handler_ref = handler_ref
        log.debug("hotkey event handler installed ref=%#x", handler_ref)

    def _remove_handler(self) -> None:
        handler_ref, self._handler_ref = self._handler_ref, 0
        if not handler_ref:
            return
        if self._in_handler:
            # RemoveEventHandler from inside its own dispatch is not safe; with no hot key
            # registered the handler is inert anyway.
            self._handler_ref = handler_ref
            log.debug("skipping RemoveEventHandler while inside the handler")
            return
        status = self._api.remove_event_handler(handler_ref)
        if status != NO_ERR:
            log.warning("RemoveEventHandler failed, OSStatus=%d", status)

    def _do_register(self, hotkey: Hotkey) -> None:
        api = self._api
        self._ensure_handler()
        with self._lock:
            current, old_id, old_ref = self._current, self._active_id, self._active_ref
        if current is not None and old_id is not None and current.vk == hotkey.vk \
                and current.modifiers == hotkey.modifiers:
            with self._lock:
                self._current = hotkey
            log.info("hotkey %s already registered", hotkey.display)
            return
        new_id = self._hotkey_ids[1] if old_id == self._hotkey_ids[0] else self._hotkey_ids[0]
        new_ref, status = api.register_event_hotkey(hotkey.vk, carbon_modifiers(hotkey), new_id)
        if status != NO_ERR or not new_ref:
            taken = " (combination already taken)" if status == EVENT_HOT_KEY_EXISTS_ERR else ""
            log.warning("RegisterEventHotKey failed for %s (OSStatus=%d%s); keeping %s",
                        hotkey.display, status, taken, current.display if current else "none")
            raise HotkeyConflictError(hotkey.display, win32_error=status)
        if old_ref:
            old_status = api.unregister_event_hotkey(old_ref)
            if old_status != NO_ERR:
                log.warning("UnregisterEventHotKey(id=%s) failed, OSStatus=%d", old_id, old_status)
        with self._lock:
            self._current = hotkey
            self._active_id = new_id
            self._active_ref = new_ref
            self._last_fire = None
        log.info("hotkey registered %s", hotkey.display)

    def _do_unregister(self) -> None:
        with self._lock:
            old_ref, old_id, current = self._active_ref, self._active_id, self._current
            self._active_ref = 0
            self._active_id = None
            self._current = None
        if not old_ref:
            return
        status = self._api.unregister_event_hotkey(old_ref)
        if status != NO_ERR:
            log.warning("UnregisterEventHotKey(id=%s) failed, OSStatus=%d", old_id, status)
        log.info("hotkey unregistered %s", current.display if current else "")

    def _on_hotkey(self, hotkey_id: int) -> None:
        """Carbon kEventHotKeyPressed callback. Main thread."""
        api = self._api
        with self._lock:
            hotkey = self._current
            active = self._active_id
        if hotkey is None or hotkey_id != active:
            return
        now = api.monotonic()
        if self._last_fire is not None and now - self._last_fire < self._debounce_s:
            log.debug("hotkey %s debounced", hotkey.display)
            return
        self._last_fire = now
        log.info("hotkey fired %s", hotkey.display)
        self._in_handler = True
        try:
            self._on_fire()
        except Exception:
            log.exception("hotkey on_fire callback raised")
        finally:
            self._in_handler = False
