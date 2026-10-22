"""Global hotkey via RegisterHotKey on a dedicated message-loop thread (``contracts.HotkeyManager``).

RegisterHotKey/UnregisterHotKey are per-thread, so they only ever run on the hotkey thread.
Other threads post WM_APP+n requests to it (PostThreadMessage) and wait for the reply.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.contracts import (
    AppError,
    Hotkey,
    HotkeyConflictError,
    KeyEvent,
    Modifier,
    VK_MENU_MASK,
    WinApi,
)

log = logging.getLogger("roenspeak.win32.hotkey")

WM_HOTKEY = 0x0312
WM_APP = 0x8000
WM_APP_READY = WM_APP + 1  # hotkey thread -> itself: queue exists, loop is running
WM_APP_REGISTER = WM_APP + 2  # wparam = request id
WM_APP_UNREGISTER = WM_APP + 3  # wparam = request id
WM_APP_STOP = WM_APP + 4  # wparam = request id

VK_MENU, VK_LMENU, VK_RMENU = 0x12, 0xA4, 0xA5
_ALT_VKS = (VK_MENU, VK_LMENU, VK_RMENU)

#: The two RegisterHotKey ids used alternately, so a new combo can be registered before the
#: old one is removed (on failure the old one simply stays).
_HOTKEY_IDS = (1, 2)

REQUEST_TIMEOUT_S = 2.0
_START_TIMEOUT_S = 2.0
_JOIN_TIMEOUT_S = 2.0
_POST_RETRY_S = 0.005


@dataclass
class _Request:
    kind: int
    hotkey: Optional[Hotkey] = None
    done: threading.Event = field(default_factory=threading.Event)
    error: Optional[BaseException] = None


class Win32HotkeyManager:
    """See ``contracts.HotkeyManager``."""

    def __init__(self, api: WinApi, on_fire: Callable[[], None], debounce_s: float = 0.3) -> None:
        self._api = api
        self._on_fire = on_fire
        self._debounce_s = float(debounce_s)

        self._lifecycle = threading.RLock()  # start/stop
        self._lock = threading.Lock()  # state below
        self._thread: Optional[threading.Thread] = None
        self._thread_id: Optional[int] = None
        self._ready = threading.Event()
        self._requests: dict[int, _Request] = {}
        self._req_ids = itertools.count(1)

        # Owned by the hotkey thread (read under _lock by other threads).
        self._current: Optional[Hotkey] = None
        self._active_id: Optional[int] = None
        self._last_fire: Optional[float] = None

    # --- public API (any thread) ---------------------------------------------------------

    @property
    def current(self) -> Optional[Hotkey]:
        with self._lock:
            return self._current

    def start(self) -> None:
        with self._lifecycle:
            if self._thread is not None and self._thread.is_alive():
                return
            self._ready = threading.Event()
            self._thread_id = None
            t = threading.Thread(target=self._run, name="roenspeak-hotkey", daemon=True)
            self._thread = t
            t.start()
            if not self._ready.wait(_START_TIMEOUT_S):
                log.warning("hotkey thread did not report ready within %.1f s", _START_TIMEOUT_S)
            if not t.is_alive():
                self._thread = None
                raise AppError("The hotkey service could not be started.", details="hotkey thread exited at startup")

    def register(self, hotkey: Hotkey) -> None:
        self._request(_Request(WM_APP_REGISTER, hotkey))

    def unregister(self) -> None:
        if not self._is_running():
            return
        self._request(_Request(WM_APP_UNREGISTER))

    def stop(self) -> None:
        with self._lifecycle:
            t = self._thread
            if t is None:
                return
            if t is threading.current_thread():
                # Called from on_fire: can't join ourselves. Unregister and quit the loop directly.
                self._do_unregister()
                self._api.post_quit_message()
                self._thread = None
                return
            if t.is_alive():
                try:
                    self._request(_Request(WM_APP_STOP))
                except (AppError, RuntimeError) as exc:
                    log.warning("hotkey stop request failed: %s", getattr(exc, "details", "") or exc)
                t.join(_JOIN_TIMEOUT_S)
                if t.is_alive():
                    log.warning("hotkey thread did not exit within %.1f s", _JOIN_TIMEOUT_S)
            self._thread = None
            with self._lock:
                self._current = None
                self._active_id = None

    # --- request plumbing ----------------------------------------------------------------

    def _is_running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive() and self._thread_id is not None

    def _request(self, req: _Request) -> None:
        t = self._thread
        if t is not None and t is threading.current_thread():
            self._execute(req)  # re-entrant call from on_fire
            if req.error is not None:
                raise req.error
            return
        if not self._is_running():
            raise RuntimeError("Win32HotkeyManager is not started")
        req_id = next(self._req_ids)
        with self._lock:
            self._requests[req_id] = req
        deadline = time.perf_counter() + REQUEST_TIMEOUT_S
        posted = False
        while True:
            if self._api.post_thread_message(self._thread_id, req.kind, req_id, 0):
                posted = True
                break
            if time.perf_counter() >= deadline or not t.is_alive():
                break
            time.sleep(_POST_RETRY_S)
        if posted:
            req.done.wait(max(0.0, deadline - time.perf_counter()))
        if not req.done.is_set():
            with self._lock:
                taken = self._requests.pop(req_id, None) is None
            if taken:
                # The hotkey thread picked it up just now; let it finish.
                req.done.wait(1.0)
            if not req.done.is_set():
                err = self._api.get_last_error() if not posted else 0
                raise AppError(
                    "The hotkey service is not responding. Restart RoEnSpeak.",
                    details=f"hotkey request {req.kind:#x} timed out (posted={posted}, GetLastError={err})",
                )
        if req.error is not None:
            raise req.error

    # --- hotkey thread ------------------------------------------------------------------

    def _run(self) -> None:
        api = self._api
        try:
            self._thread_id = api.get_current_thread_id()
            # Posting to ourselves makes this a GUI thread with a message queue; the READY message
            # comes back through get_message once the loop is running.
            if not api.post_thread_message(self._thread_id, WM_APP_READY, 0, 0):
                log.warning("self-post failed (GetLastError=%d); continuing", api.get_last_error())
                self._ready.set()
            log.debug("hotkey thread started tid=%s", self._thread_id)
            while True:
                try:
                    msg = api.get_message()
                except Exception:
                    log.exception("GetMessage failed; hotkey thread exits")
                    break
                if msg is None:
                    break
                try:
                    self._dispatch(msg.message, msg.wparam)
                except Exception:
                    log.exception("error handling message %#x", msg.message)
        finally:
            try:
                self._do_unregister()
            except Exception:
                log.exception("unregister on hotkey thread exit failed")
            self._ready.set()
            with self._lock:
                pending = list(self._requests.values())
                self._requests.clear()
            for req in pending:
                req.error = AppError("The hotkey service stopped.", details="hotkey thread exited")
                req.done.set()
            self._thread_id = None
            log.debug("hotkey thread exited")

    def _dispatch(self, message: int, wparam: int) -> None:
        if message == WM_HOTKEY:
            self._on_hotkey(wparam)
        elif message == WM_APP_READY:
            self._ready.set()
        elif message in (WM_APP_REGISTER, WM_APP_UNREGISTER, WM_APP_STOP):
            with self._lock:
                req = self._requests.pop(wparam, None)
            if req is None:
                return  # caller gave up already
            self._execute(req)

    def _execute(self, req: _Request) -> None:
        try:
            if req.kind == WM_APP_REGISTER:
                assert req.hotkey is not None
                self._do_register(req.hotkey)
            elif req.kind == WM_APP_UNREGISTER:
                self._do_unregister()
            elif req.kind == WM_APP_STOP:
                self._do_unregister()
                self._api.post_quit_message()
        except BaseException as exc:  # handed back to the caller
            req.error = exc
        finally:
            req.done.set()

    def _do_register(self, hotkey: Hotkey) -> None:
        api = self._api
        with self._lock:
            current, old_id = self._current, self._active_id
        if current is not None and old_id is not None and current.vk == hotkey.vk \
                and current.modifiers == hotkey.modifiers:
            with self._lock:
                self._current = hotkey
            log.info("hotkey %s already registered", hotkey.display)
            return
        new_id = _HOTKEY_IDS[1] if old_id == _HOTKEY_IDS[0] else _HOTKEY_IDS[0]
        if not api.register_hotkey(new_id, hotkey.win32_modifiers, hotkey.vk):
            err = api.get_last_error()
            log.warning("RegisterHotKey failed for %s (GetLastError=%d); keeping %s",
                        hotkey.display, err, current.display if current else "none")
            raise HotkeyConflictError(hotkey.display, win32_error=err)
        if old_id is not None:
            if not api.unregister_hotkey(old_id):
                log.warning("UnregisterHotKey(%d) failed, GetLastError=%d", old_id, api.get_last_error())
        with self._lock:
            self._current = hotkey
            self._active_id = new_id
            self._last_fire = None
        log.info("hotkey registered %s", hotkey.display)

    def _do_unregister(self) -> None:
        with self._lock:
            old_id, current = self._active_id, self._current
            self._active_id = None
            self._current = None
        if old_id is None:
            return
        if not self._api.unregister_hotkey(old_id):
            log.warning("UnregisterHotKey(%d) failed, GetLastError=%d", old_id, self._api.get_last_error())
        log.info("hotkey unregistered %s", current.display if current else "")

    def _on_hotkey(self, hotkey_id: int) -> None:
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

        if Modifier.ALT in hotkey.modifiers or any(api.get_async_key_state(vk) & 0x8000 for vk in _ALT_VKS):
            sent = api.send_input([KeyEvent(vk=VK_MENU_MASK), KeyEvent(vk=VK_MENU_MASK, key_up=True)])
            if sent != 2:
                log.debug("Alt-menu suppression SendInput inserted %d/2 events", sent)

        log.info("hotkey fired %s", hotkey.display)
        try:
            self._on_fire()
        except Exception:
            log.exception("hotkey on_fire callback raised")
