"""Single-instance guard via a named mutex (``contracts.SingleInstance``)."""

from __future__ import annotations

import logging
import threading
from typing import Optional

from app.contracts import SHOW_WINDOW_MESSAGE_NAME, WinApi, single_instance_name

log = logging.getLogger("roenspeak.win32.instance")

ASFW_ANY = 0xFFFFFFFF


class NamedMutexSingleInstance:
    def __init__(self, api: WinApi, name: Optional[str] = None) -> None:
        self._api = api
        self._name = name if name is not None else single_instance_name()
        self._handle = 0
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    def acquire(self) -> bool:
        with self._lock:
            if self._handle:
                return True
            handle, existed = self._api.create_mutex(self._name)
            if existed:
                if handle:
                    self._api.close_handle(handle)
                log.info("another instance holds %s", self._name)
                return False
            if not handle:
                # Could not create or open the mutex for another reason; don't refuse to run.
                log.warning("CreateMutex(%s) failed (GetLastError=%d); single-instance check skipped",
                            self._name, self._api.get_last_error())
                return True
            self._handle = handle
            log.info("single-instance mutex acquired %s", self._name)
            return True

    def notify_existing(self) -> None:
        msg = self._api.register_window_message(SHOW_WINDOW_MESSAGE_NAME)
        if not msg:
            log.warning("RegisterWindowMessage(%s) failed", SHOW_WINDOW_MESSAGE_NAME)
            return
        # Let the running instance take the foreground when it handles the message.
        if not self._api.allow_set_foreground_window(ASFW_ANY):
            log.debug("AllowSetForegroundWindow(ASFW_ANY) failed, GetLastError=%d", self._api.get_last_error())
        self._api.broadcast_message(msg)
        log.info("asked the running instance to show its window")

    def release(self) -> None:
        with self._lock:
            handle, self._handle = self._handle, 0
        if handle:
            self._api.close_handle(handle)
            log.info("single-instance mutex released")
