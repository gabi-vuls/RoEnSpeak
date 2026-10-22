"""QtUiFacade: the UiFacade core calls from any thread.

Every call is turned into a queued signal, so it runs on the Qt thread and the caller returns
immediately. Calls that arrive before run_ui() binds the facade to the UI are kept in order and
replayed on bind.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, Protocol

from PySide6.QtCore import QObject, Qt, Signal, Slot

log = logging.getLogger("roenspeak.ui.facade")


class UiTarget(Protocol):
    def show_error(self, title: str, message: str, details: str = "") -> None: ...
    def show_download_dialog(self, model: str, size_mb: int) -> None: ...
    def show_main_window(self) -> None: ...
    def exit_app(self) -> None: ...


class QtUiFacade(QObject):
    """Implements contracts.UiFacade. Construct on the Qt thread after QApplication exists."""

    _call = Signal(str, tuple)

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._target: Optional[UiTarget] = None
        self._pending: list[tuple[str, tuple]] = []
        self._call.connect(self._dispatch, Qt.QueuedConnection)

    # -- UiFacade (any thread) ----------------------------------------------
    def show_error(self, title: str, message: str, details: str = "") -> None:
        self._call.emit("show_error", (title, message, details))

    def show_download_dialog(self, model: str, size_mb: int) -> None:
        self._call.emit("show_download_dialog", (model, int(size_mb)))

    def show_main_window(self) -> None:
        self._call.emit("show_main_window", ())

    def exit_app(self) -> None:
        self._call.emit("exit_app", ())

    # -- binding (Qt thread) ------------------------------------------------
    def bind(self, target: UiTarget) -> None:
        """Attach the live UI and replay queued calls in arrival order."""
        with self._lock:
            self._target = target
            pending, self._pending = self._pending, []
        for name, args in pending:
            self._invoke(target, name, args)

    def unbind(self) -> None:
        with self._lock:
            self._target = None

    @property
    def bound(self) -> bool:
        return self._target is not None

    @Slot(str, tuple)
    def _dispatch(self, name: str, args: tuple) -> None:
        with self._lock:
            target = self._target
            if target is None:
                self._pending.append((name, args))
                log.debug("queued %s until the UI is bound", name)
                return
        self._invoke(target, name, args)

    @staticmethod
    def _invoke(target: UiTarget, name: str, args: tuple) -> None:
        try:
            getattr(target, name)(*args)
        except Exception:  # noqa: BLE001 (a UI failure must never propagate into core)
            log.exception("UI facade call %s failed", name)
