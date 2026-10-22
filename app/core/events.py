"""In-process synchronous event bus (core -> UI)."""

from __future__ import annotations

import logging
import threading
from typing import Callable

from app.contracts import AppEvent

log = logging.getLogger("roenspeak.core.events")


class SimpleEventBus:
    """EventBus: publish() calls every subscriber synchronously on the publishing thread, in
    subscription order. A failing subscriber is logged and does not affect the others."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[AppEvent], None]] = []

    def subscribe(self, callback: Callable[[AppEvent], None]) -> Callable[[], None]:
        token = _Subscription(callback)
        with self._lock:
            self._subscribers.append(token)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(token)
                except ValueError:
                    pass

        return unsubscribe

    def publish(self, event: AppEvent) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for sub in subscribers:
            try:
                sub(event)
            except Exception:
                log.exception("Event subscriber %r failed on %s", sub, type(event).__name__)


class _Subscription:
    """Wraps a callback so the same callable can be subscribed twice and removed individually."""

    __slots__ = ("callback",)

    def __init__(self, callback: Callable[[AppEvent], None]) -> None:
        self.callback = callback

    def __call__(self, event: AppEvent) -> None:
        self.callback(event)

    def __repr__(self) -> str:
        return repr(self.callback)
