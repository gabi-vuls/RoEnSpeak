"""Thread-safe app state machine enforcing contracts.ALLOWED_TRANSITIONS."""

from __future__ import annotations

import logging
import threading

from app.contracts import ALLOWED_TRANSITIONS, AppState, EventBus, InvalidTransition, StateChanged

log = logging.getLogger("roenspeak.core.state")


class AppStateMachine:
    """StateMachine. transition() to the current state is a no-op returning False; an allowed
    transition publishes StateChanged after the lock is released; anything else raises
    InvalidTransition and leaves the state unchanged."""

    def __init__(self, bus: EventBus, initial: AppState = AppState.DISARMED) -> None:
        self._bus = bus
        self._state = AppState(initial)
        self._lock = threading.Lock()

    @property
    def state(self) -> AppState:
        return self._state

    def can_transition(self, new: AppState) -> bool:
        new = AppState(new)
        with self._lock:
            return new == self._state or new in ALLOWED_TRANSITIONS[self._state]

    def transition(self, new: AppState, detail: str = "") -> bool:
        new = AppState(new)
        with self._lock:
            old = self._state
            if new == old:
                return False
            if new not in ALLOWED_TRANSITIONS[old]:
                raise InvalidTransition(old, new)
            self._state = new
        log.info("state %s -> %s%s", old.value, new.value, f" ({detail})" if detail else "")
        self._bus.publish(StateChanged(old, new, detail))
        return True
