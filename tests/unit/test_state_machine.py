"""Spec/contract tests for app.core.state.AppStateMachine and app.core.events.SimpleEventBus.

Derived from contracts.ALLOWED_TRANSITIONS, StateMachine, EventBus docstrings and
ARCHITECTURE.md section 6 (scenarios).
"""

from __future__ import annotations

import itertools
import logging
import threading
from collections import Counter

import pytest

from app.contracts import (
    ALLOWED_TRANSITIONS,
    AppError,
    AppState,
    EventBus,
    InvalidTransition,
    Notification,
    StateChanged,
    StateMachine,
)
from app.core.events import SimpleEventBus
from app.core.state import AppStateMachine

TIMEOUT = 5.0
ALL_PAIRS = list(itertools.product(list(AppState), list(AppState)))


class RecordingBus:
    """Minimal EventBus fake that records published events."""

    def __init__(self) -> None:
        self.events: list = []
        self._subs: list = []
        self._lock = threading.Lock()

    def publish(self, event) -> None:
        with self._lock:
            self.events.append(event)
            subs = list(self._subs)
        for cb in subs:
            cb(event)

    def subscribe(self, callback):
        self._subs.append(callback)

        def unsubscribe():
            if callback in self._subs:
                self._subs.remove(callback)

        return unsubscribe


def run_with_timeout(fn, timeout=TIMEOUT):
    """Run fn in a daemon thread; fail (instead of hanging) on deadlock."""
    result: dict = {}

    def target():
        try:
            result["value"] = fn()
        except BaseException as e:  # re-raised below
            result["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), "deadlock: call did not return within timeout"
    if "error" in result:
        raise result["error"]
    return result.get("value")


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


def test_protocols():
    bus = SimpleEventBus()
    assert isinstance(bus, EventBus)
    assert isinstance(AppStateMachine(bus), StateMachine)


def test_default_initial_state_is_disarmed():
    assert AppStateMachine(RecordingBus()).state == AppState.DISARMED


@pytest.mark.parametrize("state", list(AppState))
def test_initial_state_param_no_event(state):
    bus = RecordingBus()
    sm = AppStateMachine(bus, initial=state)
    assert sm.state == state
    assert bus.events == []


def test_invalid_transition_is_app_error():
    assert issubclass(InvalidTransition, AppError)


# ---------------------------------------------------------------------------
# Exhaustive transition table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("old,new", ALL_PAIRS, ids=[f"{a.value}->{b.value}" for a, b in ALL_PAIRS])
def test_transition_table(old, new):
    bus = RecordingBus()
    sm = AppStateMachine(bus, initial=old)
    if old == new:
        assert sm.transition(new, "same") is False
        assert sm.state == old
        assert bus.events == []
    elif new in ALLOWED_TRANSITIONS[old]:
        assert sm.can_transition(new) is True
        assert sm.transition(new, "detail-x") is True
        assert sm.state == new
        assert bus.events == [StateChanged(old, new, "detail-x")]
    else:
        assert sm.can_transition(new) is False
        with pytest.raises(InvalidTransition) as ei:
            sm.transition(new, "nope")
        assert ei.value.old == old
        assert ei.value.new == new
        assert sm.state == old
        assert bus.events == []


@pytest.mark.parametrize("old,new", [(a, b) for a, b in ALL_PAIRS if b in ALLOWED_TRANSITIONS[a]])
def test_can_transition_has_no_side_effects(old, new):
    bus = RecordingBus()
    sm = AppStateMachine(bus, initial=old)
    sm.can_transition(new)
    assert sm.state == old
    assert bus.events == []


def test_default_detail_is_empty_string():
    bus = RecordingBus()
    sm = AppStateMachine(bus)
    sm.transition(AppState.LOADING_MODEL)
    assert bus.events == [StateChanged(AppState.DISARMED, AppState.LOADING_MODEL, "")]


def test_disarm_reachable_from_every_state():
    for s in AppState:
        if s is AppState.DISARMED:
            continue
        bus = RecordingBus()
        sm = AppStateMachine(bus, initial=s)
        assert sm.transition(AppState.DISARMED) is True
        assert sm.state == AppState.DISARMED


def test_error_reachable_from_every_state_except_error():
    for s in AppState:
        sm = AppStateMachine(RecordingBus(), initial=s)
        if s is AppState.ERROR:
            assert sm.transition(AppState.ERROR) is False
        else:
            assert sm.transition(AppState.ERROR, "boom") is True


# ---------------------------------------------------------------------------
# Spec-level scenarios
# ---------------------------------------------------------------------------


def _walk(sm, states):
    for s in states:
        assert sm.transition(s) is True, s


def test_startup_and_dictation_cycle_with_simple_bus():
    bus = SimpleEventBus()
    got: list = []
    bus.subscribe(got.append)
    sm = AppStateMachine(bus)
    path = [AppState.LOADING_MODEL, AppState.READY, AppState.LISTENING, AppState.PROCESSING, AppState.READY]
    _walk(sm, path)
    prev = [AppState.DISARMED] + path[:-1]
    assert got == [StateChanged(o, n, "") for o, n in zip(prev, path)]


def test_first_run_download_path():
    sm = AppStateMachine(RecordingBus())
    _walk(sm, [AppState.DOWNLOADING, AppState.LOADING_MODEL, AppState.READY])


def test_idle_unload_and_reload_on_hotkey():
    sm = AppStateMachine(RecordingBus(), initial=AppState.READY)
    _walk(sm, [AppState.UNLOADED, AppState.LOADING_MODEL, AppState.LISTENING, AppState.PROCESSING,
               AppState.READY])
    with pytest.raises(InvalidTransition):
        AppStateMachine(RecordingBus(), initial=AppState.UNLOADED).transition(AppState.LISTENING)


def test_hotkey_again_while_processing_starts_new_session():
    sm = AppStateMachine(RecordingBus(), initial=AppState.PROCESSING)
    assert sm.transition(AppState.LISTENING) is True


def test_error_recovery():
    sm = AppStateMachine(RecordingBus())
    _walk(sm, [AppState.LOADING_MODEL, AppState.ERROR, AppState.LOADING_MODEL, AppState.READY])
    _walk(sm, [AppState.LISTENING, AppState.ERROR])
    with pytest.raises(InvalidTransition):
        sm.transition(AppState.LISTENING)
    with pytest.raises(InvalidTransition):
        sm.transition(AppState.PROCESSING)
    assert sm.state == AppState.ERROR
    _walk(sm, [AppState.DISARMED, AppState.LOADING_MODEL])


def test_listening_cannot_jump_to_ready():
    sm = AppStateMachine(RecordingBus(), initial=AppState.LISTENING)
    with pytest.raises(InvalidTransition):
        sm.transition(AppState.READY)
    assert sm.state == AppState.LISTENING


# ---------------------------------------------------------------------------
# Locking / reentrancy
# ---------------------------------------------------------------------------


def test_subscriber_sees_new_state_and_can_reenter():
    bus = SimpleEventBus()
    sm = AppStateMachine(bus)
    seen: list = []

    def on_event(ev):
        if isinstance(ev, StateChanged):
            seen.append((ev.new, sm.state))
            if ev.new == AppState.LOADING_MODEL:
                sm.transition(AppState.READY, "nested")  # reentrant transition from the callback

    bus.subscribe(on_event)
    assert run_with_timeout(lambda: sm.transition(AppState.LOADING_MODEL)) is True
    assert sm.state == AppState.READY
    assert seen == [(AppState.LOADING_MODEL, AppState.LOADING_MODEL), (AppState.READY, AppState.READY)]


def test_event_published_after_lock_released():
    """A subscriber that waits for ANOTHER thread to read/transition the machine must not
    deadlock (catches publishing while holding a lock, even a reentrant one)."""
    bus = SimpleEventBus()
    sm = AppStateMachine(bus)
    other_results: list = []

    def on_event(ev):
        if isinstance(ev, StateChanged) and ev.new == AppState.LOADING_MODEL:
            def other():
                other_results.append(sm.state)
                other_results.append(sm.can_transition(AppState.READY))
                other_results.append(sm.transition(AppState.READY))

            t = threading.Thread(target=other, daemon=True)
            t.start()
            t.join(2.0)
            other_results.append("alive" if t.is_alive() else "done")

    bus.subscribe(on_event)
    run_with_timeout(lambda: sm.transition(AppState.LOADING_MODEL))
    assert other_results == [AppState.LOADING_MODEL, True, True, "done"]
    assert sm.state == AppState.READY


def test_concurrent_transitions_stay_consistent():
    bus = RecordingBus()
    sm = AppStateMachine(bus, initial=AppState.READY)
    cycle = [AppState.LISTENING, AppState.PROCESSING, AppState.READY]
    successes = Counter()
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker(offset):
        barrier.wait()
        n_ok = 0
        for i in range(300):
            target = cycle[(i + offset) % 3]
            try:
                if sm.transition(target):
                    n_ok += 1
            except InvalidTransition:
                pass
        with lock:
            successes["ok"] += n_ok

    threads = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT)
        assert not t.is_alive()

    events = bus.events
    assert len(events) == successes["ok"]
    for ev in events:
        assert isinstance(ev, StateChanged)
        assert ev.new in ALLOWED_TRANSITIONS[ev.old], ev
        assert ev.old != ev.new
    # The multiset of transitions must form one walk from READY to the final state:
    # out-degree - in-degree is +1 at the start, -1 at the end, 0 elsewhere (or all 0 if equal).
    balance = Counter()
    for ev in events:
        balance[ev.old] += 1
        balance[ev.new] -= 1
    final = sm.state
    expected = Counter()
    if final != AppState.READY:
        expected[AppState.READY] = 1
        expected[final] = -1
    assert {k: v for k, v in balance.items() if v} == {k: v for k, v in expected.items() if v}


# ---------------------------------------------------------------------------
# SimpleEventBus
# ---------------------------------------------------------------------------


def test_bus_delivers_in_subscription_order_on_publishing_thread():
    bus = SimpleEventBus()
    calls: list = []
    for name in "abc":
        bus.subscribe(lambda ev, n=name: calls.append((n, ev, threading.get_ident())))
    ev = Notification("t", "m")
    bus.publish(ev)
    me = threading.get_ident()
    assert calls == [("a", ev, me), ("b", ev, me), ("c", ev, me)]


def test_bus_publish_from_other_thread_runs_there():
    bus = SimpleEventBus()
    idents: list = []
    bus.subscribe(lambda ev: idents.append(threading.get_ident()))
    t = threading.Thread(target=lambda: bus.publish(Notification("t", "m")))
    t.start()
    t.join(TIMEOUT)
    assert idents == [t.ident]


def test_bus_subscriber_exception_isolated_and_logged(caplog):
    bus = SimpleEventBus()
    calls: list = []

    def bad(ev):
        raise ValueError("subscriber blew up")

    bus.subscribe(lambda ev: calls.append("first"))
    bus.subscribe(bad)
    bus.subscribe(lambda ev: calls.append("third"))
    with caplog.at_level(logging.DEBUG):
        bus.publish(Notification("t", "m"))  # must not raise
    assert calls == ["first", "third"]
    assert any(r.levelno >= logging.WARNING for r in caplog.records), "subscriber exception must be logged"


def test_bus_unsubscribe():
    bus = SimpleEventBus()
    calls: list = []
    unsub_a = bus.subscribe(lambda ev: calls.append("a"))
    bus.subscribe(lambda ev: calls.append("b"))
    assert callable(unsub_a)
    bus.publish(Notification("t", "1"))
    unsub_a()
    bus.publish(Notification("t", "2"))
    assert calls == ["a", "b", "b"]


def test_bus_publish_without_subscribers():
    SimpleEventBus().publish(Notification("t", "m"))


def test_bus_subscribe_from_other_threads():
    bus = SimpleEventBus()
    count = Counter()
    lock = threading.Lock()

    def cb(ev):
        with lock:
            count["n"] += 1

    threads = [threading.Thread(target=lambda: bus.subscribe(cb)) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT)
    bus.publish(Notification("t", "m"))
    assert count["n"] == 20


def test_state_machine_with_simple_bus_subscriber_failure_does_not_break_transition():
    bus = SimpleEventBus()
    good: list = []

    def bad(ev):
        raise RuntimeError("x")

    bus.subscribe(bad)
    bus.subscribe(good.append)
    sm = AppStateMachine(bus)
    assert sm.transition(AppState.LOADING_MODEL) is True
    assert good == [StateChanged(AppState.DISARMED, AppState.LOADING_MODEL, "")]
