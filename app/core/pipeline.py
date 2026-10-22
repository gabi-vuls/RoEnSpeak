"""Capture -> VAD -> inference -> injection pipeline (ARCHITECTURE §2).

Threads (all daemon, all joined by stop()):
  * the audio source's own thread calls feed_audio(), which only enqueues;
  * "core-segmenter": q_audio (chunks + BeginSession/Flush/Reset) -> Segmenter -> q_infer;
  * "core-inference": sole owner of the SpeechEngine. q_infer is a priority queue: engine
    commands (load/unload/shutdown) run before queued segments, segments and flush markers
    keep their FIFO order. Also runs the idle-unload timer;
  * "core-injection": q_inject -> TextInjector (or a result sink in CLI mode), logs
    LOG_SEGMENT_LATENCY, publishes SegmentProcessed, reports flush completion.

Discarding (disarm/quit) bumps an epoch. Every queued item carries the epoch it was created
in; stale items are skipped by every stage, so no queue has to be drained under a race.
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from app.contracts import (
    LOG_SEGMENT_LATENCY,
    AppError,
    AudioSegment,
    ComputeType,
    DecodeConfig,
    EngineInfo,
    EngineNotLoadedError,
    EventBus,
    HallucinationFilter,
    InjectionBlockedError,
    ModifiersHeldError,
    Notification,
    SegmentCloseReason,
    Segmenter,
    SegmentProcessed,
    SpeechEngine,
    Task,
    TextInjector,
    TranslationResult,
    VadConfig,
)

log = logging.getLogger("roenspeak.core.pipeline")


# -- q_audio commands ------------------------------------------------------


@dataclass(frozen=True)
class BeginSession:
    session_id: int


@dataclass(frozen=True)
class Flush:
    session_id: int


@dataclass(frozen=True)
class Reset:
    pass


# -- q_infer / q_inject items ---------------------------------------------


@dataclass(frozen=True)
class _FlushDone:
    session_id: int


@dataclass(frozen=True)
class _LoadCmd:
    model: str
    compute_type: ComputeType
    cpu_mode: bool
    decode: DecodeConfig
    token: Any
    task: Optional[Task] = None


@dataclass(frozen=True)
class _UnloadCmd:
    reason: str


@dataclass(frozen=True)
class _ShutdownCmd:
    unload: bool


@dataclass(frozen=True)
class _Processed:
    result: TranslationResult
    reason: SegmentCloseReason


_PRIO_SHUTDOWN, _PRIO_CMD, _PRIO_DATA = 0, 1, 2
_IDLE_RECHECK_S = 5.0


def _fmt_ms(value: Optional[float]) -> str:
    return "na" if value is None else f"{value:.1f}"


def format_latency_line(
    *,
    session_id: int,
    segment_id: int,
    reason: SegmentCloseReason,
    audio_s: float,
    infer_ms: float,
    latency_ms: Optional[float],
    chars: int,
    filtered: bool,
) -> str:
    """The LOG_SEGMENT_LATENCY line: "segment_latency session= id= reason= audio_ms= infer_ms=
    latency_ms= chars= filtered=". latency_ms is "na" when nothing was injected."""
    return (
        f"{LOG_SEGMENT_LATENCY} session={session_id} id={segment_id} reason={reason.value} "
        f"audio_ms={audio_s * 1000.0:.0f} infer_ms={infer_ms:.1f} latency_ms={_fmt_ms(latency_ms)} "
        f"chars={chars} filtered={1 if filtered else 0}"
    )


def _noop(*_a: Any, **_k: Any) -> None:
    return None


#: Notification title per InjectionBlockedError.reason. The message itself comes from
#: InjectionBlockedError.user_message, which contracts.py words per reason and platform.
_BLOCKED_TITLES: dict[str, str] = {
    # Windows
    "elevated": "Text not typed: administrator window",
    "unknown_integrity": "Text not typed: administrator window",
    "sendinput_partial": "Text not typed: input blocked",
    # macOS
    "no_accessibility": "Text not typed: permission needed",
    "secure_input": "Text not typed: password field",
    "cgevent_failed": "Text not typed: input blocked",
}
_BLOCKED_TITLE_DEFAULT = "Text not typed"

#: "sendinput_partial" is the one reason contracts.py has no specific wording for (it is a
#: SendInput return-value mismatch, not a named OS refusal), so the cause is spelled out here.
_SENDINPUT_PARTIAL_MESSAGE = (
    "Windows blocked the typed input (locked screen, secure desktop or an app that "
    "blocks simulated keys)."
)


def _blocked_message(exc: InjectionBlockedError) -> str:
    if exc.reason == "sendinput_partial":
        return _SENDINPUT_PARTIAL_MESSAGE
    return exc.user_message


class Pipeline:
    """Threads + queues between an AudioSource and a TextInjector.

    injector=None selects "sink" mode (CLI): nothing is typed, latency is
    t_infer_end - t_vad_close, and results are only delivered via on_segment_done.

    Callbacks run on pipeline threads and must not block for long:
      on_engine_loaded(info, token)       inference thread, after load + warmup succeeded
      on_engine_error(exc, phase, token)  inference thread; phase "load" | "process"
      on_engine_unloaded(reason)          inference thread ("idle", "request", "shutdown")
      idle_unload_check() -> bool         inference thread; True = go ahead and unload now
      on_session_flushed(session_id)      injection thread, after the last segment of a
                                          flushed session was handled (also when it had none)
      on_segment_done(event, result, reason)  injection thread, after each segment
    """

    def __init__(
        self,
        *,
        segmenter: Segmenter,
        engine: SpeechEngine,
        filt: HallucinationFilter,
        injector: Optional[TextInjector],
        bus: Optional[EventBus] = None,
        task: Task = Task.TRANSLATE,
        decode: DecodeConfig = DecodeConfig(),
        idle_unload_minutes: int = 0,
        on_engine_loaded: Callable[[EngineInfo, Any], None] = _noop,
        on_engine_error: Callable[[BaseException, str, Any], None] = _noop,
        on_engine_unloaded: Callable[[str], None] = _noop,
        idle_unload_check: Callable[[], bool] = lambda: True,
        on_session_flushed: Callable[[int], None] = _noop,
        on_segment_done: Callable[[SegmentProcessed, TranslationResult, SegmentCloseReason], None] = _noop,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._segmenter = segmenter
        self._engine = engine
        self._filter = filt
        self._injector = injector
        self._bus = bus
        self._settings: tuple[Task, DecodeConfig] = (Task(task), decode)
        self._idle_s = max(0, int(idle_unload_minutes)) * 60.0
        self._on_engine_loaded = on_engine_loaded
        self._on_engine_error = on_engine_error
        self._on_engine_unloaded = on_engine_unloaded
        self._idle_unload_check = idle_unload_check
        self._on_session_flushed = on_session_flushed
        self._on_segment_done = on_segment_done
        self._clock = clock

        self._epoch = 0
        self._epoch_lock = threading.Lock()
        self._seq = itertools.count()
        self._q_audio: "queue.Queue[Optional[tuple[int, Any]]]" = queue.Queue()
        self._q_infer: "queue.PriorityQueue[tuple[int, int, Any]]" = queue.PriorityQueue()
        self._q_inject: "queue.Queue[Optional[tuple[int, Any]]]" = queue.Queue()
        self._last_activity = clock()
        self._threads: list[threading.Thread] = []
        self._started = False
        self._stopped = False
        self._lifecycle_lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                return
            self._started = True
            self._threads = [
                threading.Thread(target=self._segmenter_loop, name="core-segmenter", daemon=True),
                threading.Thread(target=self._inference_loop, name="core-inference", daemon=True),
                threading.Thread(target=self._injection_loop, name="core-injection", daemon=True),
            ]
            for t in self._threads:
                t.start()
        log.info("pipeline started (%s mode)", "sink" if self._injector is None else "inject")

    def stop(self, *, unload: bool = True, timeout: float = 10.0) -> bool:
        """Discards pending work, stops all threads (unloading the engine on the inference
        thread when unload=True) and joins them. Returns True if every thread exited in time.
        Idempotent."""
        with self._lifecycle_lock:
            if self._stopped:
                return all(not t.is_alive() for t in self._threads)
            self._stopped = True
            started = self._started
        if not started:
            if unload:
                try:
                    self._engine.unload()
                except Exception:
                    log.exception("engine unload failed")
            return True
        self.discard()
        self._q_audio.put(None)
        self._q_infer.put((_PRIO_SHUTDOWN, next(self._seq), _ShutdownCmd(unload)))
        deadline = time.monotonic() + timeout
        ok = True
        # The injection thread is stopped after inference so nothing new reaches it.
        for t in self._threads:
            if t.name == "core-injection":
                continue
            t.join(max(0.0, deadline - time.monotonic()))
            ok = ok and not t.is_alive()
        self._q_inject.put(None)
        for t in self._threads:
            if t.name == "core-injection":
                t.join(max(0.0, deadline - time.monotonic()))
                ok = ok and not t.is_alive()
        if not ok:
            log.warning("pipeline stop: threads still alive after %.1fs: %s", timeout,
                        [t.name for t in self._threads if t.is_alive()])
        else:
            log.info("pipeline stopped")
        return ok

    # -- producer side (any thread) --------------------------------------

    @property
    def epoch(self) -> int:
        return self._epoch

    def feed_audio(self, chunk: np.ndarray) -> None:
        """AudioSource on_audio callback. Never blocks."""
        samples = np.array(chunk, dtype=np.float32, copy=True).reshape(-1)
        self._q_audio.put((self._epoch, samples))

    def begin_session(self, session_id: int) -> None:
        self._q_audio.put((self._epoch, BeginSession(session_id)))
        self.touch()

    def flush(self, session_id: int) -> None:
        """Emit the buffered speech of session_id; on_session_flushed(session_id) follows once
        every segment of it has been handled."""
        self._q_audio.put((self._epoch, Flush(session_id)))
        self.touch()

    def discard(self) -> None:
        """Drops every pending chunk/segment/result and resets the segmenter. An inference
        already running finishes, but its result is not injected."""
        with self._epoch_lock:
            self._epoch += 1
            epoch = self._epoch
        self._q_audio.put((epoch, Reset()))
        log.info("pipeline: pending work discarded (epoch %d)", epoch)

    def load_model(self, model: str, compute_type: ComputeType, cpu_mode: bool, decode: DecodeConfig,
                   token: Any = None, task: Optional[Task] = None) -> None:
        """Queues a load (+ warmup). When task is given it becomes the active task once the load
        succeeded, so a task change that needs a different model never runs on the old one."""
        self._q_infer.put((_PRIO_CMD, next(self._seq), _LoadCmd(model, ComputeType(compute_type), cpu_mode,
                                                                 decode, token, Task(task) if task else None)))

    def unload_model(self, reason: str = "request") -> None:
        self._q_infer.put((_PRIO_CMD, next(self._seq), _UnloadCmd(reason)))

    def set_decode_settings(self, task: Task, decode: DecodeConfig) -> None:
        self._settings = (Task(task), decode)

    @property
    def task(self) -> Task:
        return self._settings[0]

    def set_vad_config(self, cfg: VadConfig) -> None:
        setter = getattr(self._segmenter, "set_config", None)
        if setter is not None:
            setter(cfg)

    def set_idle_unload_minutes(self, minutes: int) -> None:
        self._idle_s = max(0, int(minutes)) * 60.0
        self.touch()
        # Wake the inference thread so it recomputes its timeout.
        self._q_infer.put((_PRIO_CMD, next(self._seq), None))

    def touch(self) -> None:
        """Marks activity (resets the idle-unload timer)."""
        self._last_activity = self._clock()

    # -- segmenter thread --------------------------------------------------

    def _put_data(self, epoch: int, item: Any) -> None:
        self._q_infer.put((_PRIO_DATA, next(self._seq), (epoch, item)))

    def _segmenter_loop(self) -> None:
        seg = self._segmenter
        while True:
            item = self._q_audio.get()
            if item is None:
                break
            epoch, obj = item
            if isinstance(obj, Reset):
                self._safe_reset()
                continue
            if epoch != self._epoch:
                continue
            try:
                if isinstance(obj, BeginSession):
                    seg.reset()
                    seg.begin_session(obj.session_id)
                    log.info("session %d: begin", obj.session_id)
                elif isinstance(obj, Flush):
                    segments = seg.flush()  # on failure the except below still sends the marker
                    for s in segments:
                        self._put_data(epoch, s)
                    log.info("session %d: flushed (%d final segment(s))", obj.session_id, len(segments))
                    self._put_data(epoch, _FlushDone(obj.session_id))
                else:
                    for s in seg.feed(obj):
                        self._put_data(epoch, s)
            except Exception:
                log.exception("segmenter failed on %s; state reset", type(obj).__name__)
                self._safe_reset()
                if isinstance(obj, Flush):
                    self._put_data(epoch, _FlushDone(obj.session_id))
        log.debug("segmenter thread exit")

    def _safe_reset(self) -> None:
        try:
            self._segmenter.reset()
        except Exception:
            log.exception("segmenter reset failed")

    # -- inference thread --------------------------------------------------

    def _inference_loop(self) -> None:
        while True:
            try:
                _prio, _seq, item = self._q_infer.get(timeout=self._idle_timeout())
            except queue.Empty:
                self._maybe_idle_unload()
                continue
            if isinstance(item, _ShutdownCmd):
                if item.unload:
                    self._unload("shutdown")
                break
            if item is None:  # wake-up only
                continue
            if isinstance(item, _LoadCmd):
                self._load(item)
                continue
            if isinstance(item, _UnloadCmd):
                self._unload(item.reason)
                continue
            epoch, obj = item
            if epoch != self._epoch:
                if isinstance(obj, AudioSegment):
                    log.info("segment %d (session %d) discarded", obj.id, obj.session_id)
                continue
            if isinstance(obj, _FlushDone):
                self._q_inject.put((epoch, obj))
                continue
            self._process(epoch, obj)
        log.debug("inference thread exit")

    def _idle_timeout(self) -> Optional[float]:
        if self._idle_s <= 0 or not self._engine.is_loaded:
            return None
        return max(0.05, self._last_activity + self._idle_s - self._clock())

    def _maybe_idle_unload(self) -> None:
        if self._idle_s <= 0 or not self._engine.is_loaded:
            return
        if self._clock() - self._last_activity < self._idle_s:
            return
        try:
            allowed = bool(self._idle_unload_check())
        except Exception:
            log.exception("idle_unload_check failed")
            allowed = False
        if allowed:
            log.info("idle for %.0f min; unloading model", self._idle_s / 60.0)
            self._unload("idle")
            self.touch()
        else:
            # Busy (listening/processing/disarmed...): look again shortly.
            self._last_activity = self._clock() - self._idle_s + _IDLE_RECHECK_S

    def _load(self, cmd: _LoadCmd) -> None:
        log.info("loading model %s (%s, cpu_mode=%s)", cmd.model, cmd.compute_type.value, cmd.cpu_mode)
        try:
            info = self._engine.load(cmd.model, cmd.compute_type, cmd.cpu_mode)
            t0 = time.perf_counter()
            self._engine.warmup(cmd.decode)
            log.info("warmup done in %.0f ms", (time.perf_counter() - t0) * 1000.0)
        except Exception as e:  # reported to core; never kills the thread
            log.error("model load failed: %s", getattr(e, "user_message", e), exc_info=not isinstance(e, AppError))
            self._safe_call(self._on_engine_error, e, "load", cmd.token)
            return
        finally:
            self.touch()
        if cmd.task is not None:
            self._settings = (cmd.task, self._settings[1])
        self._safe_call(self._on_engine_loaded, info, cmd.token)

    def _unload(self, reason: str) -> None:
        was_loaded = self._engine.is_loaded
        try:
            self._engine.unload()
        except Exception:
            log.exception("engine unload failed")
        if was_loaded:
            log.info("model unloaded (%s)", reason)
        self._safe_call(self._on_engine_unloaded, reason)

    def _process(self, epoch: int, seg: AudioSegment) -> None:
        task, decode = self._settings
        try:
            result = self._engine.process(seg, task, decode, self._filter)
        except EngineNotLoadedError as e:
            # Never a silent drop: core decides how to surface it (and reloads on the next press).
            log.error("segment %d (session %d) dropped: no model loaded", seg.id, seg.session_id)
            self._safe_call(self._on_engine_error, e, "process", None)
            return
        except Exception as e:
            log.error("inference failed on segment %d: %s", seg.id, getattr(e, "user_message", e),
                      exc_info=not isinstance(e, AppError))
            self._safe_call(self._on_engine_error, e, "process", None)
            return
        finally:
            self.touch()
        self._q_inject.put((epoch, _Processed(result, seg.reason)))

    # -- injection thread --------------------------------------------------

    def _injection_loop(self) -> None:
        while True:
            item = self._q_inject.get()
            if item is None:
                break
            epoch, obj = item
            if epoch != self._epoch:
                continue
            if isinstance(obj, _FlushDone):
                log.info("session %d: flush complete", obj.session_id)
                self._safe_call(self._on_session_flushed, obj.session_id)
                continue
            try:
                self._deliver(obj)
            except Exception:
                log.exception("injection stage failed on segment %d", obj.result.segment_id)
        log.debug("injection thread exit")

    def _deliver(self, p: _Processed) -> None:
        r = p.result
        injected = ""
        latency_ms: Optional[float] = None
        if self._injector is None:
            injected = r.text
            latency_ms = (r.t_infer_end - r.t_vad_close) * 1000.0
        elif not r.filtered and r.text.strip():
            try:
                outcome = self._injector.inject(r.text, r.session_id)
            except InjectionBlockedError as e:
                log.warning("injection blocked (%s) for segment %d", e.reason, r.segment_id)
                self._notify(_BLOCKED_TITLES.get(e.reason, _BLOCKED_TITLE_DEFAULT),
                             f"{_blocked_message(e)}\n\nText: {e.text}")
            except ModifiersHeldError as e:
                log.warning("injection skipped for segment %d: modifiers held", r.segment_id)
                self._notify("Text not typed: keys held down", f"{e.user_message}\n\nText: {e.text}")
            except AppError as e:
                log.warning("injection failed for segment %d: %s (%s)", r.segment_id, e.user_message, e.details)
                self._notify("Text not typed", f"{e.user_message}\n\nText: {r.text}")
            except Exception as e:
                log.exception("injection failed for segment %d", r.segment_id)
                self._notify("Text not typed", f"Typing failed: {e}\n\nText: {r.text}")
            else:
                if outcome is not None:
                    injected = outcome.injected_text
                    latency_ms = (outcome.t_injected - r.t_vad_close) * 1000.0

        log.info(format_latency_line(
            session_id=r.session_id, segment_id=r.segment_id, reason=p.reason, audio_s=r.audio_s,
            infer_ms=r.infer_ms, latency_ms=latency_ms, chars=len(injected), filtered=r.filtered,
        ))
        if r.filtered:
            log.info("segment %d filtered (%s)", r.segment_id, r.filter_reason)
        event = SegmentProcessed(
            session_id=r.session_id, segment_id=r.segment_id, text="" if r.filtered else injected,
            filtered=r.filtered, filter_reason=r.filter_reason, infer_ms=r.infer_ms, latency_ms=latency_ms,
        )
        if self._bus is not None:
            self._bus.publish(event)
        self._safe_call(self._on_segment_done, event, r, p.reason)

    def _notify(self, title: str, message: str) -> None:
        if self._bus is not None:
            self._bus.publish(Notification(title=title, message=message, warning=True))

    @staticmethod
    def _safe_call(fn: Callable[..., Any], *args: Any) -> Any:
        try:
            return fn(*args)
        except Exception:
            log.exception("pipeline callback %r failed", fn)
            return None
