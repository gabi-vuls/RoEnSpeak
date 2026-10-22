"""Streaming VAD segmenter. Semantics are specified in app.contracts.Segmenter."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

from app.contracts import (
    SAMPLE_RATE,
    VAD_FRAME_SAMPLES,
    AudioSegment,
    SegmentCloseReason,
    VadConfig,
    VadModel,
)

log = logging.getLogger("roenspeak.audio.segmenter")

F = VAD_FRAME_SAMPLES
FRAME_MS = F * 1000 // SAMPLE_RATE  # 32
#: History always kept before the current position while idle (max valid speech_pad_ms).
MIN_HISTORY_SAMPLES = SAMPLE_RATE


class _Buffer:
    """Growable float32 buffer addressed by absolute stream sample index."""

    def __init__(self) -> None:
        self._data = np.zeros(F * 64, dtype=np.float32)
        self.start = 0  # absolute index of _data[0]
        self.end = 0  # absolute index one past the last received sample

    def append(self, samples: np.ndarray) -> None:
        n = samples.shape[0]
        used = self.end - self.start
        if used + n > self._data.shape[0]:
            cap = self._data.shape[0]
            while cap < used + n:
                cap *= 2
            grown = np.zeros(cap, dtype=np.float32)
            grown[:used] = self._data[:used]
            self._data = grown
        self._data[used:used + n] = samples
        self.end += n

    def view(self, a: int, b: int) -> np.ndarray:
        return self._data[a - self.start:b - self.start]

    def copy(self, a: int, b: int) -> np.ndarray:
        a = max(a, self.start)
        b = min(b, self.end)
        return self._data[a - self.start:b - self.start].copy()

    def discard_before(self, pos: int) -> None:
        pos = min(max(pos, self.start), self.end)
        if pos == self.start:
            return
        used = self.end - pos
        self._data[:used] = self._data[pos - self.start:self.end - self.start]
        self.start = pos

    def clear(self) -> None:
        self.start = 0
        self.end = 0


class VadSegmenter:
    """Segmenter implementation. Owned by one thread, except set_config (any thread)."""

    def __init__(self, cfg: VadConfig, vad: VadModel, clock: Callable[[], float] = time.perf_counter) -> None:
        self._cfg = cfg
        self._pending_cfg: Optional[VadConfig] = None
        self._cfg_lock = threading.Lock()
        self._vad = vad
        self._clock = clock
        self._next_id = 1
        self._session_id = 0
        self._buf = _Buffer()
        self._reset_state()

    # -- public API -------------------------------------------------------

    def set_config(self, cfg: VadConfig) -> None:
        with self._cfg_lock:
            self._pending_cfg = cfg

    def begin_session(self, session_id: int) -> None:
        self._session_id = session_id

    def feed(self, samples: np.ndarray) -> list[AudioSegment]:
        self._apply_pending_config()
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        out: list[AudioSegment] = []
        if samples.shape[0]:
            self._buf.append(samples)
        while self._pos + F <= self._buf.end:
            frame_start = self._pos
            prob = float(self._vad.speech_prob(self._buf.view(frame_start, frame_start + F)))
            self._pos = frame_start + F
            seg = self._step(frame_start, prob)
            if seg is not None:
                out.append(seg)
        self._trim()
        return out

    def flush(self) -> list[AudioSegment]:
        out: list[AudioSegment] = []
        if self._in_speech:
            seg = self._close(SegmentCloseReason.FLUSH, self._audio_start(), self._buf.end)
            if seg is not None:
                out.append(seg)
        self.reset()
        return out

    def reset(self) -> None:
        self._buf.clear()
        self._reset_state()
        self._vad.reset()

    # -- internals --------------------------------------------------------

    def _reset_state(self) -> None:
        self._pos = 0  # absolute index one past the last VAD-processed frame
        self._in_speech = False
        self._speech_start = 0
        self._speech_end = 0
        self._pre_pad = True  # False for the continuation after a MAX_LENGTH cut
        self._silence_frames = 0

    def _apply_pending_config(self) -> None:
        with self._cfg_lock:
            if self._pending_cfg is not None:
                self._cfg = self._pending_cfg
                self._pending_cfg = None

    def _pad_samples(self) -> int:
        return self._cfg.speech_pad_ms * SAMPLE_RATE // 1000

    def _audio_start(self) -> int:
        if not self._pre_pad:
            return self._speech_start
        return max(0, self._speech_start - self._pad_samples())

    def _step(self, frame_start: int, prob: float) -> Optional[AudioSegment]:
        cfg = self._cfg
        if not self._in_speech:
            if prob < cfg.threshold:
                return None
            self._in_speech = True
            self._speech_start = frame_start
            self._speech_end = self._pos
            self._pre_pad = True
            self._silence_frames = 0
        elif prob < cfg.threshold - cfg.neg_threshold_offset:
            self._silence_frames += 1
            if self._silence_frames * FRAME_MS >= cfg.min_silence_ms:
                start = self._audio_start()
                end = min(self._speech_end + self._pad_samples(), self._buf.end)
                seg = self._close(SegmentCloseReason.SILENCE, start, end)
                self._in_speech = False
                self._silence_frames = 0
                return seg
        else:
            self._silence_frames = 0
            self._speech_end = self._pos

        max_samples = int(round(cfg.max_segment_s * SAMPLE_RATE))
        if self._pos - self._speech_start >= max_samples:
            seg = self._close(SegmentCloseReason.MAX_LENGTH, self._audio_start(), self._pos)
            self._speech_start = self._pos
            self._speech_end = self._pos
            self._pre_pad = False
            return seg
        return None

    def _close(self, reason: SegmentCloseReason, start: int, end: int) -> Optional[AudioSegment]:
        speech_samples = self._speech_end - self._speech_start
        if speech_samples * 1000 < self._cfg.min_speech_ms * SAMPLE_RATE:
            log.debug(
                "segment discarded reason=%s speech_ms=%.0f",
                reason.value, speech_samples * 1000 / SAMPLE_RATE,
            )
            return None
        seg = AudioSegment(
            id=self._next_id,
            session_id=self._session_id,
            samples=self._buf.copy(start, end),
            reason=reason,
            speech_start_s=self._speech_start / SAMPLE_RATE,
            speech_end_s=self._speech_end / SAMPLE_RATE,
            t_vad_close=self._clock(),
        )
        self._next_id += 1
        log.debug(
            "segment closed id=%d session=%d reason=%s audio_ms=%.0f speech=%.3f..%.3f",
            seg.id, seg.session_id, reason.value, seg.duration_s * 1000, seg.speech_start_s, seg.speech_end_s,
        )
        return seg

    def _trim(self) -> None:
        if self._in_speech:
            keep_from = self._audio_start()
        else:
            keep_from = self._pos - max(self._pad_samples(), MIN_HISTORY_SAMPLES)
        self._buf.discard_before(keep_from)
