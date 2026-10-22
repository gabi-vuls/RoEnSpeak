"""Spec/contract tests for app.audio.segmenter.VadSegmenter.

Written from app/contracts.py (Segmenter, VadConfig, AudioSegment docstrings) only.

Technique: the audio is a "ramp" whose sample i has the exact float32 value i / 2**20, so
every sample identifies its own stream position. The fake VadModel derives the frame index
from the frame's first sample and returns a scripted probability for that frame. Segment
boundaries are then checked sample-exactly by comparing ``segment.samples`` with the ramp
slice the contract defines. Segment sample ranges are half-open: [start, end).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.contracts import (
    SAMPLE_RATE,
    VAD_FRAME_SAMPLES,
    AudioSegment,
    SegmentCloseReason,
    Segmenter,
    VadConfig,
    VadModel,
)
from app.audio.segmenter import VadSegmenter

F = VAD_FRAME_SAMPLES  # 512 samples = 32 ms
SCALE = float(2**20)  # exact in float32 for indices < 2**24
SPEECH = 0.9
SIL = 0.05
PAD = 200 * SAMPLE_RATE // 1000  # default speech_pad_ms = 200 -> 3200 samples
T0 = 1000.0


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def ramp(n: int, start: int = 0) -> np.ndarray:
    return (np.arange(start, start + n, dtype=np.float64) / SCALE).astype(np.float32)


def sample_index(v) -> int:
    return int(round(float(v) * SCALE))


def script(*runs: tuple[int, float]) -> list[float]:
    """script((10, SIL), (20, SPEECH), ...) -> per-frame probability list."""
    out: list[float] = []
    for n, p in runs:
        out.extend([p] * n)
    return out


class FakeVad:
    """Scripted VadModel. The probability is chosen by the absolute frame index encoded in
    the frame's samples, so results do not depend on how audio was chunked."""

    def __init__(self, probs, default: float = SIL) -> None:
        self.probs = list(probs)
        self.default = default
        self.frames: list[int] = []
        self.reset_calls = 0
        self.errors: list[str] = []

    def speech_prob(self, frame: np.ndarray) -> float:
        arr = np.asarray(frame)
        if arr.shape != (F,):
            self.errors.append(f"frame shape {arr.shape} != ({F},)")
            raise AssertionError(self.errors[-1])
        start = sample_index(arr[0])
        if start % F != 0 or not np.array_equal(arr.astype(np.float32), ramp(F, start)):
            self.errors.append(f"frame starting at sample {start} is not a contiguous, aligned 512-sample slice")
            raise AssertionError(self.errors[-1])
        idx = start // F
        self.frames.append(idx)
        return float(self.probs[idx]) if idx < len(self.probs) else self.default

    def reset(self) -> None:
        self.reset_calls += 1


class FakeClock:
    def __init__(self, t: float = T0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def make(probs, cfg: VadConfig | None = None, clock: FakeClock | None = None):
    vad = FakeVad(probs)
    clock = clock or FakeClock()
    seg = VadSegmenter(cfg or VadConfig(), vad, clock=clock)
    return seg, vad, clock


def feed_frames(seg, audio: np.ndarray, clock: FakeClock | None = None, first_frame: int = 0):
    """Feed one frame per feed() call. Before feeding frame i, clock = T0 + i.
    Returns [(frame_index_whose_feed_returned_it, segment), ...]."""
    out = []
    n = len(audio) // F
    for i in range(n):
        if clock is not None:
            clock.t = T0 + first_frame + i
        for s in seg.feed(audio[i * F:(i + 1) * F]):
            out.append((first_frame + i, s))
    rem = audio[n * F:]
    if len(rem):
        if clock is not None:
            clock.t = T0 + first_frame + n
        for s in seg.feed(rem):
            out.append((first_frame + n, s))
    return out


def assert_span(s: AudioSegment, start: int, end: int) -> None:
    assert isinstance(s.samples, np.ndarray)
    assert s.samples.dtype == np.float32
    got_start = sample_index(s.samples[0]) if len(s.samples) else None
    assert len(s.samples) == end - start, (
        f"expected samples [{start}, {end}) (len {end - start}), got len {len(s.samples)} starting at {got_start}"
    )
    np.testing.assert_array_equal(s.samples, ramp(end - start, start))


def frames_needed(min_silence_ms: int) -> int:
    return math.ceil(min_silence_ms / 32)


# ---------------------------------------------------------------------------
# Construction / protocol
# ---------------------------------------------------------------------------


def test_is_a_segmenter():
    seg, _, _ = make([])
    assert isinstance(seg, Segmenter)
    assert isinstance(FakeVad([]), VadModel)


def test_runs_vad_on_every_complete_frame_and_buffers_partial():
    seg, vad, _ = make([])
    audio = ramp(4 * F)
    assert seg.feed(audio[: 3 * F + 100]) == []
    assert vad.frames == [0, 1, 2]
    assert seg.feed(audio[3 * F + 100: 4 * F - 1]) == []
    assert vad.frames == [0, 1, 2]  # still partial
    seg.feed(audio[4 * F - 1:])
    assert vad.frames == [0, 1, 2, 3]
    assert vad.errors == []


def test_empty_feed_is_harmless():
    seg, vad, _ = make([])
    assert seg.feed(np.zeros(0, dtype=np.float32)) == []
    assert vad.frames == []


# ---------------------------------------------------------------------------
# Speech start / threshold / hysteresis
# ---------------------------------------------------------------------------


def test_idle_below_threshold_never_starts_speech():
    seg, vad, _ = make(script((80, 0.49)))
    assert feed_frames(seg, ramp(80 * F)) == []
    assert seg.flush() == []
    assert vad.errors == []


def test_speech_starts_at_prob_equal_threshold_at_frame_first_sample():
    # frames 20..29 have prob exactly 0.5 (threshold), then silence.
    seg, vad, clock = make(script((20, SIL), (10, 0.5), (40, SIL)))
    out = feed_frames(seg, ramp(70 * F), clock)
    assert len(out) == 1
    _, s = out[0]
    assert s.speech_start_s == pytest.approx(20 * F / SAMPLE_RATE)
    assert s.speech_end_s == pytest.approx(30 * F / SAMPLE_RATE)
    assert_span(s, 20 * F - PAD, 30 * F + PAD)


def test_prob_between_neg_threshold_and_threshold_does_not_start_speech_in_idle():
    # 0.4 is >= threshold - offset (0.35) but < threshold (0.5): idle stays idle.
    seg, _, _ = make(script((40, 0.4)))
    assert feed_frames(seg, ramp(40 * F)) == []
    assert seg.flush() == []


def test_hysteresis_keeps_speech_and_extends_end():
    # idle 0..9 at 0.4 (no start); speech 10..19; 20..29 at 0.36 (>= 0.35 -> speech);
    # 30.. at 0.34 (< 0.35 -> silence). Closes on the 16th silence frame (index 45).
    probs = script((10, 0.4), (10, SPEECH), (10, 0.36), (30, 0.34))
    seg, vad, clock = make(probs)
    out = feed_frames(seg, ramp(60 * F), clock)
    assert [(i, s.reason) for i, s in out] == [(45, SegmentCloseReason.SILENCE)]
    s = out[0][1]
    assert s.speech_start_s == pytest.approx(10 * F / SAMPLE_RATE)
    assert s.speech_end_s == pytest.approx(30 * F / SAMPLE_RATE)
    assert_span(s, 10 * F - PAD, 30 * F + PAD)
    assert s.t_vad_close == T0 + 45


def test_speech_frame_resets_silence_counter():
    # speech 10..19, 15 silence frames (20..34, not enough), speech 35..39, then silence.
    probs = script((10, SIL), (10, SPEECH), (15, SIL), (5, SPEECH), (30, SIL))
    seg, _, clock = make(probs)
    out = feed_frames(seg, ramp(70 * F), clock)
    assert len(out) == 1
    i, s = out[0]
    assert i == 39 + 16
    assert s.reason == SegmentCloseReason.SILENCE
    assert s.speech_start_s == pytest.approx(10 * F / SAMPLE_RATE)
    assert s.speech_end_s == pytest.approx(40 * F / SAMPLE_RATE)
    assert_span(s, 10 * F - PAD, 40 * F + PAD)


# ---------------------------------------------------------------------------
# Silence close
# ---------------------------------------------------------------------------


def test_default_500ms_silence_closes_on_16th_frame_not_15th():
    probs = script((20, SIL), (10, SPEECH), (40, SIL))
    seg, vad, clock = make(probs)
    audio = ramp(70 * F)
    # speech frames 20..29, then 15 silence frames 30..44: must NOT close (15*32=480 < 500)
    assert feed_frames(seg, audio[: 45 * F], clock) == []
    clock.t = 4242.5
    out = seg.feed(audio[45 * F: 46 * F])  # 16th silence frame: 512 ms >= 500 ms
    assert len(out) == 1
    s = out[0]
    assert s.reason == SegmentCloseReason.SILENCE
    assert s.id == 1
    assert s.session_id == 0
    assert s.t_vad_close == 4242.5
    assert s.speech_start_s == pytest.approx(20 * F / SAMPLE_RATE)
    assert s.speech_end_s == pytest.approx(30 * F / SAMPLE_RATE)
    assert_span(s, 20 * F - PAD, 30 * F + PAD)
    assert s.duration_s == pytest.approx((10 * F + 2 * PAD) / SAMPLE_RATE)
    # nothing else afterwards
    assert feed_frames(seg, audio[46 * F:]) == []
    assert vad.errors == []


@pytest.mark.parametrize("min_silence_ms,n_frames", [(100, 4), (300, 10), (500, 16), (512, 16), (513, 17), (1000, 32)])
def test_configurable_min_silence_closes_after_whole_frames(min_silence_ms, n_frames):
    assert frames_needed(min_silence_ms) == n_frames
    cfg = VadConfig(min_silence_ms=min_silence_ms, speech_pad_ms=0)
    probs = script((5, SIL), (10, SPEECH), (60, SIL))
    seg, _, clock = make(probs, cfg)
    out = feed_frames(seg, ramp(75 * F), clock)
    assert len(out) == 1
    i, s = out[0]
    assert i == 14 + n_frames, f"closed after {i - 14} silence frames, expected {n_frames}"
    assert s.reason == SegmentCloseReason.SILENCE
    assert s.t_vad_close == T0 + i
    assert_span(s, 5 * F, 15 * F)


def test_multiple_segments_each_closed_by_silence():
    probs = script((5, SIL), (10, SPEECH), (20, SIL), (12, SPEECH), (20, SIL))
    seg, _, clock = make(probs)
    out = feed_frames(seg, ramp(67 * F), clock)
    assert [(i, s.reason, s.id) for i, s in out] == [
        (14 + 16, SegmentCloseReason.SILENCE, 1),
        (46 + 16, SegmentCloseReason.SILENCE, 2),
    ]
    assert_span(out[0][1], 5 * F - PAD if 5 * F >= PAD else 0, 15 * F + PAD)
    assert_span(out[1][1], 35 * F - PAD, 47 * F + PAD)


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


def test_pre_pad_clipped_when_speech_starts_at_sample_zero():
    probs = script((10, SPEECH), (20, SIL))
    seg, _, _ = make(probs)
    out = feed_frames(seg, ramp(30 * F))
    assert len(out) == 1
    s = out[0][1]
    assert s.speech_start_s == 0.0
    assert_span(s, 0, 10 * F + PAD)


def test_pre_pad_partially_clipped():
    # speech at frame 3 = sample 1536 < PAD -> starts at sample 0
    probs = script((3, SIL), (10, SPEECH), (20, SIL))
    seg, _, _ = make(probs)
    s = feed_frames(seg, ramp(33 * F))[0][1]
    assert_span(s, 0, 13 * F + PAD)


def test_pre_roll_history_kept_across_many_chunks():
    probs = script((50, SIL), (10, SPEECH), (20, SIL))
    seg, vad, _ = make(probs)
    audio = ramp(80 * F)
    out = []
    for k in range(0, len(audio), 1000):
        out.extend(seg.feed(audio[k:k + 1000]))
    assert len(out) == 1
    assert_span(out[0], 50 * F - PAD, 60 * F + PAD)
    assert vad.errors == []


def test_post_pad_clipped_to_received_audio_when_silence_shorter_than_pad():
    # min_silence 100 ms -> closes after 4 frames (2048 samples) < pad 3200 samples:
    # the segment can only extend to the audio received so far.
    cfg = VadConfig(min_silence_ms=100, speech_pad_ms=200)
    probs = script((20, SIL), (10, SPEECH), (20, SIL))
    seg, _, _ = make(probs, cfg)
    out = feed_frames(seg, ramp(50 * F))
    assert len(out) == 1
    i, s = out[0]
    assert i == 33
    assert_span(s, 20 * F - PAD, 34 * F)


@pytest.mark.parametrize("pad_ms", [0, 64, 100, 1000])
def test_configurable_pad(pad_ms):
    pad = pad_ms * SAMPLE_RATE // 1000
    cfg = VadConfig(speech_pad_ms=pad_ms, min_silence_ms=1000)  # 32 frames of silence > 1000 ms pad
    probs = script((70, SIL), (10, SPEECH), (40, SIL))
    seg, _, _ = make(probs, cfg)
    out = feed_frames(seg, ramp(120 * F))
    assert len(out) == 1
    assert_span(out[0][1], 70 * F - pad, 80 * F + pad)


# ---------------------------------------------------------------------------
# min_speech_ms discard
# ---------------------------------------------------------------------------


def test_short_speech_discarded_and_consumes_no_id():
    # 7 frames = 224 ms < 250 ms -> discarded; then 8 frames = 256 ms -> kept with id 1.
    probs = script((10, SIL), (7, SPEECH), (30, SIL), (8, SPEECH), (30, SIL))
    seg, _, clock = make(probs)
    out = feed_frames(seg, ramp(85 * F), clock)
    assert len(out) == 1
    i, s = out[0]
    assert s.id == 1
    assert s.speech_start_s == pytest.approx(47 * F / SAMPLE_RATE)
    assert_span(s, 47 * F - PAD, 55 * F + PAD)


@pytest.mark.parametrize("min_speech_ms,kept", [(256, True), (257, False), (0, True)])
def test_min_speech_boundary(min_speech_ms, kept):
    # speech duration is exactly 8 frames = 256 ms; "below min_speech_ms" -> discarded
    cfg = VadConfig(min_speech_ms=min_speech_ms)
    probs = script((10, SIL), (8, SPEECH), (30, SIL))
    seg, _, _ = make(probs, cfg)
    out = feed_frames(seg, ramp(48 * F))
    assert len(out) == (1 if kept else 0)


def test_min_speech_zero_keeps_single_frame():
    cfg = VadConfig(min_speech_ms=0, speech_pad_ms=0)
    probs = script((10, SIL), (1, SPEECH), (30, SIL))
    seg, _, _ = make(probs, cfg)
    out = feed_frames(seg, ramp(41 * F))
    assert len(out) == 1
    assert_span(out[0][1], 10 * F, 11 * F)


def test_hysteresis_frames_count_as_speech_for_min_speech():
    # 3 strong + 5 hysteresis frames = 8 frames = 256 ms >= 250 -> kept
    probs = script((10, SIL), (3, SPEECH), (5, 0.4), (30, SIL))
    seg, _, _ = make(probs)
    out = feed_frames(seg, ramp(48 * F))
    assert len(out) == 1
    assert out[0][1].speech_end_s == pytest.approx(18 * F / SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Max-length cut
# ---------------------------------------------------------------------------


def test_default_max_length_cut_at_15s_no_overlap_then_silence_close():
    n_speech = 550  # 17.6 s of continuous speech starting at sample 0
    probs = script((n_speech, SPEECH), (30, SIL))
    seg, vad, clock = make(probs)
    audio = ramp((n_speech + 30) * F)
    out = feed_frames(seg, audio, clock)
    assert [s.reason for _, s in out] == [SegmentCloseReason.MAX_LENGTH, SegmentCloseReason.SILENCE]
    (i1, s1), (i2, s2) = out
    # cut at the first frame end where position - start >= 15 s = 240000 samples -> 469 frames
    max_samples = 15 * SAMPLE_RATE
    assert max_samples <= len(s1.samples) < max_samples + F
    cut = 469 * F
    assert i1 == 468
    assert s1.t_vad_close == T0 + 468
    assert_span(s1, 0, cut)
    assert s1.speech_start_s == 0.0
    # next segment starts exactly at the cut, without pre-pad, state stayed "speech"
    assert s2.speech_start_s == pytest.approx(cut / SAMPLE_RATE)
    assert s2.speech_end_s == pytest.approx(n_speech * F / SAMPLE_RATE)
    assert_span(s2, cut, n_speech * F + PAD)
    assert i2 == n_speech - 1 + 16
    assert (s1.id, s2.id) == (1, 2)
    # tiling: no duplicated or missing samples
    np.testing.assert_array_equal(np.concatenate([s1.samples, s2.samples]), audio[: n_speech * F + PAD])
    assert vad.errors == []


def test_max_length_repeated_cuts_tile_audio_and_flush_remainder():
    # 2.04 s = 32640 samples = 63.75 frames -> cut after 64 frames; speech from sample 0 for 200 frames.
    cfg = VadConfig(max_segment_s=2.04)
    probs = script((200, SPEECH))
    seg, _, clock = make(probs, cfg)
    audio = ramp(200 * F + 77)
    out = feed_frames(seg, audio, clock)
    assert [i for i, _ in out] == [63, 127, 191]
    assert all(s.reason == SegmentCloseReason.MAX_LENGTH for _, s in out)
    for k, (_, s) in enumerate(out):
        assert_span(s, k * 64 * F, (k + 1) * 64 * F)
        assert s.speech_start_s == pytest.approx(k * 64 * F / SAMPLE_RATE)
    clock.t = 9999.0
    tail = seg.flush()
    assert len(tail) == 1
    assert tail[0].reason == SegmentCloseReason.FLUSH
    assert tail[0].t_vad_close == 9999.0
    assert_span(tail[0], 192 * F, 200 * F + 77)
    segs = [s for _, s in out] + tail
    assert [s.id for s in segs] == [1, 2, 3, 4]
    np.testing.assert_array_equal(np.concatenate([s.samples for s in segs]), audio)


def test_max_length_configured_5s():
    cfg = VadConfig(max_segment_s=5.0)
    probs = script((200, SPEECH), (30, SIL))
    seg, _, _ = make(probs, cfg)
    audio = ramp(230 * F)
    out = feed_frames(seg, audio)
    assert out[0][1].reason == SegmentCloseReason.MAX_LENGTH
    n = len(out[0][1].samples)
    assert 5 * SAMPLE_RATE <= n < 5 * SAMPLE_RATE + F
    assert n % F == 0
    assert_span(out[0][1], 0, n)
    # following segment starts where the first ended
    assert sample_index(out[1][1].samples[0]) == n


def test_max_length_with_pre_pad_no_overlap():
    # Speech starts at frame 20 (pre-pad available). Exact cut position depends on whether
    # "segment start" includes the pre-pad (flagged ambiguity), so it is checked loosely;
    # the no-overlap / tiling / no-pre-pad-on-continuation rules are checked exactly.
    cfg = VadConfig(max_segment_s=2.04)
    probs = script((20, SIL), (100, SPEECH), (30, SIL))
    seg, _, _ = make(probs, cfg)
    audio = ramp(150 * F)
    out = feed_frames(seg, audio)
    reasons = [s.reason for _, s in out]
    assert reasons[0] == SegmentCloseReason.MAX_LENGTH
    assert reasons[-1] == SegmentCloseReason.SILENCE
    s1 = out[0][1]
    start1 = 20 * F - PAD
    assert sample_index(s1.samples[0]) == start1
    cut = start1 + len(s1.samples)
    assert cut % F == 0, "cut happens at a frame boundary (current position)"
    assert 2.04 * SAMPLE_RATE - PAD - F < len(s1.samples) <= 2.04 * SAMPLE_RATE + PAD + F
    segs = [s for _, s in out]
    for a, b in zip(segs, segs[1:]):
        end_a = sample_index(a.samples[0]) + len(a.samples)
        assert sample_index(b.samples[0]) == end_a  # contiguous: no overlap, no gap, no pre-pad
        assert b.speech_start_s == pytest.approx(end_a / SAMPLE_RATE)
    np.testing.assert_array_equal(np.concatenate([s.samples for s in segs]), audio[start1: 120 * F + PAD])


def test_after_max_length_cut_tail_closes_on_silence():
    # After a cut the state stays "speech": a short tail is closed by silence normally and
    # the tail's speech_start is the cut position.
    cfg = VadConfig(max_segment_s=2.04, speech_pad_ms=0)
    probs = script((64 + 10, SPEECH), (30, SIL))
    seg, _, _ = make(probs, cfg)
    out = feed_frames(seg, ramp(104 * F))
    assert [s.reason for _, s in out] == [SegmentCloseReason.MAX_LENGTH, SegmentCloseReason.SILENCE]
    assert_span(out[1][1], 64 * F, 74 * F)
    assert out[1][1].speech_start_s == pytest.approx(64 * F / SAMPLE_RATE)


# ---------------------------------------------------------------------------
# flush()
# ---------------------------------------------------------------------------


def test_flush_in_speech_emits_flush_segment_with_partial_frame():
    probs = script((20, SIL), (10, SPEECH), (5, SIL))
    seg, vad, clock = make(probs)
    audio = ramp(35 * F + 100)
    assert feed_frames(seg, audio, clock) == []
    resets_before = vad.reset_calls
    clock.t = 777.25
    out = seg.flush()
    assert len(out) == 1
    s = out[0]
    assert s.reason == SegmentCloseReason.FLUSH
    assert s.id == 1
    assert s.t_vad_close == 777.25
    assert s.speech_start_s == pytest.approx(20 * F / SAMPLE_RATE)
    assert s.speech_end_s == pytest.approx(30 * F / SAMPLE_RATE)
    assert_span(s, 20 * F - PAD, 35 * F + 100)
    assert vad.reset_calls == resets_before + 1
    # flush behaves like reset afterwards
    assert seg.flush() == []


def test_flush_in_speech_without_trailing_silence():
    probs = script((10, SIL), (20, SPEECH))
    seg, _, _ = make(probs)
    feed_frames(seg, ramp(30 * F + 3))
    out = seg.flush()
    assert len(out) == 1
    assert_span(out[0], 10 * F - PAD, 30 * F + 3)


def test_flush_in_idle_returns_empty_resets_and_drops_partial_frame():
    seg, vad, _ = make(script((10, SIL)))
    seg.feed(ramp(300))  # partial frame, idle
    assert seg.flush() == []
    assert vad.reset_calls == 1
    # the 300 buffered samples were dropped: the next frames start at the new audio
    seg.feed(ramp(2 * F))
    assert vad.frames == [0, 1]
    assert vad.errors == []


def test_flush_discards_short_speech_and_consumes_no_id():
    probs = script((10, SIL), (5, SPEECH), (3, SIL))  # 160 ms < 250 ms
    seg, vad, _ = make(probs)
    feed_frames(seg, ramp(18 * F))
    assert seg.flush() == []
    assert vad.reset_calls == 1
    probs2 = script((10, SPEECH), (20, SIL))
    vad.probs = probs2
    out = feed_frames(seg, ramp(30 * F))
    assert [s.id for _, s in out] == [1]


def test_flush_clears_state_so_next_stream_is_fresh():
    probs = script((10, SIL), (10, SPEECH))
    seg, vad, _ = make(probs)
    feed_frames(seg, ramp(20 * F + 50))
    assert len(seg.flush()) == 1
    # New stream: silence only -> nothing, the previous speech must not linger.
    vad.probs = script((40, SIL))
    assert feed_frames(seg, ramp(40 * F)) == []
    assert seg.flush() == []
    assert vad.errors == []


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_drops_speech_in_progress_and_calls_vad_reset():
    seg, vad, _ = make(script((10, SPEECH)))
    feed_frames(seg, ramp(10 * F + 200))
    seg.reset()
    assert vad.reset_calls == 1
    vad.probs = script((40, SIL))
    assert feed_frames(seg, ramp(40 * F)) == []  # also checks frame alignment after reset
    assert seg.flush() == []
    assert vad.errors == []


def test_times_are_relative_to_last_reset():
    seg, vad, _ = make(script((50, SIL)))
    feed_frames(seg, ramp(50 * F))
    seg.reset()
    vad.probs = script((5, SIL), (10, SPEECH), (20, SIL))
    out = feed_frames(seg, ramp(35 * F))
    assert len(out) == 1
    assert out[0][1].speech_start_s == pytest.approx(5 * F / SAMPLE_RATE)
    assert out[0][1].speech_end_s == pytest.approx(15 * F / SAMPLE_RATE)


def test_ids_strictly_increase_and_survive_reset_and_flush():
    probs = script((10, SIL), (10, SPEECH), (20, SIL))
    seg, vad, _ = make(probs)
    ids = []
    ids += [s.id for _, s in feed_frames(seg, ramp(40 * F))]
    seg.reset()
    ids += [s.id for _, s in feed_frames(seg, ramp(40 * F))]
    vad.probs = script((10, SIL), (10, SPEECH))
    feed_frames(seg, ramp(20 * F))
    ids += [s.id for s in seg.flush()]
    vad.probs = script((10, SIL), (10, SPEECH), (20, SIL))
    ids += [s.id for _, s in feed_frames(seg, ramp(40 * F))]
    assert ids == [1, 2, 3, 4]


def test_fresh_segmenter_first_id_is_one():
    seg, _, _ = make(script((10, SPEECH), (20, SIL)))
    out = feed_frames(seg, ramp(30 * F))
    assert [s.id for _, s in out] == [1]


# ---------------------------------------------------------------------------
# session_id
# ---------------------------------------------------------------------------


def test_session_id_defaults_to_zero_and_follows_begin_session():
    probs = script((10, SIL), (10, SPEECH), (20, SIL))
    seg, vad, _ = make(probs)
    assert feed_frames(seg, ramp(40 * F))[0][1].session_id == 0
    seg.reset()
    seg.begin_session(7)
    assert feed_frames(seg, ramp(40 * F))[0][1].session_id == 7
    seg.flush()
    seg.begin_session(8)
    vad.probs = script((10, SIL), (10, SPEECH))
    feed_frames(seg, ramp(20 * F))
    out = seg.flush()
    assert [(s.session_id, s.reason) for s in out] == [(8, SegmentCloseReason.FLUSH)]


# ---------------------------------------------------------------------------
# set_config / configurable thresholds
# ---------------------------------------------------------------------------


def test_set_config_applies_to_next_feed():
    probs = script((10, SIL), (10, SPEECH), (40, SIL))
    seg, _, clock = make(probs)  # default 500 ms
    audio = ramp(60 * F)
    assert feed_frames(seg, audio[: 20 * F], clock) == []
    seg.set_config(VadConfig(min_silence_ms=300))
    out = feed_frames(seg, audio[20 * F:], clock, first_frame=20)
    assert len(out) == 1
    assert out[0][0] == 19 + 10  # 10 frames (320 ms >= 300 ms), not 16
    assert_span(out[0][1], 10 * F - PAD, 20 * F + PAD)


def test_set_config_max_segment_applies_to_next_feed():
    probs = script((300, SPEECH))
    seg, _, _ = make(probs)  # default 15 s
    seg.set_config(VadConfig(max_segment_s=2.04))
    out = feed_frames(seg, ramp(70 * F))
    assert [(i, s.reason) for i, s in out] == [(63, SegmentCloseReason.MAX_LENGTH)]


def test_custom_threshold_and_offset():
    cfg = VadConfig(threshold=0.7, neg_threshold_offset=0.0, speech_pad_ms=0)
    # 0.69 must not start speech; 0.7 starts; in speech 0.69 (< 0.7 - 0) is silence.
    probs = script((10, 0.69), (10, 0.7), (40, 0.69))
    seg, _, _ = make(probs, cfg)
    out = feed_frames(seg, ramp(60 * F))
    assert len(out) == 1
    i, s = out[0]
    assert i == 19 + 16
    assert_span(s, 10 * F, 20 * F)


# ---------------------------------------------------------------------------
# Chunk-size invariance
# ---------------------------------------------------------------------------


def _signature(segs):
    return [
        (s.id, s.session_id, s.reason, round(s.speech_start_s, 9), round(s.speech_end_s, 9), s.t_vad_close,
         s.samples.dtype, s.samples.tobytes())
        for s in segs
    ]


def _run_chunked(probs, cfg, audio, sizes):
    seg, vad, _ = make(probs, cfg, FakeClock(5.0))
    segs = []
    pos = 0
    k = 0
    while pos < len(audio):
        n = sizes[k % len(sizes)]
        segs.extend(seg.feed(audio[pos:pos + n]))
        pos += n
        k += 1
    segs.extend(seg.flush())
    assert vad.errors == []
    return segs


BIG_SCENARIO = (
    VadConfig(max_segment_s=2.0),
    script((5, SIL), (80, SPEECH), (5, 0.4), (20, SIL), (5, SPEECH), (20, SIL), (10, SPEECH)),
    145 * F + 200,
)
SMALL_SCENARIO = (
    VadConfig(min_silence_ms=300, speech_pad_ms=100),
    script((2, SIL), (18, SPEECH), (12, SIL), (3, SPEECH), (12, SIL), (8, SPEECH)),
    55 * F + 33,
)


@pytest.mark.parametrize("sizes", [[511], [512], [513], [4096], [1000, 1, 7, 2500, 333], "random"])
def test_chunk_size_invariance(sizes):
    cfg, probs, n = BIG_SCENARIO
    audio = ramp(n)
    reference = _run_chunked(probs, cfg, audio, [n])
    reasons = [s.reason for s in reference]
    # sanity: the scenario exercises max-length, silence and flush closes
    assert SegmentCloseReason.MAX_LENGTH in reasons
    assert SegmentCloseReason.SILENCE in reasons
    assert reasons[-1] == SegmentCloseReason.FLUSH
    if sizes == "random":
        rng = np.random.default_rng(1234)
        sizes = [int(x) for x in rng.integers(1, 3000, size=500)]
    assert _signature(_run_chunked(probs, cfg, audio, sizes)) == _signature(reference)


def test_chunk_size_one_sample_invariance():
    cfg, probs, n = SMALL_SCENARIO
    audio = ramp(n)
    reference = _run_chunked(probs, cfg, audio, [n])
    assert [s.reason for s in reference] == [SegmentCloseReason.SILENCE, SegmentCloseReason.FLUSH]
    assert _signature(_run_chunked(probs, cfg, audio, [1])) == _signature(reference)
