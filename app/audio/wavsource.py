"""WAV file input: a finite AudioSource for the CLI test mode and the desktop test.

The RIFF parser handles PCM 8/16/24/32-bit, IEEE float 32/64-bit and WAVE_FORMAT_EXTENSIBLE
wrappers of those, at any sample rate and channel count. Channels are averaged to mono and
the result is resampled to 16 kHz with soxr.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from app.contracts import SAMPLE_RATE, AppError, AudioDeviceError

log = logging.getLogger("roenspeak.audio.wavsource")

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

PathLike = Union[str, Path]


def _bad(path: PathLike, why: str) -> AudioDeviceError:
    return AudioDeviceError(f"Can't read WAV file '{path}': {why}", details=why)


def _decode(data: bytes, fmt: int, bits: int, channels: int) -> np.ndarray:
    width = bits // 8
    usable = len(data) - len(data) % (width * channels)
    data = data[:usable]
    if fmt == WAVE_FORMAT_PCM:
        if bits == 8:
            x = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif bits == 16:
            x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        elif bits == 24:
            b = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            v = np.where(v >= 1 << 23, v - (1 << 24), v)
            x = v.astype(np.float32) / float(1 << 23)
        elif bits == 32:
            x = (np.frombuffer(data, dtype="<i4").astype(np.float64) / 2147483648.0).astype(np.float32)
        else:
            raise ValueError(f"unsupported PCM bit depth {bits}")
    elif fmt == WAVE_FORMAT_IEEE_FLOAT:
        if bits == 32:
            x = np.frombuffer(data, dtype="<f4").astype(np.float32)
        elif bits == 64:
            x = np.frombuffer(data, dtype="<f8").astype(np.float32)
        else:
            raise ValueError(f"unsupported float bit depth {bits}")
    else:
        raise ValueError(f"unsupported WAV format tag 0x{fmt:04x} (only PCM and IEEE float)")
    return x.reshape(-1, channels)


def _parse(raw: bytes) -> tuple[np.ndarray, int]:
    """Returns (frames x channels float32, sample_rate). Raises ValueError on bad input."""
    if len(raw) < 12 or raw[0:4] not in (b"RIFF", b"RF64") or raw[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos = 12
    fmt_info: Optional[tuple[int, int, int, int]] = None
    data: Optional[bytes] = None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        size = struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body_start = pos + 8
        body_end = min(body_start + size, len(raw))
        if cid == b"fmt ":
            body = raw[body_start:body_end]
            if len(body) < 16:
                raise ValueError("fmt chunk too short")
            tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == WAVE_FORMAT_EXTENSIBLE:
                if len(body) < 26:
                    raise ValueError("WAVE_FORMAT_EXTENSIBLE fmt chunk too short")
                tag = struct.unpack("<H", body[24:26])[0]  # first 2 bytes of the SubFormat GUID
            fmt_info = (tag, channels, rate, bits)
        elif cid == b"data":
            # Streaming writers may leave size 0 / 0xFFFFFFFF; take what is there.
            if size in (0, 0xFFFFFFFF):
                body_end = len(raw)
            data = raw[body_start:body_end]
            if fmt_info is not None:
                break
        pos = body_start + size + (size & 1)
    if fmt_info is None:
        raise ValueError("missing fmt chunk")
    if data is None:
        raise ValueError("missing data chunk")
    tag, channels, rate, bits = fmt_info
    if channels < 1 or rate < 1000 or bits % 8:
        raise ValueError(f"invalid format: channels={channels} rate={rate} bits={bits}")
    return _decode(data, tag, bits, channels), rate


def read_wav_16k_mono(path: PathLike) -> np.ndarray:
    """Whole file as float32 mono 16 kHz in [-1, 1]. Raises AudioDeviceError on bad input."""
    try:
        raw = Path(path).read_bytes()
    except OSError as e:
        raise _bad(path, e.strerror or str(e)) from e
    try:
        frames, rate = _parse(raw)
    except (ValueError, struct.error) as e:
        raise _bad(path, str(e)) from e
    mono = frames.mean(axis=1, dtype=np.float32) if frames.shape[1] > 1 else frames[:, 0]
    if rate != SAMPLE_RATE and mono.shape[0]:
        import soxr

        mono = soxr.resample(mono, rate, SAMPLE_RATE, quality="HQ")
    out = np.clip(np.ascontiguousarray(mono, dtype=np.float32), -1.0, 1.0)
    log.debug("wav loaded path=%s rate=%d channels=%d seconds=%.2f", path, rate, frames.shape[1],
              out.shape[0] / SAMPLE_RATE)
    return out


class WavFileSource:
    """Finite AudioSource. Delivers chunk_ms chunks from its own thread, as fast as possible
    or paced at 1x when realtime=True, then calls on_end once."""

    def __init__(self, path: PathLike, realtime: bool = False, chunk_ms: int = 32) -> None:
        if chunk_ms <= 0:
            raise ValueError("chunk_ms must be positive")
        self._path = path
        self._realtime = realtime
        self._chunk = max(1, chunk_ms * SAMPLE_RATE // 1000)
        self._audio: Optional[np.ndarray] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    def start(
        self,
        on_audio: Callable[[np.ndarray], None],
        on_end: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[AppError], None]] = None,
    ) -> None:
        if self._thread is not None:
            raise RuntimeError("WavFileSource already started")
        if self._audio is None:
            self._audio = read_wav_16k_mono(self._path)
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, args=(self._audio, on_audio, on_end), name="roenspeak-wav-source", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join()
        self._thread = None

    def _run(
        self, audio: np.ndarray, on_audio: Callable[[np.ndarray], None], on_end: Optional[Callable[[], None]]
    ) -> None:
        chunk_s = self._chunk / SAMPLE_RATE
        t0 = time.perf_counter()
        n = 0
        for start in range(0, audio.shape[0], self._chunk):
            if self._stop_evt.is_set():
                return
            if self._realtime:
                delay = t0 + (n + 1) * chunk_s - time.perf_counter()
                if delay > 0 and self._stop_evt.wait(delay):
                    return
            try:
                on_audio(audio[start:start + self._chunk].copy())
            except Exception:
                log.exception("on_audio failed; wav delivery stopped")
                return
            n += 1
        if self._stop_evt.is_set() or on_end is None:
            return
        try:
            on_end()
        except Exception:
            log.exception("on_end failed")
