"""Microphone capture via sounddevice (PortAudio).

Host API choice: Windows WASAPI (shared mode) when present, because it reports full device
names (MME truncates them to 31 characters) and follows the Windows default recording
device. The stream is first opened at 16 kHz mono with WASAPI auto_convert (the Windows
audio engine resamples and downmixes). If the device rejects that, it is reopened at the
device's default sample rate and channel count; channels are averaged to mono and the
audio is resampled to 16 kHz with a streaming soxr resampler. Without WASAPI (or if WASAPI
can't open the device at all) the same sequence is tried on MME, then on the PortAudio
default host API.

Device loss: PortAudio reports it as a stream that finishes without us stopping it, as
callbacks that stop arriving, or as an exception in the callback. A watchdog
thread checks the stream every 250 ms; if the stream finished on its own, raised, or
delivered no callback for STALL_TIMEOUT_S, delivery stops and on_error(AudioDeviceError)
is called from the watchdog thread.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from app.contracts import SAMPLE_RATE, AppError, AudioDeviceError

log = logging.getLogger("roenspeak.audio.capture")

BLOCK_MS = 32
STALL_TIMEOUT_S = 2.0
WATCHDOG_PERIOD_S = 0.25
HOSTAPI_PREFERENCE = ("Windows WASAPI", "MME", "Windows DirectSound", "ALSA", "PulseAudio", "Core Audio")


def _sd() -> Any:
    try:
        import sounddevice
    except (ImportError, OSError) as e:
        raise AudioDeviceError(
            "Audio capture is not available (PortAudio could not be loaded).", details=repr(e)
        ) from e
    return sounddevice


# PortAudio enumerates devices only at initialization. To see devices plugged in (or a new
# Windows default) after startup, it is re-initialized before listing and before opening,
# but only while none of our streams is open: Pa_Terminate closes open streams under them.
_PA_LOCK = threading.RLock()
_open_streams = 0


def _refresh_devices(sd: Any) -> bool:
    """Re-enumerate devices if no stream of ours is open. Caller holds _PA_LOCK."""
    if _open_streams:
        return False
    try:
        sd._terminate()
        sd._initialize()
    except Exception as e:
        raise AudioDeviceError("Could not refresh the list of audio devices.", details=repr(e)) from e
    return True


def _stream_opened() -> None:
    global _open_streams
    with _PA_LOCK:
        _open_streams += 1


def _stream_closed() -> None:
    global _open_streams
    with _PA_LOCK:
        _open_streams = max(0, _open_streams - 1)


@dataclass(frozen=True)
class _DeviceChoice:
    index: int
    name: str
    hostapi: str
    max_channels: int
    default_samplerate: float


def _hostapi_order(sd: Any) -> list[int]:
    apis = list(sd.query_hostapis())
    order: list[int] = []
    for wanted in HOSTAPI_PREFERENCE:
        order += [i for i, api in enumerate(apis) if api["name"] == wanted]
    try:
        default_api = int(sd.default.hostapi)
    except Exception:
        default_api = -1
    if 0 <= default_api < len(apis) and default_api not in order:
        order.append(default_api)
    return order


def _input_devices(sd: Any, hostapi: int) -> list[_DeviceChoice]:
    api = sd.query_hostapis(hostapi)
    out = []
    for idx in api["devices"]:
        info = sd.query_devices(idx)
        if info["max_input_channels"] > 0:
            out.append(
                _DeviceChoice(idx, info["name"], api["name"], int(info["max_input_channels"]),
                              float(info["default_samplerate"]))
            )
    return out


def _default_input(sd: Any, hostapi: int) -> Optional[_DeviceChoice]:
    idx = sd.query_hostapis(hostapi)["default_input_device"]
    if idx is None or idx < 0:
        return None
    for d in _input_devices(sd, hostapi):
        if d.index == idx:
            return d
    return None


def _candidates(sd: Any, device: Optional[str]) -> list[_DeviceChoice]:
    """Devices to try, best first. Empty if nothing matches."""
    out: list[_DeviceChoice] = []
    for api in _hostapi_order(sd):
        if device is None:
            d = _default_input(sd, api)
            if d is not None:
                out.append(d)
        else:
            out += [d for d in _input_devices(sd, api) if d.name == device]
    return out


class MicSource:
    """AudioSource for a microphone. device=None means the Windows default recording device."""

    def __init__(self, device: Optional[str] = None) -> None:
        self._device = device
        self._lock = threading.Lock()  # guards _delivering and serializes delivery vs. stop
        self._delivering = False
        self._running = False
        self._stream: Any = None
        self._resampler: Any = None
        self._channels = 1
        self._on_audio: Optional[Callable[[np.ndarray], None]] = None
        self._on_error: Optional[Callable[[AppError], None]] = None
        self._last_callback = 0.0
        self._fault: Optional[str] = None
        self._stop_evt = threading.Event()
        self._watchdog: Optional[threading.Thread] = None
        self.opened_with: str = ""

    @staticmethod
    def list_input_devices() -> list[str]:
        sd = _sd()
        with _PA_LOCK:
            _refresh_devices(sd)
            for api in _hostapi_order(sd):
                devices = _input_devices(sd, api)
                if devices:
                    names: list[str] = []
                    for d in devices:
                        if d.name not in names:
                            names.append(d.name)
                    return names
        return []

    def start(
        self,
        on_audio: Callable[[np.ndarray], None],
        on_end: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[AppError], None]] = None,
    ) -> None:
        if self._running:
            raise RuntimeError("MicSource already started")
        sd = _sd()
        with _PA_LOCK:
            self._start_locked(sd, on_audio, on_error)
        self._running = True
        self._watchdog = threading.Thread(target=self._watch, name="roenspeak-mic-watchdog", daemon=True)
        self._watchdog.start()

    def _start_locked(
        self, sd: Any, on_audio: Callable[[np.ndarray], None], on_error: Optional[Callable[[AppError], None]]
    ) -> None:
        _refresh_devices(sd)
        try:
            candidates = _candidates(sd, self._device)
        except Exception as e:
            raise AudioDeviceError("Could not list audio input devices.", details=repr(e)) from e
        if not candidates:
            if self._device is None:
                raise AudioDeviceError("No microphone found. Connect a microphone and try again.")
            raise AudioDeviceError(
                f"Microphone '{self._device}' was not found. Choose another microphone in Settings.",
                details=f"device={self._device!r}",
            )
        self._on_audio = on_audio
        self._on_error = on_error
        self._fault = None
        self._stop_evt.clear()
        errors: list[str] = []
        for cand in candidates:
            if self._open(sd, cand, errors):
                break
        else:
            raise AudioDeviceError(
                "The microphone could not be opened. It may be in use or disconnected.",
                details="; ".join(errors),
            )

    def stop(self) -> None:
        with self._lock:
            self._delivering = False
        self._stop_evt.set()
        wd = self._watchdog
        if wd is not None and wd is not threading.current_thread():
            wd.join()
        self._watchdog = None
        self._close_stream()
        self._running = False

    # -- internals --------------------------------------------------------

    def _open(self, sd: Any, cand: _DeviceChoice, errors: list[str]) -> bool:
        attempts: list[tuple[float, int, bool]] = [(float(SAMPLE_RATE), 1, True)]
        native = (cand.default_samplerate, max(1, min(cand.max_channels, 2)), False)
        if native[:2] != (float(SAMPLE_RATE), 1):
            attempts.append(native)
        for rate, channels, try_convert in attempts:
            extra = None
            if cand.hostapi == "Windows WASAPI" and try_convert:
                extra = sd.WasapiSettings(auto_convert=True)
            try:
                self._prepare(rate, channels)
                stream = sd.InputStream(
                    device=cand.index, samplerate=rate, channels=channels, dtype="float32",
                    blocksize=int(round(rate * BLOCK_MS / 1000)), latency="low",
                    extra_settings=extra, callback=self._callback, finished_callback=self._finished,
                )
                self._stream = stream
                _stream_opened()
                with self._lock:
                    self._delivering = True
                self._last_callback = time.monotonic()
                stream.start()
            except Exception as e:
                with self._lock:
                    self._delivering = False
                self._close_stream()
                errors.append(f"{cand.hostapi}/{cand.name}@{rate:g}Hz/{channels}ch: {e}")
                log.info("mic open failed api=%s device=%r rate=%g ch=%d: %s",
                         cand.hostapi, cand.name, rate, channels, e)
                continue
            self.opened_with = f"{cand.hostapi}|{cand.name}|{rate:g}Hz|{channels}ch"
            log.info("mic_opened api=%s device=%r rate=%g channels=%d resample=%s",
                     cand.hostapi, cand.name, rate, channels, self._resampler is not None)
            return True
        return False

    def _prepare(self, rate: float, channels: int) -> None:
        self._channels = channels
        if int(round(rate)) == SAMPLE_RATE:
            self._resampler = None
        else:
            import soxr

            self._resampler = soxr.ResampleStream(rate, SAMPLE_RATE, 1, dtype="float32", quality="HQ")

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        self._last_callback = time.monotonic()
        if status and status.input_overflow:
            log.debug("mic input overflow")
        try:
            if self._channels == 1:
                mono = indata[:, 0].astype(np.float32, copy=True)
            else:
                mono = indata.mean(axis=1, dtype=np.float32)
            if self._resampler is not None:
                mono = np.asarray(self._resampler.resample_chunk(mono), dtype=np.float32)
            if mono.shape[0] == 0:
                return
            with self._lock:
                if self._delivering and self._on_audio is not None:
                    self._on_audio(mono)
        except Exception as e:  # never let an exception kill the PortAudio thread silently
            log.exception("mic callback failed")
            self._fault = f"callback error: {e!r}"

    def _finished(self) -> None:
        if not self._stop_evt.is_set():
            self._fault = self._fault or "stream finished unexpectedly"

    def _watch(self) -> None:
        while not self._stop_evt.wait(WATCHDOG_PERIOD_S):
            reason = self._fault
            stream = self._stream
            if reason is None and stream is not None:
                try:
                    if not stream.active:
                        reason = "stream became inactive"
                except Exception as e:
                    reason = f"stream query failed: {e!r}"
            if reason is None and time.monotonic() - self._last_callback > STALL_TIMEOUT_S:
                reason = f"no audio callbacks for {STALL_TIMEOUT_S:.0f} s"
            if reason is None:
                continue
            with self._lock:
                if not self._delivering:
                    return
                self._delivering = False
            if self._stop_evt.is_set():
                return
            log.error("mic_lost device=%r reason=%s", self._device, reason)
            self._close_stream()
            self._running = False
            cb = self._on_error
            if cb is not None:
                try:
                    cb(AudioDeviceError(
                        "The microphone stopped working (unplugged or taken by another app).",
                        details=reason,
                    ))
                except Exception:
                    log.exception("on_error callback failed")
            return

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.abort(ignore_errors=True)
        except Exception:
            log.debug("stream abort failed", exc_info=True)
        try:
            stream.close(ignore_errors=True)
        except Exception:
            log.debug("stream close failed", exc_info=True)
        _stream_closed()
