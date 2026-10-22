"""Streaming Silero VAD on the silero_vad_v6.onnx model.

faster_whisper.vad.SileroVADModel resets its LSTM state on every call (batch use only).
SileroStreamingVad runs the same silero_vad_v6.onnx one 512-sample frame at a time and
carries the h/c state and the 64-sample context between calls. The tensor layout mirrors
upstream: input (1, 64 + 512) = previous frame's last 64 samples + current frame,
h/c (1, 1, 128). The first frame after reset() gets a zero context, exactly like the
first row of the upstream batch.

Where the .onnx comes from differs per platform, which is all silero_model_path() does.
On Windows the asset ships inside the faster-whisper package (which is a dependency there).
macOS has no faster-whisper at all, so the identical file -- same model, taken from the
pinned faster-whisper commit, see assets/vad/README.md -- is vendored in assets/vad/ and
resolved the way app/ui/assets.py resolves assets (sys._MEIPASS when frozen, else the
repo). The ONNX signature is byte-for-byte the same (input [seq_len, 576], h/c [1, 1, 128]),
so nothing below this function changes between platforms. Both sources are tried on both
platforms, preferring the platform's native one, so neither build can break if a spec file
stops bundling one of them.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np

from app.contracts import IS_WINDOWS, VAD_FRAME_SAMPLES, AudioDeviceError

log = logging.getLogger("roenspeak.audio.vad")

CONTEXT_SAMPLES = 64
STATE_SHAPE = (1, 1, 128)
MODEL_FILENAME = "silero_vad_v6.onnx"
#: Where the vendored copy lives, relative to the bundle root / the repo root.
VENDORED_SUBDIR = ("assets", "vad")


def _vendored_model_path() -> str:
    """<bundle>/assets/vad/<model> when frozen (sys._MEIPASS, i.e. the onedir _internal
    folder), otherwise <repo>/assets/vad/<model>. Mirrors app.ui.assets.assets_dir()."""
    meipass = getattr(sys, "_MEIPASS", None)
    if getattr(sys, "frozen", False) and meipass:
        root = meipass
    else:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, *VENDORED_SUBDIR, MODEL_FILENAME)


def _package_model_path() -> str:
    """The copy inside the faster-whisper package. "" when faster-whisper is absent
    (always on macOS, where it is not and must not be a dependency)."""
    try:
        from faster_whisper.utils import get_assets_path
    except Exception:
        return ""
    try:
        return os.path.join(get_assets_path(), MODEL_FILENAME)
    except Exception:  # pragma: no cover - get_assets_path is a path join
        return ""


def silero_model_candidates() -> list[str]:
    """Candidate paths of the VAD model, best first."""
    vendored, package = _vendored_model_path(), _package_model_path()
    order = [package, vendored] if IS_WINDOWS else [vendored, package]
    return [p for p in order if p]


def silero_model_path() -> str:
    """The first candidate that exists. Raises AudioDeviceError when none does."""
    candidates = silero_model_candidates()
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise AudioDeviceError(
        "The voice-activity model is missing from the installation.",
        details="not found: " + ", ".join(candidates or [MODEL_FILENAME]),
    )


class SileroStreamingVad:
    """VadModel: speech probability per 512-sample frame with recurrent state carried."""

    def __init__(self, model_path: str | None = None) -> None:
        import onnxruntime

        path = model_path or silero_model_path()
        if not os.path.isfile(path):
            raise AudioDeviceError(
                "The voice-activity model is missing from the installation.",
                details=f"not found: {path}",
            )
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self._input = np.zeros((1, CONTEXT_SAMPLES + VAD_FRAME_SAMPLES), dtype=np.float32)
        self._h = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._c = np.zeros(STATE_SHAPE, dtype=np.float32)
        log.debug("silero streaming VAD loaded from %s", path)

    def reset(self) -> None:
        self._input.fill(0.0)
        self._h = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._c = np.zeros(STATE_SHAPE, dtype=np.float32)

    def speech_prob(self, frame: np.ndarray) -> float:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.shape[0] != VAD_FRAME_SAMPLES:
            raise ValueError(f"VAD frame must be {VAD_FRAME_SAMPLES} samples, got {frame.shape[0]}")
        # _input[0, :64] already holds the previous frame's tail (zeros after reset).
        self._input[0, CONTEXT_SAMPLES:] = frame
        out, self._h, self._c = self._session.run(None, {"input": self._input, "h": self._h, "c": self._c})
        self._input[0, :CONTEXT_SAMPLES] = frame[-CONTEXT_SAMPLES:]
        return float(np.clip(np.asarray(out).reshape(-1)[0], 0.0, 1.0))
