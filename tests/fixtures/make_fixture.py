"""Generate tests/fixtures/ro_sample.wav, the Romanian speech fixture for integration tests.

Provenance
----------
* Voice: Microsoft Edge neural TTS, ``ro-RO-AlinaNeural`` (approved by the owner in Phase 0:
  "Test fixture generated via a ro-RO neural TTS voice"), through the ``edge-tts`` package.
* Generated: 2026-10-22 with edge-tts 7.3.1 and PyAV 19.0.2 (decode + resample).
* Each sentence is synthesized separately and the sentences are joined with
  ``GAP_S`` seconds of digital silence. Leading and trailing silence is added too. That
  gives guaranteed pauses (> 0.7 s) between sentences, so the streaming VAD
  (min_silence_ms=500) closes one segment per sentence.
* Output: 16 kHz, mono, PCM16 wav (stdlib ``wave``).

The script and the expected meaning are in ``ro_sample.txt``.

This is a one-off tool. Its dependencies are NOT project requirements. To run it:

    python3 -m venv /tmp/fixture-venv
    /tmp/fixture-venv/bin/pip install edge-tts av numpy
    /tmp/fixture-venv/bin/python tests/fixtures/make_fixture.py

It needs network access (edge-tts calls Microsoft's online TTS service). The output is not
bit-for-bit reproducible because the service may change its voices over time.
"""

from __future__ import annotations

import asyncio
import io
import sys
import wave
from pathlib import Path

import numpy as np

VOICE = "ro-RO-AlinaNeural"
SENTENCES = (
    "Bună ziua.",
    "Astăzi vremea este frumoasă și vreau să merg în parc.",
    "Mulțumesc pentru ajutor.",
)
SAMPLE_RATE = 16_000
GAP_S = 1.2  # silence between sentences (> 0.7 s required)
LEAD_S = 0.5  # silence before the first sentence
TRAIL_S = 1.0  # silence after the last sentence
OUT = Path(__file__).with_name("ro_sample.wav")


async def _tts_mp3(text: str) -> bytes:
    import edge_tts

    buf = bytearray()
    async for chunk in edge_tts.Communicate(text, VOICE).stream():
        if chunk["type"] == "audio":
            buf.extend(chunk["data"])
    return bytes(buf)


def _decode_16k_mono(mp3: bytes) -> np.ndarray:
    import av

    out: list[np.ndarray] = []
    with av.open(io.BytesIO(mp3)) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(audio=0):
            for rf in resampler.resample(frame):
                out.append(rf.to_ndarray().reshape(-1))
        for rf in resampler.resample(None):
            out.append(rf.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.int16)


def _trim(x: np.ndarray, thresh: int = 200) -> np.ndarray:
    """Trim the TTS engine's own leading/trailing near-silence so our gaps are exact."""
    idx = np.flatnonzero(np.abs(x.astype(np.int32)) > thresh)
    if idx.size == 0:
        return x
    pad = int(0.05 * SAMPLE_RATE)
    return x[max(0, idx[0] - pad): idx[-1] + pad]


def main() -> int:
    parts: list[np.ndarray] = [np.zeros(int(LEAD_S * SAMPLE_RATE), np.int16)]
    for i, sentence in enumerate(SENTENCES):
        pcm = _trim(_decode_16k_mono(asyncio.run(_tts_mp3(sentence))))
        parts.append(pcm)
        gap = TRAIL_S if i == len(SENTENCES) - 1 else GAP_S
        parts.append(np.zeros(int(gap * SAMPLE_RATE), np.int16))
    audio = np.concatenate(parts)
    with wave.open(str(OUT), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())
    print(f"wrote {OUT} ({len(audio) / SAMPLE_RATE:.2f} s, {OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
