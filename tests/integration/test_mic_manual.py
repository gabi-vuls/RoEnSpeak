"""Manual microphone test (marker ``mic``). Needs a human. Skipped unless ROENSPEAK_MIC_TEST=1.

    ROENSPEAK_MIC_TEST=1 scripts/win.sh -m pytest tests/integration -m mic -s

The CLI only accepts a .wav, so this records 6 s from the default input device with
sounddevice (a project dependency) into a temp wav, then runs it through the same
``python -m app.cli --json`` path as the gpu tests. When prompted, say the fixture sentence
(Romanian): "Astăzi vremea este frumoasă și vreau să merg în parc."
It doesn't exercise MicSource itself (device selection, resampling). That is covered only by
using the app by hand.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from app import contracts

from .helpers import assert_english, require_model, run_cli, safe_print, write_wav

pytestmark = [pytest.mark.mic, pytest.mark.gpu]

RECORD_S = 6


@pytest.mark.skipif(os.environ.get("ROENSPEAK_MIC_TEST") != "1", reason="manual: set ROENSPEAK_MIC_TEST=1")
def test_spoken_romanian_from_default_mic(tmp_path):
    require_model()
    sd = pytest.importorskip("sounddevice")
    safe_print(f"\n>>> Speak now for {RECORD_S} s: 'Astăzi vremea este frumoasă și vreau să merg în parc.'")
    rec = sd.rec(RECORD_S * contracts.SAMPLE_RATE, samplerate=contracts.SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    audio = rec.reshape(-1).astype(np.float32)
    assert float(np.abs(audio).max()) > 0.01, "microphone recorded silence"
    wav = write_wav(tmp_path / "mic.wav", audio)
    r = run_cli([wav, "--json"], tmp_path)
    assert r.returncode == 0, r.describe()
    doc = r.json()
    text = " ".join(s["text"] for s in doc["segments"] if not s["filtered"])
    safe_print(f">>> heard: {text!r}  segments={[(s['infer_ms'], s['latency_ms']) for s in doc['segments']]}")
    assert doc["engine"]["device"] == "cuda"
    assert_english(text)
