"""Manual microphone test (marker ``mic``). Needs a human. Skipped unless ROENSPEAK_MIC_TEST=1.

    Windows:  ROENSPEAK_MIC_TEST=1 scripts/win.sh -m pytest tests/integration -m mic -s
    macOS:    ROENSPEAK_MIC_TEST=1 .venv/bin/python -m pytest tests/integration -m mic -s

The CLI only accepts a .wav, so this records 6 s from the default input device with
sounddevice (a project dependency) into a temp wav, then runs it through the same
``python -m app.cli --json`` path as the gpu/metal tests. When prompted, say the fixture
sentence (Romanian): "Astăzi vremea este frumoasă și vreau să merg în parc."
It doesn't exercise MicSource itself (device selection, resampling). That is covered only by
using the app by hand.

Markers: ``mic`` pairs with the host's engine marker -- ``gpu`` on Windows (faster-whisper on
CUDA), ``metal`` on macOS (whisper.cpp on Metal) -- which is what makes tests/conftest.py run
it on the right machine. The expected ``EngineInfo.device`` follows from that
(``helpers.EXPECTED_DEVICE``: "cuda" / "metal").

**macOS only:** the first run pops the system microphone prompt for the interpreter
(``.venv/bin/python``, or the terminal that launched it) and sounddevice records silence until
you allow it -- so the first attempt can fail the "microphone recorded silence" assertion.
Allow it and run again; the grant is remembered per binary. Check it any time with
``app.macos.permissions.microphone_status()``, and reopen the pane with
``open_privacy_settings("microphone")`` if you denied it by accident.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from app import contracts

from .helpers import (
    EXPECTED_DEVICE,
    IS_MACOS,
    assert_english,
    require_platform_model,
    run_cli,
    safe_print,
    write_wav,
)

pytestmark = [pytest.mark.mic, pytest.mark.metal if IS_MACOS else pytest.mark.gpu]

RECORD_S = 6


@pytest.mark.skipif(os.environ.get("ROENSPEAK_MIC_TEST") != "1", reason="manual: set ROENSPEAK_MIC_TEST=1")
def test_spoken_romanian_from_default_mic(tmp_path):
    require_platform_model()
    sd = pytest.importorskip("sounddevice")
    if IS_MACOS:
        from app.macos.permissions import DENIED, microphone_status

        if microphone_status() == DENIED:
            pytest.skip(
                "microphone access is denied for this interpreter; allow it in System Settings "
                "> Privacy & Security > Microphone and run again"
            )
    safe_print(f"\n>>> Speak now for {RECORD_S} s: 'Astăzi vremea este frumoasă și vreau să merg în parc.'")
    rec = sd.rec(RECORD_S * contracts.SAMPLE_RATE, samplerate=contracts.SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    audio = rec.reshape(-1).astype(np.float32)
    assert float(np.abs(audio).max()) > 0.01, (
        "microphone recorded silence"
        + (" (did macOS prompt for microphone access? allow it and run again)" if IS_MACOS else "")
    )
    wav = write_wav(tmp_path / "mic.wav", audio)
    r = run_cli([wav, "--json"], tmp_path)
    assert r.returncode == 0, r.describe()
    doc = r.json()
    text = " ".join(s["text"] for s in doc["segments"] if not s["filtered"])
    safe_print(f">>> heard: {text!r}  segments={[(s['infer_ms'], s['latency_ms']) for s in doc['segments']]}")
    assert doc["engine"]["device"] == EXPECTED_DEVICE, doc["engine"]
    assert_english(text)
