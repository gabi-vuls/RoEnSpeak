"""Spec/contract tests for the VAD asset resolution in app.audio.vad.

ARCHITECTURE.md section 4 pins the VAD to the silero_vad_v6.onnx that faster-whisper
bundles; section 12.1 / binding decision 9 say macOS has no faster-whisper at all, so the
identical file is vendored in assets/vad/ and resolved from there. Both sources are tried
on both platforms (preferring the platform's native one) so neither build can break if a
spec file stops bundling one of them, and a missing asset is an AudioDeviceError naming
every candidate.

The last two tests read the real vendored file. They are deliberately not hardware tests:
they are a cheap guard that the vendored asset is the model SileroStreamingVad feeds
(input [seq_len, 576], h/c [1, 1, 128]) -- a silently wrong model would break segmentation
in a way the segmenter's fake-VAD tests cannot see.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from app.contracts import VAD_FRAME_SAMPLES, AudioDeviceError
from app.audio import vad as vad_mod
from app.audio.vad import (
    CONTEXT_SAMPLES,
    MODEL_FILENAME,
    STATE_SHAPE,
    VENDORED_SUBDIR,
    silero_model_candidates,
    silero_model_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED = REPO_ROOT.joinpath(*VENDORED_SUBDIR, MODEL_FILENAME)


@pytest.fixture
def two_sources(tmp_path, monkeypatch):
    """Both candidate locations, neither existing yet. Returns (vendored, package)."""
    vendored = tmp_path / "bundle" / "assets" / "vad" / MODEL_FILENAME
    package = tmp_path / "site-packages" / "faster_whisper" / "assets" / MODEL_FILENAME
    monkeypatch.setattr(vad_mod, "_vendored_model_path", lambda: str(vendored))
    monkeypatch.setattr(vad_mod, "_package_model_path", lambda: str(package))
    return vendored, package


def create(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ONNX (not really)")
    return path


# ---------------------------------------------------------------------------
# candidate order
# ---------------------------------------------------------------------------


def test_macos_prefers_the_vendored_asset(two_sources, monkeypatch):
    vendored, package = two_sources
    monkeypatch.setattr(vad_mod, "IS_WINDOWS", False)
    assert silero_model_candidates() == [str(vendored), str(package)]


def test_windows_prefers_the_faster_whisper_package_asset(two_sources, monkeypatch):
    vendored, package = two_sources
    monkeypatch.setattr(vad_mod, "IS_WINDOWS", True)
    assert silero_model_candidates() == [str(package), str(vendored)]


@pytest.mark.parametrize("is_windows", [True, False])
def test_both_platforms_list_both_sources(two_sources, monkeypatch, is_windows):
    """Neither platform's build can break if one source stops shipping the file."""
    vendored, package = two_sources
    monkeypatch.setattr(vad_mod, "IS_WINDOWS", is_windows)
    assert set(silero_model_candidates()) == {str(vendored), str(package)}


def test_a_missing_faster_whisper_leaves_only_the_vendored_candidate(monkeypatch, tmp_path):
    """faster-whisper is absent on macOS and must not be a dependency there."""
    vendored = tmp_path / MODEL_FILENAME
    monkeypatch.setattr(vad_mod, "_vendored_model_path", lambda: str(vendored))
    monkeypatch.setattr(vad_mod, "_package_model_path", lambda: "")
    monkeypatch.setattr(vad_mod, "IS_WINDOWS", True)
    assert silero_model_candidates() == [str(vendored)]


def test_package_path_is_empty_without_faster_whisper(monkeypatch):
    """_package_model_path() swallows the ImportError instead of breaking the import."""
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", None)
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    assert vad_mod._package_model_path() == ""


# ---------------------------------------------------------------------------
# resolution and fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("is_windows", [True, False])
def test_the_preferred_source_wins_when_both_exist(two_sources, monkeypatch, is_windows):
    vendored, package = two_sources
    create(vendored)
    create(package)
    monkeypatch.setattr(vad_mod, "IS_WINDOWS", is_windows)
    assert silero_model_path() == str(package if is_windows else vendored)


@pytest.mark.parametrize("is_windows", [True, False])
@pytest.mark.parametrize("which", ["vendored", "package"])
def test_each_platform_falls_back_to_the_other_source(two_sources, monkeypatch, is_windows, which):
    """macOS falls back to the package asset and Windows to the vendored one, so a spec
    file that stops bundling one source cannot break that platform's build."""
    sources = dict(zip(("vendored", "package"), two_sources))
    existing = create(sources[which])
    monkeypatch.setattr(vad_mod, "IS_WINDOWS", is_windows)
    assert silero_model_path() == str(existing)


def test_a_directory_is_not_a_candidate(two_sources, monkeypatch):
    vendored, package = two_sources
    vendored.mkdir(parents=True)  # a directory with the right name is not the model
    create(package)
    monkeypatch.setattr(vad_mod, "IS_WINDOWS", False)
    assert silero_model_path() == str(package)


def test_no_candidate_raises_audio_device_error_listing_the_candidates(two_sources, monkeypatch):
    vendored, package = two_sources
    monkeypatch.setattr(vad_mod, "IS_WINDOWS", False)
    with pytest.raises(AudioDeviceError) as e:
        silero_model_path()
    assert e.value.user_message
    assert str(vendored) in e.value.details
    assert str(package) in e.value.details
    assert e.value.details.startswith("not found: ")


def test_no_candidate_at_all_still_names_the_file(monkeypatch):
    monkeypatch.setattr(vad_mod, "_vendored_model_path", lambda: "")
    monkeypatch.setattr(vad_mod, "_package_model_path", lambda: "")
    assert silero_model_candidates() == []
    with pytest.raises(AudioDeviceError) as e:
        silero_model_path()
    assert MODEL_FILENAME in e.value.details


def test_the_streaming_vad_rejects_a_missing_explicit_path(tmp_path):
    with pytest.raises(AudioDeviceError):
        vad_mod.SileroStreamingVad(str(tmp_path / "nope.onnx"))


# ---------------------------------------------------------------------------
# where the vendored copy is looked for
# ---------------------------------------------------------------------------


def test_vendored_path_is_repo_assets_vad_when_not_frozen(monkeypatch):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert Path(vad_mod._vendored_model_path()) == VENDORED


def test_vendored_path_is_the_bundle_when_frozen(monkeypatch, tmp_path):
    """Mirrors app.ui.assets.assets_dir(): sys._MEIPASS is the onedir _internal folder."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert Path(vad_mod._vendored_model_path()) == tmp_path.joinpath(*VENDORED_SUBDIR, MODEL_FILENAME)


def test_meipass_is_ignored_when_not_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert Path(vad_mod._vendored_model_path()) == VENDORED


def test_this_checkout_resolves_a_real_model_file():
    path = silero_model_path()
    assert os.path.isfile(path)
    assert os.path.basename(path) == MODEL_FILENAME


# ---------------------------------------------------------------------------
# the vendored asset really is the model SileroStreamingVad feeds
# ---------------------------------------------------------------------------


def test_the_vendored_asset_is_present_in_the_repo():
    assert VENDORED.is_file()
    assert VENDORED.stat().st_size > 1_000_000  # ~1.2 MB; a LFS pointer or stub would not be
    assert (VENDORED.parent / "README.md").is_file()  # provenance


def test_the_vendored_asset_has_the_signature_the_streaming_vad_relies_on():
    import onnxruntime

    session = onnxruntime.InferenceSession(str(VENDORED), providers=["CPUExecutionProvider"])
    inputs = {i.name: i.shape for i in session.get_inputs()}
    assert set(inputs) == {"input", "h", "c"}
    assert inputs["input"][1:] == [CONTEXT_SAMPLES + VAD_FRAME_SAMPLES]  # 64 + 512 = 576
    assert isinstance(inputs["input"][0], str)  # dynamic: "seq_len"
    assert inputs["h"] == list(STATE_SHAPE) == [1, 1, 128]
    assert inputs["c"] == list(STATE_SHAPE) == [1, 1, 128]
    assert [o.name for o in session.get_outputs()][1:] == ["hn", "cn"]


def test_the_vendored_asset_runs_one_streaming_frame():
    v = vad_mod.SileroStreamingVad(str(VENDORED))
    prob = v.speech_prob(np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32))
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
    v.reset()
    with pytest.raises(ValueError):
        v.speech_prob(np.zeros(VAD_FRAME_SAMPLES + 1, dtype=np.float32))
