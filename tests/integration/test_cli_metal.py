"""End-to-end CLI tests on the real Metal GPU and ggml model (marker ``metal``).

The macOS counterpart of test_cli_gpu.py. Every test runs ``python -m app.cli`` as a subprocess
(cwd = repo root) with isolated ROENSPEAK_DATA_DIR / ROENSPEAK_CONFIG_DIR temp dirs (empty
config -> Config() defaults) and ROENSPEAK_MODELS_DIR = the real models dir, and asserts only
the CLI output contract in app/contracts.py plus the LOG_* lines in
<DATA_DIR>/logs/roenspeak.log.

Run on macOS:  .venv/bin/python -m pytest tests/integration -m metal -q -s
(-s prints the measured latencies and the English text.)

What differs from the Windows contract, and why (see also ARCHITECTURE.md section 12.1):

* ``EngineInfo.device`` is ``"metal"``, not ``"cuda"``.
* ``EngineInfo.compute_type`` is the **ggml** type baked into the model file (``f16`` for
  ``--compute-type float16``, ``q5_0`` for ``int8_float16``), because whisper.cpp has no
  runtime compute type. ``helpers.expected_compute_type()`` owns that mapping.
* ``cublas_version`` / ``cudnn_version`` are always None, and the ``cuda_probe`` log line
  carries ``cublas=none cudnn=none`` so the same grep works on both platforms.
* ``--cpu`` still reports ``device=cpu``, but ``compute_type`` stays the ggml file's type
  (``f16``), not ``contracts.CPU_COMPUTE_TYPE`` -- there is no separate CPU quantization:
  whisper.cpp runs the very same file on the CPU backend.

Why the error-path tests (exit 3 / exit 4 / broken engine library) are also marked ``metal``:
they spawn the real CLI, which imports app.core.platform, builds a MetalProbe and dlopens
libroenspeak_whisper.dylib. The contract doesn't say whether the model-presence check comes
before or after the GPU probe (in app/cli.py today the probe is step 2 and the model check is
step 3), so "model missing -> 3" is only deterministic on a machine where Metal works. The
``--cpu`` variants remove that ambiguity. They don't need the model itself and skip nothing.
"""

from __future__ import annotations

import dataclasses
import re
import sys

import numpy as np
import pytest

from app import contracts

from .helpers import (
    REAL_MODELS_DIR,
    RO_SAMPLE_WAV,
    assert_english,
    assert_romanian,
    broken_gpu_env,
    expected_compute_type,
    ggml_model_downloaded,
    ggml_model_file_path,
    log_records,
    require_ggml_model,
    run_cli,
    safe_print,
    wav_duration_s,
    write_wav,
)

#: skipif as well as the marker: the ``metal`` marker says "needs Metal", and tests/conftest.py
#: enforces that, but a bare ``pytest -m metal`` on a non-macOS host would otherwise still try
#: to import and collect this module's macOS-only expectations.
pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only (whisper.cpp on Metal)"),
]

#: Generous per-segment bound for warm large-v3 f16 on Apple silicon.
#:
#: Measured on this M1 Max with the 9.4 s / 3-phrase fixture: 511 / 640 / 511 ms per segment
#: (ARCHITECTURE.md section 12.1 records ~0.9 s per phrase and a <=1.5 s budget). 4000 ms is
#: ~6x the measured value and ~2.7x the architectural budget. The headroom is deliberate and
#: covers the three things that legitimately make a run slower without being a regression:
#: the extra ~450 ms of Metal pipeline-state creation on the first inference if the warm-up
#: ever stops absorbing it, thermal throttling on a fanless Mac, and a developer machine that
#: is busy doing something else. A tighter bound would make the suite flaky, which is worse
#: than missing a 2x slowdown -- the printed numbers (-s) are what you read for real timings.
MAX_INFER_MS = 4000.0
#: In --realtime mode latency_ms (t_infer_end - t_vad_close) should be close to infer_ms.
MAX_REALTIME_LATENCY_MS = 4000.0

SEGMENT_KEYS = {"id", "reason", "audio_s", "infer_ms", "latency_ms", "filtered", "filter_reason", "text"}
SUMMARY_KEYS = {"segments", "kept", "total_audio_s", "total_infer_ms", "wall_ms"}
ENGINE_KEYS = {f.name for f in dataclasses.fields(contracts.EngineInfo)}
REASONS = {r.value for r in contracts.SegmentCloseReason}

#: The fixture is three sentences separated by 1.2 s pauses (> min_silence_ms=500).
EXPECTED_SEGMENTS = 3

ENGINE_RE = re.compile(
    r"^ENGINE model=(?P<model>\S+) device=(?P<device>\S+) compute_type=(?P<compute_type>\S+) "
    r"gpu=(?P<gpu>.*) load_ms=(?P<load_ms>[\d.]+)$"
)
SEGMENT_RE = re.compile(
    r"^SEGMENT id=(?P<id>\d+) reason=(?P<reason>\w+) audio_s=(?P<audio_s>[\d.]+) "
    r"infer_ms=(?P<infer_ms>[\d.]+) filtered=(?P<filtered>[01]) text=(?P<text>.*)$"
)
SUMMARY_RE = re.compile(
    r"^SUMMARY segments=(?P<segments>\d+) kept=(?P<kept>\d+) total_audio_s=(?P<total_audio_s>[\d.]+) "
    r"total_infer_ms=(?P<total_infer_ms>[\d.]+) wall_ms=(?P<wall_ms>[\d.]+)$"
)


def kept_text(doc: dict) -> str:
    return " ".join(s["text"] for s in doc["segments"] if not s["filtered"] and s["text"])


def report(label: str, doc: dict) -> None:
    """Print measured numbers (visible with -s / in the captured output on failure)."""
    eng = doc["engine"]
    safe_print(f"\n[{label}] engine device={eng['device']} compute_type={eng['compute_type']} "
               f"gpu={eng.get('gpu_name')} macos={eng.get('driver_version')} load_ms={eng.get('load_ms')}")
    for s in doc["segments"]:
        safe_print(f"[{label}]   seg id={s['id']} reason={s['reason']} audio_s={s['audio_s']} "
                   f"infer_ms={s['infer_ms']} latency_ms={s['latency_ms']} filtered={s['filtered']} "
                   f"text={s['text']!r}")
    safe_print(f"[{label}] summary {doc['summary']}")


def check_json_shape(doc: dict) -> None:
    assert set(doc) >= {"engine", "segments", "summary"}, doc.keys()
    assert set(doc["engine"]) >= ENGINE_KEYS, f"engine keys {set(doc['engine'])} lack {ENGINE_KEYS}"
    assert set(doc["summary"]) >= SUMMARY_KEYS
    ids = []
    for s in doc["segments"]:
        assert set(s) >= SEGMENT_KEYS, f"segment keys {set(s)}"
        assert s["reason"] in REASONS, s
        assert isinstance(s["infer_ms"], (int, float)) and s["infer_ms"] > 0, s
        assert isinstance(s["filtered"], bool), s
        if s["filtered"]:
            assert s["text"] == "" and s["filter_reason"], s
        else:
            assert s["filter_reason"] is None, s
        # CLI mode: latency_ms = t_infer_end - t_vad_close >= infer_ms
        assert isinstance(s["latency_ms"], (int, float)), s
        assert s["latency_ms"] >= s["infer_ms"] - 1.0, s
        ids.append(s["id"])
    assert ids == sorted(ids) and len(set(ids)) == len(ids), f"segment ids not strictly increasing: {ids}"
    summ = doc["summary"]
    assert summ["segments"] == len(doc["segments"])
    assert summ["kept"] == sum(1 for s in doc["segments"] if not s["filtered"])
    # Ambiguous in the contract whether totals cover only segments or also input audio / warm-up,
    # so only lower bounds are checked.
    assert summ["total_infer_ms"] >= sum(s["infer_ms"] for s in doc["segments"]) - 2.0, summ
    assert summ["total_audio_s"] >= 0, summ
    assert summ["wall_ms"] > 0, summ


def assert_metal_engine(doc: dict, compute_type=contracts.ComputeType.FLOAT16) -> None:
    """The macOS analogue of assert_cuda_engine()."""
    eng = doc["engine"]
    assert eng["device"] == "metal", eng
    assert eng["compute_type"] == expected_compute_type(compute_type), eng
    assert eng["gpu_name"], eng
    # MetalProbe puts the macOS product version where CUDA puts the driver version, and leaves
    # the two CUDA-only fields empty on purpose (see app/engine/metal_probe.py).
    assert eng["driver_version"], eng
    assert eng["cublas_version"] is None and eng["cudnn_version"] is None, eng


# ---------------------------------------------------------------------------
# Default run (shared by several assertions; the model is loaded once for these)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def default_run(tmp_path_factory):
    require_ggml_model()
    r = run_cli([RO_SAMPLE_WAV, "--json"], tmp_path_factory.mktemp("cli_default"))
    assert r.returncode == 0, r.describe()
    doc = r.json()
    report("default", doc)
    return r, doc


def test_default_json_shape_and_metal_f16(default_run):
    r, doc = default_run
    check_json_shape(doc)
    eng = doc["engine"]
    assert eng["model"] == contracts.DEFAULT_MODEL
    assert_metal_engine(doc)
    assert eng["compute_type"] == "f16", f"expected the real ggml compute type, got {eng}"
    assert "WARNING: CPU mode" not in r.stderr


def test_default_translation_is_english_and_segmented(default_run):
    _, doc = default_run
    assert len(doc["segments"]) == EXPECTED_SEGMENTS, (
        f"expected {EXPECTED_SEGMENTS} VAD segments (one per fixture sentence), got {doc['segments']}"
    )
    assert doc["summary"]["kept"] == EXPECTED_SEGMENTS, doc["summary"]
    text = kept_text(doc)
    safe_print(f"[default] english: {text!r}")
    assert_english(text)
    # The fixture's middle sentence is the one with the content words; require more than a
    # single lucky keyword so a half-broken translation can't pass.
    lowered = text.casefold()
    assert "weather" in lowered and "park" in lowered, f"middle sentence not translated: {text!r}"


def test_default_latency_within_bound(default_run):
    _, doc = default_run
    slow = [s for s in doc["segments"] if s["infer_ms"] > MAX_INFER_MS]
    assert not slow, f"infer_ms above {MAX_INFER_MS} ms: {slow}"
    safe_print("[default] infer_ms:", [s["infer_ms"] for s in doc["segments"]])


def test_default_latency_and_engine_logged(default_run):
    r, doc = default_run
    log = r.log_text()
    assert log, f"no log file at {r.log_path}"

    loaded = log_records(log, contracts.LOG_ENGINE_LOADED)
    assert loaded, f"no '{contracts.LOG_ENGINE_LOADED}' line in log:\n{log[-3000:]}"
    rec = loaded[-1]
    assert rec.get("device") == "metal", rec
    assert rec.get("compute_type") == "f16", rec
    assert rec.get("model") == contracts.DEFAULT_MODEL, rec
    # Same keys as on Windows, with the CUDA-only ones explicitly "none".
    assert {"device", "compute_type", "model", "gpu", "driver", "cublas", "cudnn", "load_ms"} <= set(rec), rec
    assert rec["cublas"] == "none" and rec["cudnn"] == "none", rec
    float(rec["load_ms"])

    probe = log_records(log, contracts.LOG_CUDA_PROBE)
    assert probe, f"no '{contracts.LOG_CUDA_PROBE}' line in log:\n{log[-3000:]}"
    p = probe[-1]
    assert {"ok", "gpu", "driver", "cublas", "cudnn", "error"} <= set(p), p
    assert p["ok"] == "1", p
    assert p["cublas"] == "none" and p["cudnn"] == "none", p
    assert p["error"] == "none", p

    lat = log_records(log, contracts.LOG_SEGMENT_LATENCY)
    assert len(lat) >= len(doc["segments"]), (
        f"{len(lat)} segment_latency lines for {len(doc['segments'])} segments"
    )
    for rec in lat:
        assert {"session", "id", "reason", "audio_ms", "infer_ms", "latency_ms", "chars", "filtered"} <= set(rec), rec
        float(rec["latency_ms"])
        float(rec["infer_ms"])
    safe_print("\n[default] log segment_latency:",
               [(x.get("id"), x.get("infer_ms"), x.get("latency_ms")) for x in lat])


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


def test_text_mode_output_format(tmp_path):
    require_ggml_model()
    r = run_cli([RO_SAMPLE_WAV], tmp_path)
    assert r.returncode == 0, r.describe()
    lines = [ln.rstrip("\r") for ln in r.stdout.splitlines() if ln.strip()]
    engine = [ENGINE_RE.match(ln) for ln in lines if ln.startswith("ENGINE")]
    segs = [SEGMENT_RE.match(ln) for ln in lines if ln.startswith("SEGMENT")]
    summ = [SUMMARY_RE.match(ln) for ln in lines if ln.startswith("SUMMARY")]
    other = [ln for ln in lines if not ln.startswith(("ENGINE", "SEGMENT", "SUMMARY"))]
    assert not other, f"unexpected stdout lines: {other}\n{r.describe()}"
    assert len(engine) == 1 and engine[0], f"bad ENGINE line:\n{r.stdout}"
    assert len(summ) == 1 and summ[0], f"bad SUMMARY line:\n{r.stdout}"
    assert all(segs), f"bad SEGMENT line(s):\n{r.stdout}"
    assert lines[0].startswith("ENGINE") and lines[-1].startswith("SUMMARY")
    e = engine[0]
    assert (e["model"], e["device"], e["compute_type"]) == (contracts.DEFAULT_MODEL, "metal", "f16")
    assert e["gpu"] and e["gpu"] != "none", f"ENGINE line has no gpu name: {e.group(0)!r}"
    assert len(segs) == EXPECTED_SEGMENTS
    assert all(m["reason"] in REASONS for m in segs)
    s = summ[0]
    assert int(s["segments"]) == len(segs)
    assert int(s["kept"]) == sum(m["filtered"] == "0" for m in segs)
    assert_english(" ".join(m["text"] for m in segs if m["filtered"] == "0"))


def test_transcribe_task_returns_romanian(tmp_path):
    require_ggml_model()
    r = run_cli([RO_SAMPLE_WAV, "--json", "--task", "transcribe"], tmp_path)
    assert r.returncode == 0, r.describe()
    doc = r.json()
    report("transcribe", doc)
    check_json_shape(doc)
    assert_metal_engine(doc)
    assert_romanian(kept_text(doc))


def test_transcribe_text_mode_survives_non_utf8_pipe(tmp_path):
    """Romanian output (ș, ț, ă) printed to a pipe with PYTHONIOENCODING/PYTHONUTF8 removed
    must not crash the CLI. macOS defaults to UTF-8, so this is weaker here than on Windows
    (where the default pipe encoding is cp1252), but _utf8_stdio() is shared code and this
    keeps the two suites symmetrical. Contract: exit 0 = ok."""
    require_ggml_model()
    r = run_cli([RO_SAMPLE_WAV, "--task", "transcribe"], tmp_path, utf8=False)
    assert r.returncode == 0, r.describe()
    assert "SUMMARY" in r.stdout


def test_beam_1(tmp_path):
    require_ggml_model()
    r = run_cli([RO_SAMPLE_WAV, "--json", "--beam", "1"], tmp_path)
    assert r.returncode == 0, r.describe()
    doc = r.json()
    report("beam1", doc)
    check_json_shape(doc)
    assert_metal_engine(doc)
    assert_english(kept_text(doc))


def test_int8_float16_compute_type(tmp_path):
    """int8_float16 selects the q5_0 ggml file, which is a *separate download*. Skips cleanly
    (never downloads) when it is not on disk -- a 1.1 GB fetch is not a test's business."""
    require_ggml_model(contracts.DEFAULT_MODEL, contracts.ComputeType.INT8_FLOAT16)
    r = run_cli([RO_SAMPLE_WAV, "--json", "--compute-type", "int8_float16"], tmp_path)
    assert r.returncode == 0, r.describe()
    doc = r.json()
    report("int8_float16", doc)
    check_json_shape(doc)
    assert_metal_engine(doc, contracts.ComputeType.INT8_FLOAT16)
    assert doc["engine"]["compute_type"] == "q5_0", doc["engine"]
    assert_english(kept_text(doc))
    loaded = log_records(r.log_text(), contracts.LOG_ENGINE_LOADED)
    assert loaded and loaded[-1].get("compute_type") == "q5_0", loaded


def test_realtime_latency(tmp_path):
    """--realtime feeds at 1x speed, so each segment is inferred as soon as it closes and
    latency_ms (VAD close -> text) is the meaningful end-of-phrase latency."""
    require_ggml_model()
    r = run_cli([RO_SAMPLE_WAV, "--json", "--realtime"], tmp_path)
    assert r.returncode == 0, r.describe()
    doc = r.json()
    report("realtime", doc)
    check_json_shape(doc)
    assert doc["summary"]["wall_ms"] >= wav_duration_s(RO_SAMPLE_WAV) * 1000 * 0.9, doc["summary"]
    slow = [s for s in doc["segments"] if s["latency_ms"] > MAX_REALTIME_LATENCY_MS]
    assert not slow, f"latency_ms above {MAX_REALTIME_LATENCY_MS} ms in realtime mode: {slow}"
    assert_english(kept_text(doc))
    safe_print("[realtime] latency_ms:", [s["latency_ms"] for s in doc["segments"]])


# ---------------------------------------------------------------------------
# Hallucinations on silence / noise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["silence", "noise"])
def test_no_hallucinated_text_on_silence_or_noise(tmp_path, kind):
    require_ggml_model()
    n = 6 * contracts.SAMPLE_RATE
    if kind == "silence":
        audio = np.zeros(n, np.float32)
    else:
        rng = np.random.default_rng(1234)
        audio = (rng.standard_normal(n) * 0.01).astype(np.float32)  # ~ -40 dBFS white noise
    wav = write_wav(tmp_path / f"{kind}.wav", audio)
    r = run_cli([wav, "--json"], tmp_path)
    assert r.returncode == 0, r.describe()
    doc = r.json()
    report(kind, doc)
    check_json_shape(doc)
    assert doc["summary"]["kept"] == 0, f"hallucinated text kept for {kind}: {kept_text(doc)!r}"
    assert kept_text(doc) == ""


# ---------------------------------------------------------------------------
# Metal unavailable: never a silent CPU fallback
# ---------------------------------------------------------------------------


def test_broken_engine_library_without_cpu_flag_fails_with_exit_2(tmp_path):
    """There is no "hide the GPU" switch on a Mac -- Metal is part of the OS. The honest
    stand-in for "no usable Metal device" is to break what actually answers that question:
    MetalProbe loads libroenspeak_whisper.dylib and asks it, and $ROENSPEAK_WHISPER_LIB is
    first in whispercpp.library_candidates(). helpers.broken_gpu_env() points it at a file
    that is not a Mach-O object. Expected per contract: CudaUnavailableError -> exit 2, a
    message naming Metal, no inference, and no CPU fallback."""
    require_ggml_model()
    r = run_cli([RO_SAMPLE_WAV], tmp_path, extra_env=broken_gpu_env(tmp_path))
    assert r.returncode == 2, r.describe()
    assert re.search(r"metal", r.stderr, re.I), f"stderr lacks a Metal explanation:\n{r.stderr}"
    assert re.search(r"macos", r.stderr, re.I), f"stderr lacks the macOS requirement:\n{r.stderr}"
    assert "SEGMENT" not in r.stdout, r.describe()
    assert "device=cpu" not in r.stdout, r.describe()
    for rec in log_records(r.log_text(), contracts.LOG_ENGINE_LOADED):
        assert rec.get("device") != "cpu", f"silent CPU fallback logged: {rec}"
    probe = log_records(r.log_text(), contracts.LOG_CUDA_PROBE)
    assert probe and probe[-1].get("ok") == "0", probe
    assert probe[-1].get("error") not in (None, "none"), probe


def _cpu_model() -> str:
    if ggml_model_downloaded("medium"):
        return "medium"
    require_ggml_model(contracts.DEFAULT_MODEL)
    return contracts.DEFAULT_MODEL


@pytest.mark.slow
def test_cpu_flag_runs_on_cpu_with_warning(tmp_path):
    """Opt-in CPU mode. Uses medium if downloaded, else large-v3, beam 1 (whisper.cpp on the
    CPU backend is ~6x slower than Metal: ~3.2 s instead of ~0.55 s per phrase here, so the
    9 s fixture takes ~15 s). Note the compute type stays the ggml file's type: unlike
    CTranslate2 on Windows, whisper.cpp has no separate CPU quantization, so
    contracts.CPU_COMPUTE_TYPE does not apply on macOS."""
    model = _cpu_model()
    r = run_cli([RO_SAMPLE_WAV, "--json", "--cpu", "--model", model, "--beam", "1"], tmp_path, timeout=900)
    assert r.returncode == 0, r.describe()
    assert "WARNING: CPU mode (slow)" in r.stderr, r.stderr
    doc = r.json()
    report(f"cpu-{model}", doc)
    check_json_shape(doc)
    assert doc["engine"]["device"] == "cpu", doc["engine"]
    assert doc["engine"]["compute_type"] == "f16", doc["engine"]
    assert doc["engine"]["model"] == model
    assert_english(kept_text(doc))
    loaded = log_records(r.log_text(), contracts.LOG_ENGINE_LOADED)
    assert loaded and loaded[-1].get("device") == "cpu", loaded


def test_cpu_flag_does_not_blame_the_gpu_for_a_broken_library(tmp_path):
    """--cpu skips the Metal probe entirely (app/cli.py step 2), so a broken engine library in
    CPU mode must not be reported as "GPU unavailable". whisper.cpp still lives in that same
    dylib, so the load fails -- but as a broken *installation*: exit 1 ("other error") with the
    "build it with ./native/whispercpp/build.sh / reinstall" wording, not exit 2."""
    require_ggml_model()
    r = run_cli([RO_SAMPLE_WAV, "--cpu"], tmp_path, extra_env=broken_gpu_env(tmp_path))
    assert r.returncode != 2, f"--cpu still reported the GPU as the problem:\n{r.describe()}"
    assert r.returncode == 1, r.describe()
    assert "WARNING: CPU mode (slow)" in r.stderr, r.stderr
    assert re.search(r"build\.sh|reinstall", r.stderr, re.I), (
        f"stderr does not point at a broken installation:\n{r.stderr}"
    )
    assert "SEGMENT" not in r.stdout, r.describe()


# ---------------------------------------------------------------------------
# Error exit codes
# ---------------------------------------------------------------------------


def test_model_not_downloaded_exit_3(tmp_path):
    empty = tmp_path / "empty_models"
    empty.mkdir()
    r = run_cli([RO_SAMPLE_WAV], tmp_path, models_dir=empty)
    assert r.returncode == 3, r.describe()
    assert "SEGMENT" not in r.stdout
    assert not any(empty.iterdir()), "CLI must not download without --download"


def test_model_not_downloaded_exit_3_cpu(tmp_path):
    """Same, with --cpu, so the result doesn't depend on whether Metal is probed first."""
    empty = tmp_path / "empty_models"
    empty.mkdir()
    r = run_cli([RO_SAMPLE_WAV, "--cpu"], tmp_path, models_dir=empty)
    assert r.returncode == 3, r.describe()
    assert not any(empty.iterdir())


def test_partial_model_dir_counts_as_not_downloaded(tmp_path):
    """A model dir with the ggml file but no MODEL_COMPLETE_MARKER (interrupted download) is
    "not downloaded"."""
    models = tmp_path / "partial_models"
    d = models / contracts.DEFAULT_MODEL
    d.mkdir(parents=True)
    (d / contracts.ggml_model_file(contracts.DEFAULT_MODEL, contracts.ComputeType.FLOAT16)).write_bytes(b"ggml")
    r = run_cli([RO_SAMPLE_WAV, "--cpu"], tmp_path, models_dir=models)
    assert r.returncode == 3, r.describe()


def test_other_quantization_on_disk_counts_as_not_downloaded(tmp_path):
    """The marker is shared by both quantizations in one model dir, so "downloaded" has to
    mean "the *selected* ggml file is there" (GgmlModelDownloader.is_downloaded). A complete
    f16 download must still report q5_0 as missing."""
    models = tmp_path / "f16_only"
    d = models / contracts.DEFAULT_MODEL
    d.mkdir(parents=True)
    (d / contracts.ggml_model_file(contracts.DEFAULT_MODEL, contracts.ComputeType.FLOAT16)).write_bytes(b"ggml")
    (d / contracts.MODEL_COMPLETE_MARKER).write_text("{}", encoding="utf-8")
    r = run_cli([RO_SAMPLE_WAV, "--cpu", "--compute-type", "int8_float16"], tmp_path, models_dir=models)
    assert r.returncode == 3, r.describe()


def test_missing_wav_exit_4(tmp_path):
    require_ggml_model()  # exit-code precedence vs. model check is unspecified
    r = run_cli([tmp_path / "does_not_exist.wav"], tmp_path)
    assert r.returncode == 4, r.describe()
    assert "SEGMENT" not in r.stdout


def test_unreadable_wav_exit_4(tmp_path):
    require_ggml_model()  # exit-code precedence vs. model check is unspecified
    bad = tmp_path / "garbage.wav"
    bad.write_bytes(b"this is not a RIFF/WAVE file" * 10)
    r = run_cli([bad], tmp_path)
    assert r.returncode == 4, r.describe()
    assert "SEGMENT" not in r.stdout


@pytest.mark.parametrize(
    "bad_args",
    [["--compute-type", "float32"], ["--task", "summarize"], ["--beam", "notanumber"], ["--beam", "0"]],
    ids=["compute_type", "task", "beam_text", "beam_range"],
)
def test_invalid_args_exit_4(tmp_path, bad_args):
    """Contract: invalid args -> 4. (argparse's default exit code for usage errors is 2,
    which would collide with "Metal/GPU unavailable".)"""
    r = run_cli([RO_SAMPLE_WAV, *bad_args], tmp_path)
    assert r.returncode == 4, r.describe()


def test_models_dir_is_real_for_metal_tests():
    """Sanity: tells you up front why the model-dependent tests would skip, and guards against
    a stray ROENSPEAK_TEST_MODELS_DIR pointing at a fixture dir instead of the real one."""
    if not ggml_model_downloaded(contracts.DEFAULT_MODEL):
        pytest.skip(f"ggml {contracts.DEFAULT_MODEL} not downloaded in {REAL_MODELS_DIR}")
    path = ggml_model_file_path(contracts.DEFAULT_MODEL)
    size = path.stat().st_size
    safe_print(f"\n[guard] real ggml model {path} ({size / 1e9:.2f} GB)")
    assert size > 2_000_000_000, f"{path} is {size} bytes; that is not the real large-v3 f16 model"


def test_engine_dylibs_are_built():
    """The seven dylibs native/whispercpp/build.sh stages. Reported as a skip (not a failure)
    because "not built yet" is a setup state, not a regression."""
    from app.engine.whispercpp import WHISPER_LIB_NAME, library_path

    path = library_path()
    if path is None:
        pytest.skip(f"{WHISPER_LIB_NAME} is not built; run ./native/whispercpp/build.sh")
    siblings = sorted(p.name for p in path.parent.glob("*.dylib"))
    safe_print(f"\n[guard] engine lib {path}; dylibs in that dir: {siblings}")
    assert len(siblings) >= 7, f"expected the shim + 6 ggml/whisper dylibs, found {siblings}"
