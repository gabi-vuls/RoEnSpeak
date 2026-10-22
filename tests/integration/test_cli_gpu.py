"""End-to-end CLI tests on the real GPU and model (marker ``gpu``).

Every test runs ``python -m app.cli`` as a subprocess (cwd = repo root) with isolated
ROENSPEAK_DATA_DIR / ROENSPEAK_CONFIG_DIR temp dirs (empty config -> Config() defaults) and
ROENSPEAK_MODELS_DIR = the real models dir, and asserts only the CLI output contract in
app/contracts.py plus the LOG_* lines in <DATA_DIR>/logs/roenspeak.log.

Run on Windows:  scripts/win.sh -m pytest tests/integration -m gpu -q -s
(-s prints the measured latencies.)

Why the error-path tests (exit 3 / exit 4 / CUDA hidden) are also marked ``gpu``: they spawn the
real CLI, which imports faster-whisper/ctranslate2 and probes CUDA. The contract doesn't say
whether the model-presence check comes before or after the CUDA probe, so "model missing -> 3"
is only deterministic on a machine where CUDA works. The --cpu variant removes that ambiguity.
They don't need the model itself and skip nothing.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import pytest

from app import contracts

from .helpers import (
    safe_print,
    RO_SAMPLE_WAV,
    REAL_MODELS_DIR,
    assert_english,
    assert_romanian,
    log_records,
    model_downloaded,
    require_model,
    run_cli,
    wav_duration_s,
    write_wav,
)

pytestmark = pytest.mark.gpu

#: Generous per-segment bound for warm large-v3 float16 beam 5 on an RTX 4060 Ti
#: (budget in ARCHITECTURE.md is ~300-700 ms). The first segment may pay CUDA kernel/JIT
#: warm-up if the CLI does not warm up, which is why the bound is loose.
MAX_INFER_MS = 3000.0
#: In --realtime mode latency_ms (t_infer_end - t_vad_close) should be close to infer_ms.
MAX_REALTIME_LATENCY_MS = 3000.0

SEGMENT_KEYS = {"id", "reason", "audio_s", "infer_ms", "latency_ms", "filtered", "filter_reason", "text"}
SUMMARY_KEYS = {"segments", "kept", "total_audio_s", "total_infer_ms", "wall_ms"}
ENGINE_KEYS = {f.name for f in dataclasses.fields(contracts.EngineInfo)}
REASONS = {r.value for r in contracts.SegmentCloseReason}

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
          f"gpu={eng.get('gpu_name')} load_ms={eng.get('load_ms')}")
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


def assert_cuda_engine(doc: dict, compute_type: str) -> None:
    eng = doc["engine"]
    assert eng["device"] == "cuda", eng
    assert eng["compute_type"] == compute_type, eng
    assert eng["gpu_name"], eng


# ---------------------------------------------------------------------------
# Default run (shared by several assertions; the model is loaded once for these)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def default_run(tmp_path_factory):
    require_model()
    r = run_cli([RO_SAMPLE_WAV, "--json"], tmp_path_factory.mktemp("cli_default"))
    assert r.returncode == 0, r.describe()
    doc = r.json()
    report("default", doc)
    return r, doc


def test_default_json_shape_and_cuda_float16(default_run):
    r, doc = default_run
    check_json_shape(doc)
    eng = doc["engine"]
    assert eng["model"] == contracts.DEFAULT_MODEL
    assert_cuda_engine(doc, "float16")
    assert "WARNING: CPU mode" not in r.stderr


def test_default_translation_is_english_and_segmented(default_run):
    _, doc = default_run
    # The fixture has three sentences separated by 1.2 s pauses (> min_silence_ms=500).
    assert len(doc["segments"]) >= 2, f"expected >= 2 VAD segments, got {doc['segments']}"
    assert doc["summary"]["kept"] >= 2, doc["summary"]
    assert_english(kept_text(doc))


def test_default_latency_within_bound(default_run):
    _, doc = default_run
    slow = [s for s in doc["segments"] if s["infer_ms"] > MAX_INFER_MS]
    assert not slow, f"infer_ms above {MAX_INFER_MS} ms: {slow}"


def test_default_latency_and_engine_logged(default_run):
    r, doc = default_run
    log = r.log_text()
    assert log, f"no log file at {r.log_path}"
    loaded = log_records(log, contracts.LOG_ENGINE_LOADED)
    assert loaded, f"no '{contracts.LOG_ENGINE_LOADED}' line in log:\n{log[-3000:]}"
    assert loaded[-1].get("device") == "cuda", loaded[-1]
    assert loaded[-1].get("compute_type") == "float16", loaded[-1]
    lat = log_records(log, contracts.LOG_SEGMENT_LATENCY)
    assert len(lat) >= len(doc["segments"]), f"{len(lat)} segment_latency lines for {len(doc['segments'])} segments"
    for rec in lat:
        assert "latency_ms" in rec and "infer_ms" in rec, rec
        float(rec["latency_ms"])
        float(rec["infer_ms"])
    safe_print("\n[default] log segment_latency:", [(x.get("id"), x.get("infer_ms"), x.get("latency_ms")) for x in lat])


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


def test_text_mode_output_format(tmp_path):
    require_model()
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
    assert (e["model"], e["device"], e["compute_type"]) == (contracts.DEFAULT_MODEL, "cuda", "float16")
    assert len(segs) >= 2
    assert all(m["reason"] in REASONS for m in segs)
    s = summ[0]
    assert int(s["segments"]) == len(segs)
    assert int(s["kept"]) == sum(m["filtered"] == "0" for m in segs)
    assert_english(" ".join(m["text"] for m in segs if m["filtered"] == "0"))


def test_transcribe_task_returns_romanian(tmp_path):
    require_model()
    r = run_cli([RO_SAMPLE_WAV, "--json", "--task", "transcribe"], tmp_path)
    assert r.returncode == 0, r.describe()
    doc = r.json()
    report("transcribe", doc)
    check_json_shape(doc)
    assert_cuda_engine(doc, "float16")
    assert_romanian(kept_text(doc))


def test_transcribe_text_mode_survives_non_utf8_pipe(tmp_path):
    """Romanian output (ș, ț, ă) printed to a pipe with the Windows default (cp1252) encoding
    must not crash the CLI. Contract: exit 0 = ok. Not stated explicitly; flagged as robustness."""
    require_model()
    r = run_cli([RO_SAMPLE_WAV, "--task", "transcribe"], tmp_path, utf8=False)
    assert r.returncode == 0, r.describe()
    assert "SUMMARY" in r.stdout


def test_beam_1(tmp_path):
    require_model()
    r = run_cli([RO_SAMPLE_WAV, "--json", "--beam", "1"], tmp_path)
    assert r.returncode == 0, r.describe()
    doc = r.json()
    report("beam1", doc)
    check_json_shape(doc)
    assert_cuda_engine(doc, "float16")
    assert_english(kept_text(doc))


def test_int8_float16_compute_type(tmp_path):
    require_model()
    r = run_cli([RO_SAMPLE_WAV, "--json", "--compute-type", "int8_float16"], tmp_path)
    assert r.returncode == 0, r.describe()
    doc = r.json()
    report("int8_float16", doc)
    assert_cuda_engine(doc, "int8_float16")
    assert_english(kept_text(doc))
    loaded = log_records(r.log_text(), contracts.LOG_ENGINE_LOADED)
    assert loaded and loaded[-1].get("compute_type") == "int8_float16", loaded


def test_realtime_latency(tmp_path):
    """--realtime feeds at 1x speed, so each segment is inferred as soon as it closes and
    latency_ms (VAD close -> text) is the meaningful end-of-phrase latency."""
    require_model()
    r = run_cli([RO_SAMPLE_WAV, "--json", "--realtime"], tmp_path)
    assert r.returncode == 0, r.describe()
    doc = r.json()
    report("realtime", doc)
    check_json_shape(doc)
    assert doc["summary"]["wall_ms"] >= wav_duration_s(RO_SAMPLE_WAV) * 1000 * 0.9, doc["summary"]
    slow = [s for s in doc["segments"] if s["latency_ms"] > MAX_REALTIME_LATENCY_MS]
    assert not slow, f"latency_ms above {MAX_REALTIME_LATENCY_MS} ms in realtime mode: {slow}"
    assert_english(kept_text(doc))


# ---------------------------------------------------------------------------
# Hallucinations on silence / noise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["silence", "noise"])
def test_no_hallucinated_text_on_silence_or_noise(tmp_path, kind):
    require_model()
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
# CUDA unavailable: never a silent CPU fallback
# ---------------------------------------------------------------------------


def test_cuda_hidden_without_cpu_flag_fails_with_exit_2(tmp_path):
    """CUDA_VISIBLE_DEVICES=-1 makes the CUDA runtime that ctranslate2 uses (and anything else
    using cudart) see zero devices, while the driver/NVML still exist. That's the closest
    reproducible stand-in for "no usable CUDA device" on this machine. Expected per contract:
    either the probe or ctranslate2 fails -> CudaUnavailableError -> exit 2, a message about
    the NVIDIA GPU/driver requirement, no inference, and no CPU fallback."""
    require_model()
    r = run_cli([RO_SAMPLE_WAV], tmp_path, extra_env={"CUDA_VISIBLE_DEVICES": "-1"})
    assert r.returncode == 2, r.describe()
    assert re.search(r"nvidia|driver", r.stderr, re.I), f"stderr lacks an NVIDIA/driver explanation:\n{r.stderr}"
    assert "SEGMENT" not in r.stdout, r.describe()
    assert "device=cpu" not in r.stdout, r.describe()
    for rec in log_records(r.log_text(), contracts.LOG_ENGINE_LOADED):
        assert rec.get("device") != "cpu", f"silent CPU fallback logged: {rec}"


def _cpu_model() -> str:
    if model_downloaded("medium"):
        return "medium"
    require_model(contracts.DEFAULT_MODEL)
    return contracts.DEFAULT_MODEL


@pytest.mark.slow
def test_cpu_flag_runs_on_cpu_with_warning(tmp_path):
    """Opt-in CPU mode. Uses medium if downloaded, else large-v3 (int8 on CPU, beam 1 to
    keep it to ~1-2 min on the 9 s fixture)."""
    model = _cpu_model()
    r = run_cli([RO_SAMPLE_WAV, "--json", "--cpu", "--model", model, "--beam", "1"], tmp_path, timeout=900)
    assert r.returncode == 0, r.describe()
    assert "WARNING: CPU mode (slow)" in r.stderr, r.stderr
    doc = r.json()
    report(f"cpu-{model}", doc)
    assert doc["engine"]["device"] == "cpu", doc["engine"]
    assert doc["engine"]["compute_type"] == contracts.CPU_COMPUTE_TYPE, doc["engine"]
    assert doc["engine"]["model"] == model
    assert_english(kept_text(doc))
    loaded = log_records(r.log_text(), contracts.LOG_ENGINE_LOADED)
    assert loaded and loaded[-1].get("device") == "cpu", loaded


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
    """Same, with --cpu, so the result doesn't depend on whether CUDA is probed first."""
    empty = tmp_path / "empty_models"
    empty.mkdir()
    r = run_cli([RO_SAMPLE_WAV, "--cpu"], tmp_path, models_dir=empty)
    assert r.returncode == 3, r.describe()
    assert not any(empty.iterdir())


def test_partial_model_dir_counts_as_not_downloaded(tmp_path):
    """A model dir without MODEL_COMPLETE_MARKER (interrupted download) is "not downloaded"."""
    models = tmp_path / "partial_models"
    d = models / contracts.DEFAULT_MODEL
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    r = run_cli([RO_SAMPLE_WAV, "--cpu"], tmp_path, models_dir=models)
    assert r.returncode == 3, r.describe()


def test_missing_wav_exit_4(tmp_path):
    require_model()  # exit-code precedence vs. model check is unspecified
    r = run_cli([tmp_path / "does_not_exist.wav"], tmp_path)
    assert r.returncode == 4, r.describe()
    assert "SEGMENT" not in r.stdout


def test_unreadable_wav_exit_4(tmp_path):
    require_model()  # exit-code precedence vs. model check is unspecified
    bad = tmp_path / "garbage.wav"
    bad.write_bytes(b"this is not a RIFF/WAVE file" * 10)
    r = run_cli([bad], tmp_path)
    assert r.returncode == 4, r.describe()
    assert "SEGMENT" not in r.stdout


@pytest.mark.parametrize(
    "bad_args",
    [["--compute-type", "float32"], ["--task", "summarize"], ["--beam", "notanumber"]],
    ids=["compute_type", "task", "beam"],
)
def test_invalid_args_exit_4(tmp_path, bad_args):
    """Contract: invalid args -> 4. (argparse's default exit code for usage errors is 2,
    which would collide with "CUDA unavailable".)"""
    r = run_cli([RO_SAMPLE_WAV, *bad_args], tmp_path)
    assert r.returncode == 4, r.describe()


def test_models_dir_is_real_for_gpu_tests():
    """Sanity: tells you up front why the model-dependent tests would skip."""
    if not model_downloaded(contracts.DEFAULT_MODEL):
        pytest.skip(f"{contracts.DEFAULT_MODEL} not downloaded in {REAL_MODELS_DIR}")
    assert (REAL_MODELS_DIR / contracts.DEFAULT_MODEL / "model.bin").stat().st_size > 1_000_000_000
