"""Spec/contract tests for app.engine.whispercpp.WhisperCppEngine (contracts.SpeechEngine).

Expectations come from contracts.SpeechEngine / EngineInfo / SegmentStats /
TranslationResult / LOG_ENGINE_LOADED / ggml_model_file / MODEL_COMPLETE_MARKER, from
SPEC.md ("Inference must run on CUDA. ... never fall back to CPU silently", "Do NOT offer
large-v3-turbo ... for translation", the decoding defaults) and from ARCHITECTURE.md
sections 4 and 12.1, which make behaviour parity with the Windows WhisperEngine the
specification.

The ctypes shim is faked, so there is no dylib, no model file, no GPU and no network here:
every test runs on any platform. Anything that needs a real dylib or a real ggml model is
an integration test (marker `metal`).
"""

from __future__ import annotations

import logging
import re
import time
import zlib
from pathlib import Path

import numpy as np
import pytest

from app.contracts import (
    LOG_ENGINE_LOADED,
    MODEL_COMPLETE_MARKER,
    MODELS,
    SAMPLE_RATE,
    SOURCE_LANGUAGE,
    AppError,
    AudioSegment,
    ComputeType,
    CudaReport,
    CudaUnavailableError,
    DecodeConfig,
    EngineInfo,
    EngineNotLoadedError,
    GpuInfo,
    GpuOutOfMemoryError,
    HallucinationFilter,
    ModelNotDownloadedError,
    SegmentCloseReason,
    SegmentStats,
    SpeechEngine,
    Task,
    TaskNotSupportedError,
    TranslationResult,
    ggml_model_file,
)
from app.engine import whispercpp
from app.engine.whispercpp import WhisperCppEngine

# ---------------------------------------------------------------------------
# Fakes: the ctypes shim, the probe and the hallucination filter
# ---------------------------------------------------------------------------

HEALTHY_GPU = GpuInfo(name="Apple M1 Max", total_vram_mb=25559, free_vram_mb=25558, driver_version="15.1")
HEALTHY_REPORT = CudaReport(True, HEALTHY_GPU, None, None, None)


class FakeLib:
    """The subset of libroenspeak_whisper.dylib WhisperCppEngine calls, scriptable.

    Every entry point records its name in ``calls`` so the tests can assert both *that*
    the shim was used and that it was *not* used on the paths that must fail early.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.init_args: list[tuple[str, int, int]] = []
        self.runs: list[dict[str, object]] = []
        self.freed: list[int] = []
        self.opts_i: dict[str, int] = {}
        self.opts_f: dict[str, float] = {}
        self.aborts = 0
        # scripting
        self.next_handle = 0x1000
        self.init_fails = False
        self.reject_opt: str | None = None
        self.segments: list[tuple[str, float, float, int]] = []
        self.run_rc = 0
        self.run_raises: BaseException | None = None
        self.backend = b"Metal"
        self.model_type = b"large"
        self.multilingual = 1
        self.last_error_text = ""

    # -- lifecycle ---------------------------------------------------------------

    def rs_whisper_init_ex(self, path: bytes, use_gpu: int, flash_attn: int) -> int:
        self.calls.append("init")
        self.init_args.append((path.decode(), int(use_gpu), int(flash_attn)))
        if self.init_fails:
            return 0
        self.next_handle += 1
        return self.next_handle

    def rs_whisper_free(self, handle: int) -> None:
        self.calls.append("free")
        self.freed.append(handle)

    def rs_whisper_handle_backend(self, handle: int) -> bytes:
        return self.backend

    def rs_whisper_model_type(self, handle: int) -> bytes:
        return self.model_type

    def rs_whisper_is_multilingual(self, handle: int) -> int:
        return self.multilingual

    # -- sticky decoding options -------------------------------------------------

    def rs_whisper_set_opt_i(self, handle: int, key: bytes, value: int) -> int:
        name = key.decode()
        if self.reject_opt == name:
            return -1
        self.opts_i[name] = int(value)
        return 0

    def rs_whisper_set_opt_f(self, handle: int, key: bytes, value: float) -> int:
        name = key.decode()
        if self.reject_opt == name:
            return -1
        self.opts_f[name] = float(value)
        return 0

    def rs_whisper_reset_opts(self, handle: int) -> None:
        self.opts_i.clear()
        self.opts_f.clear()

    # -- inference ---------------------------------------------------------------

    def rs_whisper_run(self, handle, audio, n, lang, translate, beam, n_threads) -> int:
        self.calls.append("run")
        self.runs.append(
            {
                "handle": handle,
                "n": int(n),
                "lang": lang.decode(),
                "translate": int(translate),
                "beam": int(beam),
                "n_threads": int(n_threads),
            }
        )
        if self.run_raises is not None:
            raise self.run_raises
        return self.run_rc

    def rs_whisper_request_abort(self, handle: int) -> None:
        self.aborts += 1

    def rs_whisper_n_segments(self, handle: int) -> int:
        return len(self.segments)

    def rs_whisper_segment_text(self, handle: int, i: int) -> bytes:
        return self.segments[i][0].encode("utf-8")

    def rs_whisper_segment_avg_logprob(self, handle: int, i: int) -> float:
        return self.segments[i][1]

    def rs_whisper_segment_no_speech_prob(self, handle: int, i: int) -> float:
        return self.segments[i][2]

    def rs_whisper_segment_n_tokens(self, handle: int, i: int) -> int:
        return self.segments[i][3]


class FakeShim:
    """Stands in for whispercpp.WhisperShim (the bound dylib)."""

    DEVICES = "MTL0:Apple M1 Max(type=gpu,free=25558MiB,total=25559MiB)"

    def __init__(self, lib: FakeLib, path: Path | None = None, metal: bool = True) -> None:
        self.lib = lib
        self.path = path or Path("/fake/build/native/libroenspeak_whisper.dylib")
        self.whisper_version = "1.9.5"
        self.build_commit = "pinned"
        self.system_info = "METAL = 1"
        self._metal = metal
        self.log_level_refreshes = 0

    def metal_available(self) -> bool:
        return self._metal

    def devices(self) -> str:
        return self.DEVICES

    def last_error(self) -> str:
        return self.lib.last_error_text

    def refresh_log_level(self) -> None:
        self.log_level_refreshes += 1


class FakeProbe:
    """CudaProbe stand-in (contracts.CudaProbe)."""

    def __init__(self, report: CudaReport = HEALTHY_REPORT) -> None:
        self.report = report
        self.probe_calls = 0
        self.setup_calls = 0

    def setup_dll_paths(self) -> list[str]:
        self.setup_calls += 1
        return []

    def probe(self) -> CudaReport:
        self.probe_calls += 1
        return self.report


class ScriptedFilter:
    """HallucinationFilter that records every SegmentStats and returns scripted reasons.

    ``reasons[i]`` is the verdict for the i-th whisper segment; missing entries mean keep.
    """

    def __init__(self, reasons=()) -> None:
        self.reasons = list(reasons)
        self.seen: list[SegmentStats] = []

    def evaluate(self, stats: SegmentStats):
        i = len(self.seen)
        self.seen.append(stats)
        return self.reasons[i] if i < len(self.reasons) else None

    def set_config(self, cfg) -> None:  # pragma: no cover - not used by the engine
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KV_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=("(?:[^"\\]|\\.)*"|\S+)')


def kv_pairs(line: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2).strip('"')) for m in _KV_RE.finditer(line)]


def kv(line: str) -> dict[str, str]:
    return dict(kv_pairs(line))


def log_lines(caplog, key: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(key + " ")]


def put_model(
    models_root: Path,
    model: str = "large-v3",
    compute_type: ComputeType = ComputeType.FLOAT16,
    *,
    marker: bool = True,
    size: int = 1024,
) -> Path:
    """A model directory that looks downloaded: the ggml file + MODEL_COMPLETE_MARKER."""
    d = models_root / model
    d.mkdir(parents=True, exist_ok=True)
    f = d / ggml_model_file(model, compute_type)
    f.write_bytes(b"\0" * size)
    if marker:
        (d / MODEL_COMPLETE_MARKER).write_text("{}", encoding="utf-8")
    return f


def segment(
    *,
    samples: np.ndarray | None = None,
    n: int = SAMPLE_RATE,
    seg_id: int = 11,
    session_id: int = 7,
    reason: SegmentCloseReason = SegmentCloseReason.SILENCE,
) -> AudioSegment:
    if samples is None:
        samples = np.zeros(n, dtype=np.float32)
    return AudioSegment(
        id=seg_id,
        session_id=session_id,
        samples=samples,
        reason=reason,
        speech_start_s=0.1,
        speech_end_s=0.9,
        t_vad_close=time.perf_counter(),
    )


@pytest.fixture
def lib() -> FakeLib:
    return FakeLib()


@pytest.fixture
def shim(lib) -> FakeShim:
    return FakeShim(lib)


@pytest.fixture
def probe() -> FakeProbe:
    return FakeProbe()


@pytest.fixture
def models_root(tmp_path) -> Path:
    return tmp_path / "models"


@pytest.fixture
def engine(probe, models_root, shim, monkeypatch) -> WhisperCppEngine:
    """An engine whose shim is faked and whose dylib "exists"."""
    monkeypatch.setattr(whispercpp, "load_shim", lambda: shim)
    monkeypatch.setattr(whispercpp, "library_path", lambda: shim.path)
    return WhisperCppEngine(probe, models_root)


@pytest.fixture
def loaded(engine, models_root, lib):
    """A loaded large-v3 engine with the call log cleared."""
    put_model(models_root)
    engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)
    lib.calls.clear()
    yield engine
    engine.unload()


# ---------------------------------------------------------------------------
# Contract shape
# ---------------------------------------------------------------------------


def test_is_a_speech_engine(engine):
    assert isinstance(engine, SpeechEngine)


def test_module_imports_on_any_platform():
    # contract rule 7: importing must never fail on the wrong platform.
    assert whispercpp.WhisperCppEngine is WhisperCppEngine


def test_model_file_is_models_root_model_ggml_file(engine, models_root):
    assert engine.model_file("medium", ComputeType.INT8_FLOAT16) == (
        models_root / "medium" / "ggml-medium-q5_0.bin"
    )
    assert engine.model_file("large-v3", ComputeType.FLOAT16) == (
        models_root / "large-v3" / "ggml-large-v3.bin"
    )


# ---------------------------------------------------------------------------
# load(): validation
# ---------------------------------------------------------------------------


def test_load_unknown_model_raises_value_error(engine, probe, lib):
    with pytest.raises(ValueError):
        engine.load("tiny", ComputeType.FLOAT16, cpu_mode=False)
    assert (probe.probe_calls, lib.calls) == (0, [])


def test_load_bad_compute_type_raises_value_error(engine, models_root, probe, lib):
    put_model(models_root)
    with pytest.raises(ValueError):
        engine.load("large-v3", "float32", cpu_mode=False)
    assert (probe.probe_calls, lib.calls) == (0, [])


@pytest.mark.parametrize("what", ["nothing", "no_file", "no_marker", "other_quantization"])
def test_load_without_a_complete_model_raises_model_not_downloaded(
    engine, models_root, probe, lib, what
):
    """load() never downloads: a missing file or a missing marker is a hard error, and it
    is raised before anything else is touched (so no download can have been attempted)."""
    if what == "no_file":
        (models_root / "large-v3").mkdir(parents=True)
        (models_root / "large-v3" / MODEL_COMPLETE_MARKER).write_text("{}", encoding="utf-8")
    elif what == "no_marker":
        put_model(models_root, marker=False)
    elif what == "other_quantization":
        # The q5_0 file is there and complete, but float16 was asked for.
        put_model(models_root, compute_type=ComputeType.INT8_FLOAT16)

    before = sorted(p.name for p in (models_root / "large-v3").glob("*")) if models_root.is_dir() else []
    with pytest.raises(ModelNotDownloadedError) as e:
        engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)
    assert e.value.model == "large-v3"
    assert e.value.user_message
    # Nothing downloaded, no GPU touched, no library loaded.
    after = sorted(p.name for p in (models_root / "large-v3").glob("*")) if models_root.is_dir() else []
    assert after == before
    assert (probe.probe_calls, lib.calls) == (0, [])
    assert not engine.is_loaded


def test_load_accepts_the_compute_type_whose_file_is_present(engine, models_root):
    put_model(models_root, compute_type=ComputeType.INT8_FLOAT16)
    info = engine.load("large-v3", ComputeType.INT8_FLOAT16, cpu_mode=False)
    assert info.compute_type == "q5_0"


# ---------------------------------------------------------------------------
# load(): GPU policy (never a silent CPU fallback)
# ---------------------------------------------------------------------------


def test_load_without_cpu_mode_requires_a_usable_probe(engine, models_root, probe, lib):
    put_model(models_root)
    probe.report = CudaReport(False, None, None, None, "no Metal GPU device is available")
    with pytest.raises(CudaUnavailableError) as e:
        engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)
    assert probe.probe_calls == 1
    assert lib.calls == []  # the model was never initialised on the CPU behind our back
    assert "no Metal GPU device is available" in e.value.user_message
    assert not engine.is_loaded


def test_cpu_mode_skips_the_probe_and_uses_the_cpu_backend(engine, models_root, probe, lib):
    put_model(models_root)
    lib.backend = b"CPU"
    info = engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=True)
    assert probe.probe_calls == 0  # the probe is not consulted at all in CPU mode
    assert lib.init_args[0][1] == 0  # use_gpu = 0
    assert info.device == "cpu"
    assert info.gpu_name is None and info.driver_version is None


def test_gpu_mode_asks_for_the_gpu_and_reports_the_real_backend(engine, models_root, lib):
    put_model(models_root)
    info = engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)
    assert lib.init_args[0][1] == 1  # use_gpu = 1
    assert info.device == "metal"


def test_load_reports_a_cpu_fallback_instead_of_claiming_metal(engine, models_root, lib, caplog):
    """EngineInfo.device always states the device actually used (contracts.EngineInfo)."""
    caplog.set_level(logging.INFO)
    put_model(models_root)
    lib.backend = b"CPU"
    info = engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)
    assert info.device == "cpu"
    assert kv(log_lines(caplog, LOG_ENGINE_LOADED)[0])["device"] == "cpu"


@pytest.mark.parametrize("cpu_mode", [False, True])
def test_missing_library_points_at_the_build_script(engine, models_root, monkeypatch, cpu_mode):
    put_model(models_root)
    monkeypatch.setattr(whispercpp, "library_path", lambda: None)

    def boom() -> None:
        raise FileNotFoundError("libroenspeak_whisper.dylib was not found (looked in: /nowhere)")

    monkeypatch.setattr(whispercpp, "load_shim", boom)
    expected = CudaUnavailableError if not cpu_mode else AppError
    with pytest.raises(expected) as e:
        engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=cpu_mode)
    assert "build.sh" in e.value.user_message
    assert not engine.is_loaded


def test_probe_failure_with_no_library_points_at_the_build_script(engine, models_root, probe, monkeypatch):
    """The probe reports "library missing"; that is a broken install, not an old Mac."""
    put_model(models_root)
    probe.report = CudaReport(False, None, None, None, "the speech engine library is missing")
    monkeypatch.setattr(whispercpp, "library_path", lambda: None)
    with pytest.raises(CudaUnavailableError) as e:
        engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)
    assert "build.sh" in e.value.user_message


def test_unloadable_library_is_cuda_unavailable(engine, models_root, monkeypatch):
    def boom():
        raise OSError("libroenspeak_whisper.dylib has ABI version 1, but this build needs 2")

    monkeypatch.setattr(whispercpp, "load_shim", boom)
    put_model(models_root)
    with pytest.raises(CudaUnavailableError) as e:
        engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)
    assert "build.sh" in e.value.user_message
    assert "ABI version" in e.value.details


def test_init_failure_is_an_app_error_and_leaves_nothing_loaded(engine, models_root, lib):
    put_model(models_root)
    lib.init_fails = True
    lib.last_error_text = "failed to read the model header"
    with pytest.raises(AppError) as e:
        engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)
    assert "failed to read the model header" in e.value.details
    assert not engine.is_loaded and engine.info is None


def test_init_failure_mentioning_metal_is_cuda_unavailable(engine, models_root, lib):
    put_model(models_root)
    lib.init_fails = True
    lib.last_error_text = "ggml_metal_init: command buffer creation failed"
    with pytest.raises(CudaUnavailableError):
        engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)


def test_out_of_memory_at_load_is_gpu_out_of_memory(engine, models_root, lib):
    put_model(models_root)
    lib.init_fails = True
    lib.last_error_text = "ggml_metal: out of memory allocating the KV cache"
    with pytest.raises(GpuOutOfMemoryError):
        engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)


def test_a_rejected_decoding_option_frees_the_handle(engine, models_root, lib):
    """A stale dylib must not leave a half-configured context alive."""
    put_model(models_root)
    lib.reject_opt = "suppress_nst"
    with pytest.raises(AppError):
        engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)
    assert lib.freed == [lib.next_handle]
    assert not engine.is_loaded


# ---------------------------------------------------------------------------
# load(): EngineInfo + the engine_loaded log line
# ---------------------------------------------------------------------------


def test_engine_info_has_all_eight_fields(engine, models_root):
    put_model(models_root)
    info = engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)
    assert isinstance(info, EngineInfo)
    assert info.model == "large-v3"
    assert info.device == "metal"  # the real backend
    assert info.compute_type == "f16"  # the real ggml type, not "float16"
    assert info.gpu_name == "Apple M1 Max"
    assert info.driver_version == "15.1"  # macOS version: what a user can update
    assert info.cublas_version is None  # CUDA-only concepts on a Mac
    assert info.cudnn_version is None
    assert 0.0 <= info.load_ms < 60_000.0
    assert engine.info == info


@pytest.mark.parametrize(
    "compute_type,expected",
    [(ComputeType.FLOAT16, "f16"), (ComputeType.INT8_FLOAT16, "q5_0")],
)
def test_compute_type_is_the_real_ggml_type(engine, models_root, compute_type, expected):
    put_model(models_root, compute_type=compute_type)
    info = engine.load("large-v3", compute_type, cpu_mode=False)
    assert info.compute_type == expected
    assert info.compute_type != compute_type.value


def test_exactly_one_engine_loaded_line_with_the_documented_keys(engine, models_root, caplog):
    caplog.set_level(logging.INFO)
    put_model(models_root)
    info = engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)

    lines = log_lines(caplog, LOG_ENGINE_LOADED)
    assert len(lines) == 1
    keys = [k for k, _ in kv_pairs(lines[0])]
    assert keys == ["device", "compute_type", "model", "gpu", "driver", "cublas", "cudnn", "load_ms"]
    fields = kv(lines[0])
    assert fields["device"] == "metal"
    assert fields["compute_type"] == "f16"
    assert fields["model"] == "large-v3"
    assert fields["gpu"] == "Apple M1 Max"
    assert fields["driver"] == "15.1"
    # What keeps the Windows/macOS log greps identical.
    assert fields["cublas"] == "none" and fields["cudnn"] == "none"
    assert float(fields["load_ms"]) == pytest.approx(info.load_ms, abs=0.1)


def test_cpu_mode_engine_loaded_line_has_none_for_the_gpu_fields(engine, models_root, caplog):
    caplog.set_level(logging.INFO)
    put_model(models_root)
    engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=True)
    fields = kv(log_lines(caplog, LOG_ENGINE_LOADED)[0])
    assert fields["gpu"] == "none" and fields["driver"] == "none"
    assert fields["cublas"] == "none" and fields["cudnn"] == "none"


def test_load_unloads_the_previous_model_first(engine, models_root, lib, caplog):
    caplog.set_level(logging.INFO)
    put_model(models_root, "large-v3")
    put_model(models_root, "medium")
    engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)
    first_handle = lib.next_handle
    lib.calls.clear()

    info = engine.load("medium", ComputeType.FLOAT16, cpu_mode=False)
    # The old context is freed before the new one is created.
    assert lib.calls[:2] == ["free", "init"]
    assert lib.freed == [first_handle]
    assert info.model == "medium"
    assert engine.info is not None and engine.info.model == "medium"
    assert len(log_lines(caplog, LOG_ENGINE_LOADED)) == 2


# ---------------------------------------------------------------------------
# load(): fixed decoding parameters (ARCHITECTURE sections 4 / 12.1 parity decisions)
# ---------------------------------------------------------------------------


def test_whispercpp_internal_no_speech_gate_is_disabled(engine, models_root, lib):
    """whisper.cpp drops a segment when no_speech_prob > no_speech_thold AND
    avg_logprob < logprob_thold, and its defaults are exactly our filter's defaults. The
    engine must push the threshold out of reach so our ThresholdHallucinationFilter is the
    only judge -- otherwise the filter would never see the segments it exists to judge
    (ARCHITECTURE section 12.1; the Windows engine passes no_speech_threshold=None)."""
    put_model(models_root)
    engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)
    # no_speech_prob is a probability, so anything > 1.0 can never fire.
    assert lib.opts_f["no_speech_thold"] > 1.0


def test_non_speech_token_suppression_is_enabled(engine, models_root, lib):
    """faster-whisper's defaults are suppress_blank=True and suppress_tokens=[-1], which
    expands to the non-speech token set == whisper.cpp's suppress_nst."""
    put_model(models_root)
    engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)
    assert lib.opts_i["suppress_nst"] == 1
    assert lib.opts_i["suppress_blank"] == 1


def test_fixed_decoding_params_mirror_the_windows_engine(engine, models_root, lib):
    """contracts.DecodeConfig: temperature=0.0 with no fallback,
    condition_on_previous_text=False, without_timestamps=True."""
    put_model(models_root)
    engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)
    assert lib.opts_f["temperature"] == 0.0
    assert lib.opts_f["temperature_inc"] == 0.0  # temperature fallback off
    assert lib.opts_i["no_context"] == 1  # condition_on_previous_text=False
    assert lib.opts_i["no_timestamps"] == 1  # without_timestamps=True
    assert lib.opts_i["best_of"] == 1  # one pass, like temperature=0.0 with no fallback


def test_the_library_never_prints_to_stdout(engine, models_root, lib):
    """A bundled .app has nowhere to print; every whisper.cpp print option must be off."""
    put_model(models_root)
    engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)
    for key in ("print_progress", "print_realtime", "print_timestamps", "print_special"):
        assert lib.opts_i[key] == 0


# ---------------------------------------------------------------------------
# warmup()
# ---------------------------------------------------------------------------


def test_warmup_runs_one_inference_and_discards_the_result(loaded, lib):
    lib.segments = [("hallucinated warmup text", -0.2, 0.1, 5)]
    assert loaded.warmup(DecodeConfig()) is None
    assert lib.calls == ["run"]
    run = lib.runs[-1]
    assert run["n"] == int(SAMPLE_RATE * whispercpp.WARMUP_SECONDS)
    assert run["translate"] == 0  # transcription: the text is thrown away anyway
    assert run["lang"] == SOURCE_LANGUAGE


def test_warmup_without_a_model_raises_engine_not_loaded(engine):
    with pytest.raises(EngineNotLoadedError):
        engine.warmup(DecodeConfig())


def test_warmup_uses_the_configured_beam_size(loaded, lib):
    loaded.warmup(DecodeConfig(beam_size=1))
    assert lib.runs[-1]["beam"] == 1


# ---------------------------------------------------------------------------
# process(): preconditions
# ---------------------------------------------------------------------------


def test_process_without_a_model_raises_engine_not_loaded(engine, lib):
    with pytest.raises(EngineNotLoadedError):
        engine.process(segment(), Task.TRANSLATE, DecodeConfig(), ScriptedFilter())
    assert lib.calls == []


def test_translate_on_a_transcription_only_model_raises_without_touching_the_shim(
    engine, models_root, lib
):
    """SPEC: large-v3-turbo is offered for Romanian transcription only -- asked to
    translate it returns punctuation or untranslated Romanian, so the engine must refuse
    rather than run (contracts.TaskNotSupportedError)."""
    assert MODELS["large-v3-turbo"].translate_capable is False
    put_model(models_root, "large-v3-turbo")
    engine.load("large-v3-turbo", ComputeType.FLOAT16, cpu_mode=False)
    lib.calls.clear()

    with pytest.raises(TaskNotSupportedError) as e:
        engine.process(segment(), Task.TRANSLATE, DecodeConfig(), ScriptedFilter())
    assert lib.calls == []  # no inference was run
    assert "large-v3-turbo" in e.value.user_message

    # The same model transcribes happily.
    lib.segments = [("Bună dimineața", -0.3, 0.05, 6)]
    result = engine.process(segment(), Task.TRANSCRIBE, DecodeConfig(), ScriptedFilter())
    assert result.text == "Bună dimineața"
    assert lib.calls == ["run"]


@pytest.mark.parametrize("model", [m for m, s in MODELS.items() if s.translate_capable])
def test_translate_capable_models_may_translate(engine, models_root, lib, model):
    put_model(models_root, model)
    engine.load(model, ComputeType.FLOAT16, cpu_mode=False)
    lib.segments = [("Good morning", -0.3, 0.05, 3)]
    result = engine.process(segment(), Task.TRANSLATE, DecodeConfig(), ScriptedFilter())
    assert result.text == "Good morning"
    assert lib.runs[-1]["translate"] == 1


def test_empty_audio_skips_inference(loaded, lib):
    result = loaded.process(
        segment(samples=np.zeros(0, dtype=np.float32)), Task.TRANSLATE, DecodeConfig(), ScriptedFilter()
    )
    assert lib.calls == []
    assert (result.text, result.filtered, result.filter_reason) == ("", True, "empty")
    assert result.audio_s == 0.0


# ---------------------------------------------------------------------------
# process(): per-segment stats
# ---------------------------------------------------------------------------


def test_one_segment_stats_per_whisper_segment(loaded, lib):
    lib.segments = [
        ("Good morning, how are you?", -0.21, 0.03, 8),
        ("I am fine, thank you.", -0.44, 0.11, 7),
    ]
    filt = ScriptedFilter()
    loaded.process(segment(), Task.TRANSLATE, DecodeConfig(), filt)

    assert len(filt.seen) == 2
    assert [s.text for s in filt.seen] == [t for t, *_ in lib.segments]
    assert [s.avg_logprob for s in filt.seen] == pytest.approx([-0.21, -0.44])
    assert [s.no_speech_prob for s in filt.seen] == pytest.approx([0.03, 0.11])
    assert all(isinstance(s, SegmentStats) for s in filt.seen)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Good morning, how are you?", 26 / 34),
        ("Hello hello hello hello hello hello hello hello", 47 / 17),
        ("", 0.0),
    ],
)
def test_compression_ratio_formula(text, expected):
    """faster-whisper's get_compression_ratio: UTF-8 byte length / zlib-compressed length.
    Keeping the formula identical is what keeps FilterConfig.compression_ratio_threshold
    meaning the same thing on both platforms (ARCHITECTURE section 12.1)."""
    assert whispercpp.compression_ratio(text) == pytest.approx(expected)
    if text:
        raw = text.encode("utf-8")
        assert whispercpp.compression_ratio(text) == pytest.approx(len(raw) / len(zlib.compress(raw)))


def test_compression_ratio_reaches_the_filter(loaded, lib):
    text = "Hello hello hello hello hello hello hello hello"
    lib.segments = [(text, -0.3, 0.05, 9)]
    filt = ScriptedFilter()
    loaded.process(segment(), Task.TRANSLATE, DecodeConfig(), filt)
    assert filt.seen[0].compression_ratio == pytest.approx(47 / 17)


def test_a_segment_with_no_scorable_token_gets_neutral_stats(loaded, lib):
    """NaN would make every threshold comparison False, so it must be normalised; the
    filter's "empty" rule is what drops such a (textless) segment."""
    lib.segments = [("", float("nan"), float("nan"), 0)]
    filt = ScriptedFilter(reasons=["empty"])
    result = loaded.process(segment(), Task.TRANSLATE, DecodeConfig(), filt)
    assert filt.seen[0].avg_logprob == 0.0
    assert filt.seen[0].no_speech_prob == 0.0
    assert result.filter_reason == "empty"


# ---------------------------------------------------------------------------
# process(): filter verdicts -> TranslationResult
# ---------------------------------------------------------------------------


def test_kept_segments_are_stripped_and_joined_with_single_spaces(loaded, lib):
    lib.segments = [
        ("  Good morning. ", -0.2, 0.02, 4),
        ("\nHow are you?\t", -0.3, 0.04, 4),
    ]
    result = loaded.process(segment(), Task.TRANSLATE, DecodeConfig(), ScriptedFilter())
    assert result.text == "Good morning. How are you?"
    assert result.filtered is False and result.filter_reason is None


def test_dropped_segments_are_not_typed(loaded, lib):
    lib.segments = [
        ("Good morning.", -0.2, 0.02, 4),
        ("Thank you for watching!", -0.2, 0.02, 4),
        ("How are you?", -0.2, 0.02, 4),
    ]
    filt = ScriptedFilter(reasons=[None, "blocklist", None])
    result = loaded.process(segment(), Task.TRANSLATE, DecodeConfig(), filt)
    assert result.text == "Good morning. How are you?"
    assert result.filtered is False and result.filter_reason is None


def test_fully_filtered_result_reports_the_first_reason(loaded, lib):
    lib.segments = [
        ("aaaaaaaaaaaaaaaaaaaa", -0.2, 0.02, 4),
        ("Thank you for watching!", -0.2, 0.02, 4),
    ]
    filt = ScriptedFilter(reasons=["compression", "blocklist"])
    result = loaded.process(segment(), Task.TRANSLATE, DecodeConfig(), filt)
    assert result.text == ""
    assert result.filtered is True
    assert result.filter_reason == "compression"  # the FIRST dropped segment's reason


def test_no_whisper_segments_at_all_is_reason_empty(loaded, lib):
    lib.segments = []
    filt = ScriptedFilter()
    result = loaded.process(segment(), Task.TRANSLATE, DecodeConfig(), filt)
    assert (result.text, result.filtered, result.filter_reason) == ("", True, "empty")
    assert filt.seen == []


def test_kept_but_blank_text_is_reported_as_empty(loaded, lib):
    lib.segments = [("   ", -0.2, 0.02, 1)]
    result = loaded.process(segment(), Task.TRANSLATE, DecodeConfig(), ScriptedFilter())
    assert (result.text, result.filtered, result.filter_reason) == ("", True, "empty")


def test_translation_result_fields(loaded, lib):
    lib.segments = [("Good morning.", -0.2, 0.02, 4)]
    seg = segment(n=2 * SAMPLE_RATE, seg_id=42, session_id=9)
    t_before = time.perf_counter()
    result = loaded.process(seg, Task.TRANSLATE, DecodeConfig(), ScriptedFilter())
    t_after = time.perf_counter()

    assert isinstance(result, TranslationResult)
    assert result.segment_id == 42
    assert result.session_id == 9
    assert result.task is Task.TRANSLATE
    assert result.audio_s == pytest.approx(2.0)
    assert result.t_vad_close == seg.t_vad_close
    assert t_before <= result.t_infer_start <= result.t_infer_end <= t_after
    assert 0.0 <= result.infer_ms <= (t_after - t_before) * 1000.0 + 1.0


def test_task_is_passed_through_to_the_shim(loaded, lib):
    lib.segments = [("ok", -0.2, 0.02, 1)]
    loaded.process(segment(), Task.TRANSCRIBE, DecodeConfig(beam_size=3), ScriptedFilter())
    run = lib.runs[-1]
    assert run["translate"] == 0
    assert run["beam"] == 3
    assert run["lang"] == SOURCE_LANGUAGE
    assert run["n"] == SAMPLE_RATE


def test_process_accepts_the_task_as_a_plain_string(loaded, lib):
    lib.segments = [("Good morning.", -0.2, 0.02, 4)]
    result = loaded.process(segment(), "translate", DecodeConfig(), ScriptedFilter())
    assert result.task is Task.TRANSLATE


def test_a_scripted_filter_is_a_hallucination_filter():
    assert isinstance(ScriptedFilter(), HallucinationFilter)


# ---------------------------------------------------------------------------
# process(): failures are always AppError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error,expected",
    [
        ("the decoder returned no tokens", AppError),
        ("ggml_metal_graph_compute: command buffer failed", CudaUnavailableError),
        ("ggml_backend_metal_buffer: out of memory", GpuOutOfMemoryError),
    ],
)
def test_a_shim_failure_becomes_an_app_error(loaded, lib, error, expected):
    lib.run_rc = -1
    lib.last_error_text = error
    with pytest.raises(expected) as e:
        loaded.process(segment(), Task.TRANSLATE, DecodeConfig(), ScriptedFilter())
    assert isinstance(e.value, AppError)
    assert e.value.user_message
    assert error in e.value.details


def test_a_generic_shim_failure_is_not_reported_as_a_gpu_problem(loaded, lib):
    lib.run_rc = -1
    lib.last_error_text = "the decoder returned no tokens"
    with pytest.raises(AppError) as e:
        loaded.process(segment(), Task.TRANSLATE, DecodeConfig(), ScriptedFilter())
    assert not isinstance(e.value, CudaUnavailableError)


@pytest.mark.parametrize("exc", [ValueError("boom"), OSError("segmentation fault, nearly"), RuntimeError("?")])
def test_an_exception_from_the_shim_never_escapes_raw(loaded, lib, exc):
    lib.run_raises = exc
    with pytest.raises(AppError):
        loaded.process(segment(), Task.TRANSLATE, DecodeConfig(), ScriptedFilter())


def test_out_of_memory_unloads_the_model(loaded, lib):
    lib.run_rc = -1
    lib.last_error_text = "ggml: out of memory"
    with pytest.raises(GpuOutOfMemoryError):
        loaded.process(segment(), Task.TRANSLATE, DecodeConfig(), ScriptedFilter())
    # The model is gone, so core goes through load() again instead of reusing a dead context.
    assert not loaded.is_loaded and loaded.info is None


def test_a_filter_exception_is_not_swallowed(loaded, lib):
    class Boom:
        def evaluate(self, stats):
            raise RuntimeError("filter bug")

        def set_config(self, cfg):
            pass

    lib.segments = [("Good morning.", -0.2, 0.02, 4)]
    with pytest.raises(RuntimeError):
        loaded.process(segment(), Task.TRANSLATE, DecodeConfig(), Boom())


# ---------------------------------------------------------------------------
# unload() / is_loaded / info
# ---------------------------------------------------------------------------


def test_is_loaded_and_info_before_load(engine):
    assert engine.is_loaded is False
    assert engine.info is None


def test_unload_is_idempotent(engine, models_root, lib):
    put_model(models_root)
    engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)
    handle = lib.next_handle
    assert engine.is_loaded is True and engine.info is not None

    engine.unload()
    assert engine.is_loaded is False and engine.info is None
    engine.unload()
    engine.unload()
    assert lib.freed == [handle]  # freed exactly once


def test_unload_before_any_load_is_a_no_op(engine, lib):
    engine.unload()
    assert lib.calls == []
    assert engine.is_loaded is False


def test_process_after_unload_raises_engine_not_loaded(engine, models_root, lib):
    put_model(models_root)
    engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)
    engine.unload()
    with pytest.raises(EngineNotLoadedError):
        engine.process(segment(), Task.TRANSLATE, DecodeConfig(), ScriptedFilter())


def test_load_after_unload_works(engine, models_root, lib):
    put_model(models_root)
    engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)
    engine.unload()
    info = engine.load("large-v3", ComputeType.FLOAT16, cpu_mode=False)
    assert engine.is_loaded and info.model == "large-v3"
    assert lib.calls.count("init") == 2


def test_request_abort_is_safe_without_a_model(engine, lib):
    engine.request_abort()
    assert lib.aborts == 0


def test_request_abort_reaches_the_shim(loaded, lib):
    loaded.request_abort()
    assert lib.aborts == 1


# ---------------------------------------------------------------------------
# library discovery (pure path logic, no dylib)
# ---------------------------------------------------------------------------


def test_library_candidates_prefer_the_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(whispercpp.ENV_WHISPER_LIB, str(tmp_path))
    candidates = whispercpp.library_candidates()
    assert candidates[0] == tmp_path / whispercpp.WHISPER_LIB_NAME
    assert len(candidates) == len(set(candidates))


def test_library_candidates_accept_a_full_dylib_path(monkeypatch, tmp_path):
    lib_path = tmp_path / "custom.dylib"
    monkeypatch.setenv(whispercpp.ENV_WHISPER_LIB, str(lib_path))
    assert whispercpp.library_candidates()[0] == lib_path


def test_library_candidates_include_the_build_output_dir(monkeypatch):
    monkeypatch.delenv(whispercpp.ENV_WHISPER_LIB, raising=False)
    monkeypatch.delattr(whispercpp.sys, "_MEIPASS", raising=False)
    candidates = whispercpp.library_candidates()
    assert candidates
    assert candidates[-1].parts[-3:] == ("build", "native", whispercpp.WHISPER_LIB_NAME)


def test_library_path_is_none_when_nothing_exists(monkeypatch, tmp_path):
    monkeypatch.setenv(whispercpp.ENV_WHISPER_LIB, str(tmp_path / "missing"))
    monkeypatch.setattr(
        whispercpp, "library_candidates", lambda: [tmp_path / "missing" / whispercpp.WHISPER_LIB_NAME]
    )
    assert whispercpp.library_path() is None


def test_library_path_returns_the_first_existing_candidate(monkeypatch, tmp_path):
    first = tmp_path / "a" / whispercpp.WHISPER_LIB_NAME
    second = tmp_path / "b" / whispercpp.WHISPER_LIB_NAME
    second.parent.mkdir(parents=True)
    second.write_bytes(b"not really a dylib")
    monkeypatch.setattr(whispercpp, "library_candidates", lambda: [first, second])
    assert whispercpp.library_path() == second


# ---------------------------------------------------------------------------
# ggml quantization naming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("ggml-large-v3.bin", "f16"),
        ("ggml-large-v3-q5_0.bin", "q5_0"),
        ("ggml-medium-q5_0.bin", "q5_0"),
        ("ggml-large-v3-turbo.bin", "f16"),
        ("ggml-large-v3-turbo-q5_0.bin", "q5_0"),
        ("ggml-medium-q8_0.bin", "q8_0"),
    ],
)
def test_ggml_compute_type(filename, expected):
    assert whispercpp.ggml_compute_type(filename) == expected


@pytest.mark.parametrize("model", sorted(MODELS))
@pytest.mark.parametrize("compute_type", list(ComputeType))
def test_every_model_maps_to_a_ggml_file_and_its_quantization(model, compute_type):
    name = ggml_model_file(model, compute_type)
    expected = "f16" if compute_type is ComputeType.FLOAT16 else "q5_0"
    assert whispercpp.ggml_compute_type(name) == expected
