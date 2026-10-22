"""WhisperCppEngine: owns one whisper.cpp context on Metal (see contracts.SpeechEngine).

The macOS counterpart of app.engine.engine.WhisperEngine. There is no CUDA on Apple
Silicon and no CTranslate2 Metal backend, so the model runs in whisper.cpp behind the
small C shim built by native/whispercpp/build.sh (libroenspeak_whisper.dylib, see
native/README.md). The shim is bound with ctypes *lazily*, inside load(), so this module
imports fine on a machine where the dylib was never built and on Windows (unit tests).

Behaviour parity with WhisperEngine is the specification: same validation order, same
exception types, same EngineInfo fields, the same single LOG_ENGINE_LOADED line with the
same keys and quoting, the same warmup line, the same per-segment debug lines, the same
keep/drop logic and the same TranslationResult semantics. Differences are forced by the
backend and are called out where they occur:

* The model is one ggml file (``models_root/<model>/ggml_model_file(model, ct)``), not a
  directory of CTranslate2 files, so the compute type selects the *file*. EngineInfo
  reports the real ggml quantization ("f16" / "q5_0"), not a CTranslate2 name.
* whisper.cpp exposes no avg_logprob and no compression_ratio. avg_logprob comes from the
  shim (mean log token probability over the segment's non-special tokens) and
  compression_ratio is computed here exactly the way faster-whisper computes it, so the
  same FilterConfig thresholds keep meaning the same thing.
* faster-whisper is called with ``no_speech_threshold=None`` so that it never skips a
  segment and our HallucinationFilter decides with full stats. whisper.cpp has no such
  "off" value: it *drops* a segment when
  ``no_speech_prob > no_speech_thold && avg_logprob < logprob_thold`` (src/whisper.cpp,
  ``is_no_speech``), and its defaults (0.6 / -1.0) are exactly our filter's defaults, so it
  would silently swallow the segments the filter is supposed to judge. NO_SPEECH_THOLD_OFF
  is therefore pushed above 1.0, which makes that condition unsatisfiable.
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import sys
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

from app.contracts import (
    IS_MACOS,
    LOG_ENGINE_LOADED,
    MIN_MACOS_VERSION,
    MODEL_COMPLETE_MARKER,
    MODELS,
    SAMPLE_RATE,
    SOURCE_LANGUAGE,
    AppError,
    AudioSegment,
    ComputeType,
    CudaProbe,
    CudaReport,
    CudaUnavailableError,
    DecodeConfig,
    EngineInfo,
    EngineNotLoadedError,
    GpuOutOfMemoryError,
    HallucinationFilter,
    ModelNotDownloadedError,
    SegmentStats,
    Task,
    TaskNotSupportedError,
    TranslationResult,
    ggml_model_file,
    models_dir,
)
from app.engine.cuda_probe import format_kv  # platform-neutral k=v log formatting

logger = logging.getLogger("roenspeak.engine.whispercpp")

#: The shim and its six siblings, as staged by native/whispercpp/build.sh.
WHISPER_LIB_NAME = "libroenspeak_whisper.dylib"
#: Override the directory (or the full path) of the shim. Used by the build script's
#: smoke test and by anyone running against a non-default output dir.
ENV_WHISPER_LIB = "ROENSPEAK_WHISPER_LIB"
#: Where build.sh puts the dylibs by default, relative to the repo root.
DEFAULT_LIB_SUBDIR = Path("build") / "native"
#: rs_whisper_abi_version() must be exactly this; a mismatch means a stale dylib.
REQUIRED_ABI_VERSION = 2

WARMUP_SECONDS = 1.0
WARMUP_NOISE_AMPLITUDE = 1e-3

#: no_speech_thold that can never fire (no_speech_prob is a probability in [0, 1]).
#: See the module docstring: this is what makes our filter the only judge.
NO_SPEECH_THOLD_OFF = 1.1

#: Substrings (lower-cased) that identify a shim/whisper.cpp failure as a GPU-stack one.
#: The analogue of engine._CUDA_ERROR_MARKERS.
_METAL_ERROR_MARKERS = ("metal", "mtl", "ggml", "gpu", "device", "command buffer")

#: ggml_log_level -> nothing; every shim line is logged at DEBUG (errors are already
#: surfaced to the user through the exception built from rs_whisper_last_error()).
_LOG_SILENT, _LOG_ERRORS, _LOG_ALL = 0, 1, 2


def metal_requirement_message(reason: Optional[str] = None) -> str:
    """The macOS analogue of engine.cuda_requirement_message."""
    msg = (
        f"RoEnSpeak needs an Apple Silicon Mac running macOS {MIN_MACOS_VERSION} or newer to run "
        "the speech model on the GPU with Metal. "
    )
    if reason:
        msg += f"Problem: {reason.rstrip('. ')}. "
    msg += (
        "Update macOS and restart RoEnSpeak. "
        'Alternatively, enable "CPU mode (slow)" in Settings.'
    )
    return msg


def library_missing_message(reason: str) -> str:
    """Shown when the dylib is absent or unloadable: that is a broken installation, not a
    machine problem, so it must not read like "your Mac is unsupported"."""
    return (
        "RoEnSpeak's speech engine library is missing from this installation, so no speech "
        "model can be loaded. If you are running from a source checkout, build it with "
        "./native/whispercpp/build.sh and start RoEnSpeak again. If you installed "
        "RoEnSpeak from a .dmg, reinstall it. "
        f"Problem: {reason.rstrip('. ')}."
    )


def is_metal_runtime_error(exc: BaseException) -> bool:
    """The analogue of engine.is_cuda_runtime_error."""
    if not isinstance(exc, (RuntimeError, OSError, ValueError)):
        return False
    text = str(exc).lower()
    return any(m in text for m in _METAL_ERROR_MARKERS)


def is_out_of_memory_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def gpu_oom_message(stage: str) -> str:
    """Mirrors engine.gpu_oom_message, worded for unified memory."""
    return (
        f"Not enough free memory while {stage}. Another application may be using it. "
        "Close memory-heavy applications (browsers with many tabs, video editors, other AI "
        "tools), or in RoEnSpeak Settings switch the compute type to int8_float16 or choose "
        "the smaller medium model, then try again."
    )


def _metal_error_from(exc: BaseException, stage: str) -> CudaUnavailableError:
    text = str(exc)
    details = f"{type(exc).__name__}: {text}"
    if is_out_of_memory_error(exc):
        return GpuOutOfMemoryError(gpu_oom_message(stage), details=details)
    return CudaUnavailableError(
        metal_requirement_message(f"a Metal error occurred while {stage}"), details=details
    )


def compression_ratio(text: str) -> float:
    """faster-whisper's compression_ratio (faster_whisper.transcribe.get_compression_ratio):
    UTF-8 byte length over zlib-compressed length. Reimplemented so the FilterConfig
    compression_ratio_threshold keeps the exact same meaning on both platforms."""
    data = text.encode("utf-8")
    if not data:
        return 0.0
    return len(data) / len(zlib.compress(data))


# ---------------------------------------------------------------------------
# ctypes binding
# ---------------------------------------------------------------------------

_C_VOIDP = ctypes.c_void_p
_C_CHARP = ctypes.c_char_p
_C_INT = ctypes.c_int
_C_I64 = ctypes.c_int64
_C_FLOAT = ctypes.c_float
_C_DOUBLE = ctypes.c_double
_C_FLOATP = ctypes.POINTER(ctypes.c_float)

#: (name, restype, argtypes) for every symbol of roenspeak_whisper.h we use.
_SIGNATURES: tuple[tuple[str, Any, list[Any]], ...] = (
    ("rs_whisper_abi_version", _C_INT, []),
    ("rs_whisper_whisper_version", _C_CHARP, []),
    ("rs_whisper_build_commit", _C_CHARP, []),
    ("rs_whisper_system_info", _C_CHARP, []),
    ("rs_whisper_devices", _C_CHARP, []),
    ("rs_whisper_metal_available", _C_INT, []),
    ("rs_whisper_set_log_level", None, [_C_INT]),
    ("rs_whisper_last_error", _C_CHARP, []),
    ("rs_whisper_lang_id", _C_INT, [_C_CHARP]),
    ("rs_whisper_init_ex", _C_VOIDP, [_C_CHARP, _C_INT, _C_INT]),
    ("rs_whisper_free", None, [_C_VOIDP]),
    ("rs_whisper_handle_uses_gpu", _C_INT, [_C_VOIDP]),
    ("rs_whisper_handle_backend", _C_CHARP, [_C_VOIDP]),
    ("rs_whisper_model_type", _C_CHARP, [_C_VOIDP]),
    ("rs_whisper_is_multilingual", _C_INT, [_C_VOIDP]),
    ("rs_whisper_set_opt_i", _C_INT, [_C_VOIDP, _C_CHARP, _C_INT]),
    ("rs_whisper_set_opt_f", _C_INT, [_C_VOIDP, _C_CHARP, _C_FLOAT]),
    ("rs_whisper_set_opt_s", _C_INT, [_C_VOIDP, _C_CHARP, _C_CHARP]),
    ("rs_whisper_reset_opts", None, [_C_VOIDP]),
    ("rs_whisper_run", _C_INT, [_C_VOIDP, _C_FLOATP, _C_INT, _C_CHARP, _C_INT, _C_INT, _C_INT]),
    ("rs_whisper_request_abort", None, [_C_VOIDP]),
    ("rs_whisper_n_segments", _C_INT, [_C_VOIDP]),
    ("rs_whisper_segment_text", _C_CHARP, [_C_VOIDP, _C_INT]),
    ("rs_whisper_segment_t0", _C_I64, [_C_VOIDP, _C_INT]),
    ("rs_whisper_segment_t1", _C_I64, [_C_VOIDP, _C_INT]),
    ("rs_whisper_segment_no_speech_prob", _C_FLOAT, [_C_VOIDP, _C_INT]),
    ("rs_whisper_segment_avg_logprob", _C_FLOAT, [_C_VOIDP, _C_INT]),
    ("rs_whisper_segment_n_tokens", _C_INT, [_C_VOIDP, _C_INT]),
    ("rs_whisper_full_text", _C_CHARP, [_C_VOIDP]),
    ("rs_whisper_avg_logprob", _C_FLOAT, [_C_VOIDP]),
    ("rs_whisper_max_no_speech_prob", _C_FLOAT, [_C_VOIDP]),
    ("rs_whisper_detected_language", _C_CHARP, [_C_VOIDP]),
    ("rs_whisper_last_run_ms", _C_DOUBLE, [_C_VOIDP]),
    ("rs_whisper_last_audio_ms", _C_DOUBLE, [_C_VOIDP]),
    ("rs_whisper_last_rtf", _C_DOUBLE, [_C_VOIDP]),
    ("rs_whisper_last_prep_ms", _C_DOUBLE, [_C_VOIDP]),
)

#: void (*)(int level, const char * text, void * user) -- see rs_whisper_set_log_sink.
_LOG_SINK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)


def library_candidates() -> list[Path]:
    """Where the shim is looked for, in order:

    1. ``$ROENSPEAK_WHISPER_LIB`` (the dylib itself, or the directory holding it),
    2. ``sys._MEIPASS`` (and ``sys._MEIPASS/native``) in a frozen PyInstaller bundle,
    3. ``<repo>/build/native`` -- where native/whispercpp/build.sh stages by default.

    All six sibling dylibs must live next to whichever file is picked: the shim's
    ``@loader_path`` rpath is what resolves them.
    """
    out: list[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        key = os.path.normcase(str(p))
        if key not in seen:
            seen.add(key)
            out.append(p)

    env = os.environ.get(ENV_WHISPER_LIB, "").strip()
    if env:
        p = Path(env).expanduser()
        add(p if p.suffix == ".dylib" else p / WHISPER_LIB_NAME)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        add(Path(meipass) / WHISPER_LIB_NAME)
        add(Path(meipass) / "native" / WHISPER_LIB_NAME)
    add(Path(__file__).resolve().parents[2] / DEFAULT_LIB_SUBDIR / WHISPER_LIB_NAME)
    return out


def library_path() -> Optional[Path]:
    """The first existing candidate, or None."""
    for p in library_candidates():
        try:
            if p.is_file():
                return p
        except OSError:  # pragma: no cover - unreadable mount
            continue
    return None


class WhisperShim:
    """Bound ctypes view of libroenspeak_whisper.dylib. One per process (see load_shim)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        # The six siblings are found through the @loader_path rpath baked into the shim,
        # so a plain CDLL is enough: no DYLD_LIBRARY_PATH, no preloading.
        self.lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
        for name, restype, argtypes in _SIGNATURES:
            try:
                fn = getattr(self.lib, name)
            except AttributeError as e:
                raise OSError(f"{path.name} does not export {name} (stale build?)") from e
            fn.restype = restype
            fn.argtypes = argtypes
        abi = int(self.lib.rs_whisper_abi_version())
        if abi != REQUIRED_ABI_VERSION:
            raise OSError(
                f"{path.name} has ABI version {abi}, but this build of RoEnSpeak needs "
                f"{REQUIRED_ABI_VERSION}; rebuild it with ./native/whispercpp/build.sh"
            )
        self._sink: Any = None  # keeps the CFUNCTYPE trampoline alive for the process
        self._install_log_sink()

    # -- library-level accessors ---------------------------------------------------

    def _text(self, fn: Callable[[], Any]) -> str:
        raw = fn()
        return raw.decode("utf-8", "replace") if raw else ""

    @property
    def whisper_version(self) -> str:
        return self._text(self.lib.rs_whisper_whisper_version)

    @property
    def build_commit(self) -> str:
        return self._text(self.lib.rs_whisper_build_commit)

    @property
    def system_info(self) -> str:
        return self._text(self.lib.rs_whisper_system_info)

    def devices(self) -> str:
        return self._text(self.lib.rs_whisper_devices)

    def metal_available(self) -> bool:
        return bool(self.lib.rs_whisper_metal_available())

    def last_error(self) -> str:
        return self._text(self.lib.rs_whisper_last_error)

    # -- logging -------------------------------------------------------------------

    def _install_log_sink(self) -> None:
        """Route whisper.cpp/ggml logging into this module's logger at DEBUG instead of
        stderr. Without this, whisper.cpp writes model-load and progress lines straight to
        stderr from inside the dylib, which in a bundled .app goes nowhere useful."""

        def sink(level: int, text: Optional[bytes], user: Optional[int]) -> None:
            # Runs on whatever thread ggml logs on; it must never raise back into C.
            try:
                if not text:
                    return
                line = text.decode("utf-8", "replace").rstrip("\r\n")
                if line:
                    logger.debug("whispercpp[%d] %s", level, line)
            except Exception:  # pragma: no cover - last-resort guard
                pass

        set_sink = getattr(self.lib, "rs_whisper_set_log_sink", None)
        if set_sink is None:  # pragma: no cover - guaranteed by the ABI check
            self.lib.rs_whisper_set_log_level(_LOG_SILENT)
            return
        set_sink.restype = None
        set_sink.argtypes = [_LOG_SINK_TYPE, _C_VOIDP]
        self._sink = _LOG_SINK_TYPE(sink)
        set_sink(self._sink, None)
        self.refresh_log_level()

    def refresh_log_level(self) -> None:
        """Everything when our logger is at DEBUG, errors only otherwise. Errors always
        land in rs_whisper_last_error() regardless, which is what the exceptions use."""
        level = _LOG_ALL if logger.isEnabledFor(logging.DEBUG) else _LOG_ERRORS
        self.lib.rs_whisper_set_log_level(level)


_shim_lock = threading.Lock()
_shim: Optional[WhisperShim] = None


def load_shim() -> WhisperShim:
    """The process-wide WhisperShim, loading the dylib on first use.

    Raises OSError (FileNotFoundError when nothing was found) with a message naming the
    places that were searched. Callers turn that into a user-facing AppError.
    """
    global _shim
    with _shim_lock:
        if _shim is not None:
            _shim.refresh_log_level()
            return _shim
        path = library_path()
        if path is None:
            where = ", ".join(str(p) for p in library_candidates())
            raise FileNotFoundError(f"{WHISPER_LIB_NAME} was not found (looked in: {where})")
        shim = WhisperShim(path)
        _shim = shim
    logger.info(
        "whispercpp_loaded lib=%s whisper=%s commit=%s metal=%s devices=%s",
        format_kv(shim.path),
        format_kv(shim.whisper_version),
        format_kv(shim.build_commit),
        format_kv(shim.metal_available()),
        format_kv(shim.devices()),
    )
    logger.debug("whispercpp system_info: %s", shim.system_info)
    return shim


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class _Segment:
    """One whisper.cpp segment, shaped like a faster_whisper Segment for the filter."""

    __slots__ = ("text", "avg_logprob", "no_speech_prob", "compression_ratio", "n_tokens")

    def __init__(self, text: str, avg_logprob: float, no_speech_prob: float, n_tokens: int) -> None:
        self.text = text
        # NaN means "the segment had no scorable token"; such a segment is textless and the
        # filter's "empty" rule catches it, but NaN would make every threshold comparison
        # False, so it is normalised to a neutral, confident value instead.
        self.avg_logprob = 0.0 if math.isnan(avg_logprob) else avg_logprob
        self.no_speech_prob = 0.0 if math.isnan(no_speech_prob) else no_speech_prob
        self.compression_ratio = compression_ratio(text)
        self.n_tokens = n_tokens


def ggml_compute_type(filename: str) -> str:
    """The real ggml quantization of a model file, for EngineInfo.compute_type.
    "ggml-large-v3-q5_0.bin" -> "q5_0"; an unquantized "ggml-large-v3.bin" -> "f16"."""
    stem = filename[:-4] if filename.endswith(".bin") else filename
    for q in ("q4_0", "q4_1", "q5_0", "q5_1", "q8_0"):
        if stem.endswith("-" + q):
            return q
    return "f16"


class WhisperCppEngine:
    """SpeechEngine implementation for macOS. All methods except is_loaded/info run on the
    single inference thread (and so do the shim's result accessors, which do not lock)."""

    def __init__(self, probe: Optional[CudaProbe] = None, models_root: Optional[Path] = None) -> None:
        self._probe = probe
        self._models_root = Path(models_root) if models_root is not None else None
        self._lock = threading.Lock()  # guards _handle/_info for cross-thread readers
        self._shim: Optional[WhisperShim] = None
        self._handle: Optional[int] = None
        self._info: Optional[EngineInfo] = None
        self._translate_capable = False

    # -- properties (any thread) --------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._handle is not None

    @property
    def info(self) -> Optional[EngineInfo]:
        return self._info

    @property
    def models_root(self) -> Path:
        return self._models_root if self._models_root is not None else models_dir()

    def model_file(self, model: str, compute_type: Union[ComputeType, str]) -> Path:
        """The single ggml file this engine loads for a model/compute type."""
        ct = ComputeType(compute_type)
        return self.models_root / model / ggml_model_file(model, ct)

    # -- lifecycle ----------------------------------------------------------------

    def load(self, model: str, compute_type: Union[ComputeType, str], cpu_mode: bool) -> EngineInfo:
        if model not in MODELS:
            raise ValueError(f"Unknown model: {model!r}")
        ct = ComputeType(compute_type)  # ValueError for an unknown compute type
        self.unload()

        path = self.model_file(model, ct)
        # Never download here: the marker is what the downloader writes last.
        if not path.is_file() or not (path.parent / MODEL_COMPLETE_MARKER).is_file():
            raise ModelNotDownloadedError(model)

        report: Optional[CudaReport] = None
        if cpu_mode:
            use_gpu = 0
            logger.warning(
                "CPU mode (slow): loading %s on the CPU from %s", model, path.name
            )
        else:
            use_gpu = 1
            report = self._require_probe().probe()
            if not report.ok:
                # A missing dylib is a broken installation, not an unsupported Mac, and the
                # probe reports it as its error; say the useful thing in that case.
                message = (
                    library_missing_message(report.error or "the library was not found")
                    if library_path() is None
                    else metal_requirement_message(report.error)
                )
                raise CudaUnavailableError(message, details=report.error or "")

        t0 = time.perf_counter()
        try:
            shim = load_shim()
        except Exception as e:
            reason = str(e)
            if not cpu_mode:
                raise CudaUnavailableError(library_missing_message(reason), details=repr(e)) from e
            raise AppError(library_missing_message(reason), details=repr(e)) from e

        # flash_attn is on: it is supported on Apple Silicon and is what every measurement
        # in the port notes was taken with.
        handle = shim.lib.rs_whisper_init_ex(str(path).encode("utf-8"), use_gpu, 1)
        if not handle:
            err = shim.last_error() or "whisper.cpp could not initialise the model"
            exc = RuntimeError(err)
            if not cpu_mode and is_metal_runtime_error(exc):
                raise _metal_error_from(exc, "loading the model") from exc
            raise AppError(
                f"The model file for {model} could not be loaded; it may be corrupt. "
                f"Delete the file {path} and download the model again.",
                details=f"rs_whisper_init_ex: {err}",
            )
        try:
            self._apply_fixed_params(shim, handle)
        except Exception:
            shim.lib.rs_whisper_free(handle)
            raise
        load_ms = (time.perf_counter() - t0) * 1000.0

        backend = (shim.lib.rs_whisper_handle_backend(handle) or b"").decode("utf-8", "replace")
        actual_device = backend.lower() or ("metal" if use_gpu else "cpu")
        info = EngineInfo(
            model=model,
            device=actual_device,
            compute_type=ggml_compute_type(path.name),
            gpu_name=report.gpu.name if report and report.gpu else None,
            driver_version=report.gpu.driver_version if report and report.gpu else None,
            cublas_version=None,  # CUDA-only concepts; always None on macOS
            cudnn_version=None,
            load_ms=load_ms,
        )
        with self._lock:
            self._shim = shim
            self._handle = handle
            self._info = info
            self._translate_capable = MODELS[model].translate_capable
        logger.info(
            "%s device=%s compute_type=%s model=%s gpu=%s driver=%s cublas=%s cudnn=%s load_ms=%.1f",
            LOG_ENGINE_LOADED,
            format_kv(info.device),
            format_kv(info.compute_type),
            format_kv(info.model),
            format_kv(info.gpu_name),
            format_kv(info.driver_version),
            format_kv(info.cublas_version),
            format_kv(info.cudnn_version),
            info.load_ms,
        )
        if not cpu_mode and actual_device != "metal":
            logger.warning("whisper.cpp fell back to the %s backend although Metal was requested", backend)
        model_type = (shim.lib.rs_whisper_model_type(handle) or b"").decode("utf-8", "replace")
        logger.debug(
            "whispercpp_handle file=%s type=%s multilingual=%s prep=%s",
            format_kv(path.name),
            format_kv(model_type),
            format_kv(bool(shim.lib.rs_whisper_is_multilingual(handle))),
            format_kv(shim.devices()),
        )
        return info

    def _require_probe(self) -> CudaProbe:
        if self._probe is None:
            from app.engine.metal_probe import MetalProbe  # lazy: avoids an import cycle

            self._probe = MetalProbe()
        return self._probe

    def _apply_fixed_params(self, shim: WhisperShim, handle: int) -> None:
        """The sticky knobs that mirror WhisperEngine's fixed transcribe() arguments.
        Only language/task/beam_size vary per run (they are rs_whisper_run arguments)."""
        ints = {
            # without_timestamps=True
            "no_timestamps": 1,
            # condition_on_previous_text=False
            "no_context": 1,
            # faster-whisper's defaults: suppress_blank=True and suppress_tokens=[-1],
            # which expands to the non-speech token list == whisper.cpp's suppress_nst.
            "suppress_blank": 1,
            "suppress_nst": 1,
            # whisper.cpp writes these to stdout/stderr from inside the dylib; two of them
            # default to true, which is unacceptable in a bundled .app.
            "print_progress": 0,
            "print_realtime": 0,
            "print_timestamps": 0,
            "print_special": 0,
            # One pass, like faster-whisper with temperature=0.0 and no fallback. Without
            # this, GREEDY keeps whisper.cpp's best_of=5 and runs five decoders.
            "best_of": 1,
            "single_segment": 0,
            "token_timestamps": 0,
        }
        floats = {
            "temperature": 0.0,
            "temperature_inc": 0.0,  # temperature fallback off
            "no_speech_thold": NO_SPEECH_THOLD_OFF,  # see the module docstring
        }
        for key, value in ints.items():
            if shim.lib.rs_whisper_set_opt_i(handle, key.encode("ascii"), int(value)) != 0:
                raise AppError(
                    "The speech engine library is out of date and rejected a decoding "
                    "option; rebuild it with ./native/whispercpp/build.sh.",
                    details=f"rs_whisper_set_opt_i({key}) failed: {shim.last_error()}",
                )
        for key, fvalue in floats.items():
            if shim.lib.rs_whisper_set_opt_f(handle, key.encode("ascii"), float(fvalue)) != 0:
                raise AppError(
                    "The speech engine library is out of date and rejected a decoding "
                    "option; rebuild it with ./native/whispercpp/build.sh.",
                    details=f"rs_whisper_set_opt_f({key}) failed: {shim.last_error()}",
                )

    def unload(self) -> None:
        with self._lock:
            shim, handle = self._shim, self._handle
            model_name = self._info.model if self._info else None
            self._shim = None
            self._handle = None
            self._info = None
        if handle is None or shim is None:
            return
        try:
            shim.lib.rs_whisper_free(handle)  # frees the whisper_context and its Metal buffers
        except Exception as e:  # pragma: no cover - rs_whisper_free cannot throw
            logger.debug("rs_whisper_free failed (ignored): %s", e)
        logger.info("engine_unloaded%s", f" model={model_name}" if model_name else "")

    def request_abort(self) -> None:
        """Make an in-flight process()/warmup() give up early. Safe from any thread."""
        with self._lock:
            shim, handle = self._shim, self._handle
        if shim is not None and handle is not None:
            shim.lib.rs_whisper_request_abort(handle)

    # -- inference ----------------------------------------------------------------

    def warmup(self, decode: DecodeConfig) -> None:
        """One throwaway run so the first real dictation does not pay the Metal
        pipeline-state cost (measured ~450 ms the first time on a given machine)."""
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(int(SAMPLE_RATE * WARMUP_SECONDS)) * WARMUP_NOISE_AMPLITUDE).astype(np.float32)
        t0 = time.perf_counter()
        segments = self._transcribe(noise, Task.TRANSCRIBE, decode, stage="warming up the model")
        logger.info("engine_warmup ms=%.1f segments=%d", (time.perf_counter() - t0) * 1000.0, len(segments))

    def process(
        self, segment: AudioSegment, task: Task, decode: DecodeConfig, filt: HallucinationFilter
    ) -> TranslationResult:
        task = Task(task)
        if self._handle is None:
            raise EngineNotLoadedError("The speech model is not loaded.")
        if task == Task.TRANSLATE and not self._translate_capable:
            name = self._info.model if self._info else "this model"
            raise TaskNotSupportedError(
                f"{name} can only transcribe Romanian; it cannot translate to English. "
                "Choose large-v3, large-v2 or medium for translation."
            )

        audio = np.ascontiguousarray(segment.samples, dtype=np.float32).reshape(-1)
        t_start = time.perf_counter()
        if audio.size == 0:
            whisper_segments: list[_Segment] = []
        else:
            whisper_segments = self._transcribe(audio, task, decode, stage="translating speech")
        t_end = time.perf_counter()

        kept: list[str] = []
        first_reason: Optional[str] = None
        for ws in whisper_segments:
            stats = SegmentStats(
                text=ws.text,
                avg_logprob=float(ws.avg_logprob),
                no_speech_prob=float(ws.no_speech_prob),
                compression_ratio=float(ws.compression_ratio),
            )
            reason = filt.evaluate(stats)
            logger.debug(
                "whisper_segment seg=%d avg_logprob=%.3f no_speech_prob=%.3f compression_ratio=%.2f "
                "filtered=%s reason=%s text=%r",
                segment.id, stats.avg_logprob, stats.no_speech_prob, stats.compression_ratio,
                format_kv(reason is not None), reason or "none", stats.text,
            )
            if reason is None:
                kept.append(stats.text.strip())
            elif first_reason is None:
                first_reason = reason

        text = " ".join(t for t in kept if t)
        filtered = not text
        if filtered and first_reason is None:
            first_reason = "empty"  # whisper produced no segments at all
        result = TranslationResult(
            segment_id=segment.id,
            session_id=segment.session_id,
            text=text if not filtered else "",
            filtered=filtered,
            filter_reason=first_reason if filtered else None,
            task=task,
            audio_s=segment.duration_s,
            t_vad_close=segment.t_vad_close,
            t_infer_start=t_start,
            t_infer_end=t_end,
        )
        logger.debug(
            "engine_result seg=%d session=%d audio_s=%.2f infer_ms=%.1f whisper_segments=%d filtered=%s reason=%s",
            result.segment_id, result.session_id, result.audio_s, result.infer_ms,
            len(whisper_segments), format_kv(filtered), result.filter_reason or "none",
        )
        return result

    def _transcribe(
        self, audio: np.ndarray, task: Task, decode: DecodeConfig, *, stage: str
    ) -> list[_Segment]:
        shim, handle = self._shim, self._handle
        if shim is None or handle is None:
            raise EngineNotLoadedError("The speech model is not loaded.")
        on_metal = self._info is not None and self._info.device == "metal"
        # audio is kept referenced for the whole call; the shim does not retain the buffer.
        audio = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        try:
            rc = int(
                shim.lib.rs_whisper_run(
                    handle,
                    audio.ctypes.data_as(_C_FLOATP),
                    int(audio.size),
                    SOURCE_LANGUAGE.encode("ascii"),
                    1 if task == Task.TRANSLATE else 0,
                    int(decode.beam_size),
                    0,  # n_threads: let whisper.cpp pick min(4, ncpu); Metal does the work
                )
            )
            if rc != 0:
                raise RuntimeError(shim.last_error() or f"whisper.cpp failed (rc={rc})")
            return self._collect(shim, handle)
        except AppError:
            raise
        except Exception as e:
            if is_out_of_memory_error(e):
                logger.error("out of memory while %s: %s", stage, e)
                self.unload()
                raise GpuOutOfMemoryError(gpu_oom_message(stage), details=f"{type(e).__name__}: {e}") from e
            if on_metal and is_metal_runtime_error(e):
                logger.error("Metal failure while %s: %s", stage, e)
                self.unload()
                raise _metal_error_from(e, stage) from e
            raise AppError(
                f"Speech recognition failed while {stage}: {e}", details=f"{type(e).__name__}: {e}"
            ) from e

    @staticmethod
    def _collect(shim: WhisperShim, handle: int) -> list[_Segment]:
        """Read the last run's segments. Called on the inference thread only: the shim's
        result accessors do not take the handle's mutex (see roenspeak_whisper.h)."""
        n = int(shim.lib.rs_whisper_n_segments(handle))
        out: list[_Segment] = []
        for i in range(n):
            raw = shim.lib.rs_whisper_segment_text(handle, i)
            out.append(
                _Segment(
                    text=raw.decode("utf-8", "replace") if raw else "",
                    avg_logprob=float(shim.lib.rs_whisper_segment_avg_logprob(handle, i)),
                    no_speech_prob=float(shim.lib.rs_whisper_segment_no_speech_prob(handle, i)),
                    n_tokens=int(shim.lib.rs_whisper_segment_n_tokens(handle, i)),
                )
            )
        return out


# Importing this module must never fail off macOS (contract rule 7); it is only *using* it
# that cannot work, and load() already reports that through CudaUnavailableError/AppError.
if not IS_MACOS:  # pragma: no cover - exercised by the Windows test run
    logger.debug("app.engine.whispercpp imported on %s; WhisperCppEngine is macOS-only", sys.platform)
