"""WhisperEngine: owns the faster-whisper WhisperModel (see contracts.SpeechEngine).

ctranslate2 / faster_whisper are imported lazily inside load(), after the CUDA DLL paths
have been set up, so this module also imports on machines without them (unit tests).
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from app.contracts import (
    CPU_COMPUTE_TYPE,
    LOG_ENGINE_LOADED,
    MIN_DRIVER_VERSION,
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
    models_dir,
)
from app.engine.cuda_probe import format_kv
from app.engine.downloader import model_dir_complete

logger = logging.getLogger("roenspeak.engine.engine")

#: Substrings (lower-cased) that identify a ctranslate2 RuntimeError as a CUDA-stack failure.
_CUDA_ERROR_MARKERS = ("cuda", "cudnn", "cublas", "nvrtc", "gpu", "device")

WARMUP_SECONDS = 1.0
WARMUP_NOISE_AMPLITUDE = 1e-3


def cuda_requirement_message(reason: Optional[str] = None) -> str:
    msg = (
        f"RoEnSpeak needs an NVIDIA GPU with driver {MIN_DRIVER_VERSION} or newer to run the speech "
        "model on CUDA. "
    )
    if reason:
        msg += f"Problem: {reason.rstrip('. ')}. "
    msg += (
        "Install or update the NVIDIA driver from nvidia.com and restart RoEnSpeak. "
        'Alternatively, enable "CPU mode (slow)" in Settings.'
    )
    return msg


def is_cuda_runtime_error(exc: BaseException) -> bool:
    if not isinstance(exc, (RuntimeError, OSError, ValueError)):
        return False
    text = str(exc).lower()
    return any(m in text for m in _CUDA_ERROR_MARKERS)


def is_out_of_memory_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def gpu_oom_message(stage: str) -> str:
    return (
        f"Not enough free GPU memory while {stage}. Another application may be using the "
        "video memory. Close GPU-heavy applications (games, video editors, other AI tools), "
        "or in RoEnSpeak Settings switch the compute type to int8_float16 or choose the "
        "smaller medium model, then try again."
    )


def _cuda_error_from(exc: BaseException, stage: str) -> CudaUnavailableError:
    text = str(exc)
    details = f"{type(exc).__name__}: {text}"
    if is_out_of_memory_error(exc):
        return GpuOutOfMemoryError(gpu_oom_message(stage), details=details)
    return CudaUnavailableError(cuda_requirement_message(f"a CUDA error occurred while {stage}"), details=details)


class WhisperEngine:
    """SpeechEngine implementation. All methods except is_loaded/info run on the single
    inference thread."""

    def __init__(self, probe: CudaProbe, models_root: Optional[Path] = None) -> None:
        self._probe = probe
        self._models_root = Path(models_root) if models_root is not None else None
        self._lock = threading.Lock()  # guards _model/_info for cross-thread readers
        self._model: Any = None
        self._info: Optional[EngineInfo] = None
        self._translate_capable = False

    # -- properties (any thread) --------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def info(self) -> Optional[EngineInfo]:
        return self._info

    @property
    def models_root(self) -> Path:
        return self._models_root if self._models_root is not None else models_dir()

    # -- lifecycle ----------------------------------------------------------------

    def load(self, model: str, compute_type: Union[ComputeType, str], cpu_mode: bool) -> EngineInfo:
        if model not in MODELS:
            raise ValueError(f"Unknown model: {model!r}")
        ct = ComputeType(compute_type)  # ValueError for an unknown compute type
        self.unload()

        path = self.models_root / model
        if not model_dir_complete(path):
            raise ModelNotDownloadedError(model)

        report: Optional[CudaReport] = None
        if cpu_mode:
            device, ctype = "cpu", CPU_COMPUTE_TYPE
            logger.warning("CPU mode (slow): loading %s on the CPU with compute_type=%s", model, ctype)
        else:
            device, ctype = "cuda", ct.value
            report = self._probe.probe()
            if not report.ok:
                raise CudaUnavailableError(
                    cuda_requirement_message(report.error), details=report.error or ""
                )

        t0 = time.perf_counter()
        try:
            self._probe.setup_dll_paths()
            from faster_whisper import WhisperModel  # lazy: after DLL paths are set up

            fw_logger = logging.getLogger("faster_whisper")
            if fw_logger.level == logging.NOTSET:
                # It logs "Processing audio with duration ..." at INFO for every segment.
                fw_logger.setLevel(logging.WARNING)
        except Exception as e:
            if not cpu_mode:
                raise CudaUnavailableError(
                    cuda_requirement_message(f"the speech engine could not be imported: {e}"),
                    details=repr(e),
                ) from e
            raise AppError(f"The speech engine could not be started: {e}", details=repr(e)) from e

        try:
            wm = WhisperModel(str(path), device=device, compute_type=ctype, local_files_only=True)
        except Exception as e:
            if not cpu_mode and is_cuda_runtime_error(e):
                raise _cuda_error_from(e, "loading the model") from e
            raise AppError(
                f"The model files for {model} could not be loaded; they may be corrupt. "
                f"Delete the folder {path} and download the model again.",
                details=f"{type(e).__name__}: {e}",
            ) from e
        load_ms = (time.perf_counter() - t0) * 1000.0

        actual_ct = str(getattr(wm.model, "compute_type", ctype))
        if actual_ct == f"{ctype}_float32":
            actual_ct = ctype  # ctranslate2 spells CPU "int8" as "int8_float32"; same type
        actual_device = str(getattr(wm.model, "device", device))
        info = EngineInfo(
            model=model,
            device=actual_device,
            compute_type=actual_ct,
            gpu_name=report.gpu.name if report and report.gpu else None,
            driver_version=report.gpu.driver_version if report and report.gpu else None,
            cublas_version=report.cublas_version if report else None,
            cudnn_version=report.cudnn_version if report else None,
            load_ms=load_ms,
        )
        with self._lock:
            self._model = wm
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
        if actual_ct != ctype:
            logger.warning("ctranslate2 substituted compute_type %s for the requested %s", actual_ct, ctype)
        return info

    def unload(self) -> None:
        with self._lock:
            wm = self._model
            model_name = self._info.model if self._info else None
            self._model = None
            self._info = None
        if wm is None:
            return
        try:
            inner = getattr(wm, "model", None)
            if inner is not None and hasattr(inner, "unload_model"):
                inner.unload_model()  # releases the weights' device memory right away
        except Exception as e:
            logger.debug("ctranslate2 unload_model failed (ignored): %s", e)
        del wm
        gc.collect()
        logger.info("engine_unloaded%s", f" model={model_name}" if model_name else "")

    # -- inference ----------------------------------------------------------------

    def warmup(self, decode: DecodeConfig) -> None:
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(int(SAMPLE_RATE * WARMUP_SECONDS)) * WARMUP_NOISE_AMPLITUDE).astype(np.float32)
        t0 = time.perf_counter()
        segments = self._transcribe(noise, Task.TRANSCRIBE, decode, stage="warming up the model")
        logger.info("engine_warmup ms=%.1f segments=%d", (time.perf_counter() - t0) * 1000.0, len(segments))

    def process(
        self, segment: AudioSegment, task: Task, decode: DecodeConfig, filt: HallucinationFilter
    ) -> TranslationResult:
        task = Task(task)
        if self._model is None:
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
            whisper_segments: list[Any] = []
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

    def _transcribe(self, audio: np.ndarray, task: Task, decode: DecodeConfig, *, stage: str) -> list[Any]:
        wm = self._model
        if wm is None:
            raise EngineNotLoadedError("The speech model is not loaded.")
        on_cuda = self._info is not None and self._info.device == "cuda"
        try:
            segments, _info = wm.transcribe(
                audio,
                language=SOURCE_LANGUAGE,
                task=task.value,
                beam_size=int(decode.beam_size),
                temperature=0.0,
                condition_on_previous_text=False,
                without_timestamps=True,
                vad_filter=False,
                no_speech_threshold=None,
            )
            return list(segments)  # fully consume the generator
        except AppError:
            raise
        except Exception as e:
            if on_cuda and is_cuda_runtime_error(e):
                logger.error("CUDA failure while %s: %s", stage, e)
                self.unload()
                raise _cuda_error_from(e, stage) from e
            raise AppError(f"Speech recognition failed while {stage}: {e}", details=f"{type(e).__name__}: {e}") from e
