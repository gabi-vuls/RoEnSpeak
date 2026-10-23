"""CLI test mode: run a .wav through WavFileSource -> Segmenter -> SpeechEngine (no injection).

    python -m app.cli WAV [--model NAME] [--compute-type float16|int8_float16] [--beam N]
                          [--task translate|transcribe] [--cpu] [--download] [--json] [--realtime]

Exit codes: 0 ok; 1 other error; 2 the GPU backend is unavailable (and not --cpu); 3 model
not downloaded (and not --download); 4 bad input (missing/unreadable wav, invalid args).
See the CLI contract block in app/contracts.py for the output formats.
"""

from __future__ import annotations

import sys

# The probe runs setup_dll_paths() at import time: on Windows the CUDA DLL directories must be
# on the search path before anything imports ctranslate2. (No-op on macOS.)
from app.core import platform

_PROBE = platform.make_probe()
_PROBE.setup_dll_paths()

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Optional, Sequence  # noqa: E402

from app.contracts import (  # noqa: E402
    MIN_DRIVER_VERSION,
    MIN_MACOS_VERSION,
    MODELS,
    AppError,
    ComputeType,
    CudaUnavailableError,
    DownloadCancelledError,
    DownloadError,
    DownloadProgress,
    EngineInfo,
    ModelNotDownloadedError,
    SegmentProcessed,
    SegmentCloseReason,
    Task,
    TaskNotSupportedError,
    TranslationResult,
)

EXIT_OK, EXIT_ERROR, EXIT_CUDA, EXIT_NO_MODEL, EXIT_BAD_INPUT = 0, 1, 2, 3, 4

#: What exit code 2 is called in messages and logs. Unchanged on Windows.
_GPU_UNAVAILABLE = "Metal/GPU unavailable" if platform.IS_MACOS else "CUDA unavailable"

log = logging.getLogger("roenspeak.cli")


class _BadArgs(Exception):
    pass


class _ArgParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        # argparse would exit 2, which means "CUDA unavailable" in our contract.
        raise _BadArgs(message)


def _build_parser() -> argparse.ArgumentParser:
    p = _ArgParser(prog="python -m app.cli", description="Run a .wav through the RoEnSpeak pipeline (no typing).")
    p.add_argument("wav", help="input .wav file (any rate/channels)")
    p.add_argument("--model", choices=list(MODELS), help="model (default: from config)")
    p.add_argument("--compute-type", choices=[c.value for c in ComputeType], help="GPU compute type")
    p.add_argument("--beam", type=int, help="beam size 1..10 (1 = fastest)")
    p.add_argument("--task", choices=[t.value for t in Task], help="translate (RO->EN) or transcribe (RO)")
    p.add_argument("--cpu", action="store_true", help="run on CPU (slow)")
    p.add_argument("--download", action="store_true", help="download the model if missing")
    p.add_argument("--json", action="store_true", help="print one JSON object instead of text lines")
    p.add_argument("--realtime", action="store_true", help="feed audio at 1x speed")
    return p


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass


def _err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class _Run:
    """Collects pipeline callbacks (they arrive on pipeline threads)."""

    def __init__(self, json_mode: bool) -> None:
        self.json_mode = json_mode
        self.loaded = threading.Event()
        self.flushed = threading.Event()
        self.ended = threading.Event()
        self.info: Optional[EngineInfo] = None
        self.load_error: Optional[BaseException] = None
        self.process_error: Optional[BaseException] = None
        self.source_error: Optional[BaseException] = None
        self.segments: list[dict[str, Any]] = []
        self._print_lock = threading.Lock()

    def on_loaded(self, info: EngineInfo, _token: Any) -> None:
        self.info = info
        self.loaded.set()

    def on_engine_error(self, exc: BaseException, phase: str, _token: Any) -> None:
        if phase == "load":
            self.load_error = exc
            self.loaded.set()
        else:
            self.process_error = exc
            self.flushed.set()

    def on_flushed(self, _sid: int) -> None:
        self.flushed.set()

    def on_source_error(self, exc: BaseException) -> None:
        self.source_error = exc
        self.ended.set()

    def on_segment(self, ev: SegmentProcessed, r: TranslationResult, reason: SegmentCloseReason) -> None:
        seg = {
            "id": r.segment_id,
            "reason": reason.value,
            "audio_s": round(r.audio_s, 3),
            "infer_ms": round(r.infer_ms, 1),
            "latency_ms": None if ev.latency_ms is None else round(ev.latency_ms, 1),
            "filtered": r.filtered,
            "filter_reason": r.filter_reason,
            "text": r.text,
        }
        with self._print_lock:
            self.segments.append(seg)


def _wait(ev: threading.Event, abort: Optional[threading.Event] = None) -> None:
    # Short waits keep Ctrl+C responsive (on Windows a long Event.wait() swallows it).
    while not ev.wait(0.2):
        if abort is not None and abort.is_set():
            return


def main(argv: Optional[Sequence[str]] = None) -> int:
    _utf8_stdio()
    try:
        args = _build_parser().parse_args(argv)
    except _BadArgs as e:
        _err(f"error: {e}")
        return EXIT_BAD_INPUT
    except SystemExit as e:  # --help
        return int(e.code or 0)

    from app.core.logging_setup import setup_logging

    log_file = setup_logging(console_level=logging.WARNING)
    log.info("cli start argv=%s log=%s", list(sys.argv[1:] if argv is None else argv), log_file)
    try:
        return _run(args)
    except KeyboardInterrupt:
        _err("Interrupted.")
        log.warning("cli interrupted")
        return EXIT_ERROR
    except Exception as e:
        log.exception("cli failed")
        _err(f"ERROR: {getattr(e, 'user_message', e)}")
        return EXIT_ERROR


def _run(args: argparse.Namespace) -> int:
    from app.core.config import JsonConfigStore

    # 1. Arguments and input (exit 4).
    cfg = JsonConfigStore().load()
    model = args.model or cfg.model
    compute_type = ComputeType(args.compute_type) if args.compute_type else cfg.compute_type
    task = Task(args.task) if args.task else cfg.task
    beam = args.beam if args.beam is not None else cfg.decode.beam_size
    if not 1 <= beam <= 10:
        _err(f"error: --beam must be 1..10, got {beam}")
        return EXIT_BAD_INPUT
    if task == Task.TRANSLATE and not MODELS[model].translate_capable:
        _err(f"error: model {model} is transcription-only; use --task transcribe or another model")
        return EXIT_BAD_INPUT
    decode = dataclasses.replace(cfg.decode, beam_size=beam)

    wav = Path(args.wav)
    if not wav.is_file():
        _err(f"error: wav file not found: {wav}")
        return EXIT_BAD_INPUT
    from app.audio.wavsource import WavFileSource, read_wav_16k_mono

    try:
        audio = read_wav_16k_mono(wav)
    except Exception as e:
        _err(f"error: can't read wav {wav}: {getattr(e, 'user_message', e)}")
        return EXIT_BAD_INPUT
    if len(audio) == 0:
        _err(f"error: wav file has no audio: {wav}")
        return EXIT_BAD_INPUT
    log.info("cli wav=%s audio_s=%.2f model=%s compute_type=%s task=%s beam=%d cpu=%s realtime=%s",
             wav, len(audio) / 16000.0, model, compute_type.value, task.value, beam, args.cpu, args.realtime)
    del audio

    # 2. GPU backend (exit 2).
    if args.cpu:
        _err("WARNING: CPU mode (slow)")
        log.warning("CPU mode (slow) requested on the command line")
    else:
        report = _PROBE.probe()
        if not report.ok:
            return _cuda_exit(report.error)

    # 3. Model presence (exit 3).
    downloader = platform.make_downloader()
    # macOS: the ggml downloader resolves its model file from the compute type.
    if hasattr(downloader, "set_compute_type"):
        downloader.set_compute_type(compute_type)
    if not downloader.is_downloaded(model):
        if not args.download:
            _err(f"error: model '{model}' is not downloaded (expected in {downloader.model_path(model)}); "
                 f"run with --download")
            return EXIT_NO_MODEL
        rc = _download(downloader, model, compute_type)
        if rc != EXIT_OK:
            return rc

    # 4. Load + run.
    from app.audio.segmenter import VadSegmenter
    from app.audio.vad import SileroStreamingVad
    from app.core.pipeline import Pipeline
    from app.engine.filter import ThresholdHallucinationFilter

    run = _Run(args.json)

    def on_segment(ev: SegmentProcessed, r: TranslationResult, reason: SegmentCloseReason) -> None:
        run.on_segment(ev, r, reason)
        if not args.json:
            s = run.segments[-1]
            print(f"SEGMENT id={s['id']} reason={s['reason']} audio_s={r.audio_s:.2f} infer_ms={r.infer_ms:.1f} "
                  f"filtered={1 if r.filtered else 0} text={r.text}", flush=True)

    engine = platform.make_engine(_PROBE)
    pipeline = Pipeline(
        segmenter=VadSegmenter(cfg.vad, SileroStreamingVad()),
        engine=engine,
        filt=ThresholdHallucinationFilter(cfg.filter),
        injector=None,
        bus=None,
        task=task,
        decode=decode,
        on_engine_loaded=run.on_loaded,
        on_engine_error=run.on_engine_error,
        on_session_flushed=run.on_flushed,
        on_segment_done=on_segment,
    )
    pipeline.start()
    source = None
    try:
        pipeline.load_model(model, compute_type, args.cpu, decode)
        _wait(run.loaded)
        if run.load_error is not None:
            return _load_error_exit(run.load_error)
        info = run.info
        assert info is not None
        if not args.json:
            print(f"ENGINE model={info.model} device={info.device} compute_type={info.compute_type} "
                  f"gpu={info.gpu_name or 'none'} load_ms={info.load_ms:.1f}", flush=True)

        source = WavFileSource(wav, realtime=args.realtime)
        t_start = time.perf_counter()
        pipeline.begin_session(1)
        try:
            source.start(pipeline.feed_audio, on_end=run.ended.set, on_error=run.on_source_error)
        except AppError as e:
            _err(f"error: {e.user_message}")
            return EXIT_BAD_INPUT
        _wait(run.ended)
        source.stop()
        if run.source_error is not None:
            _err(f"error: {getattr(run.source_error, 'user_message', run.source_error)}")
            return EXIT_BAD_INPUT
        pipeline.flush(1)
        _wait(run.flushed)
        wall_ms = (time.perf_counter() - t_start) * 1000.0
        if run.process_error is not None:
            e = run.process_error
            _err(f"ERROR: inference failed: {getattr(e, 'user_message', e)}")
            return EXIT_CUDA if isinstance(e, CudaUnavailableError) else EXIT_ERROR

        segs = run.segments
        summary = {
            "segments": len(segs),
            "kept": sum(1 for s in segs if not s["filtered"]),
            "total_audio_s": round(sum(s["audio_s"] for s in segs), 3),
            "total_infer_ms": round(sum(s["infer_ms"] for s in segs), 1),
            "wall_ms": round(wall_ms, 1),
        }
        log.info("cli summary %s", " ".join(f"{k}={v}" for k, v in summary.items()))
        if args.json:
            out = {"engine": dataclasses.asdict(info), "segments": segs, "summary": summary}
            print(json.dumps(out, ensure_ascii=False), flush=True)
        else:
            print(f"SUMMARY segments={summary['segments']} kept={summary['kept']} "
                  f"total_audio_s={summary['total_audio_s']:.2f} total_infer_ms={summary['total_infer_ms']:.1f} "
                  f"wall_ms={summary['wall_ms']:.1f}", flush=True)
        return EXIT_OK
    finally:
        if source is not None:
            try:
                source.stop()
            except Exception:
                log.exception("source stop failed")
        pipeline.stop(unload=True, timeout=15.0)


def _cuda_exit(error: Optional[str]) -> int:
    """Exit 2. The "ERROR: " prefix and the exit code are the machine-readable part; the
    explanation after it names the platform's GPU stack."""
    if platform.IS_MACOS:
        _err(f"ERROR: {_GPU_UNAVAILABLE}: {error or 'unknown reason'}\n"
             f"RoEnSpeak needs macOS {MIN_MACOS_VERSION} or newer and a Metal-capable GPU. "
             f"Use --cpu to run on the CPU (slow).")
    else:
        _err(f"ERROR: {_GPU_UNAVAILABLE}: {error or 'unknown reason'}\n"
             f"RoEnSpeak needs an NVIDIA GPU with driver >= {MIN_DRIVER_VERSION} and loadable cuBLAS/cuDNN. "
             f"Use --cpu to run on the CPU (slow).")
    log.error("cli: %s: %s", _GPU_UNAVAILABLE, error)
    return EXIT_CUDA


def _load_error_exit(e: BaseException) -> int:
    if isinstance(e, CudaUnavailableError):  # incl. GpuOutOfMemoryError; the message is user-facing
        _err(f"ERROR: {e.user_message}" + (f"\n({e.details})" if e.details else ""))
        log.error("cli: %s: %s", type(e).__name__, e.details or e.user_message)
        return EXIT_CUDA
    if isinstance(e, ModelNotDownloadedError):
        _err(f"error: {e.user_message} Run with --download.")
        return EXIT_NO_MODEL
    if isinstance(e, (TaskNotSupportedError, ValueError)):
        _err(f"error: {getattr(e, 'user_message', e)}")
        return EXIT_BAD_INPUT
    _err(f"ERROR: model load failed: {getattr(e, 'user_message', e)}")
    return EXIT_ERROR


def _download_mb(model: str, compute_type: ComputeType) -> int:
    """Download size for the progress line: the CTranslate2 model directory on Windows, the
    single ggml file the macOS engine loads (per compute type) on macOS."""
    spec = MODELS[model]
    if platform.IS_MACOS:
        return int(spec.ggml_mb.get(compute_type, spec.download_mb))
    return spec.download_mb


def _download(downloader: Any, model: str, compute_type: ComputeType) -> int:
    cancel = threading.Event()
    last = [0.0]

    def on_progress(p: DownloadProgress) -> None:
        now = time.monotonic()
        if now - last[0] < 1.0:
            return
        last[0] = now
        total = f"{p.total_bytes / 1e6:.0f}" if p.total_bytes else "?"
        _err(f"downloading {model}: {p.downloaded_bytes / 1e6:.0f}/{total} MB "
             f"({p.bytes_per_s / 1e6:.1f} MB/s) {p.current_file}")

    _err(f"downloading model {model} ({_download_mb(model, compute_type)} MB)...")
    result: dict[str, Any] = {}

    def worker() -> None:
        try:
            result["path"] = downloader.download(model, on_progress, cancel)
        except BaseException as e:  # handed to the main thread
            result["error"] = e

    t = threading.Thread(target=worker, name="cli-download", daemon=True)
    t.start()
    try:
        while t.is_alive():
            t.join(0.2)
    except KeyboardInterrupt:
        cancel.set()
        t.join(10.0)
        raise
    err = result.get("error")
    if err is None:
        _err(f"model downloaded to {result.get('path')}")
        return EXIT_OK
    if isinstance(err, DownloadError):
        _err(f"ERROR: {err.user_message}" + (" (no internet connection?)" if err.offline else ""))
        return EXIT_ERROR
    if isinstance(err, DownloadCancelledError):
        _err("download cancelled")
        return EXIT_ERROR
    _err(f"ERROR: download failed: {err}")
    return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
