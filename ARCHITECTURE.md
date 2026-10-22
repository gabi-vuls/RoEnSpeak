# RoEnSpeak: Architecture

RoEnSpeak is a Windows tray app. You press a global hotkey and speak Romanian. Each phrase is translated to English by
faster-whisper on the GPU and typed into whichever window has focus. Output is append-only.

`app/contracts.py` is the single source of truth for every interface named below. This document explains
*why* the pieces fit together that way.

## 1. Modules and ownership

| Package | Owner | Responsibility |
|---|---|---|
| `app/contracts.py` | orchestrator | Interfaces, dataclasses, enums, errors, constants, paths |
| `app/audio/` | IMPL-AUDIO | Mic capture (`MicSource`), wav source (`WavFileSource`), streaming Silero VAD (`SileroStreamingVad`), `VadSegmenter` |
| `app/engine/` | IMPL-ENGINE | CUDA/DLL probe, `WhisperEngine`, `ThresholdHallucinationFilter`, `HfModelDownloader`, VRAM check |
| `app/win32/` | IMPL-WIN32 | `RealWinApi` (ctypes), hotkey parser, `Win32HotkeyManager`, `Win32TextInjector`, clipboard, single instance, autostart |
| `app/ui/` | IMPL-UI | PySide6 main window, tray, REC overlay, sounds, download dialog, error dialogs, `QtUiFacade` |
| `app/core/` + `app/main.py` + `app/cli.py` | IMPL-CORE | Config store and migration, logging, state machine, event bus, pipeline threads, `AppController`, entry points |
| `tests/unit/…` | TEST-UNIT-LOGIC / TEST-UNIT-WIN32 | Spec-derived unit tests with mocked hardware |
| `tests/integration/`, `tests/fixtures/` | TEST-INTEGRATION | GPU CLI end-to-end test and desktop injection test |
| `build/`, `installer/`, `BUILD.md` | PACKAGING | PyInstaller onedir, Inno Setup, signing hook |
| `README.md` | DOCS | User documentation |

Dependency direction: `ui → contracts ← core → {audio, engine, win32} → contracts`. The UI never imports
audio/engine/win32. It talks only to `Controller` and listens to `EventBus`. Audio, engine and win32 never import each
other or core.

## 2. Threads and data flow

```
                       ┌──────────── Qt main thread ─────────────┐
                       │ MainWindow · Tray · Overlay · Dialogs   │
                       │   ▲ Qt signal (QueuedConnection)        │
                       │   │            Controller calls ▼       │
                       └───┼─────────────────────────┬───────────┘
                    EventBus.publish (any thread)    │ (returns < 50 ms)
                           │                         ▼
 ┌─ hotkey thread ─┐   ┌───┴──────────── core (AppController + StateMachine) ───────────────┐
 │ RegisterHotKey  │──▶│ on_fire → control queue → start/stop session                       │
 │ GetMessage loop │   └─────┬──────────────────────────────┬────────────────────────┬──────┘
 └─────────────────┘         │ start()/stop()               │ load/unload cmds       │
                             ▼                              ▼                        │
 ┌─ PortAudio cb thread ─┐  q_audio   ┌─ segmenter thread ─┐  q_segments  ┌─ inference thread ─┐  q_inject  ┌─ injection thread ─┐
 │ MicSource / WavSource │──chunks──▶ │ VadSegmenter.feed  │──AudioSeg──▶ │ WhisperEngine      │──Result──▶ │ Win32TextInjector  │
 │ (16 kHz mono f32)     │  +FLUSH    │ (Silero streaming) │              │ .process + filter  │            │ wait mods → UIPI → │
 └───────────────────────┘  cmds      └────────────────────┘              │ (sole model owner) │            │ SendInput/clipboard│
                                                                          └────────────────────┘            └────────────────────┘
```

- **Queues** are `queue.Queue`. `q_audio` carries audio chunks interleaved with typed commands
  (`BeginSession(id)`, `Flush`, `Reset`), so ordering is exact: when you stop, the last chunk is enqueued
  before `Flush`. `q_segments` and `q_inject` carry `AudioSegment` / `TranslationResult`. Every
  queue has a `None` shutdown sentinel.
- **Inference thread** is the only thread that touches `WhisperModel`. Load, warmup, unload and process are all
  commands on its own command queue, which it checks between segments. Model load therefore never blocks
  the UI, and an unload can never race an in-flight transcribe.
- **The UI never blocks.** Controller methods only enqueue work and return. Results come back as `AppEvent`s,
  which `QtUiFacade` and the window bridge onto the Qt thread with a queued signal.
- **Hotkey thread.** `RegisterHotKey` is per-thread, so register/unregister requests are posted to the hotkey thread
  (`PostThreadMessage(WM_APP+n)`) and the caller waits on an event for the result. `on_fire` only puts a
  `Toggle` into core's control queue.
- **Session lifecycle.**
  1. Hotkey press → new `session_id`, then `BeginSession`, `MicSource.start`, state LISTENING, start sound and overlay.
  2. Second press → `MicSource.stop()` (blocks until no more callbacks) → `Flush` → state PROCESSING.
  3. When the inference queue drains past the flushed segment's id → READY.
  4. Disarm → stop the source, `Reset` (buffered audio is discarded), drain the queues, unregister the hotkey.

## 3. Latency budget (target ≤ 1.5 s from end of speech to text)

| Stage | Budget |
|---|---|
| Trailing silence to close a segment (`min_silence_ms`) | 500 ms (configurable) |
| VAD per 32 ms frame (onnxruntime CPU, 1 thread) | < 1 ms |
| large-v3 float16, beam 5, ≤ 15 s segment, RTX 4060 Ti | ~300–700 ms |
| Modifier-release wait + SendInput | ~5 ms when keys are already up |

Measured latency is logged per segment as `segment_latency … latency_ms=` (from VAD close to text injected), plus `infer_ms`.
`beam_size=1` and `medium` are the speed levers. The model stays resident and is warmed up at load.

## 4. Speech engine details (faster-whisper @ 9e472ae, v1.2.1)

- `WhisperModel(models_dir()/<model>, device="cuda", compute_type="float16" | "int8_float16", local_files_only=True)`.
  We use plain `transcribe` on each segment (no `BatchedInferencePipeline`, because segments are short and sequential) with
  `language="ro"`, `task="translate"|"transcribe"`, `beam_size=N`, `temperature=0.0` (no fallback),
  `condition_on_previous_text=False`, `without_timestamps=True`, `vad_filter=False` (the segments are already VAD-cut),
  and `no_speech_threshold=None`, so faster-whisper does not silently skip segments and our filter decides with
  full stats. The returned generator is always fully consumed.
- **Streaming VAD.** `faster_whisper.vad.SileroVADModel.__call__` resets the LSTM state on every call, so it only works
  in batch mode. `SileroStreamingVad` loads the same bundled `silero_vad_v6.onnx`
  (`faster_whisper.utils.get_assets_path()`) into an onnxruntime CPU session (1 intra / 1 inter thread, as upstream)
  and feeds 512-sample frames with the 64-sample context and the `h`/`c` state carried between calls, using the
  same tensor layout as upstream. The segmentation rules (hysteresis, silence close, max-length cut, flush) are in
  `Segmenter`'s docstring.
- **Hallucination filter.** Per whisper segment: empty, `no_speech_prob > 0.6 ∧ avg_logprob < -1.0`,
  `compression_ratio > 2.4`, or a blocklist match (exact, or prefix for entries ending in `*`). All of these are configurable.
  Dropped segments are logged with their reason and never typed.
- **Models.** large-v3 (default), large-v2 and medium can translate. large-v3-turbo is offered only for Romanian
  transcription and is labeled that way. distil-* are English-only and are not offered.
- **Idle unload** (off by default). After N minutes in READY, the model is unloaded and the state becomes UNLOADED. The next hotkey press
  starts capture immediately (LOADING_MODEL → LISTENING) and queued segments are processed once the model is loaded.

## 5. CUDA / DLL policy

Verified in Phase 0 on this machine: `ctranslate2==4.8.3` links the CUDA runtime statically, loads
`cublas64_12.dll` at runtime, and ships a cuDNN 9 dispatcher (`cudnn64_9.dll`) that loads `cudnn_*64_9.dll` by
name. `os.add_dll_directory` alone is **not** enough for the dispatcher (it failed with "Could not locate
cudnn_graph64_9.dll"). The `nvidia/*/bin` dirs must also be on `PATH`. With that, `tiny` on cuda/float16 loaded in 0.7 s,
with cuBLAS 12.9.2 and cuDNN 9.27.0 (`cudnnGetVersion()=92700`) on driver 610.88.

- `WindowsCudaProbe.setup_dll_paths()` runs first thing in `main`/`cli`, before ctranslate2 is imported. It
  covers the dev venv (`site-packages/nvidia/*/bin`) and the frozen build (`_internal/nvidia/*/bin`).
- `probe()` reads the driver version and VRAM via NVML (`nvml.dll` comes with the driver), loads cuBLAS/cuDNN via ctypes, and
  reads their versions. It never raises.
- **Policy.** If CUDA is not usable and `cpu_mode` is False, `load()` raises `CudaUnavailableError` → state ERROR → error
  dialog ("RoEnSpeak needs an NVIDIA GPU with driver ≥ 528.33 …"). CPU is used only when the user ticks
  "CPU mode (slow)". `EngineInfo.device` and the `engine_loaded` log line always state the real device and compute type.
- **First run.** If VRAM is below `MODELS[model].vram_mb[compute_type]`, a notification suggests int8_float16.
- **Size.** cuBLAS is about 772 MB and cuDNN about 1149 MB unpacked. PACKAGING will try to trim the set (e.g. drop
  `nvblas64_12.dll`, and drop cuDNN engine DLLs only if a GPU smoke test still passes without them).

## 6. State machine

States and the allowed transitions are in `contracts.ALLOWED_TRANSITIONS`.

```
DISARMED ──arm──▶ (DOWNLOADING ─▶) LOADING_MODEL ─▶ READY ◀──────────────┐
    ▲                                   │               │ hotkey          │ flush done
    │ disarm (from any state)           │ hotkey        ▼                 │
    └───────────────────────────────────┴──────────▶ LISTENING ─hotkey─▶ PROCESSING
                                     READY ─idle─▶ UNLOADED ─hotkey─▶ LOADING_MODEL ─▶ LISTENING
                       any ─fatal─▶ ERROR ─retry/arm─▶ DOWNLOADING | LOADING_MODEL | READY | DISARMED
```

The UI maps states to the status line (idle/downloading/loading model/ready/listening/processing/error) and to
the tray icon (disarmed/ready/listening/processing/error).

## 7. Text injection

1. Strip the text and collapse newlines and tabs. Add a leading space only if this session already typed something
   that did not end in whitespace and the new text does not start with closing punctuation.
2. Wait until Ctrl/Shift/Alt/Win (left and right) are all up, polling `GetAsyncKeyState` every 10 ms with a 5 s timeout.
   On timeout → `ModifiersHeldError`, and a notification shows the dropped text.
3. **UIPI.** Compare the integrity level of the foreground window's process with our own. If it is higher, or can't be read →
   `InjectionBlockedError` → tray notification showing the text. (SendInput's return value does *not* reveal
   UIPI blocking, so this check is the primary detection.)
4. Default: one `SendInput` with `KEYEVENTF_UNICODE` down/up pairs per UTF-16 code unit, so diacritics and
   surrogate pairs work in any app. Clipboard mode instead does save formats → set text → Ctrl+V → wait 150 ms → restore.
5. **Alt-menu suppression** happens in the hotkey manager, not the injector. When the hotkey fires with Alt down, it sends `VK 0xE8`
   down/up, so the focused app sees a non-Alt keystroke and does not open its menu bar when Alt is released.

## 8. Libraries

| Library | Why |
|---|---|
| faster-whisper (local clone, pinned commit) | Required engine. CTranslate2 Whisper on CUDA, plus the bundled Silero VAD asset |
| ctranslate2 4.8.3 | Newest 4.x; CUDA 12 + cuDNN 9 builds, as the faster-whisper README requires |
| nvidia-cublas-cu12 12.9.2.10, nvidia-cudnn-cu12 9.27.0.42 | Bundled DLLs, so friends need only the NVIDIA driver |
| onnxruntime | Silero VAD on CPU (already a faster-whisper dependency) |
| sounddevice (+ bundled PortAudio) | Mic capture with device selection. If the device rejects 16 kHz, we capture at its native rate and resample with **soxr** |
| PySide6-Essentials | UI and tray. Essentials only, which keeps the installer smaller than the full PySide6 |
| nvidia-ml-py | Driver version and VRAM via NVML |
| ctypes (stdlib) | All Win32 calls; no pywin32, which keeps PyInstaller simple |
| huggingface_hub | Resumable model download into `%LOCALAPPDATA%\RoEnSpeak\models\<model>` |
| PyInstaller (onedir) + Inno Setup 6 | Installer. Onedir avoids unpacking about 2 GB of DLLs on every launch |

## 9. Paths, config, logs

- Config: `%APPDATA%\RoEnSpeak\config.json`. Schema v1 (see `Config`), with migration from the flat schema 0. Invalid fields
  fall back to defaults with warnings, and a corrupt file is backed up as `config.json.bad-<ts>`.
- Models: `%LOCALAPPDATA%\RoEnSpeak\models\<model>\` (plain files plus the `.download_complete` marker).
- Logs: `%LOCALAPPDATA%\RoEnSpeak\logs\roenspeak.log` (rotating, 5 × 2 MB). The UI has an "Open log folder" button.
- Tests redirect all three locations with `ROENSPEAK_CONFIG_DIR` / `ROENSPEAK_DATA_DIR` / `ROENSPEAK_MODELS_DIR`.
- Single instance: the named mutex `Local\RoEnSpeak.SingleInstance`. A second launch broadcasts the registered message
  `RoEnSpeak.ShowWindow` and exits.

## 10. Error policy

- Every cross-module error is an `AppError` subclass carrying a `user_message`. Core turns errors into
  `ErrorRaised` (dialog; `fatal=True` → ERROR state) or `Notification` (tray balloon) events.
- Fatal errors: CUDA unavailable, download failed or offline, model files corrupt, mic unavailable when starting.
  Non-fatal: UIPI-blocked or modifier-held injection, hotkey conflict (the previous hotkey stays), VRAM warning.
- Shutdown paths are tray Quit, `WM_ENDSESSION`/logoff, an unhandled exception in `main`, and Ctrl+C in the CLI. All of them run
  `controller.quit()`: stop the source, flush nothing, stop the threads, unregister the hotkey, unload the model, release the mutex.
  `atexit` is a backstop for unregistering the hotkey and releasing the mutex.

## 11. Build and test environment

The repo lives in WSL. `scripts/win_sync.sh` rsyncs it to `C:\dev\ro-en-speak`, and `scripts/win.sh ARGS` runs
`C:\dev\ro-en-speak\.venv\Scripts\python.exe ARGS` there (Windows Python 3.12.10). The faster-whisper clone is mirrored
to `C:\dev\faster-whisper` and installed with the pinned `git+file` commit. Test markers: default = unit tests (mocked
hardware), `-m gpu` (real CUDA and the model), `-m desktop` (real keystrokes into a test window), `-m mic` (a human
speaking).
