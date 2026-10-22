# RoEnSpeak: Original specification (verbatim from the project owner)

Build a Windows desktop dictation app: I speak Romanian, it types the English translation into whatever text field currently has focus (terminal, browser, Start menu search, any app).

## Context
- Speech engine: SYSTRAN faster-whisper (https://github.com/SYSTRAN/faster-whisper), cloned at: /home/gabiv/dev/repos/faster-whisper. Install from that local clone (pip install <path>) and pin the commit in requirements. Do not substitute any other Whisper implementation (no openai-whisper, whisper.cpp, transformers pipeline).
- Read the clone's README/requirements first and pin the exact ctranslate2 / CUDA 12 / cuDNN versions it expects.
- My GPU: RTX 4060TI 16 GB. OS: Windows 10/11 x64.
- The app will be shared with friends as an installer, so it must work on a clean Windows machine that only has an NVIDIA driver installed.

## Execution plan (orchestrator-level; summarized for sub-agents)
- Implementation sub-agents edit ONLY the files they own. Need a contract change? Stop and report back to the orchestrator instead of working around it. No placeholders, TODOs or stubbed logic in delivered code. Run a smoke check of your own module (import + minimal run) before reporting. Report back: files changed, public API implemented, assumptions made, known limitations.
- Test sub-agents write tests from the SPEC and contracts.py, not from the implementation. Edit ONLY files under `tests/` that they own. Derive expected behavior from the spec; where the spec is ambiguous, write the test for the stated behavior and flag the ambiguity in the report. Mock hardware (mic, GPU, Win32) for unit tests; mark tests needing real GPU/mic/desktop with pytest markers (`gpu`, `mic`, `desktop`). Report back: test files, what's covered, what's not covered and why.
- Test sub-agents:
  - TEST-UNIT-LOGIC: `tests/unit/test_segmenter.py`, `test_hallucination_filter.py`, `test_config.py`, `test_state_machine.py` — VAD boundary cases, max-length cut, flush-on-stop, blocklist/threshold filter, config defaults (Alt+Q, large-v3, float16), migration and invalid-config handling, state transitions.
  - TEST-UNIT-WIN32: `tests/unit/test_hotkey.py`, `test_injection.py` — hotkey string parsing/validation (reject no-modifier combos, normalize "alt+q"/"Alt+Q"), debounce, modifier-release wait, space insertion between segments, clipboard save/restore, UIPI detection path (mocked).
  - TEST-INTEGRATION: `tests/integration/` + `tests/fixtures/` — end-to-end pipeline with a Romanian .wav fixture through the CLI mode (marker `gpu`), asserting English output is non-empty, latency is logged, inference device is CUDA float16, and CPU fallback is never silent; a desktop test (marker `desktop`) that opens a test text window, triggers dictation from a .wav source, and asserts injected text appears.
- Implementation sub-agents:
  - IMPL-AUDIO: `app/audio/` — mic capture (16 kHz mono, device selection), Silero VAD segmentation (from faster-whisper), silence close + max-length cut, flush on stop.
  - IMPL-ENGINE: `app/engine/` — WhisperModel lifecycle on CUDA, warm-up, translate/transcribe calls, decoding params, hallucination filter, idle unload, CUDA/cuDNN DLL load check, GPU/VRAM detection, model downloader with progress/resume.
  - IMPL-WIN32: `app/win32/` — RegisterHotKey on a dedicated message-loop thread, hotkey string parsing/validation, debounce, Alt menu-bar suppression, SendInput unicode injection with modifier-release wait, clipboard fallback, elevated-window (UIPI) detection, single-instance mutex, autostart registry.
  - IMPL-UI: `app/ui/` — PySide6 main window (arm/disarm switch, hotkey capture field, status, settings), tray icon + menu, REC overlay, sounds, first-run download dialog, error dialogs.
  - IMPL-CORE: `app/core/` + `app/main.py` + `app/cli.py` — config load/save/migrate, logging, state machine, pipeline wiring (capture → VAD → inference → injection queues), CLI .wav test mode.
- Later phases: independent REVIEW, PACKAGING (`build/`, `installer/`, `BUILD.md`: PyInstaller onedir bundling CUDA DLLs, Inno Setup, version info, icon, signing hook), DOCS (README.md).

## Core behavior
1. App starts minimized to the system tray and runs in the background. Single instance (named mutex); a second launch brings the existing window to front.
2. Main window (opened from tray) contains:
   - A large ON/OFF switch = hotkey armed/disarmed. OFF unregisters the hotkey entirely and stops any active dictation. State persists across restarts.
   - A hotkey capture field: click it, press a key combo, it records it (e.g. "Alt+Q"). Validate: require at least one modifier, reject combos that fail to register, show a clear error if another app already owns it. Apply immediately without restart. Default: Alt+Q. "Reset to default" button.
   - Status line: idle / loading model / ready / listening / processing / error, plus GPU name and last segment latency.
   - Settings section (see below).
3. Hotkey toggles dictation: first press starts listening, second press stops. Use Win32 RegisterHotKey (not a low-level keyboard hook). Debounce key auto-repeat.
4. Alt-based combos: after the hotkey fires, prevent the target app from activating its menu bar when Alt is released (e.g. inject an unassigned virtual key such as VK 0xE8 before Alt-up, or an equivalent proven technique).
5. While listening: capture mic audio (16 kHz mono), segment with the Silero VAD bundled in faster-whisper, send each speech segment to faster-whisper with task="translate", language="ro".
6. Output is APPEND-ONLY: each translated segment is typed once into the currently focused window, never edited afterward. Insert a space between segments when needed.
7. Text injection: SendInput with KEYEVENTF_UNICODE by default. Before injecting, wait until all modifier keys (Ctrl/Shift/Alt/Win) are physically released. Clipboard-paste fallback mode available in settings (save and restore previous clipboard contents).
8. On stop: flush and translate any buffered audio so the last sentence is not lost, then stop capture.
9. Hotkey works only while the app process is running; Quit from tray unregisters everything and releases the GPU cleanly.

## Speed / GPU (priority)
- WhisperModel(device="cuda", compute_type="float16") by default; option int8_float16 for GPUs with less VRAM. Default model: large-v3. Selectable alternatives: large-v2, medium. Do NOT offer large-v3-turbo or distil-* for translation (not trained for it); if added for transcription-only mode, label them accordingly.
- Use plain WhisperModel.transcribe for short segments (no BatchedInferencePipeline).
- Load the model at startup and run one warm-up inference so the first real phrase isn't slow. Keep it resident in VRAM.
- Optional setting: unload after N minutes idle, reload on next hotkey press (off by default).
- Inference must run on CUDA. If CUDA/cuDNN is unavailable, show a clear error dialog explaining the requirement (NVIDIA GPU + minimum driver version). Provide an opt-in "CPU mode (slow)" setting, off by default; never fall back to CPU silently.
- Decoding defaults: beam_size=5 (configurable; allow 1 for max speed), condition_on_previous_text=False, temperature fallback off, without_timestamps=True.
- Target latency: translated text appears ≤1.5 s after I stop speaking a phrase. Log per-segment latency (VAD close → text injected).

## Quality
- VAD: ~500 ms silence closes a segment; force a cut at ~15 s of continuous speech. Both configurable.
- Filter Whisper hallucinations on near-silent segments ("Thank you for watching", "Subtitles by…", etc.) using no_speech_prob / avg_logprob thresholds plus a small editable blocklist.

## UX
- Tray icon reflects state (disarmed / ready / listening / processing / error). Tray menu: open window, arm/disarm, quit.
- Optional start/stop sound and a small always-on-top, click-through "● REC" overlay while listening (both toggleable).
- Settings (persist to %APPDATA%\<AppName>\config.json): microphone device, model, compute type, beam size, injection mode, VAD params, idle unload, overlay/sounds, mode (Translate RO→EN | Romanian transcription only), "Start with Windows" (HKCU Run key).
- Injection into elevated (admin) windows is blocked by UIPI: detect it and show a tray notification instead of failing silently.

## Distribution (for friends)
- Python 3.11+, UI with PySide6.
- PyInstaller onedir (not onefile: CUDA DLLs are too large to unpack every launch) + Inno Setup → single `<AppName>-Setup.exe` with Start Menu shortcut, optional desktop shortcut, uninstaller (asks whether to delete downloaded models).
- Bundle cuBLAS/cuDNN DLLs from the nvidia-*-cu12 pip wheels so users need only the NVIDIA driver. Verify on startup that the DLLs load and report the version.
- Do NOT bundle the model. First run shows a download dialog with progress (Hugging Face, Systran/faster-whisper-large-v3) into %LOCALAPPDATA%\<AppName>\models, resumable, clear error if offline.
- On first run, check NVIDIA driver version and VRAM; warn if VRAM is insufficient for the selected model/compute type and suggest int8_float16.
- App icon, version info embedded in the exe, version shown in the window.
- README notes that the installer is unsigned and SmartScreen may warn; build script has an optional signtool hook.

## Engineering
- Separate threads: audio capture → VAD/segment queue → inference worker → injection worker. UI never blocks on inference.
- Rotating logs in %LOCALAPPDATA%\<AppName>\logs; "Open log folder" button in UI.
- CLI test mode: run a .wav through the pipeline and print output + timings (no injection).
- BUILD.md: exact steps from clean checkout to setup.exe. README.md: requirements, install, usage, known limitations (admin windows, apps that block synthetic input, Alt+Q overrides Office's search shortcut while the app runs).

## Decisions approved by the owner in Phase 0 (binding)
- App name RoEnSpeak; Windows Python 3.12; repo source of truth in WSL, mirrored to C:\dev\ro-en-speak for Windows runs.
- Shift-only hotkeys rejected for printable keys. First segment of a session gets no leading space. Modifiers held > 5 s → segment dropped + notification with text. Unknown target integrity → treated as blocked (verify empirically). Disarm mid-dictation discards buffered audio; normal stop flushes. Max-length cuts have no overlap. Blocklist excludes "thank you"/"you". Schema 0 is a synthetic legacy format for migration. Test fixture generated via a ro-RO neural TTS voice.
