"""Cross-module contracts for RoEnSpeak.

THIS FILE IS THE SINGLE SOURCE OF TRUTH. Sub-agents implement against it and may not
change it; contract changes go through the orchestrator.

Conventions
-----------
* Audio is always float32 mono, SAMPLE_RATE (16 kHz), values in [-1, 1].
* Times named ``t_*`` are ``time.perf_counter()`` seconds unless stated otherwise. (Not
  time.monotonic(): on Windows that is GetTickCount64 with 15.6 ms resolution, which makes
  per-segment latency numbers useless. perf_counter is monotonic and ~100 ns.)
* "Any thread" means the method is thread-safe. Otherwise the docstring names the
  single thread that may call it.
* All app-specific errors derive from ``AppError``. Implementations do not raise
  bare ``Exception``/``RuntimeError`` across module boundaries.
* Concrete implementations live at the import paths in the "Implementation map" at
  the bottom of this file. Tests import from those paths.
"""

from __future__ import annotations

import enum
import hashlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Callable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME = "RoEnSpeak"
APP_VERSION = "0.1.0"
SAMPLE_RATE = 16_000
VAD_FRAME_SAMPLES = 512  # 32 ms at 16 kHz; Silero v6 frame size used by faster-whisper
DEFAULT_HOTKEY = "Alt+Q"  # Windows: Alt+Q, macOS: Option+Q (same string, see Modifier)
SOURCE_LANGUAGE = "ro"

#: Host platform switches. Platform code lives in app/win32 (Windows) and app/macos (macOS);
#: app.core.platform is the only module that chooses between them.
IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"


def os_display_name() -> str:
    """"Windows" / "macOS" / "the system", for user-facing messages."""
    if IS_WINDOWS:
        return "Windows"
    if IS_MACOS:
        return "macOS"
    return "the system"

#: Minimum NVIDIA driver for CUDA 12.x minor-version compatibility on Windows.
#: The definitive check is CudaProbe actually loading cuBLAS/cuDNN.
MIN_DRIVER_VERSION = "528.33"

#: Minimum macOS version. Metal 3 and the Carbon hotkey / CGEvent APIs the macOS layer uses
#: would run on 13.0, but the pinned wheels (PySide6-Essentials 6.12, onnxruntime 1.31,
#: numpy 2.5.4) are all tagged macosx_14_0, so 14.0 is the real floor for a built .app.
MIN_MACOS_VERSION = "14.0"

#: Registered window message (RegisterWindowMessageW) a second instance broadcasts
#: to ask the running instance to show its main window.
SHOW_WINDOW_MESSAGE_NAME = f"{APP_NAME}.ShowWindow"

#: Unassigned virtual key injected (down+up) while Alt is held after the hotkey fires,
#: so releasing Alt does not activate the target app's menu bar.
VK_MENU_MASK = 0xE8

#: Log line prefixes that integration tests grep for. Format: "<key> k=v k=v ...".
LOG_ENGINE_LOADED = "engine_loaded"  # device= compute_type= model= gpu= driver= cublas= cudnn= load_ms=
LOG_SEGMENT_LATENCY = "segment_latency"  # session= id= reason= audio_ms= infer_ms= latency_ms= chars= filtered=
LOG_CUDA_PROBE = "cuda_probe"  # ok= gpu= driver= cublas= cudnn= error=
#: The macOS Metal probe emits the same key with cublas=none cudnn=none, so log greps are shared.


class ComputeType(str, enum.Enum):
    """Quantization of the loaded model. The values are ctranslate2 compute types (Windows).

    On macOS whisper.cpp has no runtime compute type: the quantization is baked into the
    ggml model file, so the same two choices select a different file (see ModelSpec.ggml_file)
    and EngineInfo.compute_type reports the real ggml type ("f16" / "q5_0"). Keeping the enum
    unchanged keeps one config.json valid on both platforms.
    """

    FLOAT16 = "float16"  # macOS: ggml f16
    INT8_FLOAT16 = "int8_float16"  # macOS: ggml q5_0


#: Compute type the engine always uses in CPU mode. CPU mode ignores Config.compute_type.
CPU_COMPUTE_TYPE = "int8"


class Task(str, enum.Enum):
    TRANSLATE = "translate"  # Romanian speech -> English text
    TRANSCRIBE = "transcribe"  # Romanian speech -> Romanian text


class InjectionMode(str, enum.Enum):
    SENDINPUT = "sendinput"  # SendInput + KEYEVENTF_UNICODE (default)
    CLIPBOARD = "clipboard"  # save clipboard, set text, Ctrl+V, restore


class SegmentCloseReason(str, enum.Enum):
    SILENCE = "silence"  # trailing silence >= VadConfig.min_silence_ms
    MAX_LENGTH = "max_length"  # continuous speech reached VadConfig.max_segment_s
    FLUSH = "flush"  # dictation stopped; buffered speech emitted


class AppState(str, enum.Enum):
    """Global app state shown in the status line and tray.

    DISARMED       Hotkey not registered ("idle" in the status line). The model may still be resident.
    DOWNLOADING    Model files being downloaded (first run or model change).
    LOADING_MODEL  WhisperModel being created and warmed up.
    READY          Armed, model resident, not listening.
    LISTENING      Capturing mic audio; segments are translated and injected as they close.
    PROCESSING     Dictation stopped; flushing and translating the remaining audio.
    UNLOADED       Armed, but the model was unloaded after idle timeout; next hotkey reloads it.
    ERROR          Unrecoverable for now (CUDA missing, download failed, mic gone...). See ErrorRaised.
    """

    DISARMED = "disarmed"
    DOWNLOADING = "downloading"
    LOADING_MODEL = "loading_model"
    READY = "ready"
    LISTENING = "listening"
    PROCESSING = "processing"
    UNLOADED = "unloaded"
    ERROR = "error"


#: Allowed transitions. A transition to the current state is a no-op (no event, no error).
#: Anything else not listed raises InvalidTransition.
ALLOWED_TRANSITIONS: Mapping[AppState, frozenset[AppState]] = {
    AppState.DISARMED: frozenset(
        {AppState.DOWNLOADING, AppState.LOADING_MODEL, AppState.READY, AppState.UNLOADED, AppState.ERROR}
    ),
    AppState.DOWNLOADING: frozenset({AppState.LOADING_MODEL, AppState.DISARMED, AppState.ERROR}),
    AppState.LOADING_MODEL: frozenset(
        # LISTENING: hotkey pressed while (re)loading; capture starts, segments wait for the engine.
        # DOWNLOADING: user applied a not-yet-downloaded model while another load was in progress.
        {AppState.READY, AppState.LISTENING, AppState.DISARMED, AppState.DOWNLOADING, AppState.ERROR}
    ),
    AppState.READY: frozenset(
        {AppState.LISTENING, AppState.DISARMED, AppState.UNLOADED, AppState.LOADING_MODEL,
         AppState.DOWNLOADING, AppState.ERROR}
    ),
    AppState.LISTENING: frozenset({AppState.PROCESSING, AppState.DISARMED, AppState.ERROR}),
    AppState.PROCESSING: frozenset(
        # LISTENING: hotkey pressed again before the flush finished; a new session starts.
        {AppState.READY, AppState.LISTENING, AppState.DISARMED, AppState.ERROR}
    ),
    AppState.UNLOADED: frozenset({AppState.LOADING_MODEL, AppState.DISARMED, AppState.DOWNLOADING, AppState.ERROR}),
    AppState.ERROR: frozenset(
        {AppState.DISARMED, AppState.DOWNLOADING, AppState.LOADING_MODEL, AppState.READY}
    ),
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """A selectable Whisper model.

    vram_mb is the approximate peak VRAM for load + beam-5 inference on a <=15 s segment,
    including CUDA context overhead. The engine warns if total VRAM is below it. On macOS
    (unified memory) it is read as "approximate peak memory for the Metal backend".

    hf_repo / download_mb describe the CTranslate2 model the Windows engine loads.
    ggml_file / ggml_mb describe the single-file ggml model the macOS whisper.cpp engine
    loads from GGML_REPO, per compute type.
    """

    name: str
    hf_repo: str
    translate_capable: bool
    download_mb: int
    vram_mb: Mapping[ComputeType, int]
    label: str
    ggml_file: Mapping[ComputeType, str] = field(default_factory=dict)
    ggml_mb: Mapping[ComputeType, int] = field(default_factory=dict)


MODELS: Mapping[str, ModelSpec] = {
    "large-v3": ModelSpec(
        "large-v3", "Systran/faster-whisper-large-v3", True, 3100,
        {ComputeType.FLOAT16: 4700, ComputeType.INT8_FLOAT16: 3100}, "large-v3 (best quality)",
        {ComputeType.FLOAT16: "ggml-large-v3.bin", ComputeType.INT8_FLOAT16: "ggml-large-v3-q5_0.bin"},
        {ComputeType.FLOAT16: 3095, ComputeType.INT8_FLOAT16: 1080},
    ),
    "large-v2": ModelSpec(
        "large-v2", "Systran/faster-whisper-large-v2", True, 3100,
        {ComputeType.FLOAT16: 4700, ComputeType.INT8_FLOAT16: 3100}, "large-v2",
        {ComputeType.FLOAT16: "ggml-large-v2.bin", ComputeType.INT8_FLOAT16: "ggml-large-v2-q5_0.bin"},
        {ComputeType.FLOAT16: 3094, ComputeType.INT8_FLOAT16: 1080},
    ),
    "medium": ModelSpec(
        "medium", "Systran/faster-whisper-medium", True, 1530,
        {ComputeType.FLOAT16: 2600, ComputeType.INT8_FLOAT16: 1800}, "medium (faster, lower quality)",
        {ComputeType.FLOAT16: "ggml-medium.bin", ComputeType.INT8_FLOAT16: "ggml-medium-q5_0.bin"},
        {ComputeType.FLOAT16: 1533, ComputeType.INT8_FLOAT16: 539},
    ),
    # Not trained for translation; offered only for Task.TRANSCRIBE.
    "large-v3-turbo": ModelSpec(
        "large-v3-turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo", False, 1620,
        {ComputeType.FLOAT16: 2600, ComputeType.INT8_FLOAT16: 1800}, "large-v3-turbo (transcription only)",
        {ComputeType.FLOAT16: "ggml-large-v3-turbo.bin", ComputeType.INT8_FLOAT16: "ggml-large-v3-turbo-q5_0.bin"},
        {ComputeType.FLOAT16: 1624, ComputeType.INT8_FLOAT16: 574},
    ),
}
DEFAULT_MODEL = "large-v3"

#: Files a model directory must contain to count as downloaded. Mirrors the allow_patterns
#: in faster_whisper.utils.download_model; vocabulary.* is optional (some repos ship
#: vocabulary.json, others vocabulary.txt).
MODEL_REQUIRED_FILES = ("config.json", "model.bin", "tokenizer.json")
MODEL_ALLOW_PATTERNS = ("config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*")
#: Marker file the downloader writes into the model directory after all files are verified.
MODEL_COMPLETE_MARKER = ".download_complete"

#: Hugging Face repo holding the single-file ggml Whisper models whisper.cpp loads (macOS).
GGML_REPO = "ggerganov/whisper.cpp"


def ggml_model_file(model: str, compute_type: ComputeType) -> str:
    """File name inside GGML_REPO for this model/compute type. ValueError if unknown."""
    spec = MODELS.get(model)
    if spec is None:
        raise ValueError(f"Unknown model: {model!r}")
    name = spec.ggml_file.get(ComputeType(compute_type))
    if not name:
        raise ValueError(f"No ggml model file for {model!r} / {compute_type!r}")
    return name


# ---------------------------------------------------------------------------
# Paths (env overrides exist so tests and the desktop test can isolate state)
# ---------------------------------------------------------------------------

ENV_CONFIG_DIR = "ROENSPEAK_CONFIG_DIR"
ENV_DATA_DIR = "ROENSPEAK_DATA_DIR"
ENV_MODELS_DIR = "ROENSPEAK_MODELS_DIR"


def config_dir() -> Path:
    """$ROENSPEAK_CONFIG_DIR, else %APPDATA%\\RoEnSpeak (Windows),
    ~/Library/Application Support/RoEnSpeak (macOS), ~/.config/RoEnSpeak elsewhere."""
    if os.environ.get(ENV_CONFIG_DIR):
        return Path(os.environ[ENV_CONFIG_DIR])
    if IS_WINDOWS and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / APP_NAME
    if IS_MACOS:
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path.home() / ".config" / APP_NAME


def config_path() -> Path:
    return config_dir() / "config.json"


def data_dir() -> Path:
    """$ROENSPEAK_DATA_DIR, else %LOCALAPPDATA%\\RoEnSpeak (Windows),
    ~/Library/Application Support/RoEnSpeak (macOS), ~/.local/share/RoEnSpeak elsewhere."""
    if os.environ.get(ENV_DATA_DIR):
        return Path(os.environ[ENV_DATA_DIR])
    if IS_WINDOWS and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / APP_NAME
    if IS_MACOS:
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path.home() / ".local" / "share" / APP_NAME


def models_dir() -> Path:
    """Parent of one sub-directory per model name (models_dir()/large-v3/model.bin ...)."""
    if os.environ.get(ENV_MODELS_DIR):
        return Path(os.environ[ENV_MODELS_DIR])
    return data_dir() / "models"


def logs_dir() -> Path:
    """data_dir()/logs, except ~/Library/Logs/RoEnSpeak on macOS. $ROENSPEAK_DATA_DIR wins,
    so an isolated instance always keeps its logs next to its data."""
    if IS_MACOS and not os.environ.get(ENV_DATA_DIR):
        return Path.home() / "Library" / "Logs" / APP_NAME
    return data_dir() / "logs"


def single_instance_name() -> str:
    """Named-mutex name. When ROENSPEAK_CONFIG_DIR is set, it gets a suffix so an isolated
    test instance can run next to the user's real instance."""
    base = f"Local\\{APP_NAME}.SingleInstance" if IS_WINDOWS else f"{APP_NAME}.SingleInstance"
    override = os.environ.get(ENV_CONFIG_DIR)
    if override:
        return base + "." + hashlib.sha1(override.encode("utf-8")).hexdigest()[:12]
    return base


def single_instance_lock_path() -> Path:
    """macOS lock file (flock) standing in for the Windows named mutex. The sibling
    ".sock" path is the Unix socket the running instance listens on for "show window"."""
    name = single_instance_name().replace("\\", ".")
    return data_dir() / f"{name}.lock"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AppError(Exception):
    """Base for all app errors. ``user_message`` is safe to show in a dialog."""

    def __init__(self, user_message: str, *, details: str = "") -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.details = details


class CudaUnavailableError(AppError):
    """The GPU backend is unusable, so SpeechEngine.load failed.

    Windows: no usable NVIDIA GPU, driver too old, or cuBLAS/cuDNN failed to load.
    macOS:   the Metal backend or the whisper.cpp dynamic library could not be initialised.

    Raised by SpeechEngine.load when cpu_mode is False. The engine never falls back to
    CPU on its own.
    """


class GpuOutOfMemoryError(CudaUnavailableError):
    """CUDA ran out of memory loading or running the model (e.g. another app holds VRAM).
    The UI titles it "Not enough GPU memory" and suggests int8_float16 / a smaller model /
    closing GPU-heavy apps, NOT a driver install. Subclass of CudaUnavailableError so generic
    handling (no CPU fallback, model unloaded) still applies."""


class ModelNotDownloadedError(AppError):
    """The model directory is missing or incomplete. load() never downloads."""

    def __init__(self, model: str) -> None:
        super().__init__(f"Model '{model}' is not downloaded yet.")
        self.model = model


class EngineNotLoadedError(AppError):
    """process()/warmup() called while no model is loaded."""


class TaskNotSupportedError(AppError):
    """Task.TRANSLATE requested with a model whose translate_capable is False."""


class DownloadError(AppError):
    """Model download failed. ``offline`` is True when the failure was caused by no network
    / DNS / connection refused. Partially downloaded files are kept for resume."""

    def __init__(self, user_message: str, *, offline: bool = False, details: str = "") -> None:
        super().__init__(user_message, details=details)
        self.offline = offline


class DownloadCancelledError(AppError):
    """download() stopped because its cancel event was set. Partial files are kept."""


class HotkeyParseError(AppError):
    """Hotkey string is malformed, uses an unknown key, has no modifier, has more than one
    non-modifier key, or uses Shift as its only modifier on a printable key."""


class HotkeyConflictError(AppError):
    """RegisterHotKey (Windows) / RegisterEventHotKey (macOS) failed: another app or the OS
    owns the combo. ``win32_error`` carries GetLastError on Windows and the OSStatus on macOS."""

    def __init__(self, hotkey: str, *, win32_error: int = 0, details: str = "") -> None:
        api = "RegisterHotKey failed, GetLastError" if IS_WINDOWS else "RegisterEventHotKey failed, OSStatus"
        super().__init__(
            f"{hotkey} is already in use by another application or by {os_display_name()}. "
            "Choose a different combination.",
            details=details or f"{api}={win32_error}",
        )
        self.hotkey = hotkey
        self.win32_error = win32_error


class InjectionBlockedError(AppError):
    """The OS refused the synthetic keystrokes, so the text was NOT typed.

    ``text`` holds the text that was not injected, so the UI can show it to the user.
    ``reason`` is one of:

    Windows  "elevated" / "unknown_integrity"  the foreground window runs at a higher
             integrity level than us (or it could not be read), so UIPI would silently
             drop SendInput; "sendinput_partial"  SendInput inserted fewer events.
    macOS    "no_accessibility"  Accessibility permission is not granted, so CGEventPost
             is ignored; "secure_input"  the focused field uses secure event input (a
             password field), which blocks synthetic keystrokes; "cgevent_failed"  the
             event could not be created or posted.

    ``user_message`` is derived from ``reason``, so every platform explains the real cause.
    """

    MESSAGES = {
        "elevated": (
            "Can't type into an administrator window. Run the target app without admin rights, "
            "or start RoEnSpeak as administrator."
        ),
        "unknown_integrity": (
            "Can't type into an administrator window. Run the target app without admin rights, "
            "or start RoEnSpeak as administrator."
        ),
        "no_accessibility": (
            "RoEnSpeak needs Accessibility permission to type. Open System Settings > Privacy & "
            "Security > Accessibility and switch RoEnSpeak on, then try again."
        ),
        "secure_input": (
            "Can't type into a password field (macOS blocks synthetic keystrokes there). "
            "Click into a normal text field and dictate again."
        ),
    }
    DEFAULT_MESSAGE = "The text could not be typed into the focused window."

    def __init__(self, text: str, *, reason: str) -> None:
        super().__init__(self.MESSAGES.get(reason, self.DEFAULT_MESSAGE), details=f"reason={reason}")
        self.text = text
        self.reason = reason


class ModifiersHeldError(AppError):
    """Modifier keys were still physically held after the wait timeout; nothing was
    injected (typing with modifiers held would trigger shortcuts). ``text`` holds the dropped text."""

    def __init__(self, text: str) -> None:
        keys = "Ctrl/Shift/Alt/Win" if IS_WINDOWS else "Control/Shift/Option/Command"
        super().__init__(f"Release {keys} so the dictated text can be typed.")
        self.text = text


class AudioDeviceError(AppError):
    """Microphone missing, busy, or unplugged during capture."""


class ConfigError(AppError):
    """Config could not be saved, or update_config got a semantically invalid Config."""


class InvalidTransition(AppError):
    def __init__(self, old: AppState, new: AppState) -> None:
        super().__init__(f"Invalid state transition {old.value} -> {new.value}")
        self.old = old
        self.new = new


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_SCHEMA_VERSION = 1

DEFAULT_BLOCKLIST: tuple[str, ...] = (
    "thank you for watching",
    "thanks for watching",
    "thank you for watching and see you next time",
    "please subscribe",
    "please like and subscribe",
    "subscribe to the channel",
    "subtitles by*",
    "subtitled by*",
    "subtitrare*",
    "translated by*",
    "transcribed by*",
    "amara.org*",
    "www.*",
)


@dataclass(frozen=True)
class VadConfig:
    """Streaming VAD segmentation parameters (see Segmenter for exact semantics)."""

    threshold: float = 0.5  # speech starts when prob >= threshold. Valid range 0.05..0.95
    neg_threshold_offset: float = 0.15  # in speech, frame is silence if prob < threshold - offset. 0..0.5
    min_silence_ms: int = 500  # trailing silence that closes a segment. 100..5000
    max_segment_s: float = 15.0  # forced cut of continuous speech. 2.0..30.0
    speech_pad_ms: int = 200  # audio kept before speech start / after speech end. 0..1000
    min_speech_ms: int = 250  # segments with less speech are discarded. 0..2000


@dataclass(frozen=True)
class DecodeConfig:
    """Only beam_size is user-configurable. Fixed params passed to WhisperModel.transcribe:
    language="ro", task=<Config.task>, temperature=0.0 (no fallback),
    condition_on_previous_text=False, without_timestamps=True, vad_filter=False,
    no_speech_threshold=None (our HallucinationFilter decides instead)."""

    beam_size: int = 5  # 1..10; 1 = fastest


@dataclass(frozen=True)
class FilterConfig:
    """Hallucination filter. A whisper segment is dropped when any rule matches:

    "empty"       normalized text has no letter or digit
    "no_speech"   no_speech_prob > no_speech_prob_threshold AND avg_logprob < avg_logprob_threshold
    "compression" compression_ratio > compression_ratio_threshold
    "blocklist"   normalized text == entry, or entry ends with "*" and the normalized text
                  starts with entry[:-1] (entries are normalized the same way before comparing)

    Normalization: casefold, strip, replace every char that is not a letter, digit, space, '.'
    or '*' with a space, collapse runs of whitespace, strip, then strip '.' from both ends.
    Example: "  Thank you for watching!! " -> "thank you for watching"
    """

    no_speech_prob_threshold: float = 0.6  # 0..1
    avg_logprob_threshold: float = -1.0  # -10..0
    compression_ratio_threshold: float = 2.4  # 1..10
    blocklist: tuple[str, ...] = DEFAULT_BLOCKLIST


@dataclass(frozen=True)
class UiConfig:
    overlay: bool = True  # "● REC" click-through overlay while LISTENING
    sounds: bool = True  # start/stop beeps
    start_with_windows: bool = False  # HKCU Run key
    first_run_done: bool = False  # set after the first-run download/GPU check finished


@dataclass(frozen=True)
class Config:
    """Persisted to config_path() as JSON. Enums serialize as their .value, tuples as lists,
    nested dataclasses as objects. Key "schema_version" is always written."""

    armed: bool = True
    hotkey: str = DEFAULT_HOTKEY  # canonical display string, see HotkeyParser
    mic_device: Optional[str] = None  # device name as reported by AudioSource.list_input_devices(); None = system default
    model: str = DEFAULT_MODEL  # key of MODELS
    compute_type: ComputeType = ComputeType.FLOAT16
    task: Task = Task.TRANSLATE
    injection_mode: InjectionMode = InjectionMode.SENDINPUT
    cpu_mode: bool = False  # opt-in "CPU mode (slow)"; never enabled automatically
    idle_unload_minutes: int = 0  # 0 = never unload. 0..1440
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    schema_version: int = CONFIG_SCHEMA_VERSION


# Legacy schema 0 (pre-release format), which ConfigStore must migrate. It is a flat object
# with no "schema_version" key:
#   {"hotkey": str, "model": str, "compute_type": str, "beam_size": int,
#    "translate": bool,            -> task = translate if true else transcribe
#    "clipboard_mode": bool,       -> injection_mode = clipboard if true else sendinput
#    "vad_silence_ms": int,        -> vad.min_silence_ms
#    "vad_max_s": float,           -> vad.max_segment_s
#    "overlay": bool, "sounds": bool, "autostart": bool -> ui.overlay / ui.sounds / ui.start_with_windows
#    "armed": bool, "mic": str|null -> armed / mic_device}
# Any of these keys may be missing (-> default). A file whose "schema_version" is larger than
# CONFIG_SCHEMA_VERSION is treated as invalid (backed up, defaults used).


@runtime_checkable
class ConfigStore(Protocol):
    """Loads and saves Config. Any thread, but core serializes calls itself.

    load():
      * Missing file -> Config() defaults, no warnings, nothing written.
      * Unreadable / invalid JSON / not an object / schema too new -> the file is renamed to
        "config.json.bad-<YYYYmmdd-HHMMSS>", defaults are returned, one warning added.
      * Old schema -> migrated in memory (and saved on the next save()).
      * Each invalid field (wrong type, out of range, unknown enum value, unknown model,
        hotkey that fails parse_hotkey, transcription-only model with task=translate)
        -> that field falls back to its default, one warning per field. Unknown keys are ignored.
      * Never raises because of file content.
    save(cfg): writes atomically (temp file in the same dir + os.replace), creating the dir.
      Raises ConfigError on I/O failure.
    """

    path: Path

    def load(self) -> Config: ...

    def save(self, cfg: Config) -> None: ...

    @property
    def last_warnings(self) -> Sequence[str]:
        """Warnings produced by the most recent load()."""
        ...


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class AudioSegment:
    """One closed speech segment, ready for inference.

    id is unique for the process lifetime and increases strictly (starts at 1).
    session_id identifies the dictation session (one hotkey start..stop).
    samples: float32 mono 16 kHz, including speech_pad_ms of padding where available.
    speech_start_s / speech_end_s: stream-relative seconds (since the segmenter's last reset)
    of the first and last speech frames, without padding.
    t_vad_close: time.perf_counter() when the segment was closed (start of the latency budget).
    """

    id: int
    session_id: int
    samples: np.ndarray
    reason: SegmentCloseReason
    speech_start_s: float
    speech_end_s: float
    t_vad_close: float

    @property
    def duration_s(self) -> float:
        return len(self.samples) / SAMPLE_RATE


@runtime_checkable
class AudioSource(Protocol):
    """Pushes 16 kHz mono float32 audio to a callback.

    start(on_audio, on_end=None):
      Begins delivery. on_audio(chunk) is called from the source's own thread (the PortAudio
      callback thread for the mic). It must not block; core just enqueues the chunk.
      Chunk sizes are arbitrary (usually ~32 ms). on_end() is called once when a finite
      source (wav) runs out; the mic never calls it.
      Raises AudioDeviceError if the device can't be opened. Calling start twice without
      stop raises RuntimeError.
    stop():
      Blocks until no further on_audio call will happen. Idempotent. on_end is not called
      by stop().
    Runtime device loss -> on_error(AudioDeviceError), called from the source thread, then
    delivery stops.
    """

    def start(
        self,
        on_audio: Callable[[np.ndarray], None],
        on_end: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[AppError], None]] = None,
    ) -> None: ...

    def stop(self) -> None: ...


@runtime_checkable
class VadModel(Protocol):
    """Frame-level speech probability with state carried across frames (streaming).

    speech_prob(frame): frame is exactly VAD_FRAME_SAMPLES float32 samples. Returns the
    probability in [0, 1]. Not thread-safe; owned by the segmenter thread.
    reset(): clears recurrent state and context (start of a new stream).
    """

    def speech_prob(self, frame: np.ndarray) -> float: ...

    def reset(self) -> None: ...


@runtime_checkable
class Segmenter(Protocol):
    """Streaming VAD segmenter. Pure logic, no threads. Owned by the segmenter thread.

    Exact semantics (F = VAD_FRAME_SAMPLES = 512 samples = 32 ms):
      * feed() appends samples to an internal buffer and runs the VadModel on every complete
        F-sample frame. A partial frame stays buffered until more samples arrive.
      * Idle state: a frame with prob >= threshold starts speech at that frame's first sample.
      * Speech state: a frame with prob < threshold - neg_threshold_offset is silence, any other
        frame is speech (it resets the silence counter and extends speech end to the frame's end).
      * Close on silence: when the consecutive silence reaches min_silence_ms (counted in whole
        frames: silence_frames * 32 ms >= min_silence_ms), the segment closes with reason SILENCE.
        Its samples span [speech_start - speech_pad, speech_end + speech_pad], clipped to the
        audio actually received (pre-roll is kept, at least speech_pad_ms of history).
      * Discard: a closing segment whose speech duration (speech_end - speech_start) is below
        min_speech_ms is dropped silently and consumes no id.
      * Max length: when (current position - segment start) reaches max_segment_s while in speech,
        the segment closes with reason MAX_LENGTH and samples [speech_start - pad, current position].
        A new segment starts at current position (no overlap and no pre-pad, so text is never
        duplicated). The state stays "speech".
      * flush(): if in speech, emits the current segment with reason FLUSH, samples
        [speech_start - pad, end of all received audio, including any partial frame], subject to
        the min_speech_ms discard rule. Then it behaves like reset() (ids keep increasing).
        In idle state, flush() returns [] and resets.
      * reset(): drops all buffered audio and state and calls VadModel.reset(). Does not reset ids.
      * Every emitted AudioSegment gets session_id = the value passed to begin_session() most
        recently (0 before any call) and t_vad_close = clock() at the moment it was closed.
    """

    def begin_session(self, session_id: int) -> None: ...

    def feed(self, samples: np.ndarray) -> list[AudioSegment]: ...

    def flush(self) -> list[AudioSegment]: ...

    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuInfo:
    name: str
    total_vram_mb: int
    free_vram_mb: int
    driver_version: str  # e.g. "610.88"


@dataclass(frozen=True)
class CudaReport:
    """Result of CudaProbe.probe(). ok is True only when the driver is new enough, a CUDA
    device exists, and cuBLAS + cuDNN DLLs loaded."""

    ok: bool
    gpu: Optional[GpuInfo]
    cublas_version: Optional[str]  # "12.9.2"
    cudnn_version: Optional[str]  # "9.27.0"
    error: Optional[str]  # human-readable reason when not ok


@runtime_checkable
class CudaProbe(Protocol):
    """setup_dll_paths(): must run before ctranslate2 is imported anywhere. Adds the bundled
    nvidia/*/bin directories (frozen: sys._MEIPASS/nvidia/*/bin; dev: site-packages/nvidia/*/bin)
    via os.add_dll_directory AND prepends them to PATH. The cuDNN dispatcher needs PATH to find
    its sub-libraries. Idempotent. No-op off Windows.
    probe(): loads the DLLs, reads versions, queries NVML. Never raises; failures go in the
    report. Logs one LOG_CUDA_PROBE line. Any thread; result may be cached.
    """

    def setup_dll_paths(self) -> list[str]: ...

    def probe(self) -> CudaReport: ...


@dataclass(frozen=True)
class EngineInfo:
    model: str
    device: str  # "cuda" or "cpu". The device actually used
    compute_type: str  # actual ctranslate2 compute type, e.g. "float16", "int8_float16", "int8"
    gpu_name: Optional[str]
    driver_version: Optional[str]
    cublas_version: Optional[str]
    cudnn_version: Optional[str]
    load_ms: float


@dataclass(frozen=True)
class SegmentStats:
    """One faster_whisper Segment as seen by the filter."""

    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float


@dataclass(frozen=True)
class TranslationResult:
    """Output of SpeechEngine.process for one AudioSegment.

    text: kept whisper segments' texts, each stripped, joined with single spaces; "" if filtered.
    filtered: True when every whisper segment was dropped (or whisper returned none).
    filter_reason: the reason of the first dropped segment ("empty", "no_speech",
      "compression", "blocklist"); None when not filtered.
    """

    segment_id: int
    session_id: int
    text: str
    filtered: bool
    filter_reason: Optional[str]
    task: Task
    audio_s: float
    t_vad_close: float
    t_infer_start: float
    t_infer_end: float

    @property
    def infer_ms(self) -> float:
        return (self.t_infer_end - self.t_infer_start) * 1000.0


@runtime_checkable
class HallucinationFilter(Protocol):
    """evaluate(stats) -> None to keep, or a reason string to drop (see FilterConfig).
    Pure and thread-safe. set_config replaces thresholds/blocklist atomically."""

    def evaluate(self, stats: SegmentStats) -> Optional[str]: ...

    def set_config(self, cfg: FilterConfig) -> None: ...


@runtime_checkable
class SpeechEngine(Protocol):
    """Owns the WhisperModel. Every method except is_loaded/info must be called from the
    single inference thread.

    load(model, compute_type, cpu_mode):
      Blocking. Unloads any previous model first. Loads the model from models_dir()/<model>
      without touching the network. Logs LOG_ENGINE_LOADED. Raises CudaUnavailableError
      (cpu_mode False and the GPU backend is unusable), ModelNotDownloadedError, ValueError
      for an unknown model or compute type. EngineInfo always states the device and compute
      type actually in use, never the requested ones.
      Windows: device="cuda" + compute_type, or, only if cpu_mode, device="cpu" +
      CPU_COMPUTE_TYPE, because ctranslate2 picks its kernels at load time.
      macOS: device="metal", or "cpu" if cpu_mode, and compute_type is the ggml quantization
      of the file on disk ("f16" / "q5_0"). whisper.cpp has no runtime compute type, so
      cpu_mode loads the same file and reports the same quantization; CPU_COMPUTE_TYPE does
      not apply there.
    warmup(): one transcribe of 1 s of low-level noise with the configured params; result discarded.
    process(segment, task, decode, filt):
      Raises EngineNotLoadedError or TaskNotSupportedError. Fully consumes the segments generator.
    unload(): drops the model and frees VRAM (del + gc.collect). Idempotent.
    """

    def load(self, model: str, compute_type: ComputeType, cpu_mode: bool) -> EngineInfo: ...

    def warmup(self, decode: DecodeConfig) -> None: ...

    def process(
        self, segment: AudioSegment, task: Task, decode: DecodeConfig, filt: HallucinationFilter
    ) -> TranslationResult: ...

    def unload(self) -> None: ...

    @property
    def is_loaded(self) -> bool: ...

    @property
    def info(self) -> Optional[EngineInfo]: ...


@dataclass(frozen=True)
class DownloadProgress:
    model: str
    downloaded_bytes: int
    total_bytes: int  # 0 if unknown yet
    current_file: str
    bytes_per_s: float


@runtime_checkable
class ModelDownloader(Protocol):
    """Downloads MODELS[model].hf_repo files matching MODEL_ALLOW_PATTERNS into
    models_dir()/<model> (plain files, no HF cache symlinks).

    is_downloaded(model): every MODEL_REQUIRED_FILES file and MODEL_COMPLETE_MARKER exist. Any thread.
    download(model, on_progress, cancel): blocking (core runs it on a worker thread).
      on_progress is called from that thread at most ~10x/s. Resumes partial files.
      Writes MODEL_COMPLETE_MARKER last. Returns the model directory.
      Raises DownloadError (offline=True for connectivity problems), DownloadCancelledError.
    """

    def is_downloaded(self, model: str) -> bool: ...

    def model_path(self, model: str) -> Path: ...

    def download(
        self,
        model: str,
        on_progress: Callable[[DownloadProgress], None],
        cancel: "ThreadingEvent",
    ) -> Path: ...


class ThreadingEvent(Protocol):
    def is_set(self) -> bool: ...

    def set(self) -> None: ...


# ---------------------------------------------------------------------------
# Win32 boundary
# ---------------------------------------------------------------------------


class Modifier(str, enum.Enum):
    """Hotkey modifiers. CTRL/ALT/SHIFT exist on both platforms; WIN and CMD are the
    platform-specific "super" key.

    Both parsers accept both platforms' names for those two keys and canonicalise them to
    the host's: Win|Windows|Meta|Super|Cmd|Command is WIN on Windows and CMD on macOS, and
    Alt|Option|Opt is always ALT. Neither parser ever produces the other platform's member.
    So a hotkey written on either platform loads on the other ("Cmd+F12" reads as "Win+F12"
    on Windows, "Win+F12" as "Cmd+F12" on macOS), and the shared default "Alt+Q" means
    Alt+Q on Windows and Option+Q on macOS. One config.json works on both platforms.
    """

    CTRL = "Ctrl"
    ALT = "Alt"  # macOS: Option
    SHIFT = "Shift"
    WIN = "Win"  # Windows only
    CMD = "Cmd"  # macOS only


#: Canonical modifier order in display strings.
MODIFIER_ORDER = (Modifier.CTRL, Modifier.ALT, Modifier.SHIFT, Modifier.WIN, Modifier.CMD)
#: RegisterHotKey fsModifiers bits.
MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x8, 0x4000


@dataclass(frozen=True)
class Hotkey:
    """Parsed hotkey. display is canonical: modifiers in MODIFIER_ORDER, then the key's
    canonical name, joined by "+", e.g. "Ctrl+Alt+F9", "Alt+Q".

    Key names (case-insensitive on input, canonical output in brackets):
      letters A-Z [upper], digits 0-9, F1-F24 [F<n>], Space, Enter (alias Return), Tab,
      Esc (alias Escape), Backspace, Insert (Ins), Delete (Del), Home, End, PageUp (PgUp),
      PageDown (PgDn), Up, Down, Left, Right, Pause, Numpad0-Numpad9,
      OEM punctuation by character: ; = , - . / ` [ \\ ] '
    Not every key name exists on both platforms: Mac keyboards have no Insert, Pause or
    F21-F24, so the macOS parser rejects those with a message naming the key. A Windows
    config.json using one of them therefore falls back to DEFAULT_HOTKEY on a Mac, the same
    way any other unusable hotkey does.
    Modifier aliases, accepted on both platforms: Ctrl (Control, Ctl), Shift,
    Alt (Option, Opt), and the super key as Win (Windows, Meta, Super, Cmd, Command) --
    canonicalised to Win on Windows and Cmd on macOS. The macOS key table maps the same
    canonical key names onto Carbon kVK_* codes.
    Separators: "+" with optional surrounding whitespace. "Ctrl++" is not supported;
    use "=" or "-" style keys only.
    """

    modifiers: frozenset[Modifier]
    key: str  # canonical key name
    vk: int  # Windows virtual-key code; on macOS the Carbon virtual key code (kVK_*)

    @property
    def display(self) -> str:
        mods = [m.value for m in MODIFIER_ORDER if m in self.modifiers]
        return "+".join(mods + [self.key])

    @property
    def win32_modifiers(self) -> int:
        """fsModifiers for RegisterHotKey, always including MOD_NOREPEAT."""
        bits = {
            Modifier.ALT: MOD_ALT,
            Modifier.CTRL: MOD_CONTROL,
            Modifier.SHIFT: MOD_SHIFT,
            Modifier.WIN: MOD_WIN,
            Modifier.CMD: MOD_WIN,  # never produced by the Windows parser; kept total
        }
        value = MOD_NOREPEAT
        for m in self.modifiers:
            value |= bits[m]
        return value


@runtime_checkable
class HotkeyParser(Protocol):
    """parse(text) -> Hotkey or raises HotkeyParseError. Rejects: empty string, unknown
    token, duplicate modifier, modifier-only combos, more than one non-modifier key, no
    modifier at all, and Shift as the sole modifier with a printable key (letters, digits,
    OEM punctuation, Space). parse("alt+q") == parse(" Alt + Q ") with display "Alt+Q".
    format(h) == h.display. Pure; any thread."""

    def parse(self, text: str) -> Hotkey: ...

    def format(self, hotkey: Hotkey) -> str: ...


@dataclass(frozen=True)
class KeyEvent:
    """One INPUT record for SendInput.

    unicode_char set -> KEYEVENTF_UNICODE with wScan = that UTF-16 code unit (vk = 0).
    Otherwise a virtual-key event for vk. key_up adds KEYEVENTF_KEYUP.
    """

    vk: int = 0
    unicode_char: Optional[str] = None  # exactly one UTF-16 code unit (surrogates sent separately)
    key_up: bool = False


#: Integrity levels (SECURITY_MANDATORY_*_RID).
IL_UNTRUSTED, IL_LOW, IL_MEDIUM, IL_HIGH, IL_SYSTEM = 0x0, 0x1000, 0x2000, 0x3000, 0x4000


@dataclass(frozen=True)
class WinMessage:
    message: int
    wparam: int
    lparam: int


class WinApi(Protocol):
    """Thin, mockable wrapper over the Win32 calls the app uses. The real implementation
    (app.win32._winapi.RealWinApi) uses ctypes only. Each method maps 1:1 to the named Win32
    call and adds no policy. Thread affinity follows Win32 rules: hotkeys and GetMessage are
    per-thread.
    """

    # threads / message loop
    def get_current_thread_id(self) -> int: ...
    def get_message(self) -> Optional[WinMessage]:
        """GetMessageW(NULL hwnd) on the calling thread; None when WM_QUIT is received."""
        ...
    def post_thread_message(self, thread_id: int, message: int, wparam: int = 0, lparam: int = 0) -> bool: ...
    def post_quit_message(self) -> None: ...

    # hotkeys
    def register_hotkey(self, hotkey_id: int, modifiers: int, vk: int) -> bool:
        """RegisterHotKey(NULL, ...) for the calling thread."""
        ...
    def unregister_hotkey(self, hotkey_id: int) -> bool: ...
    def get_last_error(self) -> int: ...

    # keyboard
    def get_async_key_state(self, vk: int) -> int:
        """Raw SHORT. Bit 0x8000 set = key is physically down now."""
        ...
    def send_input(self, events: Sequence[KeyEvent]) -> int:
        """One SendInput call with all events. Returns the number of events inserted."""
        ...

    # windows / processes
    def get_foreground_window(self) -> int: ...
    def get_window_process_id(self, hwnd: int) -> int: ...
    def get_process_integrity_level(self, pid: int) -> Optional[int]:
        """Mandatory label RID of the process token; None if it can't be opened/queried."""
        ...
    def get_current_process_integrity_level(self) -> int: ...

    # clipboard
    def open_clipboard(self) -> bool: ...
    def close_clipboard(self) -> None: ...
    def empty_clipboard(self) -> bool: ...
    def enum_clipboard_formats(self) -> list[int]: ...
    def get_clipboard_bytes(self, fmt: int) -> Optional[bytes]:
        """GlobalLock'd contents of an HGLOBAL-backed format; None for non-HGLOBAL formats (bitmaps etc.)."""
        ...
    def set_clipboard_bytes(self, fmt: int, data: bytes) -> bool: ...

    # single instance / misc
    def create_mutex(self, name: str) -> tuple[int, bool]:
        """CreateMutexW -> (handle, already_existed)."""
        ...
    def close_handle(self, handle: int) -> None: ...
    def register_window_message(self, name: str) -> int: ...
    def allow_set_foreground_window(self, pid: int) -> bool:
        """AllowSetForegroundWindow(pid); ASFW_ANY = 0xFFFFFFFF. Called by a second instance
        before broadcasting SHOW_WINDOW_MESSAGE_NAME so the running instance may take focus."""
        ...
    def register_clipboard_format(self, name: str) -> int:
        """RegisterClipboardFormatW(name)."""
        ...
    def broadcast_message(self, message: int) -> None:
        """PostMessageW(HWND_BROADCAST, message, 0, 0)."""
        ...

    # time (injectable for tests)
    def monotonic(self) -> float:
        """High-resolution monotonic seconds: RealWinApi returns time.perf_counter()."""
        ...
    def sleep(self, seconds: float) -> None: ...


@runtime_checkable
class HotkeyManager(Protocol):
    """Owns a dedicated thread running a GetMessage loop. RegisterHotKey/UnregisterHotKey are
    called only on that thread; other threads post WM_APP+n requests and wait for the reply.

    start(): spawns the thread, returns once its message queue exists. Idempotent.
    register(hotkey): replaces the current registration. Blocks (<=2 s) until the hotkey
      thread has tried. On failure raises HotkeyConflictError and the PREVIOUS hotkey stays
      registered (if any). Raises RuntimeError if not started.
    unregister(): removes the registration if any. Idempotent. Blocks until done.
    stop(): unregisters, posts WM_QUIT, joins the thread. Idempotent. Safe from atexit.
    current: the registered Hotkey or None. Any thread.
    on_fire (constructor arg) is invoked on the hotkey thread for each accepted WM_HOTKEY.
      It must not block. Fires within debounce_s of the previous accepted fire are dropped.
      Before calling on_fire, if Alt is part of the hotkey or physically down, the manager
      sends VK_MENU_MASK down+up so the focused app's menu bar is not activated on Alt release.
    """

    def start(self) -> None: ...
    def register(self, hotkey: Hotkey) -> None: ...
    def unregister(self) -> None: ...
    def stop(self) -> None: ...

    @property
    def current(self) -> Optional[Hotkey]: ...


@dataclass(frozen=True)
class ClipboardSnapshot:
    """Saved clipboard contents. Format ids are ints on Windows (CF_*) and UTI strings on
    macOS (NSPasteboard types); only the platform clipboard backend interprets them."""

    formats: tuple[tuple[Union[int, str], bytes], ...]  # (format id, raw bytes) in enumeration order


@runtime_checkable
class ClipboardBackend(Protocol):
    """Called only from the injection thread. open_clipboard is retried (10 x 20 ms) because
    other apps briefly hold it. Raises AppError if it can't be opened.
    save(): snapshot every format that get_clipboard_bytes can read (others are skipped).
    set_text(text): empty + set CF_UNICODETEXT (13) as UTF-16-LE with NUL terminator, plus the
      registered formats "ExcludeClipboardContentFromMonitorProcessing" and
      "CanIncludeInClipboardHistory" (DWORD 0) so dictated text stays out of Win+V history/cloud sync.
    restore(snap): empty + set every saved format back (an empty snapshot leaves the clipboard empty).
    The retry, the CF_UNICODETEXT encoding and the two history-exclusion formats are Win32
    specifics. On macOS NSPasteboard needs none of them: format ids are UTI strings, the text
    type is public.utf8-plain-text, and there is no OS clipboard history to opt out of, so
    MacClipboard implements the same three methods with the same save/set/restore semantics
    and skips the rest."""

    def save(self) -> ClipboardSnapshot: ...
    def set_text(self, text: str) -> None: ...
    def restore(self, snapshot: ClipboardSnapshot) -> None: ...


@dataclass(frozen=True)
class InjectionOutcome:
    injected_text: str  # exactly what was typed/pasted, including any leading space
    mode: InjectionMode
    #: How many OS input events the injector sent. The unit is platform-specific and is only
    #: meant for logging: Windows sends a down/up pair per UTF-16 code unit, while macOS packs
    #: up to 20 code units into each CGEvent (surrogate pairs never split across two), so the
    #: same text yields far fewer events there. Clipboard mode counts the paste chord.
    events_sent: int
    #: perf_counter, when SendInput / the paste chord returned. In clipboard mode this is
    #: sampled before the paste-settle wait and the clipboard restore, so it measures when the
    #: keystrokes reached the OS, not when the clipboard was put back.
    t_injected: float


#: Modifier VKs the injector waits on: L/R Ctrl, L/R Shift, L/R Alt, L/R Win.
MODIFIER_VKS = (0xA2, 0xA3, 0xA0, 0xA1, 0xA4, 0xA5, 0x5B, 0x5C)
#: Text starting with one of these gets no leading space between segments.
NO_SPACE_BEFORE = ".,;:!?)]}%'\"…»"


@runtime_checkable
class TextInjector(Protocol):
    """Types text into the focused window. Called only from the injection thread.

    inject(text, session_id):
      1. text = text.strip(). Internal "\\r\\n"/"\\n"/"\\t" become single spaces. Empty -> returns
         None and changes nothing.
      2. Spacing: prefix " " iff a previous inject() in the SAME session_id injected text that
         did not end in whitespace, and text does not start with a NO_SPACE_BEFORE char. A new
         session_id resets this (the first segment of a session never gets a leading space).
      3. Wait until no MODIFIER_VKS key is down (poll every poll_s). After modifier_timeout_s
         -> raise ModifiersHeldError(text) without typing.
      4. UIPI: foreground window's process integrity level is None or greater than ours ->
         raise InjectionBlockedError(text, reason="elevated" | "unknown_integrity").
      5. SENDINPUT mode: one send_input call with down+up KeyEvent pairs per UTF-16 code unit.
         If it returns fewer events than sent -> InjectionBlockedError(reason="sendinput_partial").
         CLIPBOARD mode: snap = save(); set_text(prefixed text); send Ctrl down, V down, V up,
         Ctrl up; sleep(paste_settle_s); restore(snap) (restore runs even if sending failed).
      Spacing state is only updated after a successful injection.
    set_mode(mode): takes effect on the next inject(). Any thread.
    """

    def inject(self, text: str, session_id: int) -> Optional[InjectionOutcome]: ...
    def set_mode(self, mode: InjectionMode) -> None: ...


@runtime_checkable
class SingleInstance(Protocol):
    """acquire(): True if we are the first instance (mutex single_instance_name() created
    fresh); False if another instance holds it, in which case the caller calls
    notify_existing() and exits. release() closes the handle; idempotent."""

    def acquire(self) -> bool: ...
    def notify_existing(self) -> None: ...
    def release(self) -> None: ...


@runtime_checkable
class Autostart(Protocol):
    """HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run value named APP_NAME whose data
    is the quoted path to the running exe (frozen) or pythonw + -m app.main (dev)."""

    def is_enabled(self) -> bool: ...
    def set_enabled(self, enabled: bool) -> None: ...


# ---------------------------------------------------------------------------
# Events (core -> UI). Published on any thread; UI marshals to the Qt thread.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateChanged:
    old: AppState
    new: AppState
    detail: str = ""  # e.g. model name while loading, error summary in ERROR


@dataclass(frozen=True)
class EngineLoaded:
    info: EngineInfo


@dataclass(frozen=True)
class SegmentProcessed:
    """Published after a segment's result was injected (or filtered / failed to inject)."""

    session_id: int
    segment_id: int
    text: str  # injected text ("" when filtered)
    filtered: bool
    filter_reason: Optional[str]
    infer_ms: float
    latency_ms: Optional[float]  # t_injected - t_vad_close; None when nothing was injected


@dataclass(frozen=True)
class DownloadProgressEvent:
    progress: DownloadProgress


@dataclass(frozen=True)
class DownloadFinished:
    model: str
    ok: bool
    error: Optional[str] = None
    offline: bool = False


@dataclass(frozen=True)
class HotkeyChanged:
    hotkey: Optional[str]  # display string or None when unregistered
    registered: bool


@dataclass(frozen=True)
class ConfigChanged:
    config: Config


@dataclass(frozen=True)
class ErrorRaised:
    """Something the user must see. fatal=True means state is ERROR. kind is the AppError
    subclass name (e.g. "CudaUnavailableError")."""

    kind: str
    message: str
    details: str = ""
    fatal: bool = False


@dataclass(frozen=True)
class Notification:
    """Non-blocking tray balloon (e.g. UIPI-blocked injection, VRAM warning)."""

    title: str
    message: str
    warning: bool = False


AppEvent = Union[
    StateChanged, EngineLoaded, SegmentProcessed, DownloadProgressEvent, DownloadFinished,
    HotkeyChanged, ConfigChanged, ErrorRaised, Notification,
]


@runtime_checkable
class EventBus(Protocol):
    """publish(): any thread; calls every subscriber synchronously on the publishing thread,
    in subscription order. A subscriber exception is logged and does not stop delivery to
    the others. subscribe() returns an unsubscribe callable. Any thread."""

    def publish(self, event: AppEvent) -> None: ...
    def subscribe(self, callback: Callable[[AppEvent], None]) -> Callable[[], None]: ...


@runtime_checkable
class StateMachine(Protocol):
    """Thread-safe holder of AppState enforcing ALLOWED_TRANSITIONS.

    transition(new, detail=""): same state -> no-op, returns False. Allowed -> sets the state,
    publishes StateChanged(old, new, detail) on the bus (after releasing its lock), returns True.
    Otherwise raises InvalidTransition and the state is unchanged.
    """

    @property
    def state(self) -> AppState: ...
    def can_transition(self, new: AppState) -> bool: ...
    def transition(self, new: AppState, detail: str = "") -> bool: ...


# ---------------------------------------------------------------------------
# Controller (UI -> core) and UiFacade (core -> UI)
# ---------------------------------------------------------------------------


@runtime_checkable
class Controller(Protocol):
    """Implemented by core. All methods are called from the Qt main thread and return within
    ~50 ms. Long work runs on core threads, and results arrive as events.

    set_armed(on): on -> register hotkey (+ download/load model if needed); off -> stop any
      dictation (pending audio discarded), unregister hotkey. Persists Config.armed.
      Hotkey conflict while arming (user action or startup) -> ErrorRaised event and the app runs
      disarmed for now, but Config.armed keeps the user's intent (True) so the next launch retries.
      Only an explicit user set_armed(False) persists armed=False.
    toggle_dictation(): same as a hotkey press (start if READY/UNLOADED/LOADING_MODEL, stop if LISTENING).
    set_hotkey(text): parse + register immediately + persist. Raises HotkeyParseError /
      HotkeyConflictError (previous hotkey stays). If disarmed, only validates the combo and saves it.
    reset_hotkey(): set_hotkey(DEFAULT_HOTKEY).
    update_config(cfg): validates, persists, and applies the diff (mic, injection mode, VAD,
      filter, autostart, overlay/sounds apply live; model/compute_type/cpu_mode trigger a
      reload or download). Raises ConfigError for invalid values.
    start_model_download() / cancel_model_download(): for Config.model.
    quit(): stops dictation, unregisters hotkey, unloads model, stops all threads, then calls
      UiFacade.exit_app(). Idempotent.
    """

    @property
    def config(self) -> Config: ...
    @property
    def state(self) -> AppState: ...
    @property
    def engine_info(self) -> Optional[EngineInfo]: ...
    @property
    def last_latency_ms(self) -> Optional[float]: ...

    def set_armed(self, on: bool) -> None: ...
    def toggle_dictation(self) -> None: ...
    def set_hotkey(self, text: str) -> Hotkey: ...
    def suspend_hotkey(self, suspended: bool) -> None:
        """UI calls True while the hotkey capture field is recording (so pressing the currently
        registered combo is captured instead of toggling dictation) and False when capture ends.
        Temporarily unregisters/re-registers without touching Config.armed. No-op when disarmed."""
        ...
    def reset_hotkey(self) -> Hotkey: ...
    def update_config(self, cfg: Config) -> None: ...
    def list_input_devices(self) -> list[str]: ...
    def start_model_download(self) -> None: ...
    def cancel_model_download(self) -> None: ...
    def open_log_folder(self) -> None: ...
    def quit(self) -> None: ...


@runtime_checkable
class UiFacade(Protocol):
    """Implemented by the UI. Every method is callable from ANY thread; the implementation
    marshals to the Qt thread and returns immediately (no return values, no modal waits).

    show_error: modal-less error dialog (CUDA requirement, download offline, ...).
    show_download_dialog(model, size_mb): first-run / model-change dialog. It calls
      Controller.start_model_download() and tracks DownloadProgressEvent/DownloadFinished.
    show_main_window: bring to front (second-instance launch, tray click).
    exit_app: QApplication.quit() after core shutdown.
    """

    def show_error(self, title: str, message: str, details: str = "") -> None: ...
    def show_download_dialog(self, model: str, size_mb: int) -> None: ...
    def show_main_window(self) -> None: ...
    def exit_app(self) -> None: ...


# ---------------------------------------------------------------------------
# CLI (.wav test mode) output contract, used by integration tests
# ---------------------------------------------------------------------------
#
#   python -m app.cli WAV [--model NAME] [--compute-type float16|int8_float16] [--beam N]
#                         [--task translate|transcribe] [--cpu] [--download] [--json]
#                         [--realtime]
#
# Runs WavFileSource -> Segmenter -> SpeechEngine (no injection). Uses the user's Config for
# anything not overridden on the command line. Logs go to logs_dir() as in the app.
# --realtime feeds audio at 1x speed; by default it feeds as fast as possible.
# Exit codes: 0 ok; 2 CUDA unavailable (and not --cpu); 3 model not downloaded (and not
# --download); 4 bad input (missing/unreadable wav, invalid args); 1 other error.
# With --cpu, a line "WARNING: CPU mode (slow)" goes to stderr.
# Text output (stdout):
#   ENGINE model=<m> device=<d> compute_type=<c> gpu=<g> load_ms=<f>
#   SEGMENT id=<n> reason=<r> audio_s=<f> infer_ms=<f> filtered=<0|1> text=<text>
#   SUMMARY segments=<n> kept=<n> total_audio_s=<f> total_infer_ms=<f> wall_ms=<f>
# --json prints exactly one JSON object instead:
#   {"engine": {EngineInfo fields}, "segments": [{"id", "reason", "audio_s", "infer_ms",
#    "latency_ms", "filtered", "filter_reason", "text"}], "summary": {"segments", "kept",
#    "total_audio_s", "total_infer_ms", "wall_ms"}}
#   latency_ms in CLI mode = t_infer_end - t_vad_close (no injection step).
#
# App entry point: python -m app.main [--minimized] [--debug-audio-wav PATH]
#   --debug-audio-wav replaces the microphone with WavFileSource(PATH, realtime=True) for
#   each dictation session (used by the desktop test). Normal launch is minimized to the tray.


# ---------------------------------------------------------------------------
# Implementation map (import paths + constructor signatures)
# ---------------------------------------------------------------------------
#
# app.audio.vad.SileroStreamingVad()                         -> VadModel  (bundled faster-whisper silero_vad_v6.onnx)
# app.audio.segmenter.VadSegmenter(cfg: VadConfig, vad: VadModel, clock=time.perf_counter) -> Segmenter
#     .set_config(cfg: VadConfig)  # applies from the next feed(); any thread
# app.audio.capture.MicSource(device: Optional[str])         -> AudioSource
#     MicSource.list_input_devices() -> list[str]  (staticmethod; names usable as Config.mic_device)
# app.audio.wavsource.WavFileSource(path, realtime=False, chunk_ms=32) -> AudioSource
#     (any sample rate / channel count wav; converted to 16 kHz mono float32)
# app.audio.wavsource.read_wav_16k_mono(path) -> np.ndarray
#
# app.engine.cuda_probe.WindowsCudaProbe()                   -> CudaProbe   (Windows)
# app.engine.metal_probe.MetalProbe()                        -> CudaProbe   (macOS; setup_dll_paths() -> [],
#     probe() reports the Apple chip, unified memory and the whisper.cpp/Metal availability, never raises)
# app.engine.filter.ThresholdHallucinationFilter(cfg: FilterConfig) -> HallucinationFilter
#     app.engine.filter.normalize_text(text: str) -> str
# app.engine.engine.WhisperEngine(probe: CudaProbe, models_root: Path | None = None) -> SpeechEngine
#     (Windows: faster-whisper / CTranslate2 on CUDA)
# app.engine.whispercpp.WhisperCppEngine(probe: CudaProbe | None = None, models_root: Path | None = None)
#     -> SpeechEngine  (macOS: whisper.cpp on Metal through libroenspeak_whisper.dylib)
# app.engine.downloader.HfModelDownloader(models_root: Path | None = None) -> ModelDownloader
# app.engine.ggml_downloader.GgmlModelDownloader(models_root: Path | None = None, compute_type=FLOAT16)
#     -> ModelDownloader  (macOS: one ggml_model_file() from GGML_REPO + MODEL_COMPLETE_MARKER)
# app.engine.gpu.vram_warning(gpu: GpuInfo, model: str, compute_type: ComputeType) -> Optional[str]
#     (message suggesting int8_float16 when gpu.total_vram_mb < MODELS[model].vram_mb[compute_type])
#
# app.win32._winapi.RealWinApi()                             -> WinApi
# app.win32.hotkey_parse.parse_hotkey(text) -> Hotkey ; format_hotkey(h) -> str
#     app.win32.hotkey_parse.HotkeyStringParser()            -> HotkeyParser
# app.win32.hotkey.Win32HotkeyManager(api: WinApi, on_fire: Callable[[], None], debounce_s=0.3) -> HotkeyManager
# app.win32.clipboard.Win32Clipboard(api: WinApi)            -> ClipboardBackend
# app.win32.injector.Win32TextInjector(api: WinApi, clipboard: ClipboardBackend,
#     mode=InjectionMode.SENDINPUT, modifier_timeout_s=5.0, poll_s=0.01, paste_settle_s=0.15) -> TextInjector
# app.win32.instance.NamedMutexSingleInstance(api: WinApi, name: str | None = None) -> SingleInstance
# app.win32.autostart.RunKeyAutostart(command: str | None = None) -> Autostart
#
# app.macos._mac_api.RealMacApi()                            -> the macOS counterpart of WinApi
# app.macos.hotkey_parse.parse_hotkey(text) -> Hotkey ; format_hotkey(h) -> str
#     app.macos.hotkey_parse.MacHotkeyStringParser()          -> HotkeyParser
# app.macos.hotkey.MacHotkeyManager(api, on_fire: Callable[[], None], debounce_s=0.3) -> HotkeyManager
#     (Carbon RegisterEventHotKey on the main run loop; no Alt-menu suppression needed)
# app.macos.clipboard.MacClipboard(api)                      -> ClipboardBackend  (NSPasteboard)
# app.macos.injector.MacTextInjector(api, clipboard, mode=InjectionMode.SENDINPUT,
#     modifier_timeout_s=5.0, poll_s=0.01, paste_settle_s=0.15) -> TextInjector
#     (CGEventKeyboardSetUnicodeString; secure-input and Accessibility checks replace UIPI)
# app.macos.instance.FileLockSingleInstance(api=None, name: str | None = None) -> SingleInstance
#     (flock on single_instance_lock_path() + a Unix socket for "show window")
# app.macos.autostart.LaunchAgentAutostart(command: str | None = None) -> Autostart
#     (~/Library/LaunchAgents/com.roenspeak.RoEnSpeak.plist with RunAtLoad)
# app.macos.permissions  -> accessibility/microphone permission checks and prompts
#
# app.core.platform  -- the ONLY platform dispatch. parse_hotkey/format_hotkey plus
#     make_probe/make_engine/make_downloader/make_api/make_clipboard/make_injector/
#     make_autostart/make_hotkey_factory/make_single_instance/open_folder/gpu_requirement_message.
# app.core.config.JsonConfigStore(path: Path | None = None)  -> ConfigStore
#     app.core.config.config_to_dict(cfg) -> dict ; config_from_dict(raw) -> tuple[Config, list[str]]
#     app.core.config.migrate(raw: dict) -> dict   (any schema -> current schema dict; pure)
# app.core.events.SimpleEventBus()                           -> EventBus
# app.core.state.AppStateMachine(bus: EventBus, initial=AppState.DISARMED) -> StateMachine
# app.core.logging_setup.setup_logging(log_dir: Path | None = None, level=logging.INFO) -> Path
#     (rotating file handler: <log_dir>/roenspeak.log, 5 x 2 MB, plus stderr when a console exists)
# app.core.pipeline.Pipeline(...)                            -- threads + queues (internal to core)
# app.core.controller.AppController(...)                     -> Controller
# app.main.main(argv=None) -> int ; app.cli.main(argv=None) -> int
#
# Startup order in app.main (Qt thread): create_qapplication() -> QtUiFacade() -> AppController(..., ui=facade)
#   -> run_ui(controller, bus, facade, start_minimized) -> controller.quit() on exit paths.
# app.ui.app.create_qapplication() -> QApplication  (setQuitOnLastWindowClosed(False), app icon)
# app.ui.facade.QtUiFacade()                                 -> UiFacade
#     constructed on the Qt thread after QApplication exists; run_ui binds it to the window/tray.
#     Calls made before binding are queued and replayed once bound.
# app.ui.app.run_ui(controller: Controller, bus: EventBus, facade: QtUiFacade, start_minimized: bool) -> int
#     builds tray + main window + overlay, installs a native event filter that calls
#     show_main_window() on the registered SHOW_WINDOW_MESSAGE_NAME message, returns app.exec().
