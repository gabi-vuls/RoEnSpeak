"""AppController: the Controller implementation that wires config, state, pipeline, hotkey,
download and UI together.

Threading model
  * Public Controller methods are called from the Qt thread and return quickly: anything
    slow is posted to the "core-control" thread, which runs jobs one at a time.
  * set_hotkey/update_config do their synchronous part (validate, register, persist) on the
    caller's thread under self._lock, because they must raise to the caller.
  * Pipeline/download/hotkey callbacks never block: they post a job to core-control. The one
    exception is the idle-unload check, which runs synchronously on the inference thread.
  * Nothing that holds self._lock ever waits for a pipeline thread, so there is no lock cycle.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from app.contracts import (
    DEFAULT_HOTKEY,
    MIN_DRIVER_VERSION,
    MODELS,
    AppError,
    AppEvent,
    AppState,
    AudioSource,
    Autostart,
    Config,
    ConfigChanged,
    ConfigError,
    ConfigStore,
    CudaProbe,
    CudaUnavailableError,
    DownloadCancelledError,
    DownloadError,
    DownloadFinished,
    DownloadProgress,
    DownloadProgressEvent,
    EngineInfo,
    EngineLoaded,
    EngineNotLoadedError,
    ErrorRaised,
    EventBus,
    GpuOutOfMemoryError,
    HallucinationFilter,
    Hotkey,
    HotkeyChanged,
    HotkeyConflictError,
    HotkeyManager,
    HotkeyParseError,
    InvalidTransition,
    ModelDownloader,
    Notification,
    Segmenter,
    SegmentProcessed,
    SingleInstance,
    SpeechEngine,
    StateMachine,
    TextInjector,
    TranslationResult,
    UiFacade,
    logs_dir,
)
from app.core.config import JsonConfigStore, config_from_dict, config_to_dict
from app.core.events import SimpleEventBus
from app.core.pipeline import Pipeline
from app.core.state import AppStateMachine

log = logging.getLogger("roenspeak.core.controller")

ModelKey = tuple[str, str, bool]  # (model, compute_type value, cpu_mode)

#: A hotkey press within this many seconds after an error aborted a session only acknowledges
#: the error (the user was most likely trying to stop the recording that just failed).
ERROR_ACK_WINDOW_S = 5.0
#: How long quit() waits for an in-progress model load to finish before unloading.
QUIT_LOAD_WAIT_S = 60.0


def _model_key(cfg: Config) -> ModelKey:
    return (cfg.model, cfg.compute_type.value, cfg.cpu_mode)


def _parse_hotkey(text: str) -> Hotkey:
    from app.win32.hotkey_parse import parse_hotkey

    return parse_hotkey(text)


def cuda_requirement_message(error: Optional[str]) -> str:
    return (
        f"RoEnSpeak needs an NVIDIA GPU with driver {MIN_DRIVER_VERSION} or newer, and the CUDA "
        f"libraries (cuBLAS, cuDNN) must load.\n\n"
        f"Problem: {error or 'CUDA is not available.'}\n\n"
        f"Install or update the NVIDIA driver, then restart RoEnSpeak. If this PC has no NVIDIA GPU, "
        f"you can enable \"CPU mode (slow)\" in Settings."
    )


class AppController:
    """Controller implementation. Construct on the Qt thread, then call start() once the UI
    is subscribed to the bus (main.py does this from the first event-loop iteration)."""

    def __init__(
        self,
        *,
        ui: UiFacade,
        bus: Optional[EventBus] = None,
        store: Optional[ConfigStore] = None,
        probe: Optional[CudaProbe] = None,
        engine: Optional[SpeechEngine] = None,
        downloader: Optional[ModelDownloader] = None,
        filt: Optional[HallucinationFilter] = None,
        segmenter: Optional[Segmenter] = None,
        injector: Optional[TextInjector] = None,
        hotkey_manager_factory: Optional[Callable[[Callable[[], None]], HotkeyManager]] = None,
        autostart: Optional[Autostart] = None,
        single_instance: Optional[SingleInstance] = None,
        source_factory: Optional[Callable[[Config], AudioSource]] = None,
        api: Any = None,
        debug_audio_wav: Optional[Path] = None,
        state: Optional[StateMachine] = None,
    ) -> None:
        self._ui = ui
        self._bus: EventBus = bus if bus is not None else SimpleEventBus()
        self._store: ConfigStore = store if store is not None else JsonConfigStore()
        self._sm: StateMachine = state if state is not None else AppStateMachine(self._bus)
        self._probe = probe
        self._engine = engine
        self._downloader = downloader
        self._filter = filt
        self._segmenter = segmenter
        self._injector = injector
        self._hotkey_factory = hotkey_manager_factory
        self._autostart = autostart
        self._single_instance = single_instance
        self._source_factory = source_factory
        self._api = api
        self._debug_wav = Path(debug_audio_wav) if debug_audio_wav else None

        self._lock = threading.RLock()
        self._config: Config = self._store.load()
        self._load_warnings = list(self._store.last_warnings)

        self._pipeline: Optional[Pipeline] = None
        self._hotkey_mgr: Optional[HotkeyManager] = None
        self._hotkey_registered = False  # armed: hotkey owned by us (may be suspended)
        self._hotkey_suspended = False  # temporarily unregistered while the UI captures a combo

        self._engine_info: Optional[EngineInfo] = None
        self._loaded_key: Optional[ModelKey] = None
        self._loading_key: Optional[ModelKey] = None
        self._last_latency_ms: Optional[float] = None

        self._session_counter = 0
        self._active_session: Optional[int] = None
        self._flushing_session: Optional[int] = None
        self._source: Optional[AudioSource] = None
        self._t_error_abort: Optional[float] = None  # perf_counter when an error aborted a session
        self._model_missing: Optional[str] = None  # model whose download the user still has to start

        self._download_thread: Optional[threading.Thread] = None
        self._download_cancel: Optional[threading.Event] = None
        self._download_model: Optional[str] = None

        self._started = False
        self._quitting = False
        self._quit_lock = threading.Lock()
        self._quit_started = False
        self._shutdown_done = threading.Event()

        self._jobs: "queue.Queue[Optional[Callable[[], None]]]" = queue.Queue()
        self._control_thread = threading.Thread(target=self._control_loop, name="core-control", daemon=True)
        self._control_thread.start()

    # ------------------------------------------------------------------
    # Controller properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> Config:
        return self._config

    @property
    def state(self) -> AppState:
        return self._sm.state

    @property
    def engine_info(self) -> Optional[EngineInfo]:
        return self._engine_info

    @property
    def last_latency_ms(self) -> Optional[float]:
        return self._last_latency_ms

    @property
    def bus(self) -> EventBus:
        return self._bus

    # ------------------------------------------------------------------
    # Controller methods (Qt thread)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Runs the startup sequence on the control thread. Idempotent."""
        if self._started:
            return
        self._started = True
        self._post(self._startup)

    def set_armed(self, on: bool) -> None:
        self._post(self._arm if on else self._disarm)

    def toggle_dictation(self) -> None:
        self._post(self._toggle)

    def set_hotkey(self, text: str) -> Hotkey:
        hk = _parse_hotkey(text)  # HotkeyParseError propagates
        with self._lock:
            self._register_checked(hk)  # HotkeyConflictError propagates; previous stays
            new = dataclasses.replace(self._config, hotkey=hk.display)
            self._store.save(new)
            self._config = new
            registered = self._hotkey_registered
        log.info("hotkey set to %s (registered=%s)", hk.display, registered)
        self._publish(ConfigChanged(new))
        self._publish(HotkeyChanged(hk.display if registered else None, registered))
        return hk

    def reset_hotkey(self) -> Hotkey:
        return self.set_hotkey(DEFAULT_HOTKEY)

    def _register_checked(self, hk: Hotkey) -> None:
        """Caller holds self._lock. When armed, registers hk (raises on conflict, previous stays);
        while suspended, the registration is only a conflict check and is removed again."""
        if not self._hotkey_registered or self._hotkey_mgr is None:
            return
        self._hotkey_mgr.register(hk)
        if self._hotkey_suspended:
            self._hotkey_mgr.unregister()

    def suspend_hotkey(self, suspended: bool) -> None:
        with self._lock:
            mgr = self._hotkey_mgr
            if suspended:
                if self._hotkey_registered and not self._hotkey_suspended and mgr is not None:
                    try:
                        mgr.unregister()
                    except Exception:
                        log.exception("hotkey unregister (suspend) failed")
                    self._hotkey_suspended = True
                    log.info("hotkey suspended")
                return
            if not self._hotkey_suspended:
                return
            self._hotkey_suspended = False
            if not self._hotkey_registered or mgr is None or self._quitting:
                return
            try:
                hk = _parse_hotkey(self._config.hotkey)
                mgr.register(hk)
                log.info("hotkey resumed: %s", hk.display)
            except (HotkeyConflictError, HotkeyParseError) as e:
                log.warning("could not re-register hotkey %s: %s", self._config.hotkey, e.details or e.user_message)
                self._hotkey_registered = False
                self._abort_session()
                self._publish(ErrorRaised(type(e).__name__, e.user_message, details=e.details, fatal=False))
                self._publish(HotkeyChanged(None, False))
                self._goto(AppState.DISARMED)

    def update_config(self, cfg: Config) -> None:
        if not isinstance(cfg, Config):
            raise ConfigError("Invalid settings object.")
        validated, warnings = config_from_dict(config_to_dict(cfg))
        if warnings:
            raise ConfigError("Invalid settings: " + " ".join(warnings))
        with self._lock:
            old = self._config
            hotkey_changed = validated.hotkey != old.hotkey
            if hotkey_changed:
                self._register_checked(_parse_hotkey(validated.hotkey))  # HotkeyConflictError propagates
            # Arming is applied asynchronously by _arm/_disarm, which persist it themselves.
            new = dataclasses.replace(validated, armed=old.armed)
            self._store.save(new)
            self._config = new
            self._apply_live(old, new)
            registered = self._hotkey_registered
        log.info("config updated")
        self._publish(ConfigChanged(new))
        if hotkey_changed:
            self._publish(HotkeyChanged(new.hotkey if registered else None, registered))
        want_armed = validated.armed
        self._post(lambda: self._apply_config_async(old, new, want_armed))

    def list_input_devices(self) -> list[str]:
        try:
            from app.audio.capture import MicSource

            return list(MicSource.list_input_devices())
        except Exception:
            log.exception("listing input devices failed")
            return []

    def start_model_download(self) -> None:
        self._post(self._start_download)

    def cancel_model_download(self) -> None:
        ev = self._download_cancel
        if ev is not None:
            log.info("model download cancel requested")
            ev.set()

    def open_log_folder(self) -> None:
        folder = logs_dir()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except OSError as e:
            log.error("could not open log folder %s: %s", folder, e)
            self._publish(Notification("Could not open the log folder", f"{folder}\n{e}", warning=True))

    def quit(self) -> None:
        """Starts the shutdown on a dedicated thread and returns. Idempotent."""
        with self._quit_lock:
            if self._quit_started:
                return
            self._quit_started = True
        self._quitting = True
        t = threading.Thread(target=self._shutdown_worker, name="core-shutdown", daemon=False)
        t.start()

    def shutdown(self, timeout: float = 20.0) -> bool:
        """quit() and wait until the core is fully shut down (for main's exit paths/atexit)."""
        self.quit()
        return self._shutdown_done.wait(timeout)

    # ------------------------------------------------------------------
    # Control thread
    # ------------------------------------------------------------------

    def _post(self, job: Callable[[], None]) -> None:
        if self._quitting:
            log.debug("ignoring job while quitting: %r", job)
            return
        self._jobs.put(job)

    def _control_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                break
            if self._quitting:
                continue
            try:
                job()
            except Exception as e:
                log.exception("control job %r failed", job)
                self._publish(ErrorRaised(type(e).__name__, f"Internal error: {e}", details=repr(e), fatal=False))
        log.debug("control thread exit")

    def _publish(self, event: AppEvent) -> None:
        self._bus.publish(event)

    def _goto(self, new: AppState, detail: str = "") -> bool:
        try:
            return self._sm.transition(new, detail)
        except InvalidTransition as e:
            log.warning("ignored transition: %s", e.user_message)
            return False

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _startup(self) -> None:
        if self._quitting:
            return
        if self._load_warnings:
            self._publish(Notification("Some settings were reset", "\n".join(self._load_warnings), warning=True))
        try:
            self._build_components()
        except Exception as e:
            log.exception("startup failed while creating components")
            with self._lock:
                self._enter_error(type(e).__name__, f"RoEnSpeak could not start: {getattr(e, 'user_message', e)}",
                                  details=repr(e))
            return
        if self._quitting:
            return

        cfg = self._config
        if not cfg.cpu_mode:
            report = self._probe.probe()  # never raises; logs LOG_CUDA_PROBE
            if self._quitting:
                return
            if not report.ok:
                with self._lock:
                    self._cuda_failure(report.error)
                return
            if not cfg.ui.first_run_done and report.gpu is not None:
                self._vram_check(report.gpu, cfg)
        else:
            self._publish(Notification("CPU mode (slow)", "CPU mode is enabled; translation will be slow.",
                                       warning=True))

        with self._lock:
            if self._quitting:
                return
            if cfg.armed:
                self._arm()
            if (not self._quitting and not self._hotkey_registered and self._sm.state == AppState.DISARMED
                    and self._downloader.is_downloaded(cfg.model)):
                self._request_load()  # model stays resident while disarmed; arming is then instant

    def _build_components(self) -> None:
        cfg = self._config
        if self._probe is None:
            from app.engine.cuda_probe import WindowsCudaProbe

            self._probe = WindowsCudaProbe()
            self._probe.setup_dll_paths()
        if self._engine is None:
            from app.engine.engine import WhisperEngine

            self._engine = WhisperEngine(self._probe)
        if self._downloader is None:
            from app.engine.downloader import HfModelDownloader

            self._downloader = HfModelDownloader()
        if self._filter is None:
            from app.engine.filter import ThresholdHallucinationFilter

            self._filter = ThresholdHallucinationFilter(cfg.filter)
        if self._segmenter is None:
            from app.audio.segmenter import VadSegmenter
            from app.audio.vad import SileroStreamingVad

            self._segmenter = VadSegmenter(cfg.vad, SileroStreamingVad())
        if (self._injector is None or self._hotkey_factory is None) and self._api is None:
            from app.win32._winapi import RealWinApi

            self._api = RealWinApi()
        if self._injector is None:
            from app.win32.clipboard import Win32Clipboard
            from app.win32.injector import Win32TextInjector

            self._injector = Win32TextInjector(self._api, Win32Clipboard(self._api), mode=cfg.injection_mode)
        if self._autostart is None:
            from app.win32.autostart import RunKeyAutostart

            self._autostart = RunKeyAutostart()
        if self._hotkey_factory is None:
            from app.win32.hotkey import Win32HotkeyManager

            api = self._api
            self._hotkey_factory = lambda on_fire: Win32HotkeyManager(api, on_fire)
        self._hotkey_mgr = self._hotkey_factory(self._on_hotkey_fire)
        self._hotkey_mgr.start()

        self._pipeline = Pipeline(
            segmenter=self._segmenter,
            engine=self._engine,
            filt=self._filter,
            injector=self._injector,
            bus=self._bus,
            task=cfg.task,
            decode=cfg.decode,
            idle_unload_minutes=cfg.idle_unload_minutes,
            on_engine_loaded=lambda info, token: self._post(lambda: self._on_engine_loaded(info, token)),
            on_engine_error=lambda exc, phase, token: self._post(lambda: self._on_engine_error(exc, phase, token)),
            on_engine_unloaded=lambda reason: self._post(lambda: self._on_engine_unloaded(reason)),
            idle_unload_check=self._idle_unload_check,
            on_session_flushed=lambda sid: self._post(lambda: self._on_session_flushed(sid)),
            on_segment_done=self._on_segment_done,
        )
        self._pipeline.start()

    def _vram_check(self, gpu: Any, cfg: Config) -> None:
        try:
            from app.engine.gpu import vram_warning

            msg = vram_warning(gpu, cfg.model, cfg.compute_type)
        except Exception:
            log.exception("VRAM check failed")
            return
        if msg:
            log.warning("VRAM warning: %s", msg)
            self._publish(Notification("Not enough GPU memory?", msg, warning=True))

    def _sync_autostart(self, enabled: bool) -> None:
        """Only called when the user changes ui.start_with_windows; never at startup, so an
        isolated/test config can't touch the user's real Run key."""
        if self._autostart is None:
            return
        try:
            self._autostart.set_enabled(enabled)
            log.info("start with Windows: %s", enabled)
        except Exception as e:
            log.exception("autostart update failed")
            self._publish(ErrorRaised(type(e).__name__, "Could not change the \"Start with Windows\" setting.",
                                      details=str(e), fatal=False))

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    def _enter_error(self, kind: str, message: str, details: str = "", *, discard: bool = True,
                     dialog_title: Optional[str] = None) -> None:
        if discard:
            if self._active_session is not None:
                self._t_error_abort = time.perf_counter()
            self._abort_session()
        if not self._goto(AppState.ERROR, message.splitlines()[0] if message else kind) \
                and self._sm.state != AppState.ERROR:
            log.warning("could not enter ERROR from %s for %s", self._sm.state.value, kind)
        self._publish(ErrorRaised(kind, message, details=details, fatal=True))
        if dialog_title is not None:
            self._ui.show_error(dialog_title, message, details)

    def _cuda_failure(self, error: Optional[str], details: str = "") -> None:
        """CUDA probe failed: explain the requirement around the probe's reason."""
        log.error("CUDA unavailable: %s", error)
        # Same (title, message) as the UI uses for ErrorRaised -> the UI shows a single dialog.
        self._enter_error("CudaUnavailableError", cuda_requirement_message(error), details or (error or ""),
                          dialog_title="NVIDIA GPU required")

    def _cuda_engine_error(self, exc: CudaUnavailableError) -> None:
        """The engine raised a CUDA error; its user_message is already user-facing."""
        oom = isinstance(exc, GpuOutOfMemoryError)
        log.error("%s: %s (%s)", type(exc).__name__, exc.user_message, exc.details)
        self._enter_error(type(exc).__name__, exc.user_message, exc.details,
                          dialog_title="Not enough GPU memory" if oom else "NVIDIA GPU required")

    # ------------------------------------------------------------------
    # Arm / disarm
    # ------------------------------------------------------------------

    def _arm(self) -> None:
        with self._lock:
            if self._quitting:
                return
            if self._pipeline is None:
                log.info("arm requested before startup finished; deferred")
                self._post(self._arm)
                return
            cfg = self._config
            if not cfg.cpu_mode:
                report = self._probe.probe()
                if self._quitting:
                    return
                if not report.ok:
                    self._cuda_failure(report.error)
                    return
            if not self._hotkey_registered:
                try:
                    hk = _parse_hotkey(cfg.hotkey)
                    if not self._hotkey_suspended:
                        self._hotkey_mgr.register(hk)
                except (HotkeyConflictError, HotkeyParseError) as e:
                    # Run disarmed for now, but keep Config.armed (the user's intent) so the next
                    # launch retries; only an explicit set_armed(False) persists OFF.
                    log.warning("could not register hotkey %s: %s", cfg.hotkey, e.details or e.user_message)
                    self._publish(ErrorRaised(type(e).__name__, e.user_message, details=e.details, fatal=False))
                    self._publish(HotkeyChanged(None, False))
                    if self._sm.state == AppState.ERROR:
                        self._goto(AppState.DISARMED)
                    return
                self._hotkey_registered = True
                log.info("armed: hotkey %s registered", hk.display)
                self._publish(HotkeyChanged(hk.display, True))
            self._persist_armed(True)
            self._ensure_model()

    def _disarm(self) -> None:
        with self._lock:
            self._abort_session()
            if self._hotkey_registered and self._hotkey_mgr is not None:
                try:
                    self._hotkey_mgr.unregister()
                except Exception:
                    log.exception("hotkey unregister failed")
                self._hotkey_registered = False
                self._publish(HotkeyChanged(None, False))
            self._hotkey_suspended = False
            self._persist_armed(False)
            self._goto(AppState.DISARMED)
            log.info("disarmed")

    def _persist_armed(self, armed: bool) -> None:
        if self._config.armed == armed:
            return
        new = dataclasses.replace(self._config, armed=armed)
        self._config = new
        try:
            self._store.save(new)
        except ConfigError as e:
            log.error("could not persist armed=%s: %s", armed, e.details)
        self._publish(ConfigChanged(new))

    # ------------------------------------------------------------------
    # Model loading / download
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Armed path: reach READY, loading/downloading as needed."""
        key = _model_key(self._config)
        st = self._sm.state
        if self._loaded_key == key and self._engine_info is not None and self._loading_key is None:
            if st in (AppState.DISARMED, AppState.ERROR, AppState.LOADING_MODEL):
                self._goto(AppState.READY)
            return
        if self._loading_key == key:
            self._goto(AppState.LOADING_MODEL, self._config.model)
            return
        if st in (AppState.LISTENING, AppState.PROCESSING):
            return  # reloaded when the session completes
        self._request_load()

    def _request_load(self) -> bool:
        cfg = self._config
        key = _model_key(cfg)
        if self._quitting:
            return False
        if self._loading_key == key:
            return True
        if not self._downloader.is_downloaded(cfg.model):
            self._begin_download_flow()
            return False
        self._loading_key = key
        self._model_missing = None
        # The (possibly deferred) task switches together with the model on the inference thread.
        self._pipeline.load_model(cfg.model, cfg.compute_type, cfg.cpu_mode, cfg.decode, token=key, task=cfg.task)
        if self._sm.state not in (AppState.LISTENING, AppState.PROCESSING):
            self._goto(AppState.LOADING_MODEL, cfg.model)
        return True

    def _on_engine_loaded(self, info: EngineInfo, token: Any) -> None:
        with self._lock:
            self._engine_info = info
            self._loaded_key = token
            if token != self._loading_key:
                log.info("engine loaded %s, but a newer load is pending", token)
                return
            self._loading_key = None
            self._publish(EngineLoaded(info))
            if not self._config.ui.first_run_done:
                new = dataclasses.replace(self._config, ui=dataclasses.replace(self._config.ui, first_run_done=True))
                self._config = new
                try:
                    self._store.save(new)
                except ConfigError as e:
                    log.error("could not persist first_run_done: %s", e.details)
                self._publish(ConfigChanged(new))
            if token != _model_key(self._config):
                # Settings changed while loading: load again with the current ones (unless that
                # model still waits for the user to start its download).
                if (self._sm.state not in (AppState.LISTENING, AppState.PROCESSING)
                        and self._model_missing != self._config.model):
                    self._request_load()
                return
            if self._sm.state == AppState.LOADING_MODEL:
                self._goto(AppState.READY if self._hotkey_registered else AppState.DISARMED)

    def _on_engine_error(self, exc: BaseException, phase: str, token: Any) -> None:
        with self._lock:
            if phase == "load":
                if token != self._loading_key:
                    log.info("ignoring failure of superseded load %s", token)
                    return
                self._loading_key = None
            engine_loaded = bool(getattr(self._engine, "is_loaded", False))
            if phase == "load" or not engine_loaded or isinstance(exc, (CudaUnavailableError, EngineNotLoadedError)):
                # The engine dropped (or never had) the model: the next hotkey press must reload
                # instead of recording into a pipeline that can only drop segments.
                self._loaded_key = None
                self._engine_info = None
            if isinstance(exc, EngineNotLoadedError) and self._sm.state == AppState.ERROR:
                log.warning("segment dropped: no model loaded (already in ERROR)")
                return
            if isinstance(exc, CudaUnavailableError):
                self._cuda_engine_error(exc)
                return
            if isinstance(exc, EngineNotLoadedError):
                message = ("The speech model is not loaded, so the dictated audio was dropped. "
                           "Press the hotkey again to reload the model.")
                details = exc.details or exc.user_message
            elif isinstance(exc, AppError):
                message, details = exc.user_message, exc.details
            else:
                message = f"The speech engine failed: {exc}"
                details = repr(exc)
            if phase == "load":
                message = f"Could not load the model '{self._config.model}'.\n\n{message}"
            self._enter_error(type(exc).__name__, message, details)

    def _on_engine_unloaded(self, reason: str) -> None:
        with self._lock:
            if self._loading_key is None:
                self._engine_info = None
                self._loaded_key = None

    def _idle_unload_check(self) -> bool:
        # Runs on the inference thread; must decide and change state atomically.
        with self._lock:
            if self._quitting or self._sm.state != AppState.READY or self._loading_key is not None:
                return False
            return self._goto(AppState.UNLOADED, "idle")

    def _begin_download_flow(self) -> None:
        """The model is missing: show the download dialog. The state only becomes DOWNLOADING
        once the user actually starts the download (start_model_download); until then it is
        ERROR "Model <m> not downloaded", and a hotkey press re-shows the dialog."""
        model = self._config.model
        t = self._download_thread
        if t is not None and t.is_alive() and self._download_model == model:
            self._goto(AppState.DOWNLOADING, model)
            return
        already = self._model_missing == model and self._sm.state == AppState.ERROR
        self._model_missing = model
        if not already:
            if not self._goto(AppState.ERROR, f"Model {model} not downloaded") \
                    and self._sm.state != AppState.ERROR:
                log.warning("model %s missing but state %s cannot go to ERROR", model, self._sm.state.value)
        log.info("model %s is not downloaded; showing download dialog", model)
        self._ui.show_download_dialog(model, MODELS[model].download_mb)

    def _start_download(self) -> None:
        with self._lock:
            model = self._config.model
            t = self._download_thread
            if t is not None and t.is_alive():
                if self._download_model == model:
                    return
                # Another model is still downloading: cancel it and retry when it has stopped.
                if self._download_cancel is not None:
                    self._download_cancel.set()
                retry = threading.Timer(0.5, lambda: self._post(self._start_download))
                retry.daemon = True
                retry.start()
                return
            if self._downloader.is_downloaded(model):
                self._publish(DownloadFinished(model, True))
                self._on_download_ok(model)
                return
            if self._sm.state not in (AppState.LISTENING, AppState.PROCESSING):
                if not self._goto(AppState.DOWNLOADING, model) and self._sm.state != AppState.DOWNLOADING:
                    log.warning("download of %s started but state %s cannot go to DOWNLOADING",
                                model, self._sm.state.value)
            self._model_missing = None
            cancel = threading.Event()
            self._download_cancel = cancel
            self._download_model = model
            self._download_thread = threading.Thread(
                target=self._download_worker, args=(model, cancel), name="core-download", daemon=True
            )
            self._download_thread.start()

    def _download_worker(self, model: str, cancel: threading.Event) -> None:
        def on_progress(p: DownloadProgress) -> None:
            self._publish(DownloadProgressEvent(p))

        log.info("downloading model %s", model)
        try:
            self._downloader.download(model, on_progress, cancel)
        except DownloadCancelledError:
            log.info("download of %s cancelled", model)
            self._publish(DownloadFinished(model, False, error="Download cancelled."))
            self._post(lambda: self._on_download_cancelled(model))
        except DownloadError as e:
            log.error("download of %s failed (offline=%s): %s %s", model, e.offline, e.user_message, e.details)
            self._publish(DownloadFinished(model, False, error=e.user_message, offline=e.offline))
            self._post(lambda err=e: self._on_download_failed(model, err))
        except Exception as e:
            log.exception("download of %s failed", model)
            err = DownloadError(f"Downloading the model failed: {e}", details=repr(e))
            self._publish(DownloadFinished(model, False, error=err.user_message))
            self._post(lambda: self._on_download_failed(model, err))
        else:
            log.info("download of %s finished", model)
            self._publish(DownloadFinished(model, True))
            self._post(lambda: self._on_download_ok(model))

    def _on_download_ok(self, model: str) -> None:
        with self._lock:
            if model != self._config.model:
                return
            if self._sm.state in (AppState.LISTENING, AppState.PROCESSING):
                return
            self._request_load()

    def _on_download_failed(self, model: str, e: DownloadError) -> None:
        with self._lock:
            if model != self._config.model:
                return
            self._enter_error(type(e).__name__, e.user_message, e.details)

    def _on_download_cancelled(self, model: str) -> None:
        with self._lock:
            if model != self._config.model:
                return
            if self._sm.state == AppState.DOWNLOADING:
                # Back to "not downloaded" (Config.armed is not touched); the hotkey or the
                # window can start it again.
                self._model_missing = model
                self._goto(AppState.ERROR, f"Model {model} not downloaded")

    # ------------------------------------------------------------------
    # Dictation sessions
    # ------------------------------------------------------------------

    def _on_hotkey_fire(self) -> None:
        # Hotkey thread: never block.
        self._post(self._toggle)

    def _toggle(self) -> None:
        with self._lock:
            if self._pipeline is None or not self._hotkey_registered:
                log.info("toggle ignored: not armed")
                return
            st = self._sm.state
            log.info("toggle in state %s", st.value)
            if st == AppState.LISTENING:
                self._stop_session()
            elif st in (AppState.READY, AppState.PROCESSING, AppState.LOADING_MODEL):
                self._start_session()
            elif st == AppState.UNLOADED:
                if self._request_load():
                    self._start_session()
            elif st == AppState.DOWNLOADING:
                self._publish(Notification("Model is downloading", "Dictation starts working once the download "
                                                                   "has finished."))
            elif st == AppState.ERROR:
                self._toggle_in_error()

    def _toggle_in_error(self) -> None:
        model = self._config.model
        if self._model_missing == model and not self._downloader.is_downloaded(model):
            log.info("hotkey in ERROR: model %s missing; re-showing download dialog", model)
            self._ui.show_download_dialog(model, MODELS[model].download_mb)
            return
        t_abort = self._t_error_abort
        if t_abort is not None and time.perf_counter() - t_abort < ERROR_ACK_WINDOW_S:
            # Most likely the user pressed the hotkey to stop the recording that just failed.
            log.info("hotkey in ERROR right after an aborted session: acknowledged, not starting")
            self._t_error_abort = None
            return
        if self._engine_info is not None and self._loaded_key == _model_key(self._config):
            self._goto(AppState.READY)
            self._start_session()
            return
        if self._loading_key is None and self._model_missing is None:
            log.info("hotkey in ERROR: retrying model load")
            if self._request_load() and self._sm.state == AppState.LOADING_MODEL:
                self._start_session()
            return
        self._publish(Notification("RoEnSpeak is not ready",
                                   "Fix the error shown in the RoEnSpeak window, then turn it on again.",
                                   warning=True))

    def _make_source(self) -> AudioSource:
        if self._source_factory is not None:
            return self._source_factory(self._config)
        if self._debug_wav is not None:
            from app.audio.wavsource import WavFileSource

            return WavFileSource(self._debug_wav, realtime=True)
        from app.audio.capture import MicSource

        return MicSource(self._config.mic_device)

    def _start_session(self) -> None:
        self._session_counter += 1
        sid = self._session_counter
        pipeline = self._pipeline
        pipeline.begin_session(sid)
        try:
            source = self._make_source()
            source.start(
                pipeline.feed_audio,
                on_end=lambda: self._post(lambda: self._on_source_end(sid)),
                on_error=lambda err: self._post(lambda: self._on_source_error(sid, err)),
            )
        except AppError as e:
            log.error("could not start audio capture: %s (%s)", e.user_message, e.details)
            pipeline.discard()
            self._enter_error(type(e).__name__, e.user_message, e.details)
            return
        except Exception as e:
            log.exception("could not start audio capture")
            pipeline.discard()
            self._enter_error("AudioDeviceError", f"Could not open the microphone: {e}", repr(e))
            return
        self._source = source
        self._active_session = sid
        log.info("session %d started (%s)", sid, "wav" if self._debug_wav else self._config.mic_device or "default mic")
        self._goto(AppState.LISTENING)

    def _stop_session(self) -> None:
        sid = self._active_session
        source = self._source
        self._source = None
        self._active_session = None
        if source is not None:
            try:
                source.stop()
            except Exception:
                log.exception("audio source stop failed")
        if sid is None:
            return
        self._flushing_session = sid
        self._pipeline.flush(sid)
        log.info("session %d stopped; flushing", sid)
        self._goto(AppState.PROCESSING)

    def _abort_session(self) -> None:
        """Stops capture and discards all pending audio/segments/results."""
        source = self._source
        self._source = None
        had_session = self._active_session is not None or self._flushing_session is not None
        self._active_session = None
        self._flushing_session = None
        if source is not None:
            try:
                source.stop()
            except Exception:
                log.exception("audio source stop failed")
        if self._pipeline is not None and (had_session or source is not None):
            self._pipeline.discard()

    def _on_source_end(self, sid: int) -> None:
        # A finite source (--debug-audio-wav) ran out. Like a silent mic, the session stays
        # LISTENING until the hotkey stops it; the next session replays the wav.
        log.info("session %d: audio source ended; still listening until stopped", sid)

    def _on_source_error(self, sid: int, err: AppError) -> None:
        with self._lock:
            if self._active_session != sid:
                return
            log.error("audio capture failed: %s (%s)", err.user_message, getattr(err, "details", ""))
            self._t_error_abort = time.perf_counter()
            self._stop_session()  # translate what was captured before the failure
            self._enter_error(type(err).__name__ if isinstance(err, AppError) else "AudioDeviceError",
                              getattr(err, "user_message", str(err)), getattr(err, "details", ""), discard=False)

    def _on_session_flushed(self, sid: int) -> None:
        with self._lock:
            if self._flushing_session != sid:
                return
            self._flushing_session = None
            if self._sm.state != AppState.PROCESSING:
                return
            self._goto(AppState.READY)
            cfg = self._config
            if self._loading_key is None and self._loaded_key != _model_key(cfg):
                self._request_load()  # applies the deferred task with the new model
            elif self._loading_key is None:
                self._pipeline.set_decode_settings(cfg.task, cfg.decode)  # deferred task change

    def _on_segment_done(self, event: SegmentProcessed, result: TranslationResult, reason: Any) -> None:
        if event.latency_ms is not None:
            self._last_latency_ms = event.latency_ms

    # ------------------------------------------------------------------
    # Config application
    # ------------------------------------------------------------------

    def _apply_live(self, old: Config, new: Config) -> None:
        if new.injection_mode != old.injection_mode and self._injector is not None:
            self._injector.set_mode(new.injection_mode)
        if new.filter != old.filter and self._filter is not None:
            self._filter.set_config(new.filter)
        if self._pipeline is not None:
            if new.vad != old.vad:
                self._pipeline.set_vad_config(new.vad)
            if new.task != old.task or new.decode != old.decode:
                # A task change waits for the running session to end and for a model change to
                # load (e.g. switching to a transcription-only model + transcribe).
                defer_task = (self._sm.state in (AppState.LISTENING, AppState.PROCESSING)
                              or _model_key(new) != _model_key(old) or self._loading_key is not None)
                task = self._pipeline.task if defer_task else new.task
                self._pipeline.set_decode_settings(task, new.decode)
            if new.idle_unload_minutes != old.idle_unload_minutes:
                self._pipeline.set_idle_unload_minutes(new.idle_unload_minutes)
        elif new.vad != old.vad and self._segmenter is not None and hasattr(self._segmenter, "set_config"):
            self._segmenter.set_config(new.vad)

    def _apply_config_async(self, old: Config, new: Config, want_armed: bool) -> None:
        if new.ui.start_with_windows != old.ui.start_with_windows:
            self._sync_autostart(new.ui.start_with_windows)
        with self._lock:
            if self._pipeline is None:
                return  # startup not finished; it uses the current config
            if want_armed != self._hotkey_registered:
                if want_armed:
                    self._arm()
                else:
                    self._disarm()
                return  # arming already ensured the (new) model
            if _model_key(new) == _model_key(old):
                return
            st = self._sm.state
            log.info("model settings changed to %s (state %s)", _model_key(new), st.value)
            if st in (AppState.LISTENING, AppState.PROCESSING, AppState.UNLOADED):
                return  # handled when the session ends / on the next hotkey press
            if st == AppState.ERROR:
                if self._config.armed:
                    self._arm()
                return
            if st == AppState.DISARMED and self._engine_info is None and self._loading_key is None:
                return  # nothing resident; arming loads the new model
            if not new.cpu_mode:
                report = self._probe.probe()
                if not report.ok:
                    self._cuda_failure(report.error)
                    return
                if report.gpu is not None:
                    self._vram_check(report.gpu, new)
            if st == AppState.DOWNLOADING:
                self._start_download()
                return
            self._request_load()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown_worker(self) -> None:
        log.info("shutting down")
        try:
            self._jobs.put(None)
            if threading.current_thread() is not self._control_thread:
                self._control_thread.join(5.0)
            source = self._source
            self._source = None
            self._active_session = None
            if source is not None:
                try:
                    source.stop()
                except Exception:
                    log.exception("audio source stop failed")
            if self._download_cancel is not None:
                self._download_cancel.set()
            if self._hotkey_mgr is not None:
                try:
                    self._hotkey_mgr.stop()
                except Exception:
                    log.exception("hotkey manager stop failed")
                self._hotkey_registered = False
            if self._pipeline is not None:
                # An in-progress load can't be interrupted: wait for it (bounded) so the model is
                # unloaded cleanly on the inference thread instead of being cut off at exit.
                loading = self._loading_key is not None
                if loading:
                    log.info("waiting up to %.0fs for the model load to finish before unloading", QUIT_LOAD_WAIT_S)
                self._pipeline.stop(unload=True, timeout=QUIT_LOAD_WAIT_S + 10.0 if loading else 10.0)
            elif self._engine is not None:
                try:
                    self._engine.unload()
                except Exception:
                    log.exception("engine unload failed")
            if self._single_instance is not None:
                try:
                    self._single_instance.release()
                except Exception:
                    log.exception("single-instance release failed")
            log.info("core shutdown complete")
        finally:
            self._shutdown_done.set()
            try:
                self._ui.exit_app()
            except Exception:
                log.exception("ui.exit_app failed")
