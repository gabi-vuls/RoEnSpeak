"""Qt application bootstrap: create_qapplication() and run_ui().

Startup order (app.main, Qt thread): create_qapplication() -> QtUiFacade() ->
AppController(..., ui=facade) -> run_ui(controller, bus, facade, start_minimized).

run_ui builds the tray, main window and overlay, bridges EventBus events onto the Qt thread
(queued signal) and binds the facade. What happens around that differs per platform:

Windows
  A native event filter (NativeFilter) handles three Win32-only things:
    * shows the main window when a second instance broadcasts SHOW_WINDOW_MESSAGE_NAME,
    * suppresses the Alt/system-menu side effects while the hotkey field is capturing,
    * runs controller.quit() on WM_ENDSESSION (logoff/shutdown kills the process right after).

macOS
  None of those exist, so no native event filter is installed at all.
    * A second launch connects to the running instance's Unix socket; SingleInstance exposes
      an ``on_show_window`` callback, which is relayed onto the Qt thread here.
    * Shutdown is covered by aboutToQuit (app.main connects it); commitDataRequest never
      fires on macOS.
    * The app gets a real application menu, so Cmd+Q quits through Controller.quit() instead
      of QApplication.quit(), Cmd+W hides the window and About shows the version.
    * The process is switched to the accessory activation policy, i.e. a menu-bar app with
      no Dock icon - matching the bundle's LSUIElement for dev runs.
    * Dictation silently types nothing without Accessibility permission, so the first time
      the app arms without it, the user gets a dialog with a button into System Settings.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Callable, Optional

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import QApplication, QMenuBar, QMessageBox, QSystemTrayIcon

from app.contracts import (
    APP_NAME,
    APP_VERSION,
    IS_MACOS,
    IS_WINDOWS,
    SHOW_WINDOW_MESSAGE_NAME,
    AppEvent,
    AppState,
    ConfigChanged,
    Controller,
    DownloadFinished,
    DownloadProgressEvent,
    EngineLoaded,
    ErrorRaised,
    EventBus,
    HotkeyChanged,
    InjectionBlockedError,
    Notification,
    SegmentProcessed,
    StateChanged,
)
from app.ui import _platform, _win
from app.ui.assets import app_icon
from app.ui.dialogs import (
    ACCESSIBILITY_TITLE,
    DownloadDialog,
    ErrorDialogs,
    accessibility_granted,
    present,
)
from app.ui.facade import QtUiFacade
from app.ui.hotkey_capture import modifier_names
from app.ui.main_window import MainWindow
from app.ui.overlay import RecOverlay
from app.ui.sounds import SoundPlayer
from app.ui.tray import Tray

log = logging.getLogger("roenspeak.ui.app")

APP_USER_MODEL_ID = f"{APP_NAME}.App"
#: After a capture ends, keep swallowing keyboard SC_KEYMENU for this long so releasing Alt
#: (e.g. after capturing Alt+Q) does not put the window into menu mode.
KEYMENU_GRACE_S = 3.0
#: Dialogs opened this soon after dictation ended are shown without activation.
FOCUS_GRACE_S = 2.0
#: A fatal ErrorRaised this close before a transition into ERROR belongs to it.
FATAL_PAIR_S = 1.0


def create_qapplication() -> QApplication:
    """The QApplication for the tray app (does not quit when the last window closes).

    Qt.AA_MacDontSwapCtrlAndMeta is deliberately NOT set: it is process-wide, and turning the
    swap off would move every standard shortcut off the Command key (Copy would become
    Control+C, the application menu's Quit Control+Q), which is wrong on a Mac. The hotkey
    capture field compensates locally instead - see app.ui.hotkey_capture.
    """
    existing = QApplication.instance()
    if existing is not None:
        app = existing
    else:
        _platform.set_app_user_model_id(APP_USER_MODEL_ID)
        app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setQuitOnLastWindowClosed(False)
    app.setWindowIcon(app_icon())
    if IS_MACOS:
        # Menu-bar app, no Dock icon. The bundle also sets LSUIElement, but a dev run
        # (python -m app.main) has no Info.plist and would otherwise appear in the Dock.
        _platform.set_activation_policy_accessory()
    return app


class EventBridge(QObject):
    """EventBus subscriber callable from any thread; delivers on the Qt thread."""

    received = Signal(object)

    def __init__(self, handler: Callable[[AppEvent], None]) -> None:
        super().__init__()
        self.received.connect(handler, Qt.QueuedConnection)

    def __call__(self, event: AppEvent) -> None:
        self.received.emit(event)


def _ignore_show_window() -> None:
    """What SingleInstance.on_show_window is set back to once the UI is gone."""
    log.debug("show-window ping ignored: the UI is shutting down")


class ShowWindowRelay(QObject):
    """Thread-safe bridge for the macOS "another instance launched" ping.

    SingleInstance calls its on_show_window callback from the socket thread it listens on,
    and Qt objects may only be touched on the Qt thread, so the callback just emits this
    queued signal.
    """

    ping = Signal()

    def __init__(self, handler) -> None:
        super().__init__()
        self.ping.connect(handler, Qt.QueuedConnection)


class NativeFilter(QAbstractNativeEventFilter):
    """Windows only (UiRoot installs it only there): everything it does is a Win32 concept.
    Its event_type check already makes it inert elsewhere, but an event filter that can
    never fire has no business being in the hot path of every single native event."""

    def __init__(self, root: "UiRoot") -> None:
        super().__init__()
        self._root = root
        self.show_message = _win.register_window_message(SHOW_WINDOW_MESSAGE_NAME)
        self._show_pending = False

    def nativeEventFilter(self, event_type, message):  # noqa: N802 (Qt override)
        try:
            if bytes(event_type) not in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
                return False, 0
            msg = _win.read_msg(int(message))
            if msg is None:
                return False, 0
            return self._handle(msg)
        except Exception:  # noqa: BLE001 (never let an exception escape into Qt's wndproc)
            log.exception("native event filter failed")
            return False, 0

    def _handle(self, msg: _win.MSG):
        m = msg.message
        if self.show_message and m == self.show_message:
            # A broadcast reaches every top-level HWND we own; coalesce into one show.
            if not self._show_pending:
                self._show_pending = True
                QTimer.singleShot(0, self._do_show)
            return True, 0
        root = self._root
        if m == _win.WM_SYSKEYDOWN and msg.wParam == _win.VK_SPACE and root.capture_active():
            # Windows opens the system menu on Alt+Space before Qt reports a key press.
            mods = modifier_names(QGuiApplication.queryKeyboardModifiers())
            if "Alt" not in mods:
                mods.insert(1 if "Ctrl" in mods else 0, "Alt")
            root.submit_native_combo("+".join(mods + ["Space"]))
            return True, 0
        if m == _win.WM_SYSCOMMAND and (msg.wParam & 0xFFF0) == _win.SC_KEYMENU and root.suppress_keymenu():
            return True, 0
        if m == _win.WM_ENDSESSION and msg.wParam:
            root.on_end_session()
        return False, 0

    def _do_show(self) -> None:
        self._show_pending = False
        log.info("show-window request from another instance")
        self._root.show_main_window()


class UiRoot:
    """Owns every UI object and routes events. Implements the facade target API."""

    def __init__(self, app: QApplication, controller: Controller, bus: EventBus, facade: QtUiFacade) -> None:
        self.app = app
        self.controller = controller
        self.bus = bus
        self.facade = facade
        self.window = MainWindow(controller)
        self.tray = Tray(controller)
        self.overlay = RecOverlay()
        self.sounds = SoundPlayer()
        self.errors = ErrorDialogs()
        self.download_dialog: Optional[DownloadDialog] = None
        self._state = controller.state
        self._capture_ended_at = 0.0
        self._dictation_ended_at = 0.0
        self._hard_error = False
        self._fatal_at = -1e9
        self._told_about_tray = False
        self._ending_session = False
        self._exiting = False
        self._show_pending = False
        self._accessibility_prompted = False
        self._single_instance = None
        self.native_filter: Optional[NativeFilter] = None
        self.menu_bar: Optional[QMenuBar] = None
        self.relay: Optional[ShowWindowRelay] = None

        self.tray.open_requested.connect(self.show_main_window)
        self.window.hidden_to_tray.connect(self._on_hidden_to_tray)
        self.window.hotkey_edit.capture_state_changed.connect(self._on_capture_state)
        self.window.settings.applied.connect(lambda _cfg: self._apply_ui_prefs())

        self.bridge = EventBridge(self.on_event)
        self._unsubscribe = bus.subscribe(self.bridge)

        if IS_WINDOWS:
            self.native_filter = NativeFilter(self)
            app.installNativeEventFilter(self.native_filter)
            hwnd = int(self.window.winId())  # real HWND so the hidden window gets the broadcast
            _win.allow_message_through_uipi(hwnd, self.native_filter.show_message)
        elif IS_MACOS:
            # No window messages, no UIPI, and no need to materialise the native window:
            # the ping arrives on the single instance's socket.
            self.relay = ShowWindowRelay(self.request_show_window)
            self._wire_show_window_ping()
            self.menu_bar = self._build_mac_menu()

        self._apply_state(self._state, initial=True)

    # -- startup / shutdown -------------------------------------------------
    def start(self, start_minimized: bool) -> None:
        tray_ok = QSystemTrayIcon.isSystemTrayAvailable()
        if tray_ok:
            self.tray.show()
        else:
            log.warning("system tray not available; showing the main window")
        self.facade.bind(self)
        if not start_minimized or not tray_ok:
            self.show_main_window()
        log.info("ui started minimized=%s tray=%s", start_minimized, tray_ok)

    def shutdown(self) -> None:
        try:
            self._unsubscribe()
        except Exception:  # noqa: BLE001
            log.debug("unsubscribe failed", exc_info=True)
        self.facade.unbind()
        if self.native_filter is not None:
            self.app.removeNativeEventFilter(self.native_filter)
        if self._single_instance is not None:
            try:  # stop the socket thread from calling into a dead UI
                self._single_instance.on_show_window = _ignore_show_window
            except Exception:  # noqa: BLE001
                log.debug("clearing on_show_window failed", exc_info=True)
            self._single_instance = None
        self.overlay.hide_overlay()
        self.tray.hide()

    # -- facade target (Qt thread) -----------------------------------------
    def show_error(self, title: str, message: str, details: str = "") -> None:
        self.errors.show(title, message, details, activate=self.may_take_focus())

    def show_download_dialog(self, model: str, size_mb: int) -> None:
        """Repeat calls re-show the existing dialog for the same model (keeping its phase and
        messages); a new dialog is made for another model or after a finished download."""
        dlg = self.download_dialog
        if dlg is None or dlg.model != model or dlg.phase == DownloadDialog.DONE:
            if dlg is not None:
                dlg.close()
                dlg.deleteLater()
            dlg = DownloadDialog(self.controller, model, size_mb)
            self.download_dialog = dlg
        present(dlg, self.may_take_focus())

    def may_take_focus(self) -> bool:
        """False while dictating (LISTENING/PROCESSING) and for FOCUS_GRACE_S afterwards, so a
        flushed last phrase is never typed into one of our dialogs."""
        if self._state in (AppState.LISTENING, AppState.PROCESSING):
            return False
        return time.monotonic() - self._dictation_ended_at >= FOCUS_GRACE_S

    def show_main_window(self) -> None:
        self.window.bring_to_front()
        # Windows: SetForegroundWindow. macOS: NSApp activateIgnoringOtherApps + orderFront,
        # which a menu-bar (accessory) app needs to come forward at all.
        _platform.set_foreground_window(int(self.window.winId()))
        dlg = self.download_dialog
        if dlg is not None and dlg.is_downloading() and not dlg.isVisible():
            present(dlg, activate=False)

    def exit_app(self) -> None:
        if self._exiting:
            return
        self._exiting = True
        log.info("exit_app")
        self.overlay.hide_overlay()
        self.errors.close_all()
        if self.download_dialog is not None:
            self.download_dialog.close()
        self.window.hide()
        self.tray.hide()
        self.app.quit()

    # -- second instance (macOS) ---------------------------------------------
    def _wire_show_window_ping(self) -> None:
        """Route "another launch asked us to show the window" to request_show_window().

        app.macos.instance.FileLockSingleInstance listens on a Unix socket and calls its
        settable ``on_show_window`` callback when a second launch pings it. The controller
        owns that object; it is read through a public attribute only, and a missing hook is
        a warning, never a crash (the second launch then just exits).
        """
        instance = getattr(self.controller, "single_instance", None)
        if instance is None:
            log.warning(
                "show-window ping not wired: the controller exposes no public single_instance; "
                "a second launch will exit without raising the window"
            )
            return
        if not hasattr(instance, "on_show_window"):
            log.warning(
                "show-window ping not wired: %s has no on_show_window hook", type(instance).__name__
            )
            return
        try:
            instance.on_show_window = self.relay.ping.emit  # called from the socket thread
        except Exception:  # noqa: BLE001 (never let wiring break startup)
            log.exception("could not set on_show_window on %s", type(instance).__name__)
            return
        self._single_instance = instance
        log.info("show-window ping wired to %s", type(instance).__name__)

    def request_show_window(self) -> None:
        """Another launch wants the window up (Qt thread). Coalesced, and focus-safe.

        Same shape as the Windows broadcast path: several pings collapse into one show on
        the next event-loop turn. may_take_focus() is honoured so a phrase still being
        flushed can never be typed into our own window; in that case the window is shown
        without activation.
        """
        if self._show_pending:
            return
        self._show_pending = True
        QTimer.singleShot(0, self._do_show_request)

    def _do_show_request(self) -> None:
        self._show_pending = False
        if self.may_take_focus():
            log.info("show-window request from another instance")
            self.show_main_window()
            return
        log.info("show-window request from another instance while dictating; not activating")
        present(self.window, activate=False)
        self.window.setAttribute(Qt.WA_ShowWithoutActivating, False)  # only this one show

    # -- application menu (macOS) --------------------------------------------
    def _build_mac_menu(self) -> QMenuBar:
        """The standard macOS application menu.

        Without one, Qt builds a default app menu whose Quit calls QApplication.quit()
        directly, which would kill the process without stopping dictation, unregistering the
        hotkey or unloading the model. Our QuitRole action replaces it with
        Controller.quit(). A parentless QMenuBar is the application-wide menu bar; an
        accessory app does not draw it, but its key equivalents still work while one of our
        windows is key.
        """
        bar = QMenuBar()
        menu = bar.addMenu(APP_NAME)

        about = QAction(f"About {APP_NAME}", bar)
        about.setMenuRole(QAction.AboutRole)
        about.triggered.connect(self.show_about)
        menu.addAction(about)
        menu.addSeparator()

        close = QAction("Close Window", bar)
        close.setShortcut(QKeySequence.Close)  # Cmd+W
        close.setMenuRole(QAction.NoRole)
        close.triggered.connect(self.window.close)  # hides to the menu bar, keeps running
        menu.addAction(close)

        quit_action = QAction(f"Quit {APP_NAME}", bar)
        quit_action.setShortcut(QKeySequence.Quit)  # Cmd+Q
        quit_action.setMenuRole(QAction.QuitRole)
        quit_action.triggered.connect(self._on_quit_requested)
        menu.addAction(quit_action)
        return bar

    def show_about(self) -> None:
        QMessageBox.about(
            self.window,
            f"About {APP_NAME}",
            f"<b>{APP_NAME}</b> {APP_VERSION}<br>Speak Romanian, type English.",
        )

    def _on_quit_requested(self) -> None:
        log.info("quit requested from the application menu")
        self.controller.quit()

    # -- Accessibility permission (macOS) ------------------------------------
    def _maybe_prompt_accessibility(self) -> None:
        """Explain the Accessibility permission the first time we arm without it.

        Without it macOS drops every synthetic keystroke silently, so dictation would look
        like it works and type nothing. Unknown (no permissions helper available) stays
        quiet: the injector still reports a real InjectionBlockedError if it matters.
        """
        if not IS_MACOS or self._accessibility_prompted:
            return
        if accessibility_granted() is not False:
            return
        self._accessibility_prompted = True
        log.info("Accessibility permission missing; prompting")
        self.errors.show(
            ACCESSIBILITY_TITLE,
            InjectionBlockedError.MESSAGES["no_accessibility"],
            kind="InjectionBlockedError",
            activate=self.may_take_focus(),
            accessibility_help=True,
        )

    # -- native-filter hooks (Windows) ---------------------------------------
    def capture_active(self) -> bool:
        return self.window.hotkey_edit.capturing

    def suppress_keymenu(self) -> bool:
        return self.capture_active() or (time.monotonic() - self._capture_ended_at) < KEYMENU_GRACE_S

    def submit_native_combo(self, text: str) -> None:
        self.window.hotkey_edit.submit_native_combo(text)

    def on_end_session(self) -> None:
        if self._ending_session:
            return
        self._ending_session = True
        log.info("WM_ENDSESSION: shutting down")
        self.controller.quit()

    def _on_capture_state(self, active: bool) -> None:
        if not active:
            self._capture_ended_at = time.monotonic()

    # -- events (Qt thread) -------------------------------------------------
    def on_event(self, ev: AppEvent) -> None:
        try:
            self._route(ev)
        except Exception:  # noqa: BLE001 (one bad event must not kill the UI)
            log.exception("handling %s failed", type(ev).__name__)

    def _route(self, ev: AppEvent) -> None:
        if isinstance(ev, StateChanged):
            self._on_state_changed(ev)
        elif isinstance(ev, EngineLoaded):
            self.window.set_engine_info(ev.info)
        elif isinstance(ev, SegmentProcessed):
            if ev.latency_ms is not None:
                self.window.set_latency(ev.latency_ms)
        elif isinstance(ev, DownloadProgressEvent):
            p = ev.progress
            self.window.set_download_progress(
                100.0 * p.downloaded_bytes / p.total_bytes if p.total_bytes > 0 else None
            )
            if self.download_dialog is not None:
                self.download_dialog.on_progress(p)
        elif isinstance(ev, DownloadFinished):
            if self.download_dialog is not None:
                self.download_dialog.on_finished(ev)
        elif isinstance(ev, HotkeyChanged):
            self.window.sync_from_controller()
            self.tray.refresh()
        elif isinstance(ev, ConfigChanged):
            self.window.on_config_changed(ev.config)
            self.tray.refresh()
            self._apply_ui_prefs()
        elif isinstance(ev, ErrorRaised):
            self._on_error(ev)
        elif isinstance(ev, Notification):
            self.tray.notify(ev.title, ev.message, ev.warning)
        else:
            log.debug("ignored event %r", ev)

    def _on_state_changed(self, ev: StateChanged) -> None:
        old, new = ev.old, ev.new
        dictating = (AppState.LISTENING, AppState.PROCESSING)
        if old in dictating and new not in dictating:
            self._dictation_ended_at = time.monotonic()
        if new != AppState.ERROR or old != AppState.ERROR:
            # Entering ERROR looks "hard" only if a fatal ErrorRaised came with it (core may
            # publish it just before or just after the transition).
            self._hard_error = new == AppState.ERROR and time.monotonic() - self._fatal_at < FATAL_PAIR_S
            self.window.set_hard_error(self._hard_error)
        self.window.set_state(new, ev.detail)
        self._apply_state(new, detail=ev.detail)
        if old == AppState.DISARMED and new != AppState.ERROR:
            self._maybe_prompt_accessibility()  # macOS only; no-op elsewhere
        sounds = self.controller.config.ui.sounds
        if new == AppState.LISTENING and old != AppState.LISTENING and sounds:
            self.sounds.play("start")
        elif old == AppState.LISTENING and new != AppState.LISTENING and sounds:
            self.sounds.play("stop")
        self.window.sync_from_controller()

    def _apply_state(self, state: AppState, initial: bool = False, detail: str = "") -> None:
        self._state = state
        if initial:
            self.window.set_state(state)
        self.tray.set_state(state, detail, self._hard_error)
        if state == AppState.LISTENING and self.controller.config.ui.overlay:
            self.overlay.show_overlay()
        else:
            self.overlay.hide_overlay()

    def _apply_ui_prefs(self) -> None:
        if not self.controller.config.ui.overlay:
            self.overlay.hide_overlay()
        elif self._state == AppState.LISTENING:
            self.overlay.show_overlay()

    def _on_error(self, ev: ErrorRaised) -> None:
        log.info("ErrorRaised kind=%s fatal=%s: %s", ev.kind, ev.fatal, ev.message)
        dlg = self.download_dialog
        if ev.kind in ("DownloadError", "DownloadCancelledError") and dlg is not None and dlg.isVisible():
            pass  # the download dialog shows it (via DownloadFinished) with a Retry button
        else:
            # macOS: an injection blocked for want of Accessibility gets a button straight
            # into the System Settings pane (contracts puts reason=... in details).
            needs_accessibility = (
                IS_MACOS and ev.kind == "InjectionBlockedError" and "no_accessibility" in (ev.details or "")
            )
            self._accessibility_prompted = self._accessibility_prompted or needs_accessibility
            self.errors.show(
                "", ev.message, ev.details, ev.kind,
                activate=self.may_take_focus(), accessibility_help=needs_accessibility,
            )
        if ev.fatal:
            self._fatal_at = time.monotonic()
            self._hard_error = True
            self.window.set_hard_error(True)
            self.tray.set_error()
        self.window.sync_from_controller()

    def _on_hidden_to_tray(self) -> None:
        if not self._told_about_tray and self.tray.tray.isVisible():
            self._told_about_tray = True
            self.tray.notify(APP_NAME, "RoEnSpeak keeps running in the tray. Right-click the icon to quit.")


def run_ui(controller: Controller, bus: EventBus, facade: QtUiFacade, start_minimized: bool) -> int:
    """Build the UI, bind the facade, and run the Qt event loop. Returns app.exec()'s code."""
    app = QApplication.instance() or create_qapplication()
    root = UiRoot(app, controller, bus, facade)
    root.start(start_minimized)
    try:
        return app.exec()
    finally:
        root.shutdown()
