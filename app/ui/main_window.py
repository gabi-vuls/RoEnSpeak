"""Main window: ON/OFF switch, hotkey capture (dictation + prompt adjustment), status line, settings."""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QFont, QHideEvent
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.contracts import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_HOTKEY,
    DEFAULT_PROMPT_HOTKEY,
    IS_MACOS,
    AppError,
    AppState,
    Config,
    Controller,
    EngineInfo,
    HotkeyConflictError,
    HotkeyParseError,
)
from app.core.platform import SUPPORTS_PROMPT_ADJUST
from app.ui.assets import app_icon
from app.ui.hotkey_capture import HotkeyCaptureEdit
from app.ui.settings_panel import ERROR_COLOR, HINT_COLOR, OK_COLOR, SettingsPanel
from app.ui.status import SOFT_ERROR_COLOR, STATE_COLORS, engine_text, latency_text, status_text
from app.ui.widgets import StatusDot, ToggleSwitch

log = logging.getLogger("roenspeak.ui.window")

#: PromptStatus.phase -> what the prompt hint shows while a job runs.
PROMPT_PHASE_TEXT = {
    "copying": "Copying the selection…",
    "rewriting": "Rewriting the prompt with the local model…",
    "pasting": "Pasting the rewritten prompt…",
}

#: How the user's OS names the modifiers a hotkey may use (contracts.Modifier).
MODIFIER_HINT = "Cmd, Ctrl or Option" if IS_MACOS else "Ctrl, Alt or Win"
MODIFIER_LIST = "Cmd/Ctrl/Option" if IS_MACOS else "Ctrl/Alt/Win"


def _card() -> QFrame:
    f = QFrame()
    f.setObjectName("card")
    f.setFrameShape(QFrame.StyledPanel)
    return f


class MainWindow(QWidget):
    """Closing hides to the tray; the app keeps running."""

    hidden_to_tray = Signal()

    def __init__(self, controller: Controller, parent=None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._state: AppState = controller.state
        self._detail = ""
        self._download_pct: Optional[float] = None
        self._hard_error = False
        self._prompt_registered: Optional[bool] = None  # None until the first PromptHotkeyChanged
        self._prompt_phase: Optional[str] = None  # set while a prompt adjustment runs
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.setMinimumWidth(480)

        # Header
        title = QLabel(APP_NAME)
        f = QFont(title.font())
        f.setPointSizeF(f.pointSizeF() * 1.6)
        f.setBold(True)
        title.setFont(f)
        subtitle = QLabel("Speak Romanian, type English")
        subtitle.setStyleSheet(f"color: {HINT_COLOR};")
        self.version_label = QLabel(f"v{APP_VERSION}")
        self.version_label.setStyleSheet(f"color: {HINT_COLOR};")
        head_text = QVBoxLayout()
        head_text.setSpacing(0)
        head_text.addWidget(title)
        head_text.addWidget(subtitle)
        icon = QLabel()
        icon.setPixmap(app_icon().pixmap(40, 40))
        header = QHBoxLayout()
        header.addWidget(icon)
        header.addLayout(head_text)
        header.addStretch(1)
        header.addWidget(self.version_label, 0, Qt.AlignTop)

        # Control card: switch + hotkey + status
        card = _card()
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        arm_label = QLabel("Dictation hotkey")
        bf = QFont(arm_label.font())
        bf.setBold(True)
        bf.setPointSizeF(bf.pointSizeF() * 1.1)
        arm_label.setFont(bf)
        self.arm_hint = QLabel("")
        self.arm_hint.setWordWrap(True)
        self.arm_hint.setStyleSheet(f"color: {HINT_COLOR};")
        arm_text = QVBoxLayout()
        arm_text.setSpacing(0)
        arm_text.addWidget(arm_label)
        arm_text.addWidget(self.arm_hint)
        self.switch = ToggleSwitch()
        self.switch.setToolTip("ON registers the global hotkey. OFF unregisters it and stops any dictation.")
        self.switch.setAccessibleName("Hotkey armed")
        self.switch.clicked.connect(self._on_switch_clicked)
        grid.addLayout(arm_text, 0, 0, 1, 2)
        grid.addWidget(self.switch, 0, 2, Qt.AlignRight)

        grid.addWidget(QLabel("Hotkey"), 1, 0)
        self.hotkey_edit = HotkeyCaptureEdit()
        self.hotkey_edit.setMinimumWidth(170)
        self.hotkey_edit.captured.connect(self._on_hotkey_captured)
        self.hotkey_edit.capture_cancelled.connect(lambda: self._set_hotkey_message("", None))
        self.hotkey_edit.capture_state_changed.connect(self._on_capture_state)
        self.hotkey_edit.unsupported_key.connect(
            lambda k: self._set_hotkey_message(
                f"“{k}” can't be used. Try a letter, digit or F-key with {MODIFIER_LIST}.", ERROR_COLOR
            )
        )
        grid.addWidget(self.hotkey_edit, 1, 1)
        self.reset_hotkey_btn = QPushButton("Reset to default")
        self.reset_hotkey_btn.setToolTip(f"Use {DEFAULT_HOTKEY}")
        self.reset_hotkey_btn.clicked.connect(self._on_reset_hotkey)
        grid.addWidget(self.reset_hotkey_btn, 1, 2)
        self.hotkey_message = QLabel("")
        self.hotkey_message.setWordWrap(True)
        self.hotkey_message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.hotkey_message.hide()
        grid.addWidget(self.hotkey_message, 2, 1, 1, 2)

        # Prompt adjustment hotkey (Windows): select text, press it, the selection is rewritten.
        prompt_label = QLabel("Prompt hotkey")
        prompt_label.setToolTip("Select text in any app and press this combination: the selection is "
                                "rewritten as a clear, structured prompt by the local model (Ollama).")
        self.prompt_hotkey_edit = HotkeyCaptureEdit()
        self.prompt_hotkey_edit.setMinimumWidth(170)
        self.prompt_hotkey_edit.captured.connect(self._on_prompt_hotkey_captured)
        self.prompt_hotkey_edit.capture_cancelled.connect(lambda: self._set_prompt_message("", None))
        self.prompt_hotkey_edit.capture_state_changed.connect(self._on_prompt_capture_state)
        self.prompt_hotkey_edit.unsupported_key.connect(
            lambda k: self._set_prompt_message(
                f"“{k}” can't be used. Try a letter, digit or F-key with {MODIFIER_LIST}.", ERROR_COLOR
            )
        )
        self.reset_prompt_hotkey_btn = QPushButton("Reset to default")
        self.reset_prompt_hotkey_btn.setToolTip(f"Use {DEFAULT_PROMPT_HOTKEY}")
        self.reset_prompt_hotkey_btn.clicked.connect(self._on_reset_prompt_hotkey)
        self.prompt_hint = QLabel("")
        self.prompt_hint.setWordWrap(True)
        self.prompt_hint.setStyleSheet(f"color: {HINT_COLOR};")
        self.prompt_message = QLabel("")
        self.prompt_message.setWordWrap(True)
        self.prompt_message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.prompt_message.hide()
        grid.addWidget(prompt_label, 3, 0)
        grid.addWidget(self.prompt_hotkey_edit, 3, 1)
        grid.addWidget(self.reset_prompt_hotkey_btn, 3, 2)
        grid.addWidget(self.prompt_hint, 4, 1, 1, 2)
        grid.addWidget(self.prompt_message, 5, 1, 1, 2)
        self._prompt_widgets = (prompt_label, self.prompt_hotkey_edit, self.reset_prompt_hotkey_btn,
                                self.prompt_hint)
        if not SUPPORTS_PROMPT_ADJUST:
            for w in self._prompt_widgets:
                w.hide()

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        grid.addWidget(line, 6, 0, 1, 3)

        self.status_dot = StatusDot()
        self.status_label = QLabel("")
        sf = QFont(self.status_label.font())
        sf.setBold(True)
        self.status_label.setFont(sf)
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        status_row = QHBoxLayout()
        status_row.addWidget(self.status_dot, 0, Qt.AlignVCenter)
        status_row.addWidget(self.status_label, 1)
        grid.addLayout(status_row, 7, 0, 1, 3)
        self.engine_label = QLabel("")
        self.engine_label.setStyleSheet(f"color: {HINT_COLOR};")
        self.latency_label = QLabel("")
        self.latency_label.setStyleSheet(f"color: {HINT_COLOR};")
        info_row = QHBoxLayout()
        info_row.setContentsMargins(22, 0, 0, 0)
        info_row.addWidget(self.engine_label, 1)
        info_row.addWidget(self.latency_label, 0, Qt.AlignRight)
        grid.addLayout(info_row, 8, 0, 1, 3)
        grid.setColumnStretch(1, 1)

        # Settings
        settings_title = QLabel("Settings")
        settings_title.setFont(bf)
        self.settings = SettingsPanel(controller)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)
        root.addLayout(header)
        root.addWidget(card)
        root.addWidget(settings_title)
        root.addWidget(self.settings, 1)

        self._hotkey_msg_timer = QTimer(self)
        self._hotkey_msg_timer.setSingleShot(True)
        self._hotkey_msg_timer.timeout.connect(lambda: self._set_hotkey_message("", None))
        self._prompt_msg_timer = QTimer(self)
        self._prompt_msg_timer.setSingleShot(True)
        self._prompt_msg_timer.timeout.connect(lambda: self._set_prompt_message("", None))

        self.sync_from_controller()
        self.set_engine_info(controller.engine_info)
        self.set_latency(controller.last_latency_ms)

    # -- state from core ----------------------------------------------------
    def sync_from_controller(self) -> None:
        """Re-read armed/hotkey/state from the controller (cheap; called after every event)."""
        cfg = self._controller.config
        if self.switch.isChecked() != cfg.armed:
            self.switch.setChecked(cfg.armed)
        self.arm_hint.setText(
            f"Press {cfg.hotkey} in any app to start/stop dictation." if cfg.armed else "Off - the hotkey is not registered."
        )
        if not self.hotkey_edit.capturing:
            self.hotkey_edit.set_display(cfg.hotkey)
        if not self.prompt_hotkey_edit.capturing:
            self.prompt_hotkey_edit.set_display(cfg.prompt_hotkey)
        self._render_prompt_hint()
        self._render_status()

    def set_prompt_hotkey_state(self, registered: bool) -> None:
        self._prompt_registered = registered
        self._render_prompt_hint()

    def set_prompt_status(self, busy: bool, phase: str) -> None:
        self._prompt_phase = phase if busy else None
        self._render_prompt_hint()

    def _render_prompt_hint(self) -> None:
        cfg = self._controller.config
        if self._prompt_phase is not None:
            text = PROMPT_PHASE_TEXT.get(self._prompt_phase, "Adjusting the prompt…")
        elif not cfg.llm.enabled:
            text = "Off - turn on prompt adjustment in Settings."
        elif self._prompt_registered is False:
            text = "The prompt hotkey is not registered. Choose another combination."
        else:
            text = f"Select text in any app, press {cfg.prompt_hotkey}: it is rewritten as a structured prompt."
        self.prompt_hint.setText(text)

    def set_state(self, state: AppState, detail: str = "") -> None:
        self._state = state
        self._detail = detail
        if state != AppState.DOWNLOADING:
            self._download_pct = None
        self._render_status()

    def set_hard_error(self, hard: bool) -> None:
        """True after a fatal ErrorRaised; False renders ERROR as "needs attention"."""
        self._hard_error = hard
        self._render_status()

    def set_download_progress(self, pct: Optional[float]) -> None:
        self._download_pct = pct
        self._render_status()

    def set_engine_info(self, info: Optional[EngineInfo]) -> None:
        self.engine_label.setText(engine_text(info))
        self.engine_label.setToolTip(
            ""
            if info is None
            else f"device={info.device} compute={info.compute_type} driver={info.driver_version or '-'} "
            f"cuBLAS={info.cublas_version or '-'} cuDNN={info.cudnn_version or '-'} load={info.load_ms:.0f} ms"
        )

    def set_latency(self, ms: Optional[float]) -> None:
        self.latency_label.setText(latency_text(ms))

    def on_config_changed(self, cfg: Config) -> None:
        self.settings.on_config_changed(cfg)
        self.sync_from_controller()

    def _render_status(self) -> None:
        hotkey = self._controller.config.hotkey
        self.status_label.setText(
            status_text(
                self._state, detail=self._detail, hotkey=hotkey, download_pct=self._download_pct,
                hard_error=self._hard_error,
            )
        )
        soft = self._state == AppState.ERROR and not self._hard_error
        self.status_dot.set_color(SOFT_ERROR_COLOR if soft else STATE_COLORS[self._state])

    # -- user actions -------------------------------------------------------
    def _on_switch_clicked(self, checked: bool) -> None:
        try:
            self._controller.set_armed(checked)
        except AppError as exc:
            log.warning("set_armed(%s) failed: %s", checked, exc.user_message)
            self._set_hotkey_message(exc.user_message, ERROR_COLOR)
        self.sync_from_controller()

    def _on_capture_state(self, active: bool) -> None:
        # Unregister the global hotkey while recording so the current combo can be captured.
        try:
            self._controller.suspend_hotkey(active)
        except Exception:  # noqa: BLE001 (capture must keep working; core logs the cause)
            log.exception("suspend_hotkey(%s) failed", active)
        if active:
            self._set_hotkey_message(
                f"Press the new combination (with {MODIFIER_HINT}). Esc cancels.", None
            )

    def _apply_hotkey(self, action) -> None:
        try:
            hk = action()
        except (HotkeyParseError, HotkeyConflictError) as exc:
            log.info("hotkey rejected: %s", exc.user_message)
            self._set_hotkey_message(exc.user_message, ERROR_COLOR, sticky=True)
        except AppError as exc:
            log.warning("hotkey change failed: %s (%s)", exc.user_message, exc.details)
            self._set_hotkey_message(exc.user_message, ERROR_COLOR, sticky=True)
        else:
            self.hotkey_edit.set_display(hk.display)
            suffix = "" if self._controller.config.armed else " (applies when the switch is ON)"
            self._set_hotkey_message(f"Hotkey set to {hk.display}{suffix}.", OK_COLOR)
        self.sync_from_controller()

    def _on_prompt_capture_state(self, active: bool) -> None:
        try:
            self._controller.suspend_hotkey(active)  # suspends both hotkeys
        except Exception:  # noqa: BLE001 (capture must keep working; core logs the cause)
            log.exception("suspend_hotkey(%s) failed", active)
        if active:
            self._set_prompt_message(f"Press the new combination (with {MODIFIER_HINT}). Esc cancels.", None)

    def _apply_prompt_hotkey(self, action) -> None:
        try:
            hk = action()
        except AppError as exc:  # HotkeyParseError / HotkeyConflictError / ConfigError
            log.info("prompt hotkey rejected: %s", exc.user_message)
            self._set_prompt_message(exc.user_message, ERROR_COLOR, sticky=True)
        else:
            self.prompt_hotkey_edit.set_display(hk.display)
            suffix = "" if self._controller.config.llm.enabled else " (applies when prompt adjustment is on)"
            self._set_prompt_message(f"Prompt hotkey set to {hk.display}{suffix}.", OK_COLOR)
        self.sync_from_controller()

    def _on_prompt_hotkey_captured(self, text: str) -> None:
        self._apply_prompt_hotkey(lambda: self._controller.set_prompt_hotkey(text))

    def _on_reset_prompt_hotkey(self) -> None:
        self.prompt_hotkey_edit.cancel_capture()
        self._apply_prompt_hotkey(self._controller.reset_prompt_hotkey)

    def _set_prompt_message(self, text: str, color: Optional[str], sticky: bool = False) -> None:
        self._prompt_msg_timer.stop()
        self.prompt_message.setText(text)
        self.prompt_message.setStyleSheet(f"color: {color or HINT_COLOR};")
        self.prompt_message.setVisible(bool(text))
        if text and color == OK_COLOR and not sticky:
            self._prompt_msg_timer.start(4000)

    def _on_hotkey_captured(self, text: str) -> None:
        self._apply_hotkey(lambda: self._controller.set_hotkey(text))

    def _on_reset_hotkey(self) -> None:
        self.hotkey_edit.cancel_capture()
        self._apply_hotkey(self._controller.reset_hotkey)

    def _set_hotkey_message(self, text: str, color: Optional[str], sticky: bool = False) -> None:
        self._hotkey_msg_timer.stop()
        self.hotkey_message.setText(text)
        self.hotkey_message.setStyleSheet(f"color: {color or HINT_COLOR};")
        self.hotkey_message.setVisible(bool(text))
        if text and color == OK_COLOR and not sticky:
            self._hotkey_msg_timer.start(4000)

    # -- window behaviour ---------------------------------------------------
    def bring_to_front(self) -> None:
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()
        self.raise_()
        self.activateWindow()

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        self.hotkey_edit.cancel_capture()  # also resumes the suspended hotkey
        self.prompt_hotkey_edit.cancel_capture()
        super().hideEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.hotkey_edit.cancel_capture()
        self.prompt_hotkey_edit.cancel_capture()
        event.ignore()
        self.hide()
        self.hidden_to_tray.emit()
