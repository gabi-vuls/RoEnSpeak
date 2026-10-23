"""System tray icon with an Open / Arm-Disarm / Quit menu and balloon notifications.

Windows: a state-coloured icon plus a tooltip with the status line.

macOS: a menu-bar status item, which works differently in three ways.
  * It takes a *template* icon, so the system can invert it for light/dark menu bars and for
    the highlighted (menu-open) state. A template is alpha-only, which throws away the
    colour-coded state language - so the macOS artwork distinguishes the states by
    silhouette instead (see icon_art.render_tray_template).
  * It has no tooltip, so the status line that Windows shows on hover is the first,
    disabled item of the menu ("Listening · Alt+Q").
  * A click opens the menu instead of activating the item, so "Open RoEnSpeak" in the menu
    is the complete route to the window and Trigger/DoubleClick are ignored.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from app.contracts import APP_NAME, IS_MACOS, AppError, AppState, Controller
from app.ui.assets import tray_icon, tray_template_icon
from app.ui.status import STATE_LABELS, TRAY_STATE

log = logging.getLogger("roenspeak.ui.tray")


class Tray(QObject):
    open_requested = Signal()

    def __init__(self, controller: Controller, parent=None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._state = controller.state
        self._error_flag = False
        self._detail = ""
        self.icon_state = "disarmed"
        self.status_line = ""

        self.menu = QMenu()
        # macOS status items have no tooltip: the status line lives at the top of the menu.
        self.act_status = QAction("", self.menu)
        self.act_status.setEnabled(False)
        self.act_open = QAction("Open RoEnSpeak", self.menu)
        f = self.act_open.font()
        f.setBold(True)
        self.act_open.setFont(f)
        self.act_open.triggered.connect(self.open_requested.emit)
        self.act_arm = QAction("Hotkey armed", self.menu)
        self.act_arm.setCheckable(True)
        self.act_arm.triggered.connect(self._on_arm_triggered)
        self.act_quit = QAction("Quit", self.menu)
        self.act_quit.triggered.connect(self._on_quit)
        if IS_MACOS:
            self.menu.addAction(self.act_status)
            self.menu.addSeparator()
        self.menu.addAction(self.act_open)
        self.menu.addAction(self.act_arm)
        self.menu.addSeparator()
        self.menu.addAction(self.act_quit)

        self.tray = QSystemTrayIcon(self)
        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self._on_activated)
        self.refresh()

    @property
    def available(self) -> bool:
        return QSystemTrayIcon.isSystemTrayAvailable()

    def show(self) -> None:
        self.tray.show()

    def hide(self) -> None:
        self.tray.hide()

    def set_state(self, state: AppState, detail: str = "", hard_error: bool = False) -> None:
        """hard_error: a fatal ErrorRaised accompanies this ERROR state (red "!" icon)."""
        self._state = state
        self._detail = detail
        self._error_flag = hard_error and state == AppState.ERROR
        self.refresh()

    def set_error(self) -> None:
        """Show the error icon until the next state change away from ERROR."""
        self._error_flag = True
        self.refresh()

    def refresh(self) -> None:
        cfg = self._controller.config
        if self._error_flag:
            self.icon_state = "error"
        elif self._state == AppState.ERROR:
            self.icon_state = "disarmed"  # non-fatal ERROR (e.g. model not downloaded): needs attention
        else:
            self.icon_state = TRAY_STATE[self._state]
        self.tray.setIcon(tray_template_icon(self.icon_state) if IS_MACOS else tray_icon(self.icon_state))
        label = STATE_LABELS[self._state]
        if self._state == AppState.ERROR and not self._error_flag and self._detail:
            label = self._detail
        if self._error_flag and self._state != AppState.ERROR:
            label = f"{label} (error)"
        hk = f" · {cfg.hotkey}" if cfg.armed else " · hotkey off"
        self.status_line = f"{label}{hk}"
        self.tray.setToolTip(f"{APP_NAME} - {self.status_line}")  # ignored on macOS
        self.act_status.setText(self.status_line)
        self.act_arm.setChecked(cfg.armed)
        self.act_arm.setText(f"Hotkey armed ({cfg.hotkey})")

    def notify(self, title: str, message: str, warning: bool = False) -> None:
        """Balloon / Notification Centre banner. Qt routes it to the macOS notification
        centre, which silently drops it for a process without a bundle identifier (a dev
        run). That is normal, not an error, so it is only logged at debug level - the same
        text always reaches the log and, for errors, a dialog."""
        if not self.tray.isVisible() or not QSystemTrayIcon.supportsMessages():
            log.info("notification (no tray): %s - %s", title, message)
            return
        icon = QSystemTrayIcon.Warning if warning else QSystemTrayIcon.Information
        self.tray.showMessage(title, message, icon, 6000)
        if IS_MACOS:
            log.debug("notification sent to the notification centre: %s - %s", title, message)

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if IS_MACOS:
            # A click on a status item with a menu opens the menu; opening the window too
            # would steal focus behind the user's back. The menu's first action does it.
            return
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self.open_requested.emit()

    def _on_arm_triggered(self, checked: bool) -> None:
        try:
            self._controller.set_armed(checked)
        except AppError as exc:
            log.warning("set_armed(%s) from tray failed: %s", checked, exc.user_message)
            self.notify(APP_NAME, exc.user_message, warning=True)
        self.refresh()

    def _on_quit(self) -> None:
        log.info("quit requested from tray")
        self._controller.quit()
