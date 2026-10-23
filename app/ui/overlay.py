"""Small always-on-top, click-through "● REC" overlay shown while LISTENING.

It must never take focus: dictated text is injected into the foreground window. The Qt flags
(Tool | WindowDoesNotAcceptFocus | WindowTransparentForInput, WA_ShowWithoutActivating) are
identical on both platforms but are not enough on either, so a native backstop is applied to
the real window before the first show and re-asserted after every show
(app.ui._platform.make_click_through_noactivate):

* Windows: WS_EX_NOACTIVATE | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW + topmost.
* macOS:   ignoresMouseEvents, a window level above every other app's windows,
  hidesOnDeactivate = NO (Qt.Tool maps to an NSPanel that would otherwise disappear the
  moment the target app takes focus - exactly when we are dictating) and a collection
  behaviour that joins all Spaces.

Placement uses availableGeometry(), which already excludes the macOS menu bar and the
Windows taskbar, so the overlay never sits under either.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget

from app.ui import _platform

log = logging.getLogger("roenspeak.ui.overlay")

TEXT = "REC"
MARGIN_TOP = 14


class RecOverlay(QWidget):
    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.Tool
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.WindowTransparentForInput
            | Qt.WindowDoesNotAcceptFocus
            | Qt.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setWindowTitle("RoEnSpeak REC")
        self._font = QFont(self.font())
        self._font.setBold(True)
        self._font.setPointSizeF(10.5)
        fm = QFontMetrics(self._font)
        self.setFixedSize(fm.horizontalAdvance(TEXT) + 46, fm.height() + 14)
        self._dot_on = True
        self._blink = QTimer(self)
        self._blink.setInterval(650)
        self._blink.timeout.connect(self._toggle_dot)
        self._native_ready = False
        #: Windows: the resulting WS_EX_* style. macOS: the resulting NSWindow level.
        self.exstyle = 0

    def _ensure_native(self) -> None:
        if not self._native_ready:
            wid = int(self.winId())  # creates the native window (HWND/NSView) without showing it
            self.exstyle = _platform.make_click_through_noactivate(wid)
            self._native_ready = True

    def _place(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        self.move(geo.center().x() - self.width() // 2, geo.top() + MARGIN_TOP)

    def show_overlay(self) -> None:
        if self.isVisible():
            return
        self._ensure_native()
        self._place()
        self._dot_on = True
        self.show()
        # Re-assert after show(): both platforms reset parts of it when the window is mapped.
        self._reassert_native()
        # ...and once more on the next event-loop turn, because Qt orders the window front
        # asynchronously on macOS and puts its own (floating) level back when it does.
        QTimer.singleShot(0, self._reassert_native)
        self._blink.start()

    def _reassert_native(self) -> None:
        if not self.isVisible() and self._native_ready:
            return
        style = _platform.make_click_through_noactivate(int(self.winId()))
        if style != self.exstyle:
            log.debug("overlay native_style=0x%08x", style)
        self.exstyle = style

    def hide_overlay(self) -> None:
        self._blink.stop()
        if self.isVisible():
            self.hide()

    def _toggle_dot(self) -> None:
        self._dot_on = not self._dot_on
        self.update()

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QColor(255, 255, 255, 60))
        p.setBrush(QColor(20, 20, 24, 215))
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        d = 10.0
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(239, 68, 68, 255 if self._dot_on else 90))
        p.drawEllipse(QRectF(14, (r.height() - d) / 2 + 0.5, d, d))
        p.setFont(self._font)
        p.setPen(QColor("white"))
        p.drawText(QRectF(30, 0, r.width() - 38, r.height()), Qt.AlignVCenter | Qt.AlignLeft, TEXT)
        p.end()
