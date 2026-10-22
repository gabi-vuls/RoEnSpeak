"""Small always-on-top, click-through "● REC" overlay shown while LISTENING.

It must never take focus: dictated text is injected into the foreground window. Qt flags
(Tool | WindowDoesNotAcceptFocus | WindowTransparentForInput, WA_ShowWithoutActivating) are
backed up by setting WS_EX_NOACTIVATE | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW
directly on the HWND before the first show (and re-asserted after every show).
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget

from app.ui import _win

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
        self.exstyle = 0

    def _ensure_native(self) -> None:
        if not self._native_ready:
            hwnd = int(self.winId())  # creates the native window without showing it
            self.exstyle = _win.make_click_through_noactivate(hwnd)
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
        self.exstyle = _win.make_click_through_noactivate(int(self.winId()))
        self._blink.start()
        log.debug("overlay shown exstyle=0x%08x", self.exstyle)

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
