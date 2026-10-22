"""Small custom widgets: the large ON/OFF switch and the coloured status dot."""

from __future__ import annotations

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QAbstractButton, QSizePolicy, QWidget


class ToggleSwitch(QAbstractButton):
    """Large pill switch with ON/OFF text inside the track. Checkable; emits toggled/clicked."""

    ON_COLOR = QColor("#1f9d55")
    OFF_COLOR = QColor("#9aa0a6")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._pos = 0.0
        self._anim = QPropertyAnimation(self, b"knob", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.InOutCubic)
        self.toggled.connect(self._animate)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(112, 46)

    def _get_knob(self) -> float:
        return self._pos

    def _set_knob(self, v: float) -> None:
        self._pos = v
        self.update()

    knob = Property(float, _get_knob, _set_knob)

    def _animate(self, checked: bool) -> None:
        self._anim.stop()
        if not self.isVisible():
            self._pos = 1.0 if checked else 0.0
            self.update()
            return
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def setChecked(self, checked: bool) -> None:  # noqa: N802
        if checked != self.isChecked():
            super().setChecked(checked)
        else:
            self._pos = 1.0 if checked else 0.0
            self.update()

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        radius = r.height() / 2
        off, on = self.OFF_COLOR, self.ON_COLOR
        t = self._pos
        track = QColor(
            round(off.red() + (on.red() - off.red()) * t),
            round(off.green() + (on.green() - off.green()) * t),
            round(off.blue() + (on.blue() - off.blue()) * t),
        )
        if not self.isEnabled():
            track.setAlpha(110)
        p.setPen(Qt.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(r, radius, radius)
        if self.hasFocus():
            p.setPen(QPen(self.palette().highlight().color(), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r.adjusted(-1, -1, 1, 1), radius + 1, radius + 1)

        d = r.height() - 8
        x = r.left() + 4 + (r.width() - d - 8) * t
        knob = QRectF(x, r.top() + 4, d, d)

        font = QFont(self.font())
        font.setBold(True)
        font.setPointSizeF(max(9.0, font.pointSizeF() * 1.15))
        p.setFont(font)
        p.setPen(QColor("white"))
        if self.isChecked():
            p.drawText(QRectF(r.left() + 12, r.top(), r.width() - d - 16, r.height()), Qt.AlignVCenter | Qt.AlignLeft, "ON")
        else:
            p.drawText(QRectF(r.left() + d + 8, r.top(), r.width() - d - 20, r.height()), Qt.AlignVCenter | Qt.AlignRight, "OFF")

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 40))
        p.drawEllipse(knob.translated(0, 1))
        p.setBrush(QColor("white"))
        p.drawEllipse(knob)
        p.end()


class StatusDot(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._color = QColor("#8a8f98")
        self.setFixedSize(14, 14)

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(self._color)
        p.drawEllipse(QRectF(1, 1, 12, 12))
        p.end()
