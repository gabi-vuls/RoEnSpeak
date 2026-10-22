"""QPainter artwork for the app and tray icons.

Used by ``gen_icons`` to write the committed PNG/ICO files, and by ``assets`` as a runtime
fallback if an icon file is missing. Requires a QGuiApplication.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen

#: Tray icon states and their fill colours.
TRAY_COLORS: dict[str, str] = {
    "disarmed": "#8a8f98",  # grey
    "ready": "#1f9d55",  # green
    "listening": "#e02424",  # red
    "processing": "#f59e0b",  # amber
    "loading": "#2f7fd8",  # blue (downloading / loading model)
    "error": "#8b1111",  # dark red with "!"
}
TRAY_STATES = tuple(TRAY_COLORS)
TRAY_SIZES = (16, 32, 64)
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)


def _mic_path(cx: float, top: float, s: float) -> tuple[QPainterPath, QPainterPath]:
    """Return (capsule, holder) paths for a microphone glyph of nominal size s."""
    body_w = 0.36 * s
    body_h = 0.46 * s
    capsule = QPainterPath()
    capsule.addRoundedRect(QRectF(cx - body_w / 2, top, body_w, body_h), body_w / 2, body_w / 2)

    holder = QPainterPath()
    arc_w = 0.58 * s
    arc_top = top + 0.16 * s
    arc_rect = QRectF(cx - arc_w / 2, arc_top, arc_w, 0.5 * s)
    holder.arcMoveTo(arc_rect, 180)
    holder.arcTo(arc_rect, 180, 180)
    stem_top = arc_top + 0.5 * s
    stem_bottom = top + 0.88 * s
    holder.moveTo(cx, stem_top)
    holder.lineTo(cx, stem_bottom)
    holder.moveTo(cx - 0.2 * s, stem_bottom)
    holder.lineTo(cx + 0.2 * s, stem_bottom)
    return capsule, holder


def draw_mic(p: QPainter, rect: QRectF, color: QColor) -> None:
    s = min(rect.width(), rect.height())
    cx = rect.center().x()
    top = rect.top() + (rect.height() - s) / 2 + 0.04 * s
    capsule, holder = _mic_path(cx, top, s)
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(color))
    p.drawPath(capsule)
    pen = QPen(color, max(1.0, 0.085 * s))
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawPath(holder)


def draw_exclamation(p: QPainter, rect: QRectF, color: QColor) -> None:
    s = min(rect.width(), rect.height())
    cx = rect.center().x()
    top = rect.top() + (rect.height() - s) / 2
    w = 0.2 * s
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(color))
    bar = QPainterPath()
    bar.moveTo(cx - w / 2, top + 0.06 * s)
    bar.lineTo(cx + w / 2, top + 0.06 * s)
    bar.lineTo(cx + w * 0.32, top + 0.66 * s)
    bar.lineTo(cx - w * 0.32, top + 0.66 * s)
    bar.closeSubpath()
    p.drawPath(bar)
    p.drawEllipse(QPointF(cx, top + 0.85 * s), w * 0.55, w * 0.55)


def render_tray_icon(state: str, size: int) -> QImage:
    """Coloured disc with a white mic (or "!" for error)."""
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    base = QColor(TRAY_COLORS[state])
    margin = max(0.5, size * 0.03)
    disc = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
    grad = QLinearGradient(disc.topLeft(), disc.bottomLeft())
    grad.setColorAt(0.0, base.lighter(118))
    grad.setColorAt(1.0, base.darker(112))
    p.setPen(QPen(base.darker(150), max(1.0, size / 32)) if size >= 24 else Qt.NoPen)
    p.setBrush(QBrush(grad))
    p.drawEllipse(disc)
    pad = 0.13 if size <= 16 else 0.18  # bigger glyph at 16 px so it stays legible
    inner = disc.adjusted(size * pad, size * pad, -size * pad, -size * pad)
    if state == "error":
        draw_exclamation(p, inner, QColor("white"))
    else:
        draw_mic(p, inner, QColor("white"))
    p.end()
    return img


def render_app_icon(size: int) -> QImage:
    """Rounded square with an indigo-blue gradient and a white mic plus a small RO->EN accent."""
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    r = QRectF(0, 0, size, size).adjusted(size * 0.04, size * 0.04, -size * 0.04, -size * 0.04)
    grad = QLinearGradient(r.topLeft(), r.bottomRight())
    grad.setColorAt(0.0, QColor("#3b82f6"))
    grad.setColorAt(1.0, QColor("#4338ca"))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(grad))
    radius = r.width() * 0.22
    p.drawRoundedRect(r, radius, radius)
    mic_rect = r.adjusted(r.width() * 0.2, r.height() * 0.16, -r.width() * 0.2, -r.height() * 0.16)
    draw_mic(p, mic_rect, QColor("white"))
    if size >= 32:
        # Small red "speaking" dot top-right.
        d = r.width() * 0.17
        p.setBrush(QBrush(QColor("#ef4444")))
        p.setPen(QPen(QColor("white"), max(1.0, size / 48)))
        p.drawEllipse(QRectF(r.right() - d * 1.45, r.top() + d * 0.45, d, d))
    p.end()
    return img


# ---------------------------------------------------------------------------
# macOS template (mask) artwork
# ---------------------------------------------------------------------------
#
# A macOS status item takes a *template* image: only its alpha channel is used, and the
# system paints it black on a light menu bar and white on a dark one (and inverted while the
# menu is open). That erases the colour-coded state language of the tray icons above, so the
# template set distinguishes the states by silhouette instead:
#
#   disarmed    microphone with a slash through it      (hotkey off)
#   ready       plain microphone
#   listening   microphone with sound waves
#   processing  three dots                              (thinking)
#   loading     download arrow                          (downloading / loading the model)
#   error       exclamation mark
#
# Each glyph has a clearly different outline, so they are still told apart at 16 px in a
# monochrome menu bar.

#: Sizes written for the macOS status item: 1x, 2x (Retina) and 4x.
TRAY_TEMPLATE_SIZES = (16, 32, 64)


def draw_waves(p: QPainter, rect: QRectF, color: QColor) -> None:
    """Two arcs either side of the centre, like radiating sound."""
    s = min(rect.width(), rect.height())
    pen = QPen(color, max(1.1, 0.075 * s))
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    cx, cy = rect.center().x(), rect.center().y()
    w = rect.width() * 0.92
    h = rect.height() * 0.82
    box = QRectF(cx - w / 2, cy - h / 2, w, h)
    path = QPainterPath()
    path.arcMoveTo(box, 36)  # right side, 36° to -36°
    path.arcTo(box, 36, -72)
    path.arcMoveTo(box, 144)  # left side
    path.arcTo(box, 144, 72)
    p.drawPath(path)


def draw_slash(p: QPainter, rect: QRectF, color: QColor) -> None:
    """A diagonal line with a cleared gap around it, so it reads over the glyph below."""
    s = min(rect.width(), rect.height())
    inset = 0.06 * s
    a = QPointF(rect.left() + inset, rect.top() + inset)
    b = QPointF(rect.right() - inset, rect.bottom() - inset)
    gap = QPen(QColor(0, 0, 0, 255), max(2.0, 0.17 * s))
    gap.setCapStyle(Qt.RoundCap)
    p.save()
    p.setCompositionMode(QPainter.CompositionMode_Clear)
    p.setPen(gap)
    p.drawLine(a, b)
    p.restore()
    line = QPen(color, max(1.2, 0.095 * s))
    line.setCapStyle(Qt.RoundCap)
    p.setPen(line)
    p.setBrush(Qt.NoBrush)
    p.drawLine(a, b)


def draw_dots(p: QPainter, rect: QRectF, color: QColor) -> None:
    """Three dots in a row ("working on it")."""
    s = min(rect.width(), rect.height())
    r = 0.105 * s
    cy = rect.center().y()
    cx = rect.center().x()
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(color))
    for dx in (-0.3 * s, 0.0, 0.3 * s):
        p.drawEllipse(QPointF(cx + dx, cy), r, r)


def draw_download(p: QPainter, rect: QRectF, color: QColor) -> None:
    """Arrow pointing down onto a baseline."""
    s = min(rect.width(), rect.height())
    cx = rect.center().x()
    top = rect.top() + (rect.height() - s) / 2
    shaft_w = 0.16 * s
    head_w = 0.46 * s
    head_top = top + 0.38 * s
    tip_y = top + 0.68 * s
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(color))
    arrow = QPainterPath()
    arrow.addRect(QRectF(cx - shaft_w / 2, top + 0.06 * s, shaft_w, head_top - top - 0.06 * s))
    head = QPainterPath()
    head.moveTo(cx - head_w / 2, head_top)
    head.lineTo(cx + head_w / 2, head_top)
    head.lineTo(cx, tip_y)
    head.closeSubpath()
    p.drawPath(arrow.united(head))
    bar_h = max(1.2, 0.1 * s)
    p.drawRect(QRectF(cx - head_w / 2, top + 0.84 * s, head_w, bar_h))


def render_tray_template(state: str, size: int) -> QImage:
    """Monochrome, transparent-background silhouette for a macOS status item.

    Only the alpha channel matters (QIcon.setIsMask(True) / NSImage.isTemplate), so
    everything is drawn in opaque black.
    """
    if state not in TRAY_COLORS:
        raise ValueError(f"unknown tray state: {state!r}")
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    ink = QColor(0, 0, 0, 255)
    pad = size * 0.09
    box = QRectF(0, 0, size, size).adjusted(pad, pad, -pad, -pad)
    if state == "error":
        draw_exclamation(p, box, ink)
    elif state == "processing":
        draw_dots(p, box, ink)
    elif state == "loading":
        draw_download(p, box, ink)
    else:
        narrow = box.width() * (0.17 if state == "listening" else 0.07)
        draw_mic(p, box.adjusted(narrow, 0.0, -narrow, 0.0), ink)
        if state == "listening":
            draw_waves(p, box, ink)
        elif state == "disarmed":
            draw_slash(p, box, ink)
    p.end()
    return img
