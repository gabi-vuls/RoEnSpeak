"""Hotkey capture field.

Builds the canonical display string (contracts.Hotkey docstring) from Qt key events. The
Windows virtual-key code (QKeyEvent.nativeVirtualKey) is preferred because it is keyboard-
layout independent and is exactly what RegisterHotKey uses; the Qt key code is the fallback
(synthetic events, non-Windows platforms). Validation and registration are NOT done here;
the owner calls Controller.set_hotkey(text) on ``captured``.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QFocusEvent, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QLineEdit

log = logging.getLogger("roenspeak.ui.hotkey")

PROMPT = "Press a key combination…"

#: Windows VK -> canonical key name.
VK_NAMES: dict[int, str] = {
    **{0x41 + i: chr(ord("A") + i) for i in range(26)},
    **{0x30 + i: str(i) for i in range(10)},
    **{0x70 + i: f"F{i + 1}" for i in range(24)},
    **{0x60 + i: f"Numpad{i}" for i in range(10)},
    0x20: "Space", 0x0D: "Enter", 0x09: "Tab", 0x1B: "Esc", 0x08: "Backspace",
    0x2D: "Insert", 0x2E: "Delete", 0x24: "Home", 0x23: "End", 0x21: "PageUp", 0x22: "PageDown",
    0x26: "Up", 0x28: "Down", 0x25: "Left", 0x27: "Right", 0x13: "Pause",
    0xBA: ";", 0xBB: "=", 0xBC: ",", 0xBD: "-", 0xBE: ".", 0xBF: "/", 0xC0: "`",
    0xDB: "[", 0xDC: "\\", 0xDD: "]", 0xDE: "'",
}

#: Windows VKs of modifier keys (generic + left/right variants).
MODIFIER_VKS = frozenset({0x10, 0x11, 0x12, 0x5B, 0x5C, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5})

_K = Qt.Key
#: Qt key -> canonical key name (fallback when there is no native VK).
QT_KEY_NAMES: dict[int, str] = {
    **{int(_K.Key_A) + i: chr(ord("A") + i) for i in range(26)},
    **{int(_K.Key_0) + i: str(i) for i in range(10)},
    **{int(_K.Key_F1) + i: f"F{i + 1}" for i in range(24)},
    int(_K.Key_Space): "Space", int(_K.Key_Return): "Enter", int(_K.Key_Enter): "Enter",
    int(_K.Key_Tab): "Tab", int(_K.Key_Backtab): "Tab", int(_K.Key_Escape): "Esc",
    int(_K.Key_Backspace): "Backspace", int(_K.Key_Insert): "Insert", int(_K.Key_Delete): "Delete",
    int(_K.Key_Home): "Home", int(_K.Key_End): "End", int(_K.Key_PageUp): "PageUp",
    int(_K.Key_PageDown): "PageDown", int(_K.Key_Up): "Up", int(_K.Key_Down): "Down",
    int(_K.Key_Left): "Left", int(_K.Key_Right): "Right", int(_K.Key_Pause): "Pause",
    int(_K.Key_Semicolon): ";", int(_K.Key_Equal): "=", int(_K.Key_Comma): ",", int(_K.Key_Minus): "-",
    int(_K.Key_Period): ".", int(_K.Key_Slash): "/", int(_K.Key_QuoteLeft): "`",
    int(_K.Key_BracketLeft): "[", int(_K.Key_Backslash): "\\", int(_K.Key_BracketRight): "]",
    int(_K.Key_Apostrophe): "'",
}

QT_MODIFIER_KEYS = frozenset(
    int(k) for k in (_K.Key_Control, _K.Key_Alt, _K.Key_Shift, _K.Key_Meta, _K.Key_AltGr, _K.Key_Super_L, _K.Key_Super_R)
)


def modifier_names(mods: Qt.KeyboardModifier) -> list[str]:
    """Canonical modifier names in MODIFIER_ORDER (Ctrl, Alt, Shift, Win)."""
    out = []
    if mods & Qt.ControlModifier:
        out.append("Ctrl")
    if mods & Qt.AltModifier:
        out.append("Alt")
    if mods & Qt.ShiftModifier:
        out.append("Shift")
    if mods & Qt.MetaModifier:
        out.append("Win")
    return out


_MOD_FOR_QT_KEY = {
    int(_K.Key_Control): Qt.ControlModifier, int(_K.Key_Alt): Qt.AltModifier, int(_K.Key_AltGr): Qt.AltModifier,
    int(_K.Key_Shift): Qt.ShiftModifier, int(_K.Key_Meta): Qt.MetaModifier, int(_K.Key_Super_L): Qt.MetaModifier,
    int(_K.Key_Super_R): Qt.MetaModifier,
}


def held_modifiers(event: QKeyEvent, pressed: bool) -> list[str]:
    """Modifiers held after a modifier key press/release (Qt does not consistently include
    the pressed modifier itself in modifiers())."""
    mods = event.modifiers()
    own = _MOD_FOR_QT_KEY.get(int(event.key()))
    if own is not None:
        mods = (mods | own) if pressed else (mods & ~own)
    return modifier_names(mods)


def is_modifier_event(event: QKeyEvent) -> bool:
    vk = event.nativeVirtualKey()
    return int(event.key()) in QT_MODIFIER_KEYS or (vk != 0 and vk in MODIFIER_VKS)


def key_name(event: QKeyEvent) -> Optional[str]:
    """Canonical non-modifier key name for the event, or None if the key is not supported."""
    vk = event.nativeVirtualKey()
    if vk and vk in VK_NAMES:
        return VK_NAMES[vk]
    key = int(event.key())
    if key in QT_KEY_NAMES:
        name = QT_KEY_NAMES[key]
        if event.modifiers() & Qt.KeypadModifier and name.isdigit():
            return f"Numpad{name}"
        return name
    return None


def combo_text(event: QKeyEvent) -> Optional[str]:
    """Display string ("Ctrl+Alt+F9") for a non-modifier key press, or None if unsupported."""
    name = key_name(event)
    if name is None:
        return None
    return "+".join(modifier_names(event.modifiers()) + [name])


class HotkeyCaptureEdit(QLineEdit):
    """Read-only field; click (or Enter/Space while focused) to capture a combination.

    captured(text) is emitted with the display string; the owner validates it and calls
    set_display() with the resulting (or previous) hotkey. Esc or focus loss cancels.
    capture_state_changed(active) lets the owner suppress Alt/system-menu side effects.
    """

    captured = Signal(str)
    capture_cancelled = Signal()
    capture_state_changed = Signal(bool)
    unsupported_key = Signal(str)

    def __init__(self, parent=None) -> None:
        self._display = ""
        self._capturing = False
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAlignment(Qt.AlignCenter)
        self.setContextMenuPolicy(Qt.NoContextMenu)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click, then press the new key combination (Esc cancels)")

    # -- public API ---------------------------------------------------------
    @property
    def capturing(self) -> bool:
        return self._capturing

    def set_display(self, text: str) -> None:
        self._display = text
        if not self._capturing:
            self.setText(text)

    def start_capture(self) -> None:
        if self._capturing:
            return
        self._capturing = True
        self.setText(PROMPT)
        self.setFocus(Qt.MouseFocusReason)
        self.capture_state_changed.emit(True)

    def cancel_capture(self) -> None:
        if not self._capturing:
            return
        self._end_capture()
        self.capture_cancelled.emit()

    def submit_native_combo(self, text: str) -> None:
        """Deliver a combination seen only at the Win32 level (e.g. Alt+Space, which Windows
        turns into the system menu before Qt sees a key event)."""
        if self._capturing:
            self._finish(text)

    # -- internals ----------------------------------------------------------
    def _end_capture(self) -> None:
        """Single exit path for every way a capture ends (commit, Esc, focus-out, hide)."""
        self._capturing = False
        self.setText(self._display)
        self.capture_state_changed.emit(False)

    def _finish(self, text: str) -> None:
        self._end_capture()
        log.debug("hotkey captured: %s", text)
        self.captured.emit(text)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.LeftButton:
            self.start_capture()
            event.accept()
            return
        super().mousePressEvent(event)

    def event(self, event: QEvent) -> bool:  # noqa: D401
        # While capturing, take every key first: no shortcuts, no Tab focus traversal.
        if self._capturing and event.type() in (QEvent.ShortcutOverride, QEvent.KeyPress):
            if event.type() == QEvent.ShortcutOverride:
                event.accept()
                return True
            self.keyPressEvent(event)
            return True
        return super().event(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if not self._capturing:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space) and not event.modifiers():
                self.start_capture()
                event.accept()
                return
            super().keyPressEvent(event)
            return
        event.accept()
        if event.isAutoRepeat():
            return
        if is_modifier_event(event):
            held = held_modifiers(event, pressed=True)
            self.setText("+".join(held) + "+…" if held else PROMPT)
            return
        mods = modifier_names(event.modifiers())
        if event.key() == Qt.Key_Escape and not mods:
            self.cancel_capture()
            return
        text = combo_text(event)
        if text is None:
            self.unsupported_key.emit(event.text() or f"key code {int(event.key())}")
            return
        self._finish(text)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if self._capturing:
            event.accept()
            if is_modifier_event(event):
                held = held_modifiers(event, pressed=False)
                self.setText("+".join(held) + "+…" if held else PROMPT)
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:  # noqa: N802
        if self._capturing and event.reason() not in (Qt.PopupFocusReason, Qt.MenuBarFocusReason):
            self.cancel_capture()
        super().focusOutEvent(event)
