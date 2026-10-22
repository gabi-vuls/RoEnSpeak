"""Hotkey capture field.

Builds the canonical display string (contracts.Hotkey docstring) from Qt key events.
Validation and registration are NOT done here; the owner calls Controller.set_hotkey(text)
on ``captured``.

Which key code do we trust?
---------------------------
``QKeyEvent.nativeVirtualKey()`` is preferred on both platforms because it is keyboard-layout
independent and is exactly what the OS hotkey API registers - but the two platforms put
different numbers in it, and the tables collide:

* Windows: a virtual-key code (VK_RETURN = 0x0D), looked up in VK_NAMES.
* macOS:   a Carbon virtual key code (kVK_Return = 0x24), looked up in MAC_KEY_NAMES.
  0x24 is VK_HOME on Windows, so reading a macOS event with VK_NAMES turns Return into
  "Home". Hence the hard platform branch in key_name()/is_modifier_event().

Carbon key codes describe the *physical* key (ANSI position), which is what
``RegisterEventHotKey`` takes and what app.macos.hotkey_parse maps the canonical names onto,
so a captured combination always fires on the very key the user pressed. On a non-ANSI
layout the name can therefore differ from the key cap (the AZERTY "A" key is kVK_ANSI_Q);
naming it after the character instead would register a *different* physical key, which is
worse. ``Qt.Key_*`` (QT_KEY_NAMES) is the fallback for synthetic events (no native code) and
for keys Carbon has no code for (e.g. Pause).

Ctrl and Cmd
------------
Qt swaps them on macOS: ``Qt.ControlModifier`` is Command and ``Qt.MetaModifier`` is Control,
unless ``Qt.AA_MacDontSwapCtrlAndMeta`` is set. We deliberately do **not** set that attribute
(see app.ui.app.create_qapplication): it is process-wide, and unsetting the swap would also
move every standard shortcut off the Command key - QKeySequence.Copy would become Control+C
and the application menu's Quit would become Control+Q, which is wrong on a Mac. Instead
``modifier_names()`` compensates locally: ControlModifier -> "Cmd", MetaModifier -> "Ctrl",
AltModifier -> "Alt" (Option), ShiftModifier -> "Shift", matching contracts.Modifier.

The emitted string keeps the Windows slot order (Cmd where Ctrl was, Ctrl where Win was), so
macOS combinations read as "Cmd+Shift+D". It does not have to be in MODIFIER_ORDER: the
owner feeds it to the platform HotkeyParser, which accepts any order and hands back the
canonical display string that is then shown in the field.

macOS eats some combinations system-wide (Cmd+Space, Cmd+Tab, Ctrl+Up, Cmd+Option+Esc...).
Those never reach a widget, so capture simply shows nothing and keeps waiting - the user
presses something else, or Esc to cancel. There is no Windows-style native filter to work
around it (that one exists only for Alt+Space, which Windows turns into a system menu).
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QFocusEvent, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QLineEdit

from app.contracts import IS_MACOS

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

#: Carbon virtual key code (kVK_*, Carbon/HIToolbox Events.h) -> canonical key name.
#: These are physical ANSI positions, identical on every keyboard layout.
MAC_KEY_NAMES: dict[int, str] = {
    0x00: "A", 0x0B: "B", 0x08: "C", 0x02: "D", 0x0E: "E", 0x03: "F", 0x05: "G", 0x04: "H",
    0x22: "I", 0x26: "J", 0x28: "K", 0x25: "L", 0x2E: "M", 0x2D: "N", 0x1F: "O", 0x23: "P",
    0x0C: "Q", 0x0F: "R", 0x01: "S", 0x11: "T", 0x20: "U", 0x09: "V", 0x0D: "W", 0x07: "X",
    0x10: "Y", 0x06: "Z",
    0x1D: "0", 0x12: "1", 0x13: "2", 0x14: "3", 0x15: "4", 0x17: "5", 0x16: "6", 0x1A: "7",
    0x1C: "8", 0x19: "9",
    0x7A: "F1", 0x78: "F2", 0x63: "F3", 0x76: "F4", 0x60: "F5", 0x61: "F6", 0x62: "F7",
    0x64: "F8", 0x65: "F9", 0x6D: "F10", 0x67: "F11", 0x6F: "F12", 0x69: "F13", 0x6B: "F14",
    0x71: "F15", 0x6A: "F16", 0x40: "F17", 0x4F: "F18", 0x50: "F19", 0x5A: "F20",
    0x31: "Space", 0x24: "Enter", 0x4C: "Enter", 0x30: "Tab", 0x35: "Esc", 0x33: "Backspace",
    0x72: "Insert", 0x75: "Delete", 0x73: "Home", 0x77: "End", 0x74: "PageUp", 0x79: "PageDown",
    0x7E: "Up", 0x7D: "Down", 0x7B: "Left", 0x7C: "Right",
    0x29: ";", 0x18: "=", 0x2B: ",", 0x1B: "-", 0x2F: ".", 0x2C: "/", 0x32: "`",
    0x21: "[", 0x2A: "\\", 0x1E: "]", 0x27: "'",
    0x52: "Numpad0", 0x53: "Numpad1", 0x54: "Numpad2", 0x55: "Numpad3", 0x56: "Numpad4",
    0x57: "Numpad5", 0x58: "Numpad6", 0x59: "Numpad7", 0x5B: "Numpad8", 0x5C: "Numpad9",
}

#: Carbon codes of the modifier keys (left and right Command/Shift/Option/Control, Caps
#: Lock, Fn). Needed because they collide with the Windows table: kVK_ANSI_Keypad8 (0x5B)
#: is VK_LWIN, so the Windows MODIFIER_VKS set would call the numpad "8" key a modifier.
MAC_MODIFIER_KEYCODES = frozenset({0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F})

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
    """Canonical contracts.Modifier names for a Qt modifier mask.

    Windows: Ctrl, Alt, Shift, Win. macOS: Cmd, Alt (Option), Shift, Ctrl - Qt swaps Ctrl and
    Meta there (ControlModifier is Command, MetaModifier is Control), which is compensated
    here rather than with Qt.AA_MacDontSwapCtrlAndMeta; see the module docstring.
    """
    out = []
    if mods & Qt.ControlModifier:
        out.append("Cmd" if IS_MACOS else "Ctrl")
    if mods & Qt.AltModifier:
        out.append("Alt")
    if mods & Qt.ShiftModifier:
        out.append("Shift")
    if mods & Qt.MetaModifier:
        out.append("Ctrl" if IS_MACOS else "Win")
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
    """True for a press/release of a modifier key itself (Shift, Ctrl, Alt, Win/Cmd, Fn)."""
    vk = int(event.nativeVirtualKey())
    if int(event.key()) in QT_MODIFIER_KEYS:
        return True
    native = MAC_MODIFIER_KEYCODES if IS_MACOS else MODIFIER_VKS
    return vk != 0 and vk in native


def _qt_key_name(event: QKeyEvent) -> Optional[str]:
    """Canonical name from the Qt key code alone (layout dependent; the fallback)."""
    key = int(event.key())
    name = QT_KEY_NAMES.get(key)
    if name is None:
        return None
    if event.modifiers() & Qt.KeypadModifier and name.isdigit():
        return f"Numpad{name}"
    return name


def key_name(event: QKeyEvent) -> Optional[str]:
    """Canonical non-modifier key name for the event, or None if the key is not supported.

    The native key code wins (layout independent, and it is what the OS hotkey API
    registers), read through the host's own table; Qt's key code is the fallback. Never
    guesses: an unknown key returns None and the field tells the user it cannot be used.
    """
    vk = int(event.nativeVirtualKey())
    if IS_MACOS:
        name = MAC_KEY_NAMES.get(vk)
        # kVK_ANSI_A is 0, which is also "this event carries no native key code" (synthetic
        # events, QTest). Only trust 0 when Qt agrees it was the A key.
        if name is not None and not (vk == 0 and int(event.key()) != int(_K.Key_A)):
            return name
        return _qt_key_name(event)
    if vk and vk in VK_NAMES:
        return VK_NAMES[vk]
    return _qt_key_name(event)


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
        turns into the system menu before Qt sees a key event). Unused on macOS, where the
        combinations the system keeps never reach the process at all."""
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
