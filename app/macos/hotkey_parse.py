"""Hotkey string parsing and formatting on macOS (``contracts.HotkeyParser``). Pure; any thread.

Same grammar, normalization and rejection rules as ``app.win32.hotkey_parse`` -- the two
parsers accept the same strings so one config.json is valid on both platforms. Two
differences, both of them about the host:

* Modifier aliases: ``cmd|command|win|windows|meta|super`` -> ``Modifier.CMD`` and
  ``alt|option|opt`` -> ``Modifier.ALT``. So the shared default ``"Alt+Q"`` parses here as
  Option+Q, and a Windows ``"Win+..."`` config reads as Cmd+... instead of being rejected.
* ``Hotkey.vk`` holds the Carbon virtual key code (``kVK_*`` from ``HIToolbox/Events.h``)
  for the same canonical key names, because that is what ``RegisterEventHotKey`` wants.
  Keys with no Carbon key code (Insert, Pause, F21-F24) are rejected with a message naming
  the key rather than silently mapped onto something else.
"""

from __future__ import annotations

from typing import Optional

from app.contracts import Hotkey, HotkeyParseError, Modifier

__all__ = ["parse_hotkey", "format_hotkey", "MacHotkeyStringParser", "KEYS", "PRINTABLE_KEYS",
           "UNSUPPORTED_KEYS"]

_MODIFIER_ALIASES: dict[str, Modifier] = {
    "ctrl": Modifier.CTRL, "control": Modifier.CTRL, "ctl": Modifier.CTRL,
    "alt": Modifier.ALT, "option": Modifier.ALT, "opt": Modifier.ALT,
    "shift": Modifier.SHIFT,
    "cmd": Modifier.CMD, "command": Modifier.CMD,
    "win": Modifier.CMD, "windows": Modifier.CMD, "meta": Modifier.CMD, "super": Modifier.CMD,
}

# Carbon kVK_* codes (HIToolbox/Events.h). The letter and digit codes are positional on the
# US layout, so they are spelled out rather than computed.
_LETTER_VK = {
    "A": 0x00, "B": 0x0B, "C": 0x08, "D": 0x02, "E": 0x0E, "F": 0x03, "G": 0x05, "H": 0x04,
    "I": 0x22, "J": 0x26, "K": 0x28, "L": 0x25, "M": 0x2E, "N": 0x2D, "O": 0x1F, "P": 0x23,
    "Q": 0x0C, "R": 0x0F, "S": 0x01, "T": 0x11, "U": 0x20, "V": 0x09, "W": 0x0D, "X": 0x07,
    "Y": 0x10, "Z": 0x06,
}
_DIGIT_VK = {
    "0": 0x1D, "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15, "5": 0x17, "6": 0x16, "7": 0x1A,
    "8": 0x1C, "9": 0x19,
}
_FUNCTION_VK = {
    1: 0x7A, 2: 0x78, 3: 0x63, 4: 0x76, 5: 0x60, 6: 0x61, 7: 0x62, 8: 0x64, 9: 0x65,
    10: 0x6D, 11: 0x67, 12: 0x6F, 13: 0x69, 14: 0x6B, 15: 0x71, 16: 0x6A, 17: 0x40,
    18: 0x4F, 19: 0x50, 20: 0x5A,
}
_KEYPAD_VK = {
    0: 0x52, 1: 0x53, 2: 0x54, 3: 0x55, 4: 0x56, 5: 0x57, 6: 0x58, 7: 0x59, 8: 0x5B, 9: 0x5C,
}
#: OEM punctuation, by the character the US layout prints.
_PUNCTUATION_VK = {
    ";": 0x29, "=": 0x18, ",": 0x2B, "-": 0x1B, ".": 0x2F, "/": 0x2C, "`": 0x32,
    "[": 0x21, "\\": 0x2A, "]": 0x1E, "'": 0x27,
}

#: Canonical names the Windows parser accepts that no Mac keyboard has a key code for.
#: lower-case accepted name -> canonical name (used only to build a clear error message).
UNSUPPORTED_KEYS: dict[str, str] = {
    "insert": "Insert", "ins": "Insert",
    "pause": "Pause",
    "f21": "F21", "f22": "F22", "f23": "F23", "f24": "F24",
}


def _build_keys() -> tuple[dict[str, tuple[str, int]], frozenset[str]]:
    """lower-case input name -> (canonical name, Carbon key code). Also the printable set."""
    keys: dict[str, tuple[str, int]] = {}
    printable: set[str] = set()

    def add(canonical: str, vk: int, *aliases: str, is_printable: bool = False) -> None:
        for name in (canonical, *aliases):
            keys[name.lower()] = (canonical, vk)
        if is_printable:
            printable.add(canonical)

    for ch, vk in _LETTER_VK.items():
        add(ch, vk, is_printable=True)
    for ch, vk in _DIGIT_VK.items():
        add(ch, vk, is_printable=True)
    for n, vk in _FUNCTION_VK.items():
        add(f"F{n}", vk)
    add("Space", 0x31, is_printable=True)
    add("Enter", 0x24, "Return")
    add("Tab", 0x30)
    add("Esc", 0x35, "Escape")
    add("Backspace", 0x33)  # kVK_Delete: the key labelled "delete" above Return
    add("Delete", 0x75, "Del")  # kVK_ForwardDelete: fn+delete / the full-keyboard Del
    add("Home", 0x73)
    add("End", 0x77)
    add("PageUp", 0x74, "PgUp")
    add("PageDown", 0x79, "PgDn")
    add("Up", 0x7E)
    add("Down", 0x7D)
    add("Left", 0x7B)
    add("Right", 0x7C)
    for n, vk in _KEYPAD_VK.items():
        add(f"Numpad{n}", vk)
    for ch, vk in _PUNCTUATION_VK.items():
        add(ch, vk, is_printable=True)
    return keys, frozenset(printable)


#: lower-case accepted name -> (canonical key name, Carbon virtual key code)
KEYS, PRINTABLE_KEYS = _build_keys()


def parse_hotkey(text: str) -> Hotkey:
    """Parse e.g. "cmd + shift + d" into a Hotkey (display "Shift+Cmd+D").

    Raises HotkeyParseError with a message suitable for showing in the UI.
    """
    if not isinstance(text, str) or not text.strip():
        raise HotkeyParseError("Enter a key combination, for example Alt+Q.")
    original = text.strip()
    tokens = [t.strip() for t in original.split("+")]
    if any(not t for t in tokens):
        raise HotkeyParseError(
            f'"{original}" is not a valid key combination: a key name is missing around "+". '
            'Use the "=" key instead of "+".'
        )

    modifiers: set[Modifier] = set()
    key: Optional[tuple[str, int]] = None
    for tok in tokens:
        low = tok.lower()
        mod = _MODIFIER_ALIASES.get(low)
        if mod is not None:
            if mod in modifiers:
                raise HotkeyParseError(f'"{original}" uses {mod.value} more than once.')
            modifiers.add(mod)
            continue
        found = KEYS.get(low)
        if found is None:
            unsupported = UNSUPPORTED_KEYS.get(low)
            if unsupported is not None:
                raise HotkeyParseError(
                    f'"{original}" uses {unsupported}, which Mac keyboards do not have. '
                    "Choose another key, for example Alt+Q."
                )
            raise HotkeyParseError(f'Unknown key "{tok}" in "{original}".')
        if key is not None:
            raise HotkeyParseError(
                f'"{original}" has more than one main key ({key[0]} and {found[0]}). '
                "Use modifiers plus exactly one key, for example Ctrl+Alt+D."
            )
        key = found

    if key is None:
        raise HotkeyParseError(
            f'"{original}" has only modifier keys. Add one main key, for example Alt+Q.'
        )
    if not modifiers:
        raise HotkeyParseError(
            f'"{key[0]}" needs at least one modifier (Ctrl, Alt, Shift or Cmd), for example Alt+{key[0]}.'
        )
    if modifiers == {Modifier.SHIFT} and key[0] in PRINTABLE_KEYS:
        raise HotkeyParseError(
            f"Shift+{key[0]} can't be used: it would stop that key from typing normally. Add Ctrl, Alt or Cmd."
        )
    return Hotkey(modifiers=frozenset(modifiers), key=key[0], vk=key[1])


def format_hotkey(hotkey: Hotkey) -> str:
    return hotkey.display


class MacHotkeyStringParser:
    """``contracts.HotkeyParser`` over parse_hotkey/format_hotkey."""

    def parse(self, text: str) -> Hotkey:
        return parse_hotkey(text)

    def format(self, hotkey: Hotkey) -> str:
        return format_hotkey(hotkey)
