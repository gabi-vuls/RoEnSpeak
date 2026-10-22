"""Hotkey string parsing and formatting (``contracts.HotkeyParser``). Pure; any thread."""

from __future__ import annotations

from typing import Optional

from app.contracts import Hotkey, HotkeyParseError, Modifier

__all__ = ["parse_hotkey", "format_hotkey", "HotkeyStringParser", "KEYS", "PRINTABLE_KEYS"]

_MODIFIER_ALIASES: dict[str, Modifier] = {
    "ctrl": Modifier.CTRL, "control": Modifier.CTRL, "ctl": Modifier.CTRL,
    "alt": Modifier.ALT,
    "shift": Modifier.SHIFT,
    "win": Modifier.WIN, "windows": Modifier.WIN, "meta": Modifier.WIN, "super": Modifier.WIN,
}


def _build_keys() -> tuple[dict[str, tuple[str, int]], frozenset[str]]:
    """lower-case input name -> (canonical name, VK). Also the set of printable canonical names."""
    keys: dict[str, tuple[str, int]] = {}
    printable: set[str] = set()

    def add(canonical: str, vk: int, *aliases: str, is_printable: bool = False) -> None:
        for name in (canonical, *aliases):
            keys[name.lower()] = (canonical, vk)
        if is_printable:
            printable.add(canonical)

    for i in range(26):
        ch = chr(ord("A") + i)
        add(ch, ord(ch), is_printable=True)
    for d in range(10):
        add(str(d), 0x30 + d, is_printable=True)
    for n in range(1, 25):
        add(f"F{n}", 0x70 + n - 1)
    add("Space", 0x20, is_printable=True)
    add("Enter", 0x0D, "Return")
    add("Tab", 0x09)
    add("Esc", 0x1B, "Escape")
    add("Backspace", 0x08)
    add("Insert", 0x2D, "Ins")
    add("Delete", 0x2E, "Del")
    add("Home", 0x24)
    add("End", 0x23)
    add("PageUp", 0x21, "PgUp")
    add("PageDown", 0x22, "PgDn")
    add("Up", 0x26)
    add("Down", 0x28)
    add("Left", 0x25)
    add("Right", 0x27)
    add("Pause", 0x13)
    for d in range(10):
        add(f"Numpad{d}", 0x60 + d)
    # OEM punctuation (US-layout virtual keys).
    for ch, vk in ((";", 0xBA), ("=", 0xBB), (",", 0xBC), ("-", 0xBD), (".", 0xBE), ("/", 0xBF),
                   ("`", 0xC0), ("[", 0xDB), ("\\", 0xDC), ("]", 0xDD), ("'", 0xDE)):
        add(ch, vk, is_printable=True)
    return keys, frozenset(printable)


#: lower-case accepted name -> (canonical key name, virtual-key code)
KEYS, PRINTABLE_KEYS = _build_keys()


def parse_hotkey(text: str) -> Hotkey:
    """Parse e.g. "ctrl + alt + f9" into a Hotkey (display "Ctrl+Alt+F9").

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
            f'"{key[0]}" needs at least one modifier (Ctrl, Alt, Shift or Win), for example Alt+{key[0]}.'
        )
    if modifiers == {Modifier.SHIFT} and key[0] in PRINTABLE_KEYS:
        raise HotkeyParseError(
            f"Shift+{key[0]} can't be used: it would stop that key from typing normally. Add Ctrl, Alt or Win."
        )
    return Hotkey(modifiers=frozenset(modifiers), key=key[0], vk=key[1])


def format_hotkey(hotkey: Hotkey) -> str:
    return hotkey.display


class HotkeyStringParser:
    """``contracts.HotkeyParser`` over parse_hotkey/format_hotkey."""

    def parse(self, text: str) -> Hotkey:
        return parse_hotkey(text)

    def format(self, hotkey: Hotkey) -> str:
        return format_hotkey(hotkey)
