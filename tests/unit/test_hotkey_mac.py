"""Unit tests for macOS hotkey parsing and MacHotkeyManager (TEST-UNIT-MACOS).

Written from SPEC.md and app/contracts.py only -- the Hotkey/HotkeyParser docstrings for the
grammar, the HotkeyManager protocol docstring for the manager, and the macOS notes in
Modifier / HotkeyConflictError. Every system call goes through FakeMacApi
(tests/unit/fake_mac_api.py), which is also used by test_injection_mac.py.

Covered here
------------
* ``app.macos.hotkey_parse``: the full key/modifier grammar from the ``Hotkey`` docstring
  (aliases, canonical MODIFIER_ORDER display, Carbon ``kVK_*`` codes, case and whitespace
  tolerance, format/parse round trips) and every documented rejection, plus the three
  cross-platform consequences the port depends on: ``"Alt+Q"`` (DEFAULT_HOTKEY) parses,
  ``"Win+F12"`` canonicalises to ``"Cmd+F12"``, and Cmd/Win cannot be combined.
* ``app.macos.hotkey.MacHotkeyManager`` against ``contracts.HotkeyManager``: idempotent
  blocking start/unregister/stop, register replacing the previous hotkey, a failed
  registration raising HotkeyConflictError with the PREVIOUS hotkey still registered,
  ``current``, single fire per delivered event, the debounce window, on_fire exceptions
  being swallowed, register() after stop() raising, and stale events being ignored.

Nothing here needs macOS, a run loop, Carbon or an Accessibility grant: the fake supplies the
clock, the Carbon surface and a pumpable main run loop, so the tests run on any platform.

Implementation modules are imported lazily inside helpers so a missing module fails only the
tests that need it (as tests/unit/test_hotkey.py does for Windows).
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import pytest

from app.contracts import (
    DEFAULT_HOTKEY,
    MOD_NOREPEAT,
    MOD_WIN,
    AppError,
    Hotkey,
    HotkeyConflictError,
    HotkeyParseError,
    Modifier,
)
from tests.unit.fake_mac_api import (
    CONTROL_KEY,
    EVENT_HOT_KEY_EXISTS_ERR,
    KVK_ANSI_D,
    KVK_ANSI_Q,
    KVK_F9,
    OPTION_KEY,
    PARAM_ERR,
    SHIFT_KEY,
    FakeMacApi,
)

# ---------------------------------------------------------------------------
# Expected Carbon key codes (Apple's HIToolbox/Events.h), written out here so the parser is
# compared against the platform rather than against its own table.
# ---------------------------------------------------------------------------

LETTER_VK = {
    "A": 0x00, "B": 0x0B, "C": 0x08, "D": 0x02, "E": 0x0E, "F": 0x03, "G": 0x05, "H": 0x04,
    "I": 0x22, "J": 0x26, "K": 0x28, "L": 0x25, "M": 0x2E, "N": 0x2D, "O": 0x1F, "P": 0x23,
    "Q": 0x0C, "R": 0x0F, "S": 0x01, "T": 0x11, "U": 0x20, "V": 0x09, "W": 0x0D, "X": 0x07,
    "Y": 0x10, "Z": 0x06,
}
DIGIT_VK = {
    "0": 0x1D, "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15, "5": 0x17, "6": 0x16, "7": 0x1A,
    "8": 0x1C, "9": 0x19,
}
FUNCTION_VK = {
    1: 0x7A, 2: 0x78, 3: 0x63, 4: 0x76, 5: 0x60, 6: 0x61, 7: 0x62, 8: 0x64, 9: 0x65,
    10: 0x6D, 11: 0x67, 12: 0x6F, 13: 0x69, 14: 0x6B, 15: 0x71, 16: 0x6A, 17: 0x40,
    18: 0x4F, 19: 0x50, 20: 0x5A,
}
KEYPAD_VK = {0: 0x52, 1: 0x53, 2: 0x54, 3: 0x55, 4: 0x56, 5: 0x57, 6: 0x58, 7: 0x59,
             8: 0x5B, 9: 0x5C}
PUNCTUATION_VK = {
    ";": 0x29, "=": 0x18, ",": 0x2B, "-": 0x1B, ".": 0x2F, "/": 0x2C, "`": 0x32,
    "[": 0x21, "\\": 0x2A, "]": 0x1E, "'": 0x27,
}
NAMED_VK = {
    "Space": 0x31, "Enter": 0x24, "Tab": 0x30, "Esc": 0x35, "Backspace": 0x33,
    "Delete": 0x75, "Home": 0x73, "End": 0x77, "PageUp": 0x74, "PageDown": 0x79,
    "Up": 0x7E, "Down": 0x7D, "Left": 0x7B, "Right": 0x7C,
}

#: Canonical names the shared grammar lists that a Mac keyboard has no key code for.
NO_MAC_KEY = ["Insert", "Ins", "Pause", "F21", "F22", "F23", "F24"]


def _parse(text: str) -> Hotkey:
    from app.macos.hotkey_parse import parse_hotkey

    return parse_hotkey(text)


def _format(h: Hotkey) -> str:
    from app.macos.hotkey_parse import format_hotkey

    return format_hotkey(h)


def _carbon_modifiers(h: Hotkey) -> int:
    from app.macos.hotkey import carbon_modifiers

    return carbon_modifiers(h)


# ===========================================================================
# Hotkey parsing
# ===========================================================================


class TestParseBasics:
    @pytest.mark.parametrize("text", ["alt+q", "Alt+Q", " ALT + q ", "aLt+Q", "Alt + Q",
                                      "\tAlt+q\n", " Alt + Q ", "option+q", "OPT+Q"])
    def test_alt_q_variants_normalize(self, text):
        h = _parse(text)
        assert h.display == "Alt+Q"
        assert h.key == "Q"
        assert h.vk == KVK_ANSI_Q
        assert h.modifiers == frozenset({Modifier.ALT})
        assert h == _parse("Alt+Q")

    def test_default_hotkey_parses_on_macos(self):
        # The shared DEFAULT_HOTKEY must be valid on both platforms (one config.json).
        h = _parse(DEFAULT_HOTKEY)
        assert h.display == "Alt+Q"
        assert h.modifiers == frozenset({Modifier.ALT})

    def test_whitespace_insensitive_equality(self):
        assert _parse(" Alt + Q ") == _parse("alt+q")

    def test_returns_contract_hotkey(self):
        assert isinstance(_parse("ctrl+alt+f9"), Hotkey)

    def test_windows_super_config_reads_as_cmd(self):
        h = _parse("Win+F12")
        assert h.display == "Cmd+F12"
        assert h.modifiers == frozenset({Modifier.CMD})
        assert h.vk == FUNCTION_VK[12]

    def test_macos_parser_never_produces_win(self):
        for text in ["win+e", "windows+e", "meta+e", "super+e", "cmd+e", "command+e"]:
            assert _parse(text).modifiers == frozenset({Modifier.CMD})
            assert Modifier.WIN not in _parse(text).modifiers

    @pytest.mark.parametrize(
        "text, display",
        [
            ("shift+ctrl+alt+f9", "Ctrl+Alt+Shift+F9"),
            ("cmd+shift+alt+ctrl+a", "Ctrl+Alt+Shift+Cmd+A"),
            ("alt+ctrl+z", "Ctrl+Alt+Z"),
            ("cmd+alt+d", "Alt+Cmd+D"),
            ("shift+cmd+q", "Shift+Cmd+Q"),
            ("cmd+ctrl+space", "Ctrl+Cmd+Space"),
        ],
    )
    def test_canonical_modifier_order(self, text, display):
        # contracts.MODIFIER_ORDER = (CTRL, ALT, SHIFT, WIN, CMD)
        assert _parse(text).display == display

    @pytest.mark.parametrize(
        "text, display, mods",
        [
            ("control+a", "Ctrl+A", {Modifier.CTRL}),
            ("ctl+a", "Ctrl+A", {Modifier.CTRL}),
            ("CONTROL+a", "Ctrl+A", {Modifier.CTRL}),
            ("option+d", "Alt+D", {Modifier.ALT}),
            ("opt+d", "Alt+D", {Modifier.ALT}),
            ("OPTION+d", "Alt+D", {Modifier.ALT}),
            ("command+e", "Cmd+E", {Modifier.CMD}),
            ("cmd+e", "Cmd+E", {Modifier.CMD}),
            ("windows+e", "Cmd+E", {Modifier.CMD}),
            ("meta+e", "Cmd+E", {Modifier.CMD}),
            ("super+e", "Cmd+E", {Modifier.CMD}),
            ("Super+Shift+E", "Shift+Cmd+E", {Modifier.CMD, Modifier.SHIFT}),
            ("shift+f9", "Shift+F9", {Modifier.SHIFT}),
        ],
    )
    def test_modifier_aliases(self, text, display, mods):
        h = _parse(text)
        assert h.display == display
        assert h.modifiers == frozenset(mods)

    @pytest.mark.parametrize(
        "text, display, vk",
        [
            ("ctrl+return", "Ctrl+Enter", 0x24),
            ("ctrl+enter", "Ctrl+Enter", 0x24),
            ("alt+escape", "Alt+Esc", 0x35),
            ("alt+esc", "Alt+Esc", 0x35),
            ("ctrl+del", "Ctrl+Delete", 0x75),
            ("ctrl+delete", "Ctrl+Delete", 0x75),
            ("ctrl+pgup", "Ctrl+PageUp", 0x74),
            ("ctrl+pageup", "Ctrl+PageUp", 0x74),
            ("ctrl+pgdn", "Ctrl+PageDown", 0x79),
            ("ctrl+pagedown", "Ctrl+PageDown", 0x79),
        ],
    )
    def test_key_aliases(self, text, display, vk):
        h = _parse(text)
        assert h.display == display
        assert h.vk == vk


class TestParseKeys:
    @pytest.mark.parametrize("letter", sorted(LETTER_VK))
    def test_letters(self, letter):
        h = _parse(f"alt+{letter.lower()}")
        assert h.key == letter
        assert h.vk == LETTER_VK[letter]
        assert h.display == f"Alt+{letter}"

    @pytest.mark.parametrize("digit", sorted(DIGIT_VK))
    def test_digits(self, digit):
        h = _parse(f"ctrl+{digit}")
        assert h.key == digit
        assert h.vk == DIGIT_VK[digit]
        assert h.display == f"Ctrl+{digit}"

    @pytest.mark.parametrize("n", sorted(FUNCTION_VK))
    def test_function_keys(self, n):
        h = _parse(f"ctrl+f{n}")
        assert h.key == f"F{n}"
        assert h.vk == FUNCTION_VK[n]
        assert h.display == f"Ctrl+F{n}"

    @pytest.mark.parametrize("n", sorted(KEYPAD_VK))
    def test_numpad(self, n):
        h = _parse(f"ctrl+numpad{n}")
        assert h.key == f"Numpad{n}"
        assert h.vk == KEYPAD_VK[n]
        assert h.display == f"Ctrl+Numpad{n}"

    @pytest.mark.parametrize("name", sorted(NAMED_VK))
    def test_named_keys(self, name):
        h = _parse(f"ctrl+{name.lower()}")
        assert h.key == name
        assert h.vk == NAMED_VK[name]
        assert h.display == f"Ctrl+{name}"

    @pytest.mark.parametrize("char", sorted(PUNCTUATION_VK))
    def test_oem_punctuation(self, char):
        h = _parse(f"ctrl+{char}")
        assert h.key == char
        assert h.vk == PUNCTUATION_VK[char]
        assert h.display == f"Ctrl+{char}"
        assert _parse(f" Ctrl + {char} ") == h

    def test_backspace_and_delete_are_different_keys(self):
        # kVK_Delete (the key above Return) vs kVK_ForwardDelete.
        assert _parse("ctrl+backspace").vk != _parse("ctrl+delete").vk

    def test_every_key_code_is_a_byte(self):
        for text in ["alt+a", "ctrl+f20", "ctrl+numpad9", "ctrl+'", "ctrl+pagedown"]:
            assert 0 <= _parse(text).vk <= 0xFF


class TestCarbonModifiers:
    @pytest.mark.parametrize(
        "text, bits",
        [
            ("alt+q", OPTION_KEY),
            ("ctrl+f9", CONTROL_KEY),
            ("shift+f9", SHIFT_KEY),
            ("cmd+d", 0x0100),
            ("ctrl+alt+shift+cmd+a", CONTROL_KEY | OPTION_KEY | SHIFT_KEY | 0x0100),
        ],
    )
    def test_carbon_modifier_bits(self, text, bits):
        # HIToolbox/Events.h: cmdKey 0x0100, shiftKey 0x0200, optionKey 0x0800,
        # controlKey 0x1000. Carbon hot keys do not auto-repeat, so there is no NOREPEAT bit.
        assert _carbon_modifiers(_parse(text)) == bits

    def test_win32_modifiers_stays_total_for_cmd(self):
        # contracts.Hotkey.win32_modifiers maps CMD onto MOD_WIN so the property never
        # raises for a macOS hotkey; it is unused on macOS.
        assert _parse("cmd+d").win32_modifiers == MOD_WIN | MOD_NOREPEAT


ROUND_TRIP = [
    "Alt+Q", "Ctrl+Alt+F9", "Ctrl+Alt+Shift+Cmd+F20", "Ctrl+Enter", "Alt+Esc", "Ctrl+Delete",
    "Ctrl+PageUp", "Ctrl+PageDown", "Ctrl+Numpad7", "Ctrl+Space", "Cmd+Up", "Ctrl+;", "Ctrl+=",
    "Ctrl+,", "Ctrl+-", "Ctrl+.", "Ctrl+/", "Ctrl+`", "Ctrl+[", "Ctrl+\\", "Ctrl+]", "Ctrl+'",
    "Shift+F9", "Ctrl+0", "Alt+Backspace", "Alt+Tab", "Shift+Cmd+D", "Ctrl+Home", "Ctrl+End",
    "Cmd+Left", "Alt+Cmd+Right", "Ctrl+F13",
]


class TestFormat:
    @pytest.mark.parametrize("display", ROUND_TRIP)
    def test_format_parse_round_trip(self, display):
        h = _parse(display)
        assert _format(h) == display
        assert h.display == display
        assert _parse(_format(h)) == h

    @pytest.mark.parametrize("text", ["shift+ctrl+alt+f9", "control+return", "meta+pgdn",
                                      " alt + q ", "OPT+SHIFT+UP"])
    def test_format_is_display(self, text):
        h = _parse(text)
        assert _format(h) == h.display

    def test_parser_class_matches_functions(self):
        from app.macos.hotkey_parse import MacHotkeyStringParser

        p = MacHotkeyStringParser()
        for text in ["alt+q", "shift+ctrl+alt+f9", "ctrl+;", "cmd+shift+d"]:
            assert p.parse(text) == _parse(text)
            assert p.format(p.parse(text)) == _parse(text).display
        with pytest.raises(HotkeyParseError):
            p.parse("Q")


REJECTED = [
    "",
    "   ",
    "Q",                       # no modifier
    "F9",
    "Space",
    "q",
    "Ctrl",                    # modifier only
    "Alt",
    "Cmd",
    "Ctrl+Alt",
    "Ctrl+Alt+Shift+Cmd",
    "Ctrl+A+B",                # more than one main key
    "Ctrl+F9+F10",
    "Alt+Alt+Q",               # duplicate modifier
    "Ctrl+Control+Q",          # duplicate through an alias
    "Alt+Option+Q",
    "Cmd+Command+E",
    "Cmd+Meta+E",
    "Ctrl+Foo",                # unknown token
    "Foo+Q",
    "Ctrl+F25",
    "Ctrl+F0",
    "Alt+QQ",
    "Ctrl++",                  # documented as unsupported; use "=" instead
    "Alt+",
    "+Q",
    "Alt++Q",
    "Ctrl-Q",                  # "-" is not a separator
    "Shift+Q",                 # Shift-only on a printable key
    "Shift+a",
    "Shift+1",
    "Shift+Space",
    "Shift+;",
    "Shift+'",
    "Shift+\\",
]


class TestParseRejections:
    @pytest.mark.parametrize("text", REJECTED)
    def test_rejected(self, text):
        with pytest.raises(HotkeyParseError) as ei:
            _parse(text)
        assert isinstance(ei.value, AppError)
        assert ei.value.user_message

    def test_cmd_and_win_cannot_be_combined(self):
        # Win and Cmd are the same key on macOS, so naming both is a duplicate modifier.
        for text in ["Cmd+Win+E", "Win+Cmd+E", "Super+Cmd+E", "Win+Meta+E"]:
            with pytest.raises(HotkeyParseError):
                _parse(text)

    @pytest.mark.parametrize("key", NO_MAC_KEY)
    def test_keys_mac_keyboards_lack_are_rejected_by_name(self, key):
        # The shared grammar lists Insert/Pause/F21-F24, but no Mac keyboard has a Carbon key
        # code for them, so they are rejected with a message naming the key rather than
        # silently mapped onto some other key.
        with pytest.raises(HotkeyParseError) as ei:
            _parse(f"Ctrl+{key}")
        canonical = "Insert" if key in ("Insert", "Ins") else key
        assert canonical in ei.value.user_message
        assert isinstance(ei.value, AppError)

    @pytest.mark.parametrize("text, display", [("shift+f9", "Shift+F9"), ("Shift+Up", "Shift+Up"),
                                               ("shift+esc", "Shift+Esc"),
                                               ("shift+numpad1", "Shift+Numpad1")])
    def test_shift_only_allowed_on_non_printable(self, text, display):
        # Contract: Shift-only is rejected only for printable keys (letters, digits, OEM, Space).
        h = _parse(text)
        assert h.display == display
        assert h.modifiers == frozenset({Modifier.SHIFT})

    @pytest.mark.parametrize("text", ["ctrl+shift+q", "shift+cmd+1", "alt+shift+space",
                                      "ctrl+shift+;"])
    def test_shift_with_another_modifier_on_printable_ok(self, text):
        h = _parse(text)
        assert Modifier.SHIFT in h.modifiers
        assert len(h.modifiers) == 2


# ===========================================================================
# MacHotkeyManager
# ===========================================================================

ALT_Q = Hotkey(frozenset({Modifier.ALT}), "Q", KVK_ANSI_Q)
CTRL_ALT_F9 = Hotkey(frozenset({Modifier.CTRL, Modifier.ALT}), "F9", KVK_F9)
CTRL_F9 = Hotkey(frozenset({Modifier.CTRL}), "F9", KVK_F9)
CTRL_SHIFT_D = Hotkey(frozenset({Modifier.CTRL, Modifier.SHIFT}), "D", KVK_ANSI_D)

ALT_Q_COMBO = (OPTION_KEY, KVK_ANSI_Q)
CTRL_ALT_F9_COMBO = (CONTROL_KEY | OPTION_KEY, KVK_F9)
CTRL_SHIFT_D_COMBO = (CONTROL_KEY | SHIFT_KEY, KVK_ANSI_D)


def run_with_timeout(fn: Callable, *args, timeout: float = 2.0, **kwargs):
    """Run fn on a helper thread; fail (instead of hanging) if it doesn't return in time."""
    box: dict = {}

    def target() -> None:
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True, name="timeout")
    t.start()
    t.join(timeout)
    assert not t.is_alive(), f"{getattr(fn, '__qualname__', fn)} did not return within {timeout} s"
    if "error" in box:
        raise box["error"]
    return box.get("value")


def run_marshalled(api: FakeMacApi, fn: Callable, *args, timeout: float = 3.0):
    """Call fn from a worker thread while this (main) thread pumps the fake main run loop.

    This is the path a non-main-thread caller takes: queue a request, signal the run loop
    source, block until the main thread has run it.
    """
    box: dict = {}

    def target() -> None:
        try:
            box["value"] = fn(*args)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box["error"] = exc
        finally:
            box["done"] = True

    t = threading.Thread(target=target, daemon=True, name="marshalled")
    t.start()
    deadline = time.monotonic() + timeout
    while "done" not in box and time.monotonic() < deadline:
        api.pump()
        time.sleep(0.001)
    t.join(max(0.0, deadline - time.monotonic()))
    assert not t.is_alive(), "marshalled request did not complete"
    if "error" in box:
        raise box["error"]
    return box.get("value")


class FireRecorder:
    """on_fire callback. Records the calling thread and the fake time; logs into api.log."""

    def __init__(self, api: FakeMacApi, raise_on=()) -> None:
        self.api = api
        self.calls: list[tuple[int, float]] = []
        self.raise_on = set(raise_on)
        self._lock = threading.Lock()

    def __call__(self) -> None:
        with self._lock:
            idx = len(self.calls)
            self.calls.append((threading.get_ident(), self.api.monotonic()))
        self.api.record("on_fire")
        if idx in self.raise_on:
            raise RuntimeError("on_fire boom (test)")

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.calls)


@pytest.fixture
def api():
    return FakeMacApi()


@pytest.fixture
def make_manager(api):
    created = []

    def make(debounce_s: float = 0.3, on_fire: Optional[Callable[[], None]] = None,
             start: bool = True):
        from app.macos.hotkey import MacHotkeyManager

        rec = on_fire if on_fire is not None else FireRecorder(api)
        mgr = MacHotkeyManager(api, rec, debounce_s=debounce_s)
        created.append(mgr)
        if start:
            mgr.start()
        return mgr, rec

    yield make
    for mgr in created:
        try:
            mgr.stop()
        except Exception:  # noqa: BLE001 - teardown best effort; the tests assert stop()
            pass


class TestManagerStart:
    def test_start_installs_the_carbon_handler_and_a_run_loop_source(self, api, make_manager):
        mgr, _ = make_manager()
        assert api.install_calls == 1
        assert api.handlers, "no kEventHotKeyPressed handler installed"
        assert api.sources, "no main run loop source created"
        assert mgr.current is None

    def test_start_is_idempotent(self, api, make_manager):
        mgr, rec = make_manager()
        mgr.start()
        mgr.start()
        assert api.install_calls == 1
        assert len(api.sources) == 1
        mgr.register(ALT_Q)
        api.deliver_hotkey()
        assert rec.count == 1

    def test_start_returns_before_any_run_loop_spins(self, api, make_manager):
        # start() must not need a running run loop: it is called during app startup, before
        # QApplication.exec(). The fake never pumps here.
        mgr, _ = make_manager()
        assert api.pumps == 0
        mgr.register(ALT_Q)
        assert mgr.current == ALT_Q

    def test_start_raises_app_error_when_the_handler_cannot_be_installed(self, api, make_manager):
        api.install_handler_fails = True
        with pytest.raises(AppError):
            make_manager()
        assert api.handlers == {}

    def test_start_survives_a_failed_run_loop_source(self, api, make_manager):
        # Main-thread register/unregister/stop need no run loop, so start() must still work.
        api.create_source_result = 0
        mgr, rec = make_manager()
        mgr.register(ALT_Q)
        assert mgr.current == ALT_Q
        api.deliver_hotkey()
        assert rec.count == 1

    def test_restart_after_stop_works(self, api, make_manager):
        mgr, rec = make_manager()
        mgr.register(ALT_Q)
        mgr.stop()
        mgr.start()
        mgr.register(ALT_Q)
        assert api.active_combos() == {ALT_Q_COMBO}
        api.deliver_hotkey()
        assert rec.count == 1


class TestManagerRegistration:
    def test_register_calls_register_event_hotkey(self, api, make_manager):
        mgr, _ = make_manager()
        mgr.register(ALT_Q)
        assert api.active_combos() == {ALT_Q_COMBO}
        ok = [c for c in api.register_calls if c["status"] == 0]
        assert len(ok) == 1
        assert ok[0]["keycode"] == KVK_ANSI_Q
        assert ok[0]["modifiers"] == OPTION_KEY
        assert mgr.current == ALT_Q

    def test_current_none_initially(self, api, make_manager):
        mgr, _ = make_manager()
        assert mgr.current is None

    def test_register_before_start_raises_runtime_error(self, api, make_manager):
        mgr, _ = make_manager(start=False)
        with pytest.raises(RuntimeError):
            mgr.register(ALT_Q)
        assert api.registrations == []
        assert mgr.current is None

    def test_register_after_stop_raises_runtime_error(self, api, make_manager):
        mgr, _ = make_manager()
        mgr.register(ALT_Q)
        mgr.stop()
        with pytest.raises(RuntimeError):
            mgr.register(CTRL_ALT_F9)
        assert api.registrations == []
        assert mgr.current is None

    def test_register_replaces_previous(self, api, make_manager):
        mgr, rec = make_manager()
        mgr.register(ALT_Q)
        mgr.register(CTRL_ALT_F9)
        assert api.active_combos() == {CTRL_ALT_F9_COMBO}
        assert len(api.registrations) == 1
        assert mgr.current == CTRL_ALT_F9
        api.deliver_hotkey()
        assert rec.count == 1

    def test_replacement_registers_the_new_id_before_releasing_the_old(self, api, make_manager):
        # The guarantee behind "on failure the PREVIOUS hotkey stays registered": the new
        # combo is registered under the other id of the pair first, and only then is the old
        # registration released.
        mgr, _ = make_manager()
        mgr.register(ALT_Q)
        old_ref = api.registrations[0]["ref"]
        old_id = api.registrations[0]["id"]
        mgr.register(CTRL_ALT_F9)
        new_id = api.registrations[0]["id"]
        assert new_id != old_id
        order = [e for e in api.log if e[0] in ("register_hotkey", "unregister_hotkey")]
        kinds = [e[0] for e in order]
        assert kinds == ["register_hotkey", "register_hotkey", "unregister_hotkey"]
        assert order[-1][1][0] == old_ref
        assert [c["ref"] for c in api.unregister_calls] == [old_ref]

    def test_register_alternates_between_two_ids(self, api, make_manager):
        mgr, _ = make_manager()
        seen = []
        for hotkey in (ALT_Q, CTRL_ALT_F9, CTRL_SHIFT_D, ALT_Q):
            mgr.register(hotkey)
            seen.append(api.registrations[0]["id"])
        assert len(set(seen)) == 2
        assert seen[0] != seen[1] and seen[1] != seen[2] and seen[2] != seen[3]

    def test_register_same_hotkey_again_succeeds(self, api, make_manager):
        mgr, rec = make_manager()
        mgr.register(ALT_Q)
        mgr.register(ALT_Q)
        assert api.active_combos() == {ALT_Q_COMBO}
        assert len(api.registrations) == 1
        assert mgr.current == ALT_Q
        api.deliver_hotkey()
        assert rec.count == 1

    def test_conflict_raises_and_keeps_previous(self, api, make_manager):
        api.conflicts.add(CTRL_ALT_F9_COMBO)
        mgr, rec = make_manager()
        mgr.register(ALT_Q)
        with pytest.raises(HotkeyConflictError) as ei:
            mgr.register(CTRL_ALT_F9)
        err = ei.value
        assert err.win32_error == EVENT_HOT_KEY_EXISTS_ERR
        assert err.hotkey == "Ctrl+Alt+F9"
        assert isinstance(err, AppError)
        assert err.user_message
        # the previous hotkey is still registered and still works
        assert mgr.current == ALT_Q
        assert api.active_combos() == {ALT_Q_COMBO}
        assert api.unregister_calls == []
        api.deliver_hotkey()
        assert rec.count == 1

    def test_any_register_failure_raises_hotkey_conflict_error(self, api, make_manager):
        api.register_status = PARAM_ERR
        mgr, _ = make_manager()
        with pytest.raises(HotkeyConflictError) as ei:
            mgr.register(ALT_Q)
        assert ei.value.win32_error == PARAM_ERR
        assert mgr.current is None

    def test_conflict_without_previous(self, api, make_manager):
        api.conflicts.add(ALT_Q_COMBO)
        mgr, _ = make_manager()
        with pytest.raises(HotkeyConflictError) as ei:
            mgr.register(ALT_Q)
        assert ei.value.win32_error == EVENT_HOT_KEY_EXISTS_ERR
        assert mgr.current is None
        assert api.registrations == []

    def test_after_conflict_a_new_register_works(self, api, make_manager):
        api.conflicts.add(CTRL_ALT_F9_COMBO)
        mgr, rec = make_manager()
        mgr.register(ALT_Q)
        with pytest.raises(HotkeyConflictError):
            mgr.register(CTRL_ALT_F9)
        mgr.register(CTRL_SHIFT_D)
        assert api.active_combos() == {CTRL_SHIFT_D_COMBO}
        assert mgr.current == CTRL_SHIFT_D
        api.deliver_hotkey()
        assert rec.count == 1

    def test_unregister_idempotent(self, api, make_manager):
        mgr, _ = make_manager()
        mgr.unregister()  # nothing registered: no error, no call
        assert api.unregister_calls == []
        mgr.register(ALT_Q)
        mgr.unregister()
        assert api.registrations == []
        assert mgr.current is None
        mgr.unregister()
        assert api.registrations == []
        assert len(api.unregister_calls) == 1

    def test_unregister_before_start_is_safe(self, api, make_manager):
        mgr, _ = make_manager(start=False)
        mgr.unregister()
        assert api.registrations == []

    def test_unregistered_hotkey_cannot_fire(self, api, make_manager):
        mgr, rec = make_manager()
        mgr.register(ALT_Q)
        hotkey_id = api.registrations[0]["id"]
        mgr.unregister()
        api.deliver_hotkey(hotkey_id)
        assert rec.count == 0

    def test_current_readable_from_another_thread(self, api, make_manager):
        mgr, _ = make_manager()
        mgr.register(CTRL_SHIFT_D)
        assert run_with_timeout(lambda: mgr.current) == CTRL_SHIFT_D


class TestManagerMarshalling:
    def test_register_from_a_worker_thread_runs_on_the_main_thread(self, api, make_manager):
        mgr, _ = make_manager()
        run_marshalled(api, mgr.register, ALT_Q)
        assert api.active_combos() == {ALT_Q_COMBO}
        assert mgr.current == ALT_Q
        main = threading.get_ident()
        assert {c["ident"] for c in api.register_calls} == {main}

    def test_unregister_from_a_worker_thread(self, api, make_manager):
        mgr, _ = make_manager()
        mgr.register(ALT_Q)
        run_marshalled(api, mgr.unregister)
        assert api.registrations == []
        assert mgr.current is None
        assert {c["ident"] for c in api.unregister_calls} == {threading.get_ident()}

    def test_conflict_from_a_worker_thread_is_reported_to_the_caller(self, api, make_manager):
        api.conflicts.add(CTRL_ALT_F9_COMBO)
        mgr, _ = make_manager()
        mgr.register(ALT_Q)
        with pytest.raises(HotkeyConflictError):
            run_marshalled(api, mgr.register, CTRL_ALT_F9)
        assert mgr.current == ALT_Q
        assert api.active_combos() == {ALT_Q_COMBO}


class TestManagerStop:
    def test_stop_unregisters(self, api, make_manager):
        mgr, _ = make_manager()
        mgr.register(ALT_Q)
        ref = api.registrations[0]["ref"]
        mgr.stop()
        assert api.registrations == []
        assert [c["ref"] for c in api.unregister_calls] == [ref]
        assert mgr.current is None

    def test_stop_removes_the_event_handler_and_the_source(self, api, make_manager):
        mgr, _ = make_manager()
        source = next(iter(api.sources))
        mgr.stop()
        assert api.handlers == {}
        assert api.destroyed_sources == [source]

    def test_stop_twice_is_safe(self, api, make_manager):
        mgr, _ = make_manager()
        mgr.register(ALT_Q)
        mgr.stop()
        mgr.stop()
        assert api.registrations == []
        assert mgr.current is None
        assert len(api.unregister_calls) == 1

    def test_stop_without_register(self, api, make_manager):
        mgr, _ = make_manager()
        mgr.stop()
        assert api.registrations == []
        assert api.handlers == {}

    def test_stop_before_start_is_safe(self, api, make_manager):
        mgr, _ = make_manager(start=False)
        mgr.stop()
        assert api.registrations == []
        assert api.install_calls == 0

    def test_hotkey_cannot_fire_after_stop(self, api, make_manager):
        mgr, rec = make_manager()
        mgr.register(ALT_Q)
        hotkey_id = api.registrations[0]["id"]
        mgr.stop()
        assert api.handlers == {}, "handler still installed after stop()"
        assert hotkey_id not in api.active_ids()
        assert rec.count == 0


class TestManagerFire:
    def test_delivered_event_calls_on_fire_once(self, api, make_manager):
        mgr, rec = make_manager()
        mgr.register(CTRL_SHIFT_D)
        api.deliver_hotkey()
        assert rec.count == 1
        assert rec.calls[0][0] == threading.get_ident()  # delivered on the main thread

    def test_each_delivery_outside_the_window_fires_once(self, api, make_manager):
        mgr, rec = make_manager(debounce_s=0.3)
        mgr.register(CTRL_SHIFT_D)
        for _ in range(4):
            api.deliver_hotkey()
            api.advance(1.0)
        assert rec.count == 4

    def test_stale_event_for_the_inactive_id_is_ignored(self, api, make_manager):
        mgr, rec = make_manager()
        mgr.register(ALT_Q)
        stale_id = api.registrations[0]["id"]
        mgr.register(CTRL_ALT_F9)
        api.deliver_hotkey(stale_id)
        assert rec.count == 0
        api.deliver_hotkey()
        assert rec.count == 1

    def test_event_for_an_unknown_id_is_ignored(self, api, make_manager):
        # Every manager's handler sees every hot key event on the application event target.
        mgr, rec = make_manager()
        mgr.register(ALT_Q)
        api.deliver_hotkey(9999)
        assert rec.count == 0

    def test_event_with_nothing_registered_is_ignored(self, api, make_manager):
        mgr, rec = make_manager()
        api.deliver_hotkey(1)
        assert rec.count == 0

    def test_debounce_drops_second_fire_within_window(self, api, make_manager):
        mgr, rec = make_manager(debounce_s=0.3)
        mgr.register(CTRL_SHIFT_D)
        api.deliver_hotkey()
        api.advance(0.1)
        api.deliver_hotkey()
        assert rec.count == 1
        api.advance(0.4)  # 0.5 s after the accepted fire
        api.deliver_hotkey()
        assert rec.count == 2

    def test_debounce_window_measured_from_previous_accepted_fire(self, api, make_manager):
        mgr, rec = make_manager(debounce_s=0.3)
        mgr.register(CTRL_SHIFT_D)
        api.deliver_hotkey()  # t=0 accepted
        api.advance(0.2)
        api.deliver_hotkey()  # t=0.2 dropped
        api.advance(0.2)
        api.deliver_hotkey()  # t=0.4, i.e. 0.4 s after the accepted one -> accepted
        assert rec.count == 2

    def test_auto_repeat_flood_dropped(self, api, make_manager):
        mgr, rec = make_manager(debounce_s=0.3)
        mgr.register(CTRL_SHIFT_D)
        for _ in range(25):
            api.deliver_hotkey()
            api.advance(0.01)  # 25 repeats over 0.25 s, all inside the window
        assert rec.count == 1

    def test_zero_debounce_accepts_every_event(self, api, make_manager):
        mgr, rec = make_manager(debounce_s=0.0)
        mgr.register(CTRL_SHIFT_D)
        for _ in range(3):
            api.deliver_hotkey()
        assert rec.count == 3

    def test_on_fire_exception_is_swallowed_and_the_manager_keeps_working(self, api, make_manager):
        rec = FireRecorder(api, raise_on=[0])
        mgr, _ = make_manager(on_fire=rec)
        mgr.register(CTRL_SHIFT_D)
        api.deliver_hotkey()  # on_fire raises
        assert rec.count == 1
        api.advance(1.0)
        api.deliver_hotkey()
        assert rec.count == 2
        # and register/unregister/stop still work afterwards
        mgr.register(CTRL_ALT_F9)
        assert api.active_combos() == {CTRL_ALT_F9_COMBO}
        mgr.stop()
        assert api.registrations == []

    def test_no_synthetic_keys_are_posted_when_the_hotkey_fires(self, api, make_manager):
        # macOS has no menu-bar-on-Option behaviour, so the Windows VK_MENU_MASK step does
        # not exist here: firing the hotkey must post no events at all.
        mgr, rec = make_manager()
        mgr.register(ALT_Q)
        api.hold(0x00080000)  # Option physically down
        api.deliver_hotkey()
        assert rec.count == 1
        assert api.posted_unicode == []
        assert api.posted_chords == []
