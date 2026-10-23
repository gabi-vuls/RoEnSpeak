"""Unit tests for Win32SelectionGrabber (prompt adjustment: copy the selection with Ctrl+C),
the clipboard read side (Win32Clipboard.get_text / clear) and Win32TextInjector.paste_raw.

Everything goes through FakeWinApi (test_hotkey.py), extended with GetClipboardSequenceNumber
and a "foreground app" that answers Ctrl+C by putting its selection on the clipboard.
"""

from __future__ import annotations

from typing import Optional

import pytest

from app.contracts import (
    IL_HIGH,
    InjectionBlockedError,
    InjectionMode,
    KeyEvent,
    ModifiersHeldError,
    NothingSelectedError,
)
from tests.unit.test_hotkey import FakeWinApi

CF_UNICODETEXT = 13
CF_CUSTOM = 0xC123
VK_CONTROL, VK_C, VK_V, VK_LMENU = 0x11, 0x43, 0x56, 0xA4
CTRL_C = [
    KeyEvent(vk=VK_CONTROL),
    KeyEvent(vk=VK_C),
    KeyEvent(vk=VK_C, key_up=True),
    KeyEvent(vk=VK_CONTROL, key_up=True),
]


def utf16z(text: str) -> bytes:
    return text.encode("utf-16-le") + b"\x00\x00"


class SelectionFakeApi(FakeWinApi):
    """FakeWinApi + clipboard sequence number + a foreground app with a selection."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.seq = 100
        self.selection: Optional[str] = None  # what the foreground app copies on Ctrl+C
        self.copy_delay_s = 0.0  # fake time between Ctrl+C and the app writing the clipboard
        self.send_input_result = self._app_reacts

    def empty_clipboard(self) -> bool:
        ok = super().empty_clipboard()
        if ok:
            self.seq += 1
        return ok

    def set_clipboard_bytes(self, fmt: int, data: bytes) -> bool:
        ok = super().set_clipboard_bytes(fmt, data)
        if ok:
            self.seq += 1
        return ok

    def get_clipboard_sequence_number(self) -> int:
        with self._lock:
            return self.seq

    def _app_reacts(self, events) -> int:
        if list(events) == CTRL_C and self.selection is not None:
            text = self.selection

            def copy() -> None:
                self.clipboard = [[CF_UNICODETEXT, utf16z(text)]]
                self.seq += 1
                self.log.append(("app_copied", text, self.now))

            if self.copy_delay_s:
                self.at(self.now + self.copy_delay_s, copy)
            else:
                with self._lock:
                    copy()
        return len(events)


@pytest.fixture
def api():
    fake = SelectionFakeApi()
    fake.clipboard = [[CF_UNICODETEXT, utf16z("user clipboard")], [CF_CUSTOM, b"\x01\x02"]]
    yield fake
    fake.shutdown()


def make_grabber(api, **kw):
    from app.win32.clipboard import Win32Clipboard
    from app.win32.selection import Win32SelectionGrabber

    clip = Win32Clipboard(api)
    return Win32SelectionGrabber(api, clip, clip, **kw)


def user_clipboard():
    return [(CF_UNICODETEXT, utf16z("user clipboard")), (CF_CUSTOM, b"\x01\x02")]


# ---------------------------------------------------------------------------
# grab()
# ---------------------------------------------------------------------------


def test_grab_copies_the_selection_and_restores_the_clipboard(api):
    api.selection = "Fix the bug in app/core/config.py\nand add tests."
    sel = make_grabber(api).grab()
    assert sel.text == "Fix the bug in app/core/config.py\nand add tests."
    assert sel.hwnd == api.foreground_hwnd
    assert api.sent_events() == CTRL_C
    assert api.clipboard_items() == user_clipboard()
    assert api.clipboard_violations == []


def test_grab_waits_for_a_slow_copy(api):
    api.selection = "slow app"
    api.copy_delay_s = 0.3
    assert make_grabber(api).grab().text == "slow app"
    assert api.clipboard_items() == user_clipboard()


def test_nothing_selected_when_the_clipboard_never_changes(api):
    api.selection = None  # the app copies nothing (no selection)
    with pytest.raises(NothingSelectedError):
        make_grabber(api, copy_timeout_s=0.5).grab()
    assert api.clipboard_items() == user_clipboard()


def test_nothing_selected_when_only_whitespace_is_copied(api):
    api.selection = "  \n\t "
    with pytest.raises(NothingSelectedError):
        make_grabber(api).grab()
    assert api.clipboard_items() == user_clipboard()


def test_waits_for_the_hotkey_modifier_to_be_released_before_copying(api):
    api.selection = "text"
    api.press(VK_LMENU, until=api.now + 0.2)  # Alt of Alt+P still held
    make_grabber(api).grab()
    copy_call = api.send_input_calls[0]
    assert copy_call["time"] >= 1000.2


def test_modifiers_held_too_long(api):
    api.selection = "text"
    api.press(VK_LMENU)
    with pytest.raises(ModifiersHeldError):
        make_grabber(api, modifier_timeout_s=1.0).grab()
    assert api.send_input_calls == []
    assert api.clipboard_items() == user_clipboard()


def test_elevated_foreground_window_is_refused_before_touching_the_clipboard(api):
    api.selection = "text"
    api.integrity[4242] = IL_HIGH
    with pytest.raises(InjectionBlockedError) as excinfo:
        make_grabber(api).grab()
    assert excinfo.value.reason == "elevated"
    assert api.send_input_calls == []
    assert api.clipboard_items() == user_clipboard()


def test_partial_send_input_raises_and_restores(api):
    api.selection = "text"
    api.send_input_result = 2
    with pytest.raises(InjectionBlockedError):
        make_grabber(api).grab()
    assert api.clipboard_items() == user_clipboard()


# ---------------------------------------------------------------------------
# Win32Clipboard read side
# ---------------------------------------------------------------------------


def test_get_text_reads_unicode_text_up_to_the_nul(api):
    from app.win32.clipboard import Win32Clipboard

    api.clipboard = [[CF_UNICODETEXT, utf16z("Ăla bala ș ț 🙂") + b"garbage"]]
    assert Win32Clipboard(api).get_text() == "Ăla bala ș ț 🙂"
    assert api.clipboard_owner is None


def test_get_text_is_none_without_text(api):
    from app.win32.clipboard import Win32Clipboard

    api.clipboard = [[CF_CUSTOM, b"\x01"]]
    assert Win32Clipboard(api).get_text() is None


def test_clear_empties_the_clipboard(api):
    from app.win32.clipboard import Win32Clipboard

    Win32Clipboard(api).clear()
    assert api.clipboard_items() == []
    assert api.clipboard_owner is None


# ---------------------------------------------------------------------------
# paste_raw
# ---------------------------------------------------------------------------


def make_injector(api, mode=InjectionMode.SENDINPUT):
    from app.win32.clipboard import Win32Clipboard
    from app.win32.injector import Win32TextInjector

    return Win32TextInjector(api, Win32Clipboard(api), mode=mode)


@pytest.mark.parametrize("mode", list(InjectionMode))
def test_paste_raw_keeps_newlines_as_crlf_and_always_pastes(api, mode):
    text = "## Goal\nDo X\n\n## Requirements\n- a\n\t- b"
    out = make_injector(api, mode).paste_raw(text)
    assert out is not None and out.mode is InjectionMode.CLIPBOARD
    (call,) = api.send_input_calls
    assert [e.vk for e in call["events"]] == [VK_CONTROL, VK_V, VK_V, VK_CONTROL]
    pasted = dict(call["clipboard"])[CF_UNICODETEXT]
    assert pasted == utf16z(text.replace("\n", "\r\n"))
    assert api.clipboard_items() == user_clipboard()


def test_paste_raw_empty_does_nothing(api):
    assert make_injector(api).paste_raw("") is None
    assert api.send_input_calls == []


def test_paste_raw_refuses_elevated_windows(api):
    api.integrity[4242] = IL_HIGH
    with pytest.raises(InjectionBlockedError):
        make_injector(api).paste_raw("x")
    assert api.send_input_calls == []
