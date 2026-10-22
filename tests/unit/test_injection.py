"""Unit tests for Win32TextInjector and Win32Clipboard (TEST-UNIT-WIN32).

Written from SPEC.md and app/contracts.py only. Everything goes through FakeWinApi (defined
in test_hotkey.py); UIPI detection is the mocked integrity-level path.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import pytest

from app.contracts import (
    IL_HIGH,
    IL_LOW,
    IL_MEDIUM,
    IL_SYSTEM,
    IL_UNTRUSTED,
    MODIFIER_VKS,
    NO_SPACE_BEFORE,
    AppError,
    ClipboardSnapshot,
    InjectionBlockedError,
    InjectionMode,
    InjectionOutcome,
    KeyEvent,
    ModifiersHeldError,
)
from tests.unit.test_hotkey import FakeWinApi

CF_UNICODETEXT = 13
CF_BITMAP = 2
CF_CUSTOM = 0xC123  # a RegisterClipboardFormat-style id
VK_CONTROL, VK_LCONTROL, VK_V = 0x11, 0xA2, 0x56
EXCLUDE_MONITOR_FMT_NAME = "ExcludeClipboardContentFromMonitorProcessing"
CAN_INCLUDE_HISTORY_FMT_NAME = "CanIncludeInClipboardHistory"
DWORD_0 = b"\x00\x00\x00\x00"


def expected_set_text_items(api: FakeWinApi, text: str) -> dict:
    """Clipboard contents set_text must produce: CF_UNICODETEXT plus both history-exclusion
    formats as DWORD 0 (order between formats is not specified)."""
    return {
        CF_UNICODETEXT: utf16z(text),
        api.register_clipboard_format(EXCLUDE_MONITOR_FMT_NAME): DWORD_0,
        api.register_clipboard_format(CAN_INCLUDE_HISTORY_FMT_NAME): DWORD_0,
    }


def as_dict(items) -> dict:
    items = list(items)
    d = dict(items)
    assert len(d) == len(items), f"duplicate formats on clipboard: {items}"
    return d


def unicode_events(text: str) -> list[KeyEvent]:
    data = text.encode("utf-16-le", "surrogatepass")
    units = [chr(int.from_bytes(data[i : i + 2], "little")) for i in range(0, len(data), 2)]
    events: list[KeyEvent] = []
    for u in units:
        events.append(KeyEvent(vk=0, unicode_char=u, key_up=False))
        events.append(KeyEvent(vk=0, unicode_char=u, key_up=True))
    return events


def utf16z(text: str) -> bytes:
    return text.encode("utf-16-le", "surrogatepass") + b"\x00\x00"


class RecordingClipboard:
    """ClipboardBackend double that logs into the FakeWinApi log (for ordering checks)."""

    def __init__(self, api: FakeWinApi) -> None:
        self.api = api
        self.snapshot = ClipboardSnapshot(formats=((CF_UNICODETEXT, utf16z("previous")),))
        self.saved: list[ClipboardSnapshot] = []
        self.set_texts: list[str] = []
        self.restored: list[ClipboardSnapshot] = []

    def save(self) -> ClipboardSnapshot:
        self.saved.append(self.snapshot)
        self.api.record("clip_save", self.snapshot)
        return self.snapshot

    def set_text(self, text: str) -> None:
        self.set_texts.append(text)
        self.api.record("clip_set_text", text)

    def restore(self, snapshot: ClipboardSnapshot) -> None:
        self.restored.append(snapshot)
        self.api.record("clip_restore", snapshot)


@pytest.fixture
def api():
    fake = FakeWinApi()
    yield fake
    fake.shutdown()


@pytest.fixture
def clip(api):
    return RecordingClipboard(api)


def make_injector(api, clipboard=None, *, mode=InjectionMode.SENDINPUT, modifier_timeout_s=5.0, poll_s=0.01,
                  paste_settle_s=0.15):
    from app.win32.injector import Win32TextInjector

    return Win32TextInjector(
        api,
        clipboard if clipboard is not None else RecordingClipboard(api),
        mode=mode,
        modifier_timeout_s=modifier_timeout_s,
        poll_s=poll_s,
        paste_settle_s=paste_settle_s,
    )


def make_clipboard(api):
    from app.win32.clipboard import Win32Clipboard

    return Win32Clipboard(api)


def typed_text(events: list[KeyEvent]) -> str:
    """Reassemble the text typed by unicode key-down events."""
    units = "".join(ev.unicode_char for ev in events if ev.unicode_char is not None and not ev.key_up)
    return units.encode("utf-16-le", "surrogatepass").decode("utf-16-le")


# ===========================================================================
# SendInput unicode
# ===========================================================================


class TestSendInputUnicode:
    def test_single_call_down_up_pairs(self, api):
        inj = make_injector(api)
        out = inj.inject("Hello", 1)
        assert len(api.send_input_calls) == 1
        assert api.send_input_calls[0]["events"] == unicode_events("Hello")
        assert isinstance(out, InjectionOutcome)
        assert out.injected_text == "Hello"
        assert out.mode == InjectionMode.SENDINPUT
        assert out.events_sent == 10

    def test_romanian_diacritics(self, api):
        text = "ăîșțâ ĂÎȘȚÂ şţ"
        inj = make_injector(api)
        out = inj.inject(text, 1)
        events = api.send_input_calls[0]["events"]
        assert events == unicode_events(text)
        assert all(ev.vk == 0 for ev in events)
        assert typed_text(events) == text
        assert out.events_sent == 2 * len(text)

    def test_surrogate_pair_sent_as_two_code_units(self, api):
        inj = make_injector(api)
        out = inj.inject("Salut 😀", 1)
        events = api.send_input_calls[0]["events"]
        assert len(events) == 2 * 8  # "Salut " = 6 units + emoji = 2 units
        downs = [ev.unicode_char for ev in events if not ev.key_up]
        assert downs[-2:] == ["\ud83d", "\ude00"]
        assert all(len(ev.unicode_char) == 1 for ev in events)
        assert events == unicode_events("Salut 😀")
        assert out.events_sent == 16
        assert out.injected_text == "Salut 😀"

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("  Hello world  ", "Hello world"),
            ("Hello\nworld", "Hello world"),
            ("Hello\r\nworld", "Hello world"),
            ("Hello\tworld", "Hello world"),
            ("\n\tHello\r\nnew\tworld\n", "Hello new world"),
        ],
    )
    def test_strip_and_whitespace_normalization(self, api, raw, expected):
        inj = make_injector(api)
        out = inj.inject(raw, 1)
        assert out.injected_text == expected
        assert api.send_input_calls[0]["events"] == unicode_events(expected)
        sent = typed_text(api.send_input_calls[0]["events"])
        assert "\n" not in sent and "\r" not in sent and "\t" not in sent

    @pytest.mark.parametrize("raw", ["", " ", "   ", "\n", "\r\n\t ", "\t"])
    def test_empty_returns_none_without_calls(self, api, clip, raw):
        inj = make_injector(api, clip)
        assert inj.inject(raw, 1) is None
        assert api.send_input_calls == []
        assert clip.saved == [] and clip.set_texts == [] and clip.restored == []

    def test_t_injected_from_api_clock(self, api):
        api.now = 5000.0
        api.send_input_cost_s = 0.004
        inj = make_injector(api)
        out = inj.inject("Hello", 1)
        # after SendInput returned (fake clock advanced by its cost), and not time.monotonic()
        assert out.t_injected == pytest.approx(5000.004, abs=1e-6)

    def test_injects_from_any_single_worker_thread(self, api):
        inj = make_injector(api)
        from tests.unit.test_hotkey import run_with_timeout

        out = run_with_timeout(inj.inject, "Hi", 3)
        assert out.injected_text == "Hi"


# ===========================================================================
# Spacing between segments
# ===========================================================================


class TestSpacing:
    def test_first_segment_no_leading_space(self, api):
        inj = make_injector(api)
        assert inj.inject("Hello", 1).injected_text == "Hello"

    def test_second_segment_same_session_gets_space(self, api):
        inj = make_injector(api)
        inj.inject("Hello", 1)
        out = inj.inject("world", 1)
        assert out.injected_text == " world"
        assert api.send_input_calls[1]["events"] == unicode_events(" world")
        assert out.events_sent == 12

    def test_third_segment_also_gets_space(self, api):
        inj = make_injector(api)
        inj.inject("One.", 1)
        inj.inject("Two.", 1)
        assert inj.inject("Three.", 1).injected_text == " Three."

    @pytest.mark.parametrize("ch", list(NO_SPACE_BEFORE))
    def test_no_space_before_closing_punctuation(self, api, ch):
        inj = make_injector(api)
        inj.inject("Hello", 1)
        out = inj.inject(ch + "x", 1)
        assert out.injected_text == ch + "x"

    def test_new_session_resets_spacing(self, api):
        inj = make_injector(api)
        inj.inject("Hello", 1)
        assert inj.inject("World", 2).injected_text == "World"
        assert inj.inject("again", 2).injected_text == " again"

    def test_trailing_whitespace_of_previous_is_stripped_so_space_is_added(self, api):
        # The previous text is stripped before typing, so what was typed ("Hello") does not
        # end in whitespace -> the next segment gets a space.
        inj = make_injector(api)
        assert inj.inject("Hello \n", 1).injected_text == "Hello"
        assert inj.inject("world", 1).injected_text == " world"

    def test_leading_whitespace_of_new_text_does_not_double_space(self, api):
        inj = make_injector(api)
        inj.inject("Hello", 1)
        assert inj.inject("   world", 1).injected_text == " world"

    def test_empty_inject_changes_nothing(self, api):
        inj = make_injector(api)
        inj.inject("Hello", 1)
        assert inj.inject("   ", 1) is None
        assert inj.inject("", 2) is None  # empty in another session does not reset session 1
        assert inj.inject("world", 1).injected_text == " world"

    def test_blocked_injection_does_not_update_spacing(self, api):
        inj = make_injector(api)
        api.integrity[4242] = IL_HIGH
        with pytest.raises(InjectionBlockedError):
            inj.inject("Hello", 1)
        api.integrity[4242] = IL_MEDIUM
        assert inj.inject("world", 1).injected_text == "world"

    def test_partial_sendinput_does_not_update_spacing(self, api):
        inj = make_injector(api)
        api.send_input_result = lambda events: len(events) // 2
        with pytest.raises(InjectionBlockedError):
            inj.inject("Hello", 1)
        api.send_input_result = None
        assert inj.inject("world", 1).injected_text == "world"

    def test_modifier_timeout_does_not_update_spacing(self, api):
        inj = make_injector(api, modifier_timeout_s=0.2)
        api.press(0xA2)
        with pytest.raises(ModifiersHeldError):
            inj.inject("Hello", 1)
        api.release(0xA2)
        assert inj.inject("world", 1).injected_text == "world"

    def test_failure_after_success_keeps_previous_state(self, api):
        inj = make_injector(api)
        inj.inject("Hello", 1)
        api.integrity[4242] = None
        with pytest.raises(InjectionBlockedError):
            inj.inject("lost", 1)
        api.integrity[4242] = IL_MEDIUM
        assert inj.inject("world", 1).injected_text == " world"


# ===========================================================================
# Modifier-release wait
# ===========================================================================


class TestModifierWait:
    @pytest.mark.parametrize("vk", MODIFIER_VKS)
    def test_waits_until_modifier_released(self, api, vk):
        inj = make_injector(api, poll_s=0.01)
        release_at = api.now + 0.3
        api.press(vk, until=release_at)
        out = inj.inject("Hello", 1)
        assert out is not None
        assert len(api.send_input_calls) == 1
        sent_at = api.send_input_calls[0]["time"]
        assert sent_at >= release_at
        assert sent_at - release_at < 0.05  # noticed promptly (poll every 10 ms)
        assert api.sleeps, "expected polling via api.sleep"
        assert max(api.sleeps) <= 0.01 + 1e-9
        assert len(api.sleeps) >= 25

    def test_poll_interval_honored(self, api):
        inj = make_injector(api, poll_s=0.05)
        release_at = api.now + 0.5
        api.press(0xA0, until=release_at)
        inj.inject("Hello", 1)
        assert max(api.sleeps) <= 0.05 + 1e-9
        assert 8 <= len(api.sleeps) <= 12

    def test_waits_for_all_modifiers(self, api):
        inj = make_injector(api)
        t0 = api.now
        api.press(0xA2, until=t0 + 0.1)  # Ctrl
        api.press(0xA1, until=t0 + 0.25)  # RShift
        api.press(0x5C, until=t0 + 0.4)  # RWin
        inj.inject("Hello", 1)
        assert api.send_input_calls[0]["time"] >= t0 + 0.4

    def test_no_wait_when_nothing_held(self, api):
        inj = make_injector(api, poll_s=0.01)
        inj.inject("Hello", 1)
        assert sum(api.sleeps) <= 0.01 + 1e-9

    def test_non_modifier_key_held_does_not_block(self, api):
        inj = make_injector(api)
        api.press(0x51)  # Q still held (e.g. the hotkey key)
        assert inj.inject("Hello", 1) is not None

    def test_timeout_raises_without_typing(self, api, clip):
        inj = make_injector(api, clip, modifier_timeout_s=5.0, poll_s=0.01)
        api.press(0xA4)  # Alt held forever
        t0 = api.now
        wall0 = time.monotonic()
        with pytest.raises(ModifiersHeldError) as ei:
            inj.inject("Hello world", 1)
        assert ei.value.text == "Hello world"
        assert isinstance(ei.value, AppError)
        assert api.send_input_calls == []
        assert clip.saved == [] and clip.set_texts == []
        waited = api.now - t0
        assert 5.0 - 0.02 <= waited <= 5.0 + 0.1
        assert max(api.sleeps) <= 0.01 + 1e-9
        # the wait uses the injectable clock, not real time
        assert time.monotonic() - wall0 < 2.0

    def test_timeout_text_is_normalized(self, api):
        inj = make_injector(api, modifier_timeout_s=0.1)
        api.press(0xA0)
        with pytest.raises(ModifiersHeldError) as ei:
            inj.inject("  Hello\nworld ", 1)
        assert ei.value.text == "Hello world"

    def test_timeout_applies_in_clipboard_mode(self, api, clip):
        inj = make_injector(api, clip, mode=InjectionMode.CLIPBOARD, modifier_timeout_s=0.2)
        api.press(0x5B)
        with pytest.raises(ModifiersHeldError):
            inj.inject("Hello", 1)
        assert api.send_input_calls == []
        assert clip.saved == [] and clip.set_texts == [] and clip.restored == []

    def test_modifier_wait_happens_before_uipi_check(self, api):
        # Foreground is elevated while the user still holds Ctrl, then focus moves to a normal
        # window as Ctrl is released: the integrity check must use the window at typing time.
        inj = make_injector(api)
        api.window_pids[0x2000] = 7777
        api.integrity[7777] = IL_HIGH
        api.foreground_hwnd = 0x2000
        t_release = api.now + 0.2

        def switch_focus():
            api.foreground_hwnd = 0x1000

        api.press(0xA2, until=t_release)
        api.at(t_release, switch_focus)
        out = inj.inject("Hello", 1)
        assert out is not None
        assert len(api.send_input_calls) == 1


# ===========================================================================
# UIPI (mocked integrity levels)
# ===========================================================================


class TestUipi:
    @pytest.mark.parametrize("target", [IL_HIGH, IL_SYSTEM])
    def test_elevated_target_blocked(self, api, target):
        api.integrity[4242] = target
        inj = make_injector(api)
        with pytest.raises(InjectionBlockedError) as ei:
            inj.inject("Hello", 1)
        assert ei.value.reason == "elevated"
        assert ei.value.text == "Hello"
        assert isinstance(ei.value, AppError)
        assert api.send_input_calls == []

    def test_unknown_integrity_blocked(self, api):
        api.integrity[4242] = None
        inj = make_injector(api)
        with pytest.raises(InjectionBlockedError) as ei:
            inj.inject("Hello", 1)
        assert ei.value.reason == "unknown_integrity"
        assert ei.value.text == "Hello"
        assert api.send_input_calls == []

    @pytest.mark.parametrize("target", [IL_UNTRUSTED, IL_LOW, IL_MEDIUM])
    def test_lower_or_equal_target_injects(self, api, target):
        # IL_LOW / untrusted covers AppContainer windows such as Start menu search.
        api.integrity[4242] = target
        inj = make_injector(api)
        assert inj.inject("Hello", 1).injected_text == "Hello"
        assert len(api.send_input_calls) == 1

    def test_elevated_self_can_type_into_elevated_target(self, api):
        api.own_integrity = IL_HIGH
        api.integrity[4242] = IL_HIGH
        inj = make_injector(api)
        assert inj.inject("Hello", 1) is not None

    def test_integrity_of_foreground_process_is_used(self, api):
        api.window_pids[0x3000] = 9999
        api.integrity[9999] = IL_HIGH
        api.foreground_hwnd = 0x3000
        inj = make_injector(api)
        with pytest.raises(InjectionBlockedError):
            inj.inject("Hello", 1)
        assert ("integrity_query", 9999) in [(e[0], e[1]) for e in api.log]

    def test_partial_sendinput_reported(self, api):
        api.send_input_result = lambda events: len(events) - 1
        inj = make_injector(api)
        with pytest.raises(InjectionBlockedError) as ei:
            inj.inject("Hello", 1)
        assert ei.value.reason == "sendinput_partial"
        assert ei.value.text == "Hello"
        assert len(api.send_input_calls) == 1

    def test_zero_sendinput_reported(self, api):
        api.send_input_result = 0
        inj = make_injector(api)
        with pytest.raises(InjectionBlockedError) as ei:
            inj.inject("Hello", 1)
        assert ei.value.reason == "sendinput_partial"

    def test_blocked_text_of_later_segment(self, api):
        inj = make_injector(api)
        inj.inject("Hello", 1)
        api.integrity[4242] = IL_HIGH
        with pytest.raises(InjectionBlockedError) as ei:
            inj.inject("world", 1)
        # Whether the leading space is included is unspecified; the words must be there.
        assert ei.value.text.strip() == "world"

    def test_blocked_in_clipboard_mode_touches_nothing(self, api, clip):
        api.integrity[4242] = IL_HIGH
        inj = make_injector(api, clip, mode=InjectionMode.CLIPBOARD)
        with pytest.raises(InjectionBlockedError) as ei:
            inj.inject("Hello", 1)
        assert ei.value.reason == "elevated"
        assert clip.saved == [] and clip.set_texts == [] and clip.restored == []
        assert api.send_input_calls == []


# ===========================================================================
# Clipboard mode (injector side)
# ===========================================================================


def _ctrl_v_ok(events: list[KeyEvent]) -> bool:
    if len(events) != 4 or any(ev.unicode_char is not None for ev in events):
        return False
    ctrl = events[0].vk
    return (
        ctrl in (VK_CONTROL, VK_LCONTROL)
        and [(ev.vk, ev.key_up) for ev in events] == [(ctrl, False), (VK_V, False), (VK_V, True), (ctrl, True)]
    )


class TestClipboardMode:
    def test_order_save_set_paste_settle_restore(self, api, clip):
        inj = make_injector(api, clip, mode=InjectionMode.CLIPBOARD, paste_settle_s=0.15)
        out = inj.inject("Hello", 1)
        kinds = [k for k in api.kinds() if k in ("clip_save", "clip_set_text", "send_input", "clip_restore")]
        assert kinds[0] == "clip_save"
        assert kinds[1] == "clip_set_text"
        assert kinds[-1] == "clip_restore"
        assert set(kinds[2:-1]) == {"send_input"}
        assert _ctrl_v_ok(api.sent_events())
        assert clip.set_texts == ["Hello"]
        assert clip.restored == [clip.snapshot]
        assert clip.restored[0] is clip.saved[0]
        last_send = max(e[2] for e in api.log if e[0] == "send_input")
        restore_t = next(e[2] for e in api.log if e[0] == "clip_restore")
        assert restore_t - last_send >= 0.15 - 1e-9
        assert out.mode == InjectionMode.CLIPBOARD
        assert out.injected_text == "Hello"

    def test_settle_time_configurable(self, api, clip):
        inj = make_injector(api, clip, mode=InjectionMode.CLIPBOARD, paste_settle_s=0.4)
        inj.inject("Hello", 1)
        last_send = max(e[2] for e in api.log if e[0] == "send_input")
        restore_t = next(e[2] for e in api.log if e[0] == "clip_restore")
        assert restore_t - last_send >= 0.4 - 1e-9

    def test_prefixed_text_is_pasted(self, api, clip):
        inj = make_injector(api, clip, mode=InjectionMode.CLIPBOARD)
        inj.inject("Hello", 1)
        out = inj.inject("world\n", 1)
        assert clip.set_texts == ["Hello", " world"]
        assert out.injected_text == " world"

    def test_restore_runs_when_send_input_raises(self, api, clip):
        api.send_input_result = OSError("SendInput exploded")
        inj = make_injector(api, clip, mode=InjectionMode.CLIPBOARD)
        with pytest.raises(Exception):
            inj.inject("Hello", 1)
        assert clip.restored == [clip.snapshot]

    def test_restore_runs_when_send_input_partial(self, api, clip):
        api.send_input_result = 0
        inj = make_injector(api, clip, mode=InjectionMode.CLIPBOARD)
        try:
            inj.inject("Hello", 1)
        except AppError:
            pass  # unspecified whether a partial Ctrl+V raises in clipboard mode
        assert clip.restored == [clip.snapshot]

    def test_set_mode_switches_next_inject(self, api, clip):
        inj = make_injector(api, clip)
        inj.inject("One", 1)
        assert clip.set_texts == []
        inj.set_mode(InjectionMode.CLIPBOARD)
        out = inj.inject("Two", 1)
        assert out.mode == InjectionMode.CLIPBOARD
        assert clip.set_texts == [" Two"]
        inj.set_mode(InjectionMode.SENDINPUT)
        n_calls = len(api.send_input_calls)
        out = inj.inject("Three", 1)
        assert out.mode == InjectionMode.SENDINPUT
        assert api.send_input_calls[n_calls]["events"] == unicode_events(" Three")
        assert clip.set_texts == [" Two"]

    def test_set_mode_from_another_thread(self, api, clip):
        inj = make_injector(api, clip)
        t = threading.Thread(target=inj.set_mode, args=(InjectionMode.CLIPBOARD,))
        t.start()
        t.join(2.0)
        assert not t.is_alive()
        assert inj.inject("Hi", 1).mode == InjectionMode.CLIPBOARD


# ===========================================================================
# Win32Clipboard backend
# ===========================================================================


def _seed_clipboard(api: FakeWinApi, items: list[tuple[int, Optional[bytes]]]) -> None:
    api.clipboard = [[f, d] for f, d in items]


PREVIOUS = [
    (CF_UNICODETEXT, utf16z("old clipboard ăș")),
    (CF_BITMAP, None),  # not HGLOBAL-backed -> skipped by save()
    (CF_CUSTOM, b"\x00\x01\x02binary\xff"),
]


class TestWin32Clipboard:
    def test_save_snapshots_readable_formats_in_order(self, api):
        _seed_clipboard(api, PREVIOUS)
        snap = make_clipboard(api).save()
        assert isinstance(snap, ClipboardSnapshot)
        assert snap.formats == ((CF_UNICODETEXT, utf16z("old clipboard ăș")), (CF_CUSTOM, b"\x00\x01\x02binary\xff"))
        assert api.clipboard_owner is None, "clipboard left open"
        assert api.clipboard_violations == []
        assert api.clipboard_items() == PREVIOUS  # save() does not modify the clipboard

    def test_save_empty_clipboard(self, api):
        snap = make_clipboard(api).save()
        assert snap.formats == ()
        assert api.clipboard_owner is None

    def test_set_text_utf16le_nul_terminated(self, api):
        _seed_clipboard(api, PREVIOUS)
        text = "Bună ziua, ștefan 😀"
        make_clipboard(api).set_text(text)
        assert as_dict(api.clipboard_items()) == expected_set_text_items(api, text)
        # the exclusion formats are distinct registered formats (ids >= 0xC000)
        excl = [f for f, _ in api.clipboard_items() if f != CF_UNICODETEXT]
        assert len(excl) == 2 and all(f >= 0xC000 for f in excl)
        assert "empty_clipboard" in api.kinds()
        assert api.clipboard_owner is None
        assert api.clipboard_violations == []

    def test_restore_exact(self, api):
        _seed_clipboard(api, PREVIOUS)
        cb = make_clipboard(api)
        snap = cb.save()
        cb.set_text("temporary")
        assert as_dict(api.clipboard_items()) == expected_set_text_items(api, "temporary")
        cb.restore(snap)
        # exactly the snapshot: the exclusion formats set by set_text do not leak
        assert api.clipboard_items() == [(f, d) for f, d in PREVIOUS if d is not None]
        assert api.clipboard_owner is None
        assert api.clipboard_violations == []

    def test_restore_keeps_exclusion_formats_that_were_in_snapshot(self, api):
        excl = api.register_clipboard_format(EXCLUDE_MONITOR_FMT_NAME)
        previous = [(CF_UNICODETEXT, utf16z("password123")), (excl, DWORD_0)]
        _seed_clipboard(api, previous)
        cb = make_clipboard(api)
        snap = cb.save()
        assert snap.formats == tuple(previous)
        cb.set_text("dictated")
        cb.restore(snap)
        assert api.clipboard_items() == previous

    def test_restore_after_set_text_with_empty_snapshot_has_no_exclusion_formats(self, api):
        cb = make_clipboard(api)
        snap = cb.save()
        cb.set_text("dictated")
        cb.restore(snap)
        assert api.clipboard_items() == []

    def test_restore_empties_first(self, api):
        _seed_clipboard(api, [(CF_CUSTOM, b"zzz"), (0xC200, b"other")])
        snap = ClipboardSnapshot(formats=((CF_UNICODETEXT, utf16z("x")),))
        make_clipboard(api).restore(snap)
        assert api.clipboard_items() == [(CF_UNICODETEXT, utf16z("x"))]

    def test_restore_empty_snapshot_leaves_clipboard_empty(self, api):
        _seed_clipboard(api, PREVIOUS)
        make_clipboard(api).restore(ClipboardSnapshot(formats=()))
        assert api.clipboard_items() == []
        assert api.clipboard_owner is None

    @pytest.mark.parametrize("failures", [1, 5, 9])
    def test_open_retries_then_succeeds(self, api, failures):
        _seed_clipboard(api, PREVIOUS)
        api.open_fail_remaining = failures
        snap = make_clipboard(api).save()
        assert api.open_attempts == failures + 1
        assert len(snap.formats) == 2
        assert api.clipboard_owner is None

    def test_open_never_succeeds_raises_app_error(self, api):
        api.open_always_fail = True
        wall0 = time.monotonic()
        with pytest.raises(AppError):
            make_clipboard(api).set_text("Hello")
        assert 10 <= api.open_attempts <= 11
        assert api.clipboard_items() == []
        assert time.monotonic() - wall0 < 2.0

    def test_open_failure_on_restore_raises_app_error(self, api):
        api.open_always_fail = True
        with pytest.raises(AppError):
            make_clipboard(api).restore(ClipboardSnapshot(formats=((CF_UNICODETEXT, utf16z("x")),)))


# ===========================================================================
# Injector + Win32Clipboard together (clipboard round trip through the fake)
# ===========================================================================


class TestClipboardRoundTrip:
    def test_paste_then_previous_contents_restored(self, api):
        _seed_clipboard(api, PREVIOUS)
        inj = make_injector(api, make_clipboard(api), mode=InjectionMode.CLIPBOARD)
        out = inj.inject("Salut, lume ăîș", 1)
        assert out.injected_text == "Salut, lume ăîș"
        # while Ctrl+V was sent, the clipboard held exactly our text
        ctrl_v_calls = [c for c in api.send_input_calls if any(ev.vk == VK_V for ev in c["events"])]
        assert ctrl_v_calls
        assert as_dict(ctrl_v_calls[0]["clipboard"]) == expected_set_text_items(api, "Salut, lume ăîș")
        assert _ctrl_v_ok(api.sent_events())
        # afterwards: previous readable formats back, exactly, in order
        assert api.clipboard_items() == [(f, d) for f, d in PREVIOUS if d is not None]
        assert api.clipboard_owner is None
        assert api.clipboard_violations == []

    def test_previous_restored_when_send_input_raises(self, api):
        _seed_clipboard(api, PREVIOUS)
        api.send_input_result = OSError("boom")
        inj = make_injector(api, make_clipboard(api), mode=InjectionMode.CLIPBOARD)
        with pytest.raises(Exception):
            inj.inject("Hello", 1)
        assert api.clipboard_items() == [(f, d) for f, d in PREVIOUS if d is not None]
        assert api.clipboard_owner is None

    def test_empty_previous_clipboard_left_empty(self, api):
        inj = make_injector(api, make_clipboard(api), mode=InjectionMode.CLIPBOARD)
        inj.inject("Hello", 1)
        ctrl_v_calls = [c for c in api.send_input_calls if any(ev.vk == VK_V for ev in c["events"])]
        assert as_dict(ctrl_v_calls[0]["clipboard"]) == expected_set_text_items(api, "Hello")
        assert api.clipboard_items() == []
        assert api.clipboard_owner is None
