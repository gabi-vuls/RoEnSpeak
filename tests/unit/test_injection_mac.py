"""Unit tests for MacTextInjector and MacClipboard (TEST-UNIT-MACOS).

Written from SPEC.md and app/contracts.py only -- the ``TextInjector`` protocol docstring
(normalization, the leading-space rule, the modifier-release wait, the blocked-injection
check, the two injection modes), ``InjectionBlockedError``'s macOS reasons,
``ModifiersHeldError``, ``NO_SPACE_BEFORE``, ``InjectionOutcome`` and ``ClipboardBackend`` /
``ClipboardSnapshot``. Every system call goes through FakeMacApi
(tests/unit/fake_mac_api.py), so these tests need no Accessibility grant, no pasteboard and
no real clock, and run on any platform.

The macOS substitutions for the Windows steps (contracts + PORT_CONTRACT):

    modifier state   GetAsyncKeyState per VK  ->  CGEventSourceFlagsState bits
    "will it work?"  UIPI integrity levels    ->  AXIsProcessTrusted + IsSecureEventInputEnabled
    SENDINPUT mode   SendInput UNICODE        ->  CGEventKeyboardSetUnicodeString + CGEventPost
    CLIPBOARD mode   Ctrl+V                   ->  Cmd+V
"""

from __future__ import annotations

import threading
import time

import pytest

from app.contracts import (
    IS_MACOS,
    NO_SPACE_BEFORE,
    AppError,
    ClipboardSnapshot,
    InjectionBlockedError,
    InjectionMode,
    InjectionOutcome,
    ModifiersHeldError,
)
from tests.unit.fake_mac_api import (
    CG_FLAG_ALPHA_SHIFT,
    CG_FLAG_ALTERNATE,
    CG_FLAG_COMMAND,
    CG_FLAG_CONTROL,
    CG_FLAG_SHIFT,
    HELD_MODIFIER_FLAGS,
    KVK_ANSI_V,
    NON_HELD_FLAGS,
    TYPE_PNG,
    TYPE_RTF,
    TYPE_STRING,
    TYPE_URL,
    FakeMacApi,
    utf16_len,
)
from tests.unit.test_hotkey_mac import run_with_timeout

#: Maximum UTF-16 code units carried by one CGEvent unicode payload (PORT_CONTRACT: the
#: text is posted in chunks of at most this many units, and a surrogate pair is never split).
MAX_UNITS = 20

PREVIOUS = [
    (TYPE_STRING, "old clipboard ăș".encode("utf-8")),
    (TYPE_PNG, None),  # data cannot be read -> skipped by save()
    (TYPE_RTF, b"{\\rtf1\\ansi\x00\x01binary\xff}"),
]
#: PREVIOUS as save()/restore() round-trip it.
PREVIOUS_SAVED = tuple((t, d) for t, d in PREVIOUS if d is not None)


class RecordingClipboard:
    """ClipboardBackend double that logs into the FakeMacApi log (for ordering checks)."""

    def __init__(self, api: FakeMacApi) -> None:
        self.api = api
        self.snapshot = ClipboardSnapshot(formats=((TYPE_STRING, b"previous"),))
        self.saved: list[ClipboardSnapshot] = []
        self.set_texts: list[str] = []
        self.restored: list[ClipboardSnapshot] = []
        self.set_text_error: BaseException | None = None
        self.restore_error: BaseException | None = None

    def save(self) -> ClipboardSnapshot:
        self.saved.append(self.snapshot)
        self.api.record("clip_save", self.snapshot)
        return self.snapshot

    def set_text(self, text: str) -> None:
        self.set_texts.append(text)
        self.api.record("clip_set_text", text)
        if self.set_text_error is not None:
            raise self.set_text_error

    def restore(self, snapshot: ClipboardSnapshot) -> None:
        self.restored.append(snapshot)
        self.api.record("clip_restore", snapshot)
        if self.restore_error is not None:
            raise self.restore_error

    def untouched(self) -> bool:
        return self.saved == [] and self.set_texts == [] and self.restored == []


@pytest.fixture
def api():
    return FakeMacApi()


@pytest.fixture
def clip(api):
    return RecordingClipboard(api)


def make_injector(api, clipboard=None, *, mode=InjectionMode.SENDINPUT, modifier_timeout_s=5.0,
                  poll_s=0.01, paste_settle_s=0.15):
    from app.macos.injector import MacTextInjector

    return MacTextInjector(
        api,
        clipboard if clipboard is not None else RecordingClipboard(api),
        mode=mode,
        modifier_timeout_s=modifier_timeout_s,
        poll_s=poll_s,
        paste_settle_s=paste_settle_s,
    )


def make_clipboard(api):
    from app.macos.clipboard import MacClipboard

    return MacClipboard(api)


def chunks(api: FakeMacApi) -> list[str]:
    return [call["text"] for call in api.posted_unicode]


def log_time(api: FakeMacApi, kind: str, *, last: bool = False) -> float:
    entries = [e[2] for e in api.log if e[0] == kind]
    assert entries, f"no {kind} entry in the log"
    return entries[-1] if last else entries[0]


# ===========================================================================
# Unicode injection (CGEventKeyboardSetUnicodeString)
# ===========================================================================


class TestUnicodeInjection:
    def test_single_chunk_for_short_text(self, api):
        inj = make_injector(api)
        out = inj.inject("Hello", 1)
        assert chunks(api) == ["Hello"]
        assert isinstance(out, InjectionOutcome)
        assert out.injected_text == "Hello"
        assert out.mode == InjectionMode.SENDINPUT
        assert out.events_sent == 2  # one key down + one key up carrying the payload

    def test_romanian_diacritics_preserved(self, api):
        text = "ăîșțâ ĂÎȘȚÂ şţ"
        inj = make_injector(api)
        out = inj.inject(text, 1)
        assert api.typed_text() == text
        assert out.injected_text == text
        assert "".join(chunks(api)) == text

    @pytest.mark.parametrize("n", [1, 19, 20, 21, 40, 41, 97])
    def test_chunks_never_exceed_the_unicode_payload_limit(self, api, n):
        text = "".join(chr(ord("a") + i % 26) for i in range(n))
        inj = make_injector(api)
        out = inj.inject(text, 1)
        pieces = chunks(api)
        assert "".join(pieces) == text
        assert all(utf16_len(piece) <= MAX_UNITS for piece in pieces)
        assert all(piece for piece in pieces)
        assert len(pieces) == -(-n // MAX_UNITS)  # ceil
        assert out.events_sent == 2 * len(pieces)
        assert out.injected_text == text

    def test_long_diacritic_text_is_chunked_without_loss(self, api):
        text = "Bună dimineața, mă bucur să te văd astăzi la birou și mâine iar." * 3
        inj = make_injector(api)
        inj.inject(text, 1)
        assert api.typed_text() == text
        assert all(utf16_len(piece) <= MAX_UNITS for piece in chunks(api))

    def test_surrogate_pair_is_never_split_across_chunks(self, api):
        # 19 BMP chars (19 units) + an astral char (2 units) = 21 > 20, so the emoji must
        # move whole into the next chunk instead of being cut in half.
        text = "a" * 19 + "😀" + "b" * 19 + "😀"
        inj = make_injector(api)
        inj.inject(text, 1)
        pieces = chunks(api)
        assert "".join(pieces) == text
        for piece in pieces:
            assert utf16_len(piece) <= MAX_UNITS
            # a lone surrogate cannot be encoded without surrogatepass
            assert piece.encode("utf-16-le").decode("utf-16-le") == piece

    def test_astral_only_text(self, api):
        text = "😀" * 15  # 30 UTF-16 units
        inj = make_injector(api)
        inj.inject(text, 1)
        pieces = chunks(api)
        assert "".join(pieces) == text
        assert all(utf16_len(piece) <= MAX_UNITS for piece in pieces)

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("  Hello world  ", "Hello world"),
            ("Hello\nworld", "Hello world"),
            ("Hello\r\nworld", "Hello world"),
            ("Hello\tworld", "Hello world"),
            ("\n\tHello\r\nnew\tworld\n", "Hello new world"),
            ("Hello \n world", "Hello world"),
            ("Hello\n\n\nworld", "Hello world"),
        ],
    )
    def test_strip_and_whitespace_normalization(self, api, raw, expected):
        inj = make_injector(api)
        out = inj.inject(raw, 1)
        assert out.injected_text == expected
        typed = api.typed_text()
        assert typed == expected
        assert not {"\n", "\r", "\t"} & set(typed)

    def test_internal_single_spaces_are_kept(self, api):
        inj = make_injector(api)
        assert inj.inject("one two three", 1).injected_text == "one two three"

    @pytest.mark.parametrize("raw", ["", " ", "   ", "\n", "\r\n\t ", "\t", "\r"])
    def test_empty_returns_none_and_changes_nothing(self, api, clip, raw):
        inj = make_injector(api, clip)
        assert inj.inject(raw, 1) is None
        assert api.posted_unicode == []
        assert api.posted_chords == []
        assert clip.untouched()

    def test_t_injected_comes_from_the_api_clock(self, api):
        api.now = 5000.0
        api.post_cost_s = 0.004
        inj = make_injector(api)
        out = inj.inject("Hello", 1)
        # after the events were posted (the fake clock advanced by the posting cost)
        assert out.t_injected == pytest.approx(5000.004, abs=1e-6)

    def test_t_injected_after_every_chunk(self, api):
        api.now = 100.0
        api.post_cost_s = 0.001
        inj = make_injector(api)
        out = inj.inject("x" * 60, 1)  # 3 chunks
        assert out.t_injected == pytest.approx(100.003, abs=1e-6)

    def test_injects_from_a_worker_thread(self, api):
        inj = make_injector(api)
        out = run_with_timeout(inj.inject, "Hi", 3)
        assert out.injected_text == "Hi"
        assert api.posted_unicode[0]["ident"] != threading.get_ident()


# ===========================================================================
# Spacing between segments
# ===========================================================================


class TestSpacing:
    def test_first_segment_of_a_session_has_no_leading_space(self, api):
        inj = make_injector(api)
        assert inj.inject("Hello", 1).injected_text == "Hello"

    def test_second_segment_same_session_gets_a_space(self, api):
        inj = make_injector(api)
        inj.inject("Hello", 1)
        out = inj.inject("world", 1)
        assert out.injected_text == " world"
        assert api.typed_text() == "Hello world"

    def test_third_segment_also_gets_a_space(self, api):
        inj = make_injector(api)
        inj.inject("One.", 1)
        inj.inject("Two.", 1)
        assert inj.inject("Three.", 1).injected_text == " Three."
        assert api.typed_text() == "One. Two. Three."

    @pytest.mark.parametrize("ch", list(NO_SPACE_BEFORE))
    def test_no_space_before_closing_punctuation(self, api, ch):
        inj = make_injector(api)
        inj.inject("Hello", 1)
        assert inj.inject(ch + "x", 1).injected_text == ch + "x"

    def test_new_session_resets_spacing(self, api):
        inj = make_injector(api)
        inj.inject("Hello", 1)
        assert inj.inject("World", 2).injected_text == "World"
        assert inj.inject("again", 2).injected_text == " again"

    def test_returning_to_an_older_session_id_does_not_resume_its_spacing(self, api):
        # The contract keys spacing on "a previous inject() in the SAME session_id", and only
        # the last successful injection's session is remembered.
        inj = make_injector(api)
        inj.inject("Hello", 1)
        inj.inject("World", 2)
        assert inj.inject("again", 1).injected_text == "again"

    def test_trailing_whitespace_of_previous_is_stripped_so_a_space_is_added(self, api):
        inj = make_injector(api)
        assert inj.inject("Hello \n", 1).injected_text == "Hello"
        assert inj.inject("world", 1).injected_text == " world"

    def test_leading_whitespace_of_new_text_does_not_double_space(self, api):
        inj = make_injector(api)
        inj.inject("Hello", 1)
        assert inj.inject("   world", 1).injected_text == " world"
        assert api.typed_text() == "Hello world"

    def test_empty_inject_changes_nothing(self, api):
        inj = make_injector(api)
        inj.inject("Hello", 1)
        assert inj.inject("   ", 1) is None
        assert inj.inject("", 2) is None  # empty in another session does not reset session 1
        assert inj.inject("world", 1).injected_text == " world"

    def test_blocked_injection_does_not_advance_spacing_state(self, api):
        inj = make_injector(api)
        api.trusted = False
        with pytest.raises(InjectionBlockedError):
            inj.inject("Hello", 1)
        api.trusted = True
        assert inj.inject("world", 1).injected_text == "world"

    def test_failed_post_does_not_advance_spacing_state(self, api):
        inj = make_injector(api)
        api.post_unicode_result = 1  # only the key-down made it out
        with pytest.raises(InjectionBlockedError):
            inj.inject("Hello", 1)
        api.post_unicode_result = None
        assert inj.inject("world", 1).injected_text == "world"

    def test_modifier_timeout_does_not_advance_spacing_state(self, api):
        inj = make_injector(api, modifier_timeout_s=0.2)
        api.hold(CG_FLAG_CONTROL)
        with pytest.raises(ModifiersHeldError):
            inj.inject("Hello", 1)
        api.release(CG_FLAG_CONTROL)
        assert inj.inject("world", 1).injected_text == "world"

    def test_failure_after_a_success_keeps_the_previous_state(self, api):
        inj = make_injector(api)
        inj.inject("Hello", 1)
        api.secure_input = True
        with pytest.raises(InjectionBlockedError):
            inj.inject("lost", 1)
        api.secure_input = False
        assert inj.inject("world", 1).injected_text == " world"


# ===========================================================================
# Modifier-release wait (CGEventSourceFlagsState)
# ===========================================================================


class TestModifierWait:
    @pytest.mark.parametrize("flag", HELD_MODIFIER_FLAGS)
    def test_waits_until_the_modifier_is_released(self, api, flag):
        inj = make_injector(api, poll_s=0.01)
        release_at = api.now + 0.3
        api.hold(flag, until=release_at)
        out = inj.inject("Hello", 1)
        assert out is not None
        assert len(api.posted_unicode) == 1
        posted_at = api.posted_unicode[0]["time"]
        assert posted_at >= release_at
        assert posted_at - release_at < 0.05  # noticed promptly (poll every 10 ms)
        assert api.sleeps, "expected polling through api.sleep"
        assert max(api.sleeps) <= 0.01 + 1e-9
        assert len(api.sleeps) >= 25

    def test_poll_interval_honoured(self, api):
        inj = make_injector(api, poll_s=0.05)
        api.hold(CG_FLAG_SHIFT, until=api.now + 0.5)
        inj.inject("Hello", 1)
        assert max(api.sleeps) <= 0.05 + 1e-9
        assert 8 <= len(api.sleeps) <= 12

    def test_waits_for_all_four_modifiers(self, api):
        inj = make_injector(api)
        t0 = api.now
        api.hold(CG_FLAG_CONTROL, until=t0 + 0.1)
        api.hold(CG_FLAG_SHIFT, until=t0 + 0.25)
        api.hold(CG_FLAG_COMMAND, until=t0 + 0.4)
        api.hold(CG_FLAG_ALTERNATE, until=t0 + 0.2)
        inj.inject("Hello", 1)
        assert api.posted_unicode[0]["time"] >= t0 + 0.4

    def test_no_wait_when_nothing_is_held(self, api):
        inj = make_injector(api, poll_s=0.01)
        inj.inject("Hello", 1)
        assert api.sleeps == []

    @pytest.mark.parametrize("flag", NON_HELD_FLAGS)
    def test_latched_and_informational_flags_do_not_block(self, api, flag):
        # Caps Lock is a latch, not a held key: waiting for it would drop every segment for
        # as long as Caps Lock is on. The numeric-pad / help / fn / non-coalesced bits are
        # likewise not modifier keys the user can release.
        inj = make_injector(api, modifier_timeout_s=0.2, poll_s=0.01)
        api.hold(flag)
        out = inj.inject("Hello", 1)
        assert out is not None
        assert api.sleeps == []
        assert api.typed_text() == "Hello"

    def test_caps_lock_on_does_not_extend_the_wait_for_a_real_modifier(self, api):
        inj = make_injector(api, poll_s=0.01)
        api.hold(CG_FLAG_ALPHA_SHIFT)  # Caps Lock stays on for the whole test
        release_at = api.now + 0.2
        api.hold(CG_FLAG_COMMAND, until=release_at)
        out = inj.inject("Hello", 1)
        assert out is not None
        assert api.flags & CG_FLAG_ALPHA_SHIFT, "Caps Lock was expected to stay latched"
        assert api.posted_unicode[0]["time"] < release_at + 0.05

    def test_timeout_raises_without_typing(self, api, clip):
        inj = make_injector(api, clip, modifier_timeout_s=5.0, poll_s=0.01)
        api.hold(CG_FLAG_ALTERNATE)  # Option held forever
        t0 = api.now
        wall0 = time.monotonic()
        with pytest.raises(ModifiersHeldError) as ei:
            inj.inject("Hello world", 1)
        assert ei.value.text == "Hello world"
        assert isinstance(ei.value, AppError)
        assert ei.value.user_message
        assert api.posted_unicode == []
        assert api.posted_chords == []
        assert clip.untouched()
        waited = api.now - t0
        assert 5.0 - 0.02 <= waited <= 5.0 + 0.1
        assert max(api.sleeps) <= 0.01 + 1e-9
        # the wait uses the injectable clock, not real time
        assert time.monotonic() - wall0 < 2.0

    @pytest.mark.skipif(not IS_MACOS, reason="ModifiersHeldError names the host's keys")
    def test_timeout_message_names_the_macos_keys(self, api):
        inj = make_injector(api, modifier_timeout_s=0.1)
        api.hold(CG_FLAG_COMMAND)
        with pytest.raises(ModifiersHeldError) as ei:
            inj.inject("Hello", 1)
        assert "Command" in ei.value.user_message
        assert "Option" in ei.value.user_message

    def test_timeout_text_is_the_normalized_text(self, api):
        inj = make_injector(api, modifier_timeout_s=0.1)
        api.hold(CG_FLAG_SHIFT)
        with pytest.raises(ModifiersHeldError) as ei:
            inj.inject("  Hello\nworld ", 1)
        assert ei.value.text == "Hello world"

    def test_timeout_applies_in_clipboard_mode(self, api, clip):
        inj = make_injector(api, clip, mode=InjectionMode.CLIPBOARD, modifier_timeout_s=0.2)
        api.hold(CG_FLAG_COMMAND)
        with pytest.raises(ModifiersHeldError):
            inj.inject("Hello", 1)
        assert api.posted_chords == []
        assert clip.untouched()

    def test_timeout_is_configurable(self, api):
        inj = make_injector(api, modifier_timeout_s=1.0, poll_s=0.01)
        api.hold(CG_FLAG_CONTROL)
        t0 = api.now
        with pytest.raises(ModifiersHeldError):
            inj.inject("Hello", 1)
        assert 1.0 - 0.02 <= api.now - t0 <= 1.0 + 0.1

    def test_modifier_wait_happens_before_the_permission_check(self, api):
        # The user holds Ctrl while a password field has the keyboard; by the time Ctrl is
        # released focus has moved to a normal field. The check must use the state at typing
        # time, not at call time.
        inj = make_injector(api)
        api.secure_input = True
        release_at = api.now + 0.2
        api.hold(CG_FLAG_CONTROL, until=release_at)
        api.at(release_at, lambda: setattr(api, "secure_input", False))
        out = inj.inject("Hello", 1)
        assert out is not None
        assert len(api.posted_unicode) == 1


# ===========================================================================
# Blocked injection (Accessibility / secure event input / CGEvent failure)
# ===========================================================================


class TestInjectionBlocked:
    def test_no_accessibility(self, api, clip):
        api.trusted = False
        inj = make_injector(api, clip)
        with pytest.raises(InjectionBlockedError) as ei:
            inj.inject("Hello", 1)
        assert ei.value.reason == "no_accessibility"
        assert ei.value.text == "Hello"
        assert isinstance(ei.value, AppError)
        assert "Accessibility" in ei.value.user_message
        assert api.posted_unicode == []
        assert clip.untouched()

    def test_secure_input(self, api, clip):
        api.secure_input = True
        inj = make_injector(api, clip)
        with pytest.raises(InjectionBlockedError) as ei:
            inj.inject("Hello", 1)
        assert ei.value.reason == "secure_input"
        assert ei.value.text == "Hello"
        assert "password" in ei.value.user_message.lower()
        assert api.posted_unicode == []
        assert clip.untouched()

    def test_missing_accessibility_is_reported_before_secure_input(self, api):
        api.trusted = False
        api.secure_input = True
        inj = make_injector(api)
        with pytest.raises(InjectionBlockedError) as ei:
            inj.inject("Hello", 1)
        assert ei.value.reason == "no_accessibility"

    @pytest.mark.parametrize("posted", [0, 1])
    def test_cgevent_failed_when_the_event_cannot_be_posted(self, api, posted):
        api.post_unicode_result = posted
        inj = make_injector(api)
        with pytest.raises(InjectionBlockedError) as ei:
            inj.inject("Hello", 1)
        assert ei.value.reason == "cgevent_failed"
        assert ei.value.text == "Hello"
        assert ei.value.user_message

    def test_cgevent_failed_stops_at_the_failing_chunk(self, api):
        # 60 chars = 3 chunks; the second one fails, so the third is never posted.
        api.post_unicode_result = lambda text: 2 if len(api.posted_unicode) < 2 else 0
        inj = make_injector(api)
        with pytest.raises(InjectionBlockedError) as ei:
            inj.inject("x" * 60, 1)
        assert ei.value.reason == "cgevent_failed"
        assert len(api.posted_unicode) == 2

    def test_cgevent_failed_when_the_permission_state_cannot_be_read(self, api):
        api.permissions_error = OSError("ApplicationServices is unavailable")
        inj = make_injector(api)
        with pytest.raises(InjectionBlockedError) as ei:
            inj.inject("Hello", 1)
        assert ei.value.reason == "cgevent_failed"
        assert ei.value.text == "Hello"
        assert api.posted_unicode == []

    def test_blocked_text_of_a_later_segment(self, api):
        inj = make_injector(api)
        inj.inject("Hello", 1)
        api.trusted = False
        with pytest.raises(InjectionBlockedError) as ei:
            inj.inject("world", 1)
        # Whether the leading space is included is unspecified; the words must be there.
        assert ei.value.text.strip() == "world"

    @pytest.mark.parametrize("reason_setup", ["no_accessibility", "secure_input"])
    def test_blocked_in_clipboard_mode_touches_nothing(self, api, clip, reason_setup):
        if reason_setup == "no_accessibility":
            api.trusted = False
        else:
            api.secure_input = True
        inj = make_injector(api, clip, mode=InjectionMode.CLIPBOARD)
        with pytest.raises(InjectionBlockedError) as ei:
            inj.inject("Hello", 1)
        assert ei.value.reason == reason_setup
        assert clip.untouched()
        assert api.posted_chords == []

    def test_an_unreadable_frontmost_app_does_not_mask_the_real_reason(self, api):
        # frontmost_app() only decorates the log line; it must not change the outcome.
        api.trusted = False
        api.frontmost_error = OSError("NSWorkspace is unavailable")
        inj = make_injector(api)
        with pytest.raises(InjectionBlockedError) as ei:
            inj.inject("Hello", 1)
        assert ei.value.reason == "no_accessibility"

    def test_granted_accessibility_and_no_secure_input_injects(self, api):
        api.trusted = True
        api.secure_input = False
        inj = make_injector(api)
        assert inj.inject("Hello", 1).injected_text == "Hello"
        assert api.typed_text() == "Hello"


# ===========================================================================
# Clipboard mode (injector side): save -> set text -> Cmd+V -> settle -> restore
# ===========================================================================


def _cmd_v_only(api: FakeMacApi) -> bool:
    return [(c["keycode"], c["flags"]) for c in api.posted_chords] == [(KVK_ANSI_V, CG_FLAG_COMMAND)]


class TestClipboardMode:
    def test_order_save_set_paste_settle_restore(self, api, clip):
        inj = make_injector(api, clip, mode=InjectionMode.CLIPBOARD, paste_settle_s=0.15)
        out = inj.inject("Hello", 1)
        kinds = [k for k in api.kinds()
                 if k in ("clip_save", "clip_set_text", "post_chord", "clip_restore")]
        assert kinds == ["clip_save", "clip_set_text", "post_chord", "clip_restore"]
        assert _cmd_v_only(api)
        assert api.posted_unicode == [], "clipboard mode must not type the characters"
        assert clip.set_texts == ["Hello"]
        assert clip.restored == [clip.snapshot]
        assert clip.restored[0] is clip.saved[0]
        assert log_time(api, "clip_restore") - log_time(api, "post_chord") >= 0.15 - 1e-9
        assert out.mode == InjectionMode.CLIPBOARD
        assert out.injected_text == "Hello"
        assert out.events_sent == 2
        assert log_time(api, "post_chord") <= out.t_injected < log_time(api, "clip_restore")

    def test_settle_time_configurable(self, api, clip):
        inj = make_injector(api, clip, mode=InjectionMode.CLIPBOARD, paste_settle_s=0.4)
        inj.inject("Hello", 1)
        assert log_time(api, "clip_restore") - log_time(api, "post_chord") >= 0.4 - 1e-9

    def test_cmd_v_uses_the_carbon_v_key_code_with_the_command_flag(self, api, clip):
        inj = make_injector(api, clip, mode=InjectionMode.CLIPBOARD)
        inj.inject("Hello", 1)
        chord = api.posted_chords[0]
        assert chord["keycode"] == 0x09  # kVK_ANSI_V
        assert chord["flags"] == CG_FLAG_COMMAND
        assert chord["flags"] & (CG_FLAG_CONTROL | CG_FLAG_SHIFT | CG_FLAG_ALTERNATE) == 0

    def test_prefixed_text_is_what_gets_pasted(self, api, clip):
        inj = make_injector(api, clip, mode=InjectionMode.CLIPBOARD)
        inj.inject("Hello", 1)
        out = inj.inject("world\n", 1)
        assert clip.set_texts == ["Hello", " world"]
        assert out.injected_text == " world"

    def test_long_text_is_pasted_in_one_go(self, api, clip):
        # The chunk limit is a CGEvent payload limit; pasting has no such limit.
        text = "x" * 500
        inj = make_injector(api, clip, mode=InjectionMode.CLIPBOARD)
        out = inj.inject(text, 1)
        assert clip.set_texts == [text]
        assert len(api.posted_chords) == 1
        assert out.events_sent == 2

    def test_restore_runs_when_the_paste_raises(self, api, clip):
        api.post_chord_result = OSError("CGEventPost exploded")
        inj = make_injector(api, clip, mode=InjectionMode.CLIPBOARD)
        with pytest.raises(OSError):
            inj.inject("Hello", 1)
        assert clip.restored == [clip.snapshot]

    def test_restore_runs_and_the_failure_is_reported_when_the_paste_posts_nothing(self, api, clip):
        api.post_chord_result = 0
        inj = make_injector(api, clip, mode=InjectionMode.CLIPBOARD)
        with pytest.raises(InjectionBlockedError) as ei:
            inj.inject("Hello", 1)
        assert ei.value.reason == "cgevent_failed"
        assert ei.value.text == "Hello"
        assert clip.restored == [clip.snapshot]

    def test_restore_runs_when_set_text_fails(self, api, clip):
        clip.set_text_error = AppError("Could not put the text on the clipboard.")
        inj = make_injector(api, clip, mode=InjectionMode.CLIPBOARD)
        with pytest.raises(AppError):
            inj.inject("Hello", 1)
        assert clip.restored == [clip.snapshot]
        assert api.posted_chords == []

    def test_a_failing_restore_does_not_fail_the_injection(self, api, clip):
        clip.restore_error = AppError("The clipboard is not available. Try again.")
        inj = make_injector(api, clip, mode=InjectionMode.CLIPBOARD)
        out = inj.inject("Hello", 1)
        assert out.injected_text == "Hello"
        assert clip.restored == [clip.snapshot]

    def test_failed_paste_does_not_advance_spacing_state(self, api, clip):
        api.post_chord_result = 0
        inj = make_injector(api, clip, mode=InjectionMode.CLIPBOARD)
        with pytest.raises(InjectionBlockedError):
            inj.inject("Hello", 1)
        api.post_chord_result = None
        assert inj.inject("world", 1).injected_text == "world"


class TestSetMode:
    def test_set_mode_takes_effect_on_the_next_inject(self, api, clip):
        inj = make_injector(api, clip)
        inj.inject("One", 1)
        assert clip.set_texts == []
        assert chunks(api) == ["One"]

        inj.set_mode(InjectionMode.CLIPBOARD)
        out = inj.inject("Two", 1)
        assert out.mode == InjectionMode.CLIPBOARD
        assert clip.set_texts == [" Two"]
        assert chunks(api) == ["One"]  # nothing new was typed

        inj.set_mode(InjectionMode.SENDINPUT)
        out = inj.inject("Three", 1)
        assert out.mode == InjectionMode.SENDINPUT
        assert chunks(api) == ["One", " Three"]
        assert clip.set_texts == [" Two"]

    def test_set_mode_from_another_thread(self, api, clip):
        inj = make_injector(api, clip)
        run_with_timeout(inj.set_mode, InjectionMode.CLIPBOARD)
        assert inj.inject("Hi", 1).mode == InjectionMode.CLIPBOARD

    def test_set_mode_accepts_the_enum_value(self, api, clip):
        inj = make_injector(api, clip)
        inj.set_mode(InjectionMode("clipboard"))
        assert inj.inject("Hi", 1).mode == InjectionMode.CLIPBOARD


# ===========================================================================
# MacClipboard (NSPasteboard backend)
# ===========================================================================


class TestMacClipboard:
    def test_save_snapshots_readable_types_in_order(self, api):
        api.seed_pasteboard(PREVIOUS)
        snap = make_clipboard(api).save()
        assert isinstance(snap, ClipboardSnapshot)
        assert snap.formats == PREVIOUS_SAVED
        assert all(isinstance(t, str) for t, _ in snap.formats)
        assert api.pasteboard_items() == PREVIOUS  # save() does not modify the pasteboard

    def test_save_empty_pasteboard(self, api):
        snap = make_clipboard(api).save()
        assert snap.formats == ()

    def test_save_many_types(self, api):
        items = [(TYPE_STRING, b"text"), (TYPE_RTF, b"rtf"), (TYPE_URL, b"file:///tmp/x"),
                 (TYPE_PNG, b"\x89PNG")]
        api.seed_pasteboard(items)
        assert make_clipboard(api).save().formats == tuple(items)

    def test_set_text_clears_then_sets_the_string_type(self, api):
        api.seed_pasteboard(PREVIOUS)
        text = "Bună ziua, ștefan 😀"
        make_clipboard(api).set_text(text)
        assert api.pasteboard_items() == [(TYPE_STRING, text.encode("utf-8"))]
        assert api.pasteboard_string() == text
        assert "pb_clear" in api.kinds()
        assert api.kinds().index("pb_clear") < api.kinds().index("pb_set_string")
        assert api.pasteboard_violations == []

    def test_set_text_raises_app_error_when_the_pasteboard_refuses(self, api):
        api.set_string_result = False
        with pytest.raises(AppError) as ei:
            make_clipboard(api).set_text("Hello")
        assert ei.value.user_message

    def test_restore_puts_every_saved_type_back(self, api):
        api.seed_pasteboard(PREVIOUS)
        cb = make_clipboard(api)
        snap = cb.save()
        cb.set_text("temporary")
        assert api.pasteboard_items() == [(TYPE_STRING, b"temporary")]
        cb.restore(snap)
        assert api.pasteboard_items() == list(PREVIOUS_SAVED)
        assert api.pasteboard_violations == []

    def test_restore_empties_first(self, api):
        api.seed_pasteboard([(TYPE_RTF, b"zzz"), (TYPE_URL, b"other")])
        snap = ClipboardSnapshot(formats=((TYPE_STRING, b"x"),))
        make_clipboard(api).restore(snap)
        assert api.pasteboard_items() == [(TYPE_STRING, b"x")]

    def test_restore_empty_snapshot_leaves_the_pasteboard_empty(self, api):
        api.seed_pasteboard(PREVIOUS)
        make_clipboard(api).restore(ClipboardSnapshot(formats=()))
        assert api.pasteboard_items() == []

    def test_save_restore_round_trip_is_exact(self, api):
        api.seed_pasteboard(PREVIOUS)
        cb = make_clipboard(api)
        snap = cb.save()
        cb.set_text("dictated")
        cb.restore(snap)
        assert cb.save().formats == PREVIOUS_SAVED

    def test_restore_does_not_raise_when_a_type_cannot_be_set(self, api):
        api.set_data_result = False
        make_clipboard(api).restore(ClipboardSnapshot(formats=((TYPE_STRING, b"x"),)))
        # a failed restore is a warning, never an error the user sees

    def test_restore_skips_a_windows_snapshot_format_id(self, api):
        # ClipboardSnapshot.formats accepts int (CF_*) and str (UTI) ids; an int id can only
        # come from a Windows snapshot, so it is skipped rather than guessed at.
        snap = ClipboardSnapshot(formats=((13, b"\x00"), (TYPE_STRING, b"ok")))
        make_clipboard(api).restore(snap)
        assert api.pasteboard_items() == [(TYPE_STRING, b"ok")]

    @pytest.mark.parametrize("op", ["save", "set_text", "restore"])
    def test_unusable_pasteboard_raises_app_error(self, api, op):
        api.pasteboard_error = OSError("[NSPasteboard generalPasteboard] returned nil")
        cb = make_clipboard(api)
        with pytest.raises(AppError) as ei:
            if op == "save":
                cb.save()
            elif op == "set_text":
                cb.set_text("Hello")
            else:
                cb.restore(ClipboardSnapshot(formats=((TYPE_STRING, b"x"),)))
        assert ei.value.user_message
        assert ei.value.details

    def test_save_skips_a_type_whose_data_cannot_be_read(self, api):
        api.seed_pasteboard([(TYPE_STRING, b"text"), (TYPE_PNG, None)])
        assert make_clipboard(api).save().formats == ((TYPE_STRING, b"text"),)


# ===========================================================================
# Injector + MacClipboard together (real pasteboard round trip through the fake)
# ===========================================================================


class TestClipboardRoundTrip:
    def test_paste_then_previous_contents_restored(self, api):
        api.seed_pasteboard(PREVIOUS)
        inj = make_injector(api, make_clipboard(api), mode=InjectionMode.CLIPBOARD)
        text = "Salut, lume ăîș"
        out = inj.inject(text, 1)
        assert out.injected_text == text
        # while Cmd+V was posted, the pasteboard held exactly our text
        assert api.posted_chords[0]["pasteboard"] == [(TYPE_STRING, text.encode("utf-8"))]
        assert _cmd_v_only(api)
        # afterwards the previous readable types are back, in order
        assert api.pasteboard_items() == list(PREVIOUS_SAVED)
        assert api.pasteboard_violations == []

    def test_previous_restored_when_the_paste_raises(self, api):
        api.seed_pasteboard(PREVIOUS)
        api.post_chord_result = OSError("boom")
        inj = make_injector(api, make_clipboard(api), mode=InjectionMode.CLIPBOARD)
        with pytest.raises(OSError):
            inj.inject("Hello", 1)
        assert api.pasteboard_items() == list(PREVIOUS_SAVED)

    def test_empty_previous_pasteboard_is_left_empty(self, api):
        inj = make_injector(api, make_clipboard(api), mode=InjectionMode.CLIPBOARD)
        inj.inject("Hello", 1)
        assert api.posted_chords[0]["pasteboard"] == [(TYPE_STRING, b"Hello")]
        assert api.pasteboard_items() == []

    def test_second_segment_pastes_with_its_leading_space(self, api):
        api.seed_pasteboard(PREVIOUS)
        inj = make_injector(api, make_clipboard(api), mode=InjectionMode.CLIPBOARD)
        inj.inject("Hello", 1)
        inj.inject("world", 1)
        assert api.posted_chords[1]["pasteboard"] == [(TYPE_STRING, b" world")]
        assert api.pasteboard_items() == list(PREVIOUS_SAVED)
