"""Types text into the focused app (``contracts.TextInjector``). Injection thread only,
except set_mode (any thread).

Step for step the same algorithm as ``app.win32.injector.Win32TextInjector`` -- normalize,
leading-space rule, wait for the modifiers, check whether the OS will accept synthetic
keystrokes at all, then type or paste -- with the platform calls swapped:

===========================  ==================================  ================================
step                         Windows                             macOS
===========================  ==================================  ================================
modifier state               GetAsyncKeyState per modifier VK     CGEventSourceFlagsState
will the OS accept events?   UIPI integrity-level comparison      AXIsProcessTrusted + IsSecureEventInputEnabled
SENDINPUT mode               SendInput + KEYEVENTF_UNICODE        CGEventKeyboardSetUnicodeString + CGEventPost
CLIPBOARD mode               Ctrl+V                               Cmd+V
===========================  ==================================  ================================

``InjectionMode.SENDINPUT`` keeps its config value on macOS (it is "type the characters
directly", whatever the call is named).
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Optional

from app.contracts import (
    AppError,
    ClipboardBackend,
    InjectionBlockedError,
    InjectionMode,
    InjectionOutcome,
    ModifiersHeldError,
    NO_SPACE_BEFORE,
)

from app.macos._mac_api import (
    KCG_FLAG_ALTERNATE,
    KCG_FLAG_COMMAND,
    KCG_FLAG_CONTROL,
    KCG_FLAG_SHIFT,
    KVK_ANSI_V,
    MODIFIER_FLAG_MASK,
)

log = logging.getLogger("roenspeak.macos.injector")

#: Maximum UTF-16 code units per CGEvent. CGEventKeyboardSetUnicodeString accepts longer
#: strings but is not reliable with them: the event carries a fixed-size unicode payload and
#: long strings get truncated, reordered or dropped outright by the receiving app. Posting
#: the text in small chunks (one key down/up pair each) is the behaviour every app handles.
MAX_UNICODE_UNITS = 20

#: Any whitespace run that contains a CR, LF or TAB becomes one space.
_LINEBREAKS = re.compile(r"[ \r\n\t]*[\r\n\t][ \r\n\t]*")

_FLAG_NAMES = (
    (KCG_FLAG_CONTROL, "Control"),
    (KCG_FLAG_SHIFT, "Shift"),
    (KCG_FLAG_ALTERNATE, "Option"),
    (KCG_FLAG_COMMAND, "Command"),
)


def normalize_text(text: str) -> str:
    """Step 1 of inject(): strip, then turn internal newlines/tabs into single spaces."""
    return _LINEBREAKS.sub(" ", text.strip())


def utf16_chunks(text: str, limit: int = MAX_UNICODE_UNITS) -> list[str]:
    """Split text into pieces of at most ``limit`` UTF-16 code units.

    A character outside the BMP costs two code units and is never split across chunks, so a
    surrogate pair always reaches the target app inside one event.
    """
    chunks: list[str] = []
    current: list[str] = []
    units = 0
    for char in text:
        cost = 2 if ord(char) > 0xFFFF else 1
        if units + cost > limit and current:
            chunks.append("".join(current))
            current = []
            units = 0
        current.append(char)
        units += cost
    if current:
        chunks.append("".join(current))
    return chunks


def held_modifier_names(flags: int) -> list[str]:
    """The modifier keys currently held, for log lines."""
    return [name for bit, name in _FLAG_NAMES if flags & bit]


class MacTextInjector:
    def __init__(
        self,
        api,
        clipboard: ClipboardBackend,
        mode: InjectionMode = InjectionMode.SENDINPUT,
        modifier_timeout_s: float = 5.0,
        poll_s: float = 0.01,
        paste_settle_s: float = 0.15,
    ) -> None:
        self._api = api
        self._clipboard = clipboard
        self._mode_lock = threading.Lock()
        self._mode = InjectionMode(mode)
        self._modifier_timeout_s = modifier_timeout_s
        self._poll_s = poll_s
        self._paste_settle_s = paste_settle_s
        # Spacing state: session of the last successful injection and whether its text ended
        # in whitespace.
        self._last_session: Optional[int] = None
        self._last_ended_ws = True

    def set_mode(self, mode: InjectionMode) -> None:
        mode = InjectionMode(mode)
        with self._mode_lock:
            self._mode = mode
        log.info("injection mode set to %s", mode.value)

    @property
    def mode(self) -> InjectionMode:
        with self._mode_lock:
            return self._mode

    def inject(self, text: str, session_id: int) -> Optional[InjectionOutcome]:
        # 1. normalize
        text = normalize_text(text)
        if not text:
            return None
        mode = self.mode

        # 2. spacing
        prefix = ""
        if (self._last_session == session_id and not self._last_ended_ws
                and text[0] not in NO_SPACE_BEFORE):
            prefix = " "
        typed = prefix + text

        # 3. wait for modifiers to be released
        self._wait_modifiers_released(text)

        # 4. will macOS accept synthetic keystrokes at all?
        self._check_injection_allowed(text)

        # 5. inject
        if mode is InjectionMode.CLIPBOARD:
            sent, t_injected = self._paste(typed, text)
        else:
            sent, t_injected = self._send_unicode(typed, text)

        self._last_session = session_id
        self._last_ended_ws = typed[-1].isspace()
        log.debug("injected chars=%d mode=%s events=%d", len(typed), mode.value, sent)
        return InjectionOutcome(injected_text=typed, mode=mode, events_sent=sent, t_injected=t_injected)

    # --- steps ---------------------------------------------------------------------------

    def _wait_modifiers_released(self, text: str) -> None:
        """Block until Control/Shift/Option/Command are all up.

        Caps Lock and the non-coalesced bit are excluded by MODIFIER_FLAG_MASK: Caps Lock is a
        latch, so waiting for it would drop every segment while it is on.
        """
        api = self._api
        deadline: Optional[float] = None
        while True:
            held = api.modifier_flags() & MODIFIER_FLAG_MASK
            if not held:
                return
            now = api.monotonic()
            if deadline is None:
                deadline = now + self._modifier_timeout_s
                log.debug("waiting for modifiers to be released: %s", held_modifier_names(held))
            elif now >= deadline:
                log.warning("modifiers still held after %.1f s: %s; text dropped",
                            self._modifier_timeout_s, held_modifier_names(held))
                raise ModifiersHeldError(text)
            api.sleep(self._poll_s)

    def _check_injection_allowed(self, text: str) -> None:
        """The macOS stand-in for the Windows UIPI check.

        Without Accessibility permission CGEventPost is silently ignored, and while secure
        event input is on (a password field has the keyboard) synthetic keystrokes are
        dropped. Both are invisible in the return value of the posting calls, so they are
        checked up front, exactly as UIPI is on Windows.
        """
        api = self._api
        try:
            trusted = api.is_process_trusted()
            secure = api.is_secure_event_input_enabled()
        except OSError as exc:
            log.warning("could not read the accessibility/secure-input state: %s", exc)
            raise InjectionBlockedError(text, reason="cgevent_failed") from exc
        if not trusted:
            log.warning("Accessibility permission not granted%s; injection blocked",
                        self._frontmost_suffix())
            raise InjectionBlockedError(text, reason="no_accessibility")
        if secure:
            log.warning("secure event input is enabled%s; injection blocked",
                        self._frontmost_suffix())
            raise InjectionBlockedError(text, reason="secure_input")

    def _frontmost_suffix(self) -> str:
        """" (frontmost app=... pid=...)" for a log line, or "" if it cannot be read."""
        try:
            app = self._api.frontmost_app()
        except (OSError, AttributeError):
            return ""
        if app is None:
            return ""
        return f" (frontmost app={app.bundle_id or app.name} pid={app.pid})"

    def _send_unicode(self, typed: str, text: str) -> tuple[int, float]:
        api = self._api
        chunks = utf16_chunks(typed)
        sent = 0
        for chunk in chunks:
            posted = api.post_unicode_string(chunk)
            sent += posted
            if posted < 2:
                log.warning("CGEventPost inserted %d/2 events for a %d-char chunk "
                            "(%d/%d events total); text dropped",
                            posted, len(chunk), sent, 2 * len(chunks))
                raise InjectionBlockedError(text, reason="cgevent_failed")
        return sent, api.monotonic()

    def _paste(self, typed: str, text: str) -> tuple[int, float]:
        api = self._api
        snap = self._clipboard.save()
        sent = 0
        t_injected = 0.0
        try:
            self._clipboard.set_text(typed)
            sent = api.post_key_chord(KVK_ANSI_V, KCG_FLAG_COMMAND)
            t_injected = api.monotonic()
            api.sleep(self._paste_settle_s)
        finally:
            try:
                self._clipboard.restore(snap)
            except AppError as exc:
                log.warning("clipboard restore failed: %s (%s)", exc, exc.details)
        if sent < 2:
            log.warning("Cmd+V posted %d/2 events", sent)
            raise InjectionBlockedError(text, reason="cgevent_failed")
        return sent, t_injected
