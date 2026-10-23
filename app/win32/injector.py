"""Types text into the focused window (``contracts.TextInjector``). Injection thread only,
except set_mode (any thread)."""

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
    KeyEvent,
    MODIFIER_VKS,
    ModifiersHeldError,
    NO_SPACE_BEFORE,
    WinApi,
)

log = logging.getLogger("roenspeak.win32.injector")

VK_CONTROL = 0x11
VK_C = 0x43
VK_V = 0x56

# Any whitespace run that contains a CR, LF or TAB becomes one space.
_LINEBREAKS = re.compile(r"[ \r\n\t]*[\r\n\t][ \r\n\t]*")


def normalize_text(text: str) -> str:
    """Step 1 of inject(): strip, then turn internal newlines/tabs into single spaces."""
    return _LINEBREAKS.sub(" ", text.strip())


def utf16_code_units(text: str) -> list[str]:
    """Split text into single UTF-16 code units (astral chars become a surrogate pair)."""
    raw = text.encode("utf-16-le", "surrogatepass")
    return [chr(int.from_bytes(raw[i:i + 2], "little")) for i in range(0, len(raw), 2)]


def wait_modifiers_released(api: WinApi, timeout_s: float, poll_s: float, text: str) -> None:
    """Step 3 of inject(): poll until no MODIFIER_VKS key is down; ModifiersHeldError(text)
    after timeout_s."""
    deadline: Optional[float] = None
    while True:
        held = [vk for vk in MODIFIER_VKS if api.get_async_key_state(vk) & 0x8000]
        if not held:
            return
        now = api.monotonic()
        if deadline is None:
            deadline = now + timeout_s
            log.debug("waiting for modifiers to be released: %s", [hex(v) for v in held])
        elif now >= deadline:
            log.warning("modifiers still held after %.1f s: %s; text dropped",
                        timeout_s, [hex(v) for v in held])
            raise ModifiersHeldError(text)
        api.sleep(poll_s)


def check_foreground_integrity(api: WinApi, text: str) -> int:
    """Step 4 of inject() (UIPI): raise InjectionBlockedError(text) when the foreground
    window's process runs at a higher (or unknown) integrity level. Returns its hwnd."""
    try:
        own_il = api.get_current_process_integrity_level()
    except (OSError, AppError) as exc:
        log.warning("could not read own integrity level: %s", exc)
        raise InjectionBlockedError(text, reason="unknown_integrity") from exc
    hwnd = api.get_foreground_window()
    pid = api.get_window_process_id(hwnd)
    il = api.get_process_integrity_level(pid)
    if il is None:
        log.warning("foreground hwnd=%#x pid=%d integrity unknown; injection blocked", hwnd, pid)
        raise InjectionBlockedError(text, reason="unknown_integrity")
    if il > own_il:
        log.warning("foreground pid=%d integrity %#x > own %#x; injection blocked", pid, il, own_il)
        raise InjectionBlockedError(text, reason="elevated")
    return hwnd


class Win32TextInjector:
    def __init__(
        self,
        api: WinApi,
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

        # 4. UIPI
        self._check_integrity(text)

        # 5. inject
        if mode is InjectionMode.CLIPBOARD:
            sent, t_injected = self._paste(typed, text)
        else:
            sent, t_injected = self._send_unicode(typed, text)

        self._last_session = session_id
        self._last_ended_ws = typed[-1].isspace()
        log.debug("injected chars=%d mode=%s events=%d", len(typed), mode.value, sent)
        return InjectionOutcome(injected_text=typed, mode=mode, events_sent=sent, t_injected=t_injected)

    def paste_raw(self, text: str) -> Optional[InjectionOutcome]:
        """Paste text exactly as given (newlines and tabs kept, as CRLF) through the clipboard,
        whatever the injection mode. Used by prompt adjustment to replace a selection; steps 3-4
        of inject() apply, no normalization or spacing. Empty text -> None, nothing changes."""
        if not text:
            return None
        typed = re.sub(r"\r?\n", "\r\n", text)
        self._wait_modifiers_released(text)
        self._check_integrity(text)
        sent, t_injected = self._paste(typed, text)
        log.debug("pasted raw chars=%d events=%d", len(typed), sent)
        return InjectionOutcome(injected_text=typed, mode=InjectionMode.CLIPBOARD, events_sent=sent,
                                t_injected=t_injected)

    # --- steps ---------------------------------------------------------------------------

    def _wait_modifiers_released(self, text: str) -> None:
        wait_modifiers_released(self._api, self._modifier_timeout_s, self._poll_s, text)

    def _check_integrity(self, text: str) -> None:
        check_foreground_integrity(self._api, text)

    def _send_unicode(self, typed: str, text: str) -> tuple[int, float]:
        events: list[KeyEvent] = []
        for unit in utf16_code_units(typed):
            events.append(KeyEvent(unicode_char=unit))
            events.append(KeyEvent(unicode_char=unit, key_up=True))
        sent = self._api.send_input(events)
        t_injected = self._api.monotonic()
        if sent < len(events):
            log.warning("SendInput inserted %d/%d events (GetLastError=%d)",
                        sent, len(events), self._api.get_last_error())
            raise InjectionBlockedError(text, reason="sendinput_partial")
        return sent, t_injected

    def _paste(self, typed: str, text: str) -> tuple[int, float]:
        api = self._api
        events = [
            KeyEvent(vk=VK_CONTROL),
            KeyEvent(vk=VK_V),
            KeyEvent(vk=VK_V, key_up=True),
            KeyEvent(vk=VK_CONTROL, key_up=True),
        ]
        snap = self._clipboard.save()
        sent = 0
        t_injected = 0.0
        try:
            self._clipboard.set_text(typed)
            sent = api.send_input(events)
            t_injected = api.monotonic()
            api.sleep(self._paste_settle_s)
        finally:
            try:
                self._clipboard.restore(snap)
            except AppError as exc:
                log.warning("clipboard restore failed: %s (%s)", exc, exc.details)
        if sent < len(events):
            log.warning("Ctrl+V SendInput inserted %d/%d events", sent, len(events))
            raise InjectionBlockedError(text, reason="sendinput_partial")
        return sent, t_injected
