"""Reads the current text selection of the focused window by sending Ctrl+C (prompt
adjustment). Prompt-adjust thread only.

grab():
  1. Wait until no MODIFIER_VKS key is down (the user still holds Alt from Alt+P; the hotkey
     manager already sent VK_MENU_MASK so releasing Alt opens no menu). ModifiersHeldError
     after modifier_timeout_s.
  2. UIPI: an elevated foreground window cannot receive our Ctrl+C -> InjectionBlockedError.
  3. Save the clipboard, empty it, note GetClipboardSequenceNumber, send Ctrl down, C down,
     C up, Ctrl up, then poll the sequence number every poll_s for up to copy_timeout_s.
  4. Read CF_UNICODETEXT, then ALWAYS restore the saved clipboard (before the slow LLM call).
  5. No clipboard change, or only whitespace copied -> NothingSelectedError.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.contracts import (
    AppError,
    ClipboardBackend,
    ClipboardReader,
    InjectionBlockedError,
    KeyEvent,
    NothingSelectedError,
    WinApi,
)
from app.win32.injector import VK_C, VK_CONTROL, check_foreground_integrity, wait_modifiers_released

log = logging.getLogger("roenspeak.win32.selection")


@dataclass(frozen=True)
class Selection:
    text: str
    hwnd: int  # foreground window the text was copied from


class Win32SelectionGrabber:
    def __init__(
        self,
        api: WinApi,
        clipboard: ClipboardBackend,
        reader: ClipboardReader,
        modifier_timeout_s: float = 5.0,
        copy_timeout_s: float = 0.8,
        poll_s: float = 0.01,
    ) -> None:
        self._api = api
        self._clipboard = clipboard
        self._reader = reader
        self._modifier_timeout_s = modifier_timeout_s
        self._copy_timeout_s = copy_timeout_s
        self._poll_s = poll_s

    def foreground_window(self) -> int:
        return self._api.get_foreground_window()

    def grab(self) -> Selection:
        api = self._api
        wait_modifiers_released(api, self._modifier_timeout_s, self._poll_s, "")
        hwnd = check_foreground_integrity(api, "")

        snap = self._clipboard.save()
        try:
            self._reader.clear()
            seq0 = api.get_clipboard_sequence_number()
            events = [
                KeyEvent(vk=VK_CONTROL),
                KeyEvent(vk=VK_C),
                KeyEvent(vk=VK_C, key_up=True),
                KeyEvent(vk=VK_CONTROL, key_up=True),
            ]
            sent = api.send_input(events)
            if sent < len(events):
                log.warning("Ctrl+C SendInput inserted %d/%d events", sent, len(events))
                raise InjectionBlockedError("", reason="sendinput_partial")
            copied = self._wait_for_copy(seq0)
            text = self._reader.get_text() if copied else None
        finally:
            try:
                self._clipboard.restore(snap)
            except AppError as exc:
                log.warning("clipboard restore after copy failed: %s (%s)", exc, exc.details)

        if not copied:
            raise NothingSelectedError(details=f"clipboard unchanged after {self._copy_timeout_s:.1f} s")
        if text is None or not text.strip():
            raise NothingSelectedError(details="copied text is empty")
        log.debug("selection copied chars=%d hwnd=%#x", len(text), hwnd)
        return Selection(text=text, hwnd=hwnd)

    def _wait_for_copy(self, seq0: int) -> bool:
        api = self._api
        deadline = api.monotonic() + self._copy_timeout_s
        while True:
            if api.get_clipboard_sequence_number() != seq0:
                # The source app may write several formats in a row; give it a moment so
                # CF_UNICODETEXT is there before we read.
                api.sleep(self._poll_s * 3)
                return True
            if api.monotonic() >= deadline:
                return False
            api.sleep(self._poll_s)
