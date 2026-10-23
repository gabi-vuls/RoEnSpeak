"""Clipboard save / set text / restore (``contracts.ClipboardBackend``) and the read side used
by prompt adjustment (``contracts.ClipboardReader``). Injection / prompt-adjust thread only."""

from __future__ import annotations

import logging

from typing import Optional

from app.contracts import AppError, ClipboardSnapshot, WinApi

log = logging.getLogger("roenspeak.win32.clipboard")

CF_UNICODETEXT = 13
OPEN_ATTEMPTS = 10
OPEN_RETRY_S = 0.02
#: Registered formats that keep the content out of Win+V clipboard history and cloud sync.
EXCLUDE_FROM_MONITORS = "ExcludeClipboardContentFromMonitorProcessing"
CAN_INCLUDE_IN_HISTORY = "CanIncludeInClipboardHistory"
_DWORD_ZERO = (0).to_bytes(4, "little")


class Win32Clipboard:
    def __init__(self, api: WinApi) -> None:
        self._api = api

    def _open(self) -> None:
        api = self._api
        for attempt in range(OPEN_ATTEMPTS):
            if api.open_clipboard():
                return
            if attempt < OPEN_ATTEMPTS - 1:
                api.sleep(OPEN_RETRY_S)
        err = api.get_last_error()
        raise AppError(
            "The clipboard is busy (another app is holding it). Try again.",
            details=f"OpenClipboard failed {OPEN_ATTEMPTS}x, GetLastError={err}",
        )

    def save(self) -> ClipboardSnapshot:
        api = self._api
        self._open()
        try:
            saved: list[tuple[int, bytes]] = []
            skipped: list[int] = []
            for fmt in api.enum_clipboard_formats():
                data = api.get_clipboard_bytes(fmt)
                if data is None:
                    skipped.append(fmt)
                else:
                    saved.append((fmt, data))
        finally:
            api.close_clipboard()
        if skipped:
            log.debug("clipboard save skipped non-HGLOBAL formats %s", skipped)
        return ClipboardSnapshot(formats=tuple(saved))

    def set_text(self, text: str) -> None:
        api = self._api
        self._open()
        try:
            api.empty_clipboard()
            if not api.set_clipboard_bytes(CF_UNICODETEXT, (text + "\0").encode("utf-16-le")):
                raise AppError(
                    "Could not put the text on the clipboard.",
                    details=f"SetClipboardData(CF_UNICODETEXT) failed, GetLastError={api.get_last_error()}",
                )
            for name in (EXCLUDE_FROM_MONITORS, CAN_INCLUDE_IN_HISTORY):
                fmt = api.register_clipboard_format(name)
                if not fmt or not api.set_clipboard_bytes(fmt, _DWORD_ZERO):
                    log.warning("could not set clipboard format %s (id=%s); text may appear in clipboard history",
                                name, fmt)
        finally:
            api.close_clipboard()

    def get_text(self) -> Optional[str]:
        api = self._api
        self._open()
        try:
            data = api.get_clipboard_bytes(CF_UNICODETEXT)
        finally:
            api.close_clipboard()
        if data is None:
            return None
        text = data.decode("utf-16-le", "replace")
        end = text.find("\0")
        return text if end < 0 else text[:end]

    def clear(self) -> None:
        api = self._api
        self._open()
        try:
            api.empty_clipboard()
        finally:
            api.close_clipboard()

    def restore(self, snapshot: ClipboardSnapshot) -> None:
        api = self._api
        self._open()
        try:
            api.empty_clipboard()
            failed = [fmt for fmt, data in snapshot.formats if not api.set_clipboard_bytes(fmt, data)]
        finally:
            api.close_clipboard()
        if failed:
            log.warning("clipboard restore failed for formats %s", failed)
