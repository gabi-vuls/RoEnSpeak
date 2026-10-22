"""Clipboard save / set text / restore over NSPasteboard (``contracts.ClipboardBackend``).
Injection thread only.

The macOS counterpart of ``app.win32.clipboard.Win32Clipboard``, with the same three-step
contract (snapshot everything, set our text, put everything back) and the same error
contract: any failure to use the pasteboard is an ``AppError`` with a user-facing message,
and a failed *restore* is only ever a warning.

Two differences from Windows, both because NSPasteboard is a different animal:

* There is no OpenClipboard/CloseClipboard ownership, so there is no 10 x 20 ms retry loop:
  ``[NSPasteboard generalPasteboard]`` either exists or the window server is gone.
* Format ids are UTI strings (``public.utf8-plain-text``, ...) rather than CF_* integers;
  ``ClipboardSnapshot.formats`` accepts both. An int id in a snapshot can only come from a
  Windows snapshot, so restore skips it with a warning instead of guessing.

**There is no macOS analogue of the two Windows clipboard-history exclusion formats**
("ExcludeClipboardContentFromMonitorProcessing" / "CanIncludeInClipboardHistory"). macOS has
no system clipboard history to opt out of, and the ``org.nspasteboard.ConcealedType`` marker
some third-party clipboard managers honour is a community convention, not an OS feature, so
nothing is set here. In CLIPBOARD injection mode the dictated text is on the pasteboard for
``paste_settle_s`` and is then restored.
"""

from __future__ import annotations

import logging
from typing import Union

from app.contracts import AppError, ClipboardSnapshot

from app.macos._mac_api import NS_PASTEBOARD_TYPE_STRING

log = logging.getLogger("roenspeak.macos.clipboard")


class MacClipboard:
    def __init__(self, api) -> None:
        self._api = api

    def save(self) -> ClipboardSnapshot:
        """Snapshot every available pasteboard type, in the order NSPasteboard reports them."""
        api = self._api
        try:
            types = api.pasteboard_types()
        except OSError as exc:
            raise AppError(
                "The clipboard is not available. Try again.",
                details=f"[NSPasteboard types] failed: {exc}",
            ) from exc
        saved: list[tuple[Union[int, str], bytes]] = []
        skipped: list[str] = []
        for type_id in types:
            try:
                data = api.pasteboard_data(type_id)
            except OSError as exc:
                log.debug("clipboard save could not read %s: %s", type_id, exc)
                skipped.append(type_id)
                continue
            if data is None:
                skipped.append(type_id)
            else:
                saved.append((type_id, data))
        if skipped:
            log.debug("clipboard save skipped unreadable types %s", skipped)
        return ClipboardSnapshot(formats=tuple(saved))

    def set_text(self, text: str) -> None:
        """clearContents + ``setString:forType:public.utf8-plain-text``."""
        api = self._api
        try:
            api.pasteboard_clear()
            ok = api.pasteboard_set_string(text)
        except OSError as exc:
            raise AppError(
                "Could not put the text on the clipboard.",
                details=f"NSPasteboard setString: failed: {exc}",
            ) from exc
        if not ok:
            raise AppError(
                "Could not put the text on the clipboard.",
                details=f"[NSPasteboard setString:forType:{NS_PASTEBOARD_TYPE_STRING}] returned NO",
            )

    def restore(self, snapshot: ClipboardSnapshot) -> None:
        """clearContents + put every saved type back (an empty snapshot clears the clipboard)."""
        api = self._api
        try:
            api.pasteboard_clear()
        except OSError as exc:
            raise AppError(
                "The clipboard is not available. Try again.",
                details=f"[NSPasteboard clearContents] failed: {exc}",
            ) from exc
        failed: list[Union[int, str]] = []
        for type_id, data in snapshot.formats:
            if not isinstance(type_id, str):
                log.warning("clipboard restore skipping non-macOS format id %r", type_id)
                failed.append(type_id)
                continue
            try:
                if not api.pasteboard_set_data(type_id, data):
                    failed.append(type_id)
            except OSError as exc:
                log.debug("clipboard restore could not set %s: %s", type_id, exc)
                failed.append(type_id)
        if failed:
            log.warning("clipboard restore failed for formats %s", failed)
