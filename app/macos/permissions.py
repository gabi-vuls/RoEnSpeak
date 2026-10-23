"""Accessibility and microphone permission helpers for the UI.

macOS permissions have no Windows counterpart, so there is no module in ``app/win32`` to
mirror: this is the one macOS-only module in the package. The UI uses it to tell the user why
dictation or typing is not working and to send them to the right place.

Everything here is **safe to call from any thread, on any platform, at any time**: off macOS
(or if a framework cannot be bound) the checks return ``False`` / ``"unknown"`` and the
actions are no-ops. No function in this module raises -- the UI calls them from paint and
click handlers where an exception would be worse than a wrong answer.

Prompts are one-shot by OS design: ``request_accessibility()`` only shows the system prompt
while the app is untrusted and macOS has not already shown it for this app, and
``request_microphone()`` only prompts while the status is "undetermined". Afterwards the only
way in is System Settings, which ``open_privacy_settings`` opens on the right pane.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from typing import Optional

log = logging.getLogger("roenspeak.macos.permissions")

#: Microphone statuses returned by ``microphone_status()``.
GRANTED = "granted"
DENIED = "denied"
UNDETERMINED = "undetermined"
UNKNOWN = "unknown"

#: System Settings panes, by the name the UI passes to ``open_privacy_settings``.
PRIVACY_PANES = {
    "accessibility": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    "microphone": "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
}

_api = None
_api_lock = threading.Lock()


def _get_api():
    """A lazily built RealMacApi, or None off macOS / if the frameworks cannot be bound."""
    global _api
    if sys.platform != "darwin":
        return None
    if _api is None:
        with _api_lock:
            if _api is None:
                try:
                    from app.macos._mac_api import RealMacApi

                    _api = RealMacApi()
                except Exception:
                    log.exception("could not build the macOS API wrapper")
                    return None
    return _api


def accessibility_granted() -> bool:
    """True when this process has Accessibility permission (``AXIsProcessTrusted()``).

    Without it ``CGEventPost`` is silently ignored, so the injector cannot type.
    """
    api = _get_api()
    if api is None:
        return False
    try:
        return bool(api.is_process_trusted())
    except Exception:
        log.exception("AXIsProcessTrusted failed")
        return False


def request_accessibility() -> None:
    """Ask macOS to show the "allow Accessibility" prompt (once, while untrusted)."""
    api = _get_api()
    if api is None:
        return
    try:
        trusted = api.request_process_trust()
        log.info("accessibility prompt requested trusted=%s", trusted)
    except Exception:
        log.exception("AXIsProcessTrustedWithOptions failed")


def secure_input_enabled() -> bool:
    """True while a password field holds the keyboard (``IsSecureEventInputEnabled()``).

    Transient by nature: it is a snapshot of right now, for an error message.
    """
    api = _get_api()
    if api is None:
        return False
    try:
        return bool(api.is_secure_event_input_enabled())
    except Exception:
        log.exception("IsSecureEventInputEnabled failed")
        return False


def microphone_status() -> str:
    """GRANTED / DENIED / UNDETERMINED / UNKNOWN, from AVCaptureDevice.

    DENIED also covers "restricted" (blocked by a configuration profile or parental
    controls): the user cannot dictate either way, and the remedy we can offer is the same.
    """
    api = _get_api()
    if api is None:
        return UNKNOWN
    try:
        from app.macos._mac_api import (
            AV_AUTH_AUTHORIZED,
            AV_AUTH_DENIED,
            AV_AUTH_NOT_DETERMINED,
            AV_AUTH_RESTRICTED,
        )

        status = api.microphone_authorization_status()
    except Exception:
        log.exception("AVCaptureDevice authorizationStatusForMediaType: failed")
        return UNKNOWN
    if status == AV_AUTH_AUTHORIZED:
        return GRANTED
    if status in (AV_AUTH_DENIED, AV_AUTH_RESTRICTED):
        return DENIED
    if status == AV_AUTH_NOT_DETERMINED:
        return UNDETERMINED
    log.warning("unknown AVAuthorizationStatus %r", status)
    return UNKNOWN


def request_microphone() -> None:
    """Ask macOS to prompt for microphone access (only prompts while UNDETERMINED).

    The answer arrives asynchronously; read it back with ``microphone_status()``.
    """
    api = _get_api()
    if api is None:
        return
    try:
        api.request_microphone_access()
        log.info("microphone permission requested")
    except Exception:
        log.exception("AVCaptureDevice requestAccessForMediaType: failed")


def open_privacy_settings(pane: str) -> None:
    """Open System Settings on the Privacy pane named ``pane`` ("accessibility"/"microphone").

    An unknown pane name opens Privacy & Security itself rather than doing nothing.
    """
    url: Optional[str] = PRIVACY_PANES.get(pane)
    if url is None:
        log.warning("unknown privacy pane %r; opening Privacy & Security", pane)
        url = "x-apple.systempreferences:com.apple.preference.security"
    if sys.platform != "darwin":
        log.debug("not opening %s off macOS", url)
        return
    try:
        subprocess.run(["/usr/bin/open", url], check=False, capture_output=True, timeout=10.0)
        log.info("opened System Settings pane=%s", pane)
    except Exception:
        log.exception("could not open System Settings pane=%s", pane)
