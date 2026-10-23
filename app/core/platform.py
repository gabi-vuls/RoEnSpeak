"""The one platform dispatch module: picks app.win32 (Windows) or app.macos (macOS).

Why this module exists
  The shared core (controller, pipeline, main, cli) must not know which host it runs on.
  Before the macOS port, ``AppController._build_components`` imported ``app.win32.*`` and
  ``app.engine.engine`` directly, which made every core module Windows-only. Everything
  platform-specific now goes through one of the factories below.

Every function does its platform import **lazily, inside the function body**. Importing
this module therefore never imports ``app.win32`` on macOS nor ``app.macos`` on Windows,
which matters because those packages bind OS libraries (ctypes/Carbon/AppKit) and because
either package may be absent from a platform-specific build.

The ``make_*`` factories raise ``AppError`` on a host that is neither Windows nor macOS.
``parse_hotkey``/``format_hotkey``/``hotkey_parser`` raise ``ImportError`` there instead, so
``app.core.config`` keeps treating "no hotkey parser available" as "keep the string
unvalidated" exactly as it did before.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

from app.contracts import (
    IS_MACOS,
    IS_WINDOWS,
    MIN_DRIVER_VERSION,
    MIN_MACOS_VERSION,
    AppError,
    Autostart,
    ClipboardBackend,
    CudaProbe,
    Hotkey,
    HotkeyManager,
    HotkeyParser,
    InjectionMode,
    ModelDownloader,
    SingleInstance,
    SpeechEngine,
    TextInjector,
    os_display_name,
)

log = logging.getLogger("roenspeak.core.platform")

__all__ = [
    "IS_WINDOWS",
    "IS_MACOS",
    "SUPPORTS_PROMPT_ADJUST",
    "UNSUPPORTED_PLATFORM_MESSAGE",
    "parse_hotkey",
    "format_hotkey",
    "hotkey_parser",
    "make_probe",
    "make_engine",
    "make_downloader",
    "make_api",
    "make_clipboard",
    "make_injector",
    "make_selection_grabber",
    "make_autostart",
    "make_hotkey_factory",
    "make_single_instance",
    "open_folder",
    "gpu_requirement_message",
    "autostart_label",
]

#: Prompt adjustment (copy selection -> local LLM -> paste) has a Windows implementation only;
#: elsewhere the prompt hotkey is never registered and its settings are hidden.
SUPPORTS_PROMPT_ADJUST = IS_WINDOWS

#: Shown when the app is started on a host that has no platform layer.
UNSUPPORTED_PLATFORM_MESSAGE = "RoEnSpeak supports Windows and macOS."


def _unsupported() -> AppError:
    return AppError(UNSUPPORTED_PLATFORM_MESSAGE, details=f"platform={os_display_name()}")


# ---------------------------------------------------------------------------
# Hotkey strings (pure; any thread)
# ---------------------------------------------------------------------------


def parse_hotkey(text: str) -> Hotkey:
    """Parse a hotkey string with the host's key table. Raises HotkeyParseError."""
    if IS_WINDOWS:
        from app.win32.hotkey_parse import parse_hotkey as _parse
    elif IS_MACOS:
        from app.macos.hotkey_parse import parse_hotkey as _parse
    else:
        raise ImportError(UNSUPPORTED_PLATFORM_MESSAGE)
    return _parse(text)


def format_hotkey(hotkey: Hotkey) -> str:
    """Canonical display string for a Hotkey (== hotkey.display)."""
    if IS_WINDOWS:
        from app.win32.hotkey_parse import format_hotkey as _format
    elif IS_MACOS:
        from app.macos.hotkey_parse import format_hotkey as _format
    else:
        raise ImportError(UNSUPPORTED_PLATFORM_MESSAGE)
    return _format(hotkey)


def hotkey_parser() -> HotkeyParser:
    """The host's HotkeyParser implementation."""
    if IS_WINDOWS:
        from app.win32.hotkey_parse import HotkeyStringParser

        return HotkeyStringParser()
    if IS_MACOS:
        from app.macos.hotkey_parse import MacHotkeyStringParser

        return MacHotkeyStringParser()
    raise ImportError(UNSUPPORTED_PLATFORM_MESSAGE)


# ---------------------------------------------------------------------------
# Engine side
# ---------------------------------------------------------------------------


def make_probe() -> CudaProbe:
    """The GPU probe: CUDA on Windows, Metal on macOS. Callers still have to call
    setup_dll_paths() before anything imports the engine."""
    if IS_WINDOWS:
        from app.engine.cuda_probe import WindowsCudaProbe

        return WindowsCudaProbe()
    if IS_MACOS:
        from app.engine.metal_probe import MetalProbe

        return MetalProbe()
    raise _unsupported()


def make_engine(probe: CudaProbe) -> SpeechEngine:
    """faster-whisper/CTranslate2 on Windows, whisper.cpp/Metal on macOS."""
    if IS_WINDOWS:
        from app.engine.engine import WhisperEngine

        return WhisperEngine(probe)
    if IS_MACOS:
        from app.engine.whispercpp import WhisperCppEngine

        return WhisperCppEngine(probe)
    raise _unsupported()


def make_downloader() -> ModelDownloader:
    """CTranslate2 model directories on Windows, single-file ggml models on macOS.

    The macOS downloader also needs the selected compute type (it picks the ggml file);
    callers set it with ``set_compute_type(...)`` guarded by ``hasattr``.
    """
    if IS_WINDOWS:
        from app.engine.downloader import HfModelDownloader

        return HfModelDownloader()
    if IS_MACOS:
        from app.engine.ggml_downloader import GgmlModelDownloader

        return GgmlModelDownloader()
    raise _unsupported()


# ---------------------------------------------------------------------------
# Input / OS integration
# ---------------------------------------------------------------------------


def make_api() -> Any:
    """The raw OS API object the injector/clipboard/hotkey/single-instance layers share
    (``WinApi`` on Windows, its macOS counterpart on macOS)."""
    if IS_WINDOWS:
        from app.win32._winapi import RealWinApi

        return RealWinApi()
    if IS_MACOS:
        from app.macos._mac_api import RealMacApi

        return RealMacApi()
    raise _unsupported()


def make_clipboard(api: Any) -> ClipboardBackend:
    """Win32 clipboard / NSPasteboard."""
    if IS_WINDOWS:
        from app.win32.clipboard import Win32Clipboard

        return Win32Clipboard(api)
    if IS_MACOS:
        from app.macos.clipboard import MacClipboard

        return MacClipboard(api)
    raise _unsupported()


def make_injector(api: Any, clipboard: ClipboardBackend, mode: InjectionMode) -> TextInjector:
    """SendInput (Windows) / CGEventKeyboardSetUnicodeString (macOS), with the same
    clipboard-paste fallback mode."""
    if IS_WINDOWS:
        from app.win32.injector import Win32TextInjector

        return Win32TextInjector(api, clipboard, mode=mode)
    if IS_MACOS:
        from app.macos.injector import MacTextInjector

        return MacTextInjector(api, clipboard, mode=mode)
    raise _unsupported()


def make_selection_grabber(api: Any, clipboard: ClipboardBackend) -> Any:
    """Copies the focused window's selection with Ctrl+C (prompt adjustment). Windows only."""
    if IS_WINDOWS:
        from app.win32.selection import Win32SelectionGrabber

        return Win32SelectionGrabber(api, clipboard, clipboard)  # Win32Clipboard is also the reader
    raise _unsupported()


def make_autostart() -> Autostart:
    """HKCU Run key (Windows) / LaunchAgent plist (macOS)."""
    if IS_WINDOWS:
        from app.win32.autostart import RunKeyAutostart

        return RunKeyAutostart()
    if IS_MACOS:
        from app.macos.autostart import LaunchAgentAutostart

        return LaunchAgentAutostart()
    raise _unsupported()


def make_hotkey_factory(api: Any) -> Callable[[Callable[[], None]], HotkeyManager]:
    """Returns the factory the controller calls with its ``on_fire`` callback:
    RegisterHotKey (Windows) / Carbon RegisterEventHotKey (macOS)."""
    if IS_WINDOWS:
        from app.win32.hotkey import Win32HotkeyManager

        return lambda on_fire: Win32HotkeyManager(api, on_fire)
    if IS_MACOS:
        from app.macos.hotkey import MacHotkeyManager

        return lambda on_fire: MacHotkeyManager(api, on_fire)
    raise _unsupported()


def make_single_instance(api: Any) -> SingleInstance:
    """Named mutex + broadcast message (Windows) / flock + Unix socket (macOS)."""
    if IS_WINDOWS:
        from app.win32.instance import NamedMutexSingleInstance

        return NamedMutexSingleInstance(api)
    if IS_MACOS:
        from app.macos.instance import FileLockSingleInstance

        return FileLockSingleInstance(api)
    raise _unsupported()


def open_folder(path: Path) -> None:
    """Show a folder in the host's file manager. Raises OSError on failure, so callers can
    report it; never blocks (the helper is launched detached)."""
    target = str(path)
    if IS_WINDOWS:
        os.startfile(target)  # type: ignore[attr-defined]
        return
    opener = "open" if IS_MACOS else "xdg-open"
    subprocess.Popen([opener, target])


# ---------------------------------------------------------------------------
# User-facing wording
# ---------------------------------------------------------------------------


def gpu_requirement_message(reason: Optional[str] = None) -> str:
    """Explains what the GPU backend needs, around the probe's/engine's failure reason."""
    if IS_WINDOWS:
        # The CUDA probe's wording, unchanged from the Windows-only controller; engine-raised
        # CUDA errors keep their own text (app.engine.engine.cuda_requirement_message).
        return (
            f"RoEnSpeak needs an NVIDIA GPU with driver {MIN_DRIVER_VERSION} or newer, and the CUDA "
            f"libraries (cuBLAS, cuDNN) must load.\n\n"
            f"Problem: {reason or 'CUDA is not available.'}\n\n"
            f"Install or update the NVIDIA driver, then restart RoEnSpeak. If this PC has no NVIDIA GPU, "
            f"you can enable \"CPU mode (slow)\" in Settings."
        )
    if IS_MACOS:
        msg = (
            f"RoEnSpeak needs macOS {MIN_MACOS_VERSION} or newer on an Apple silicon Mac (or another "
            "Mac whose GPU supports Metal) to run the speech model on the GPU. "
        )
        if reason:
            msg += f"Problem: {reason.rstrip('. ')}. "
        msg += (
            "Update macOS and restart RoEnSpeak. "
            'Alternatively, enable "CPU mode (slow)" in Settings.'
        )
        return msg
    msg = "RoEnSpeak needs a supported GPU backend to run the speech model. "
    if reason:
        msg += f"Problem: {reason.rstrip('. ')}. "
    msg += 'Enable "CPU mode (slow)" in Settings to run without a GPU.'
    return msg


def autostart_label() -> str:
    """Label for the "launch at login" setting, as the user's OS calls it."""
    if IS_WINDOWS:
        return "Start with Windows"
    if IS_MACOS:
        return "Open at login"
    return f"Start with {os_display_name()}"
