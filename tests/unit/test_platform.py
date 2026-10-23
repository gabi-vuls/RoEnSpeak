"""Contract tests for app.core.platform -- the one platform-dispatch module.

Derived from app/contracts.py (the Protocols and the implementation map at the bottom) and
from the port contract: core code must never import app.win32 or app.macos itself, so every
platform object comes from a factory here, each factory does its import *lazily*, and on a
host that is neither Windows nor macOS the make_* factories raise AppError.

The tests run on whichever host they are on: the "this platform" tests assert against the
Protocols and against the host's concrete classes, and the other host's branches are covered
by faking the platform flags. contracts.IS_WINDOWS / IS_MACOS are computed from sys.platform
once, at import time, and app.core.platform re-exports them, so switching platform means
patching those flags (sys.platform is patched too, for consistency).
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app import contracts
from app.contracts import (
    AppError,
    Autostart,
    ClipboardBackend,
    CudaProbe,
    HotkeyManager,
    HotkeyParseError,
    InjectionMode,
    ModelDownloader,
    SingleInstance,
    SpeechEngine,
    TextInjector,
)
from app.core import platform as platform_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
PLATFORM_PACKAGE_PREFIXES = ("app.win32", "app.macos")


def _as_platform(monkeypatch, platform: str) -> None:
    monkeypatch.setattr(sys, "platform", platform, raising=False)
    for module in (contracts, platform_mod):
        monkeypatch.setattr(module, "IS_WINDOWS", platform == "win32")
        monkeypatch.setattr(module, "IS_MACOS", platform == "darwin")


@pytest.fixture
def api():
    return platform_mod.make_api()


# ---------------------------------------------------------------------------
# The module is importable everywhere, and imports no platform package
# ---------------------------------------------------------------------------


def test_platform_flags_are_the_contract_flags():
    assert platform_mod.IS_WINDOWS is contracts.IS_WINDOWS
    assert platform_mod.IS_MACOS is contracts.IS_MACOS
    assert not (platform_mod.IS_WINDOWS and platform_mod.IS_MACOS)


def test_importing_the_module_imports_no_platform_package_fresh_interpreter():
    """A fresh interpreter: importing app.core.platform must not pull in app.win32/app.macos.

    Those packages bind OS libraries (ctypes/Carbon/AppKit) and either one may be absent from
    a platform-specific build, so the lazy imports are what lets this module exist on both.
    """
    code = (
        "import sys, app.core.platform;"
        "print(sorted(m for m in sys.modules"
        " if m.startswith('app.win32') or m.startswith('app.macos')))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", result.stdout


def test_importing_the_module_imports_no_platform_package_cleaned_modules():
    saved = dict(sys.modules)
    try:
        for name in [n for n in sys.modules
                     if n == "app.core.platform" or n.startswith(PLATFORM_PACKAGE_PREFIXES)]:
            del sys.modules[name]
        importlib.import_module("app.core.platform")
        leaked = sorted(n for n in sys.modules if n.startswith(PLATFORM_PACKAGE_PREFIXES))
        assert leaked == []
    finally:
        sys.modules.clear()
        sys.modules.update(saved)


def test_calling_a_factory_does_import_the_host_package():
    """The counter-test: the imports are deferred, not missing."""
    host_parser = "app.macos.hotkey_parse" if contracts.IS_MACOS else "app.win32.hotkey_parse"
    saved = dict(sys.modules)
    try:
        # Other test modules may have imported it already; drop it so the assertion is about
        # what *this call* imports, whatever order the suite runs in.
        sys.modules.pop(host_parser, None)
        assert host_parser not in sys.modules
        platform_mod.hotkey_parser()
        assert host_parser in sys.modules, "the host's parser must be imported on first use"
    finally:
        sys.modules.clear()
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# Hotkey strings dispatch to the host's parser
# ---------------------------------------------------------------------------


def test_hotkey_parser_is_the_hosts_parser():
    parser = platform_mod.hotkey_parser()
    # contracts.HotkeyParser is a plain Protocol (not runtime_checkable), so check its shape.
    assert callable(parser.parse) and callable(parser.format)
    if contracts.IS_MACOS:
        from app.macos.hotkey_parse import MacHotkeyStringParser

        assert isinstance(parser, MacHotkeyStringParser)
    elif contracts.IS_WINDOWS:
        from app.win32.hotkey_parse import HotkeyStringParser

        assert isinstance(parser, HotkeyStringParser)


@pytest.mark.parametrize("text", ["alt+q", " Ctrl + Alt + F9 ", "ctrl+shift+1"])
def test_parse_and_format_match_the_hosts_parser(text):
    if contracts.IS_MACOS:
        from app.macos.hotkey_parse import format_hotkey as host_format
        from app.macos.hotkey_parse import parse_hotkey as host_parse
    else:
        from app.win32.hotkey_parse import format_hotkey as host_format
        from app.win32.hotkey_parse import parse_hotkey as host_parse

    hotkey = platform_mod.parse_hotkey(text)
    assert hotkey == host_parse(text)
    assert hotkey == platform_mod.hotkey_parser().parse(text)
    assert platform_mod.format_hotkey(hotkey) == host_format(hotkey) == hotkey.display


def test_default_hotkey_parses_on_this_host():
    hotkey = platform_mod.parse_hotkey(contracts.DEFAULT_HOTKEY)
    assert hotkey.display == contracts.DEFAULT_HOTKEY
    assert contracts.Modifier.ALT in hotkey.modifiers and hotkey.key == "Q"


def test_super_modifier_is_the_hosts_super_key():
    hotkey = platform_mod.parse_hotkey("Win+F12")
    if contracts.IS_MACOS:
        assert hotkey.modifiers == frozenset({contracts.Modifier.CMD})
        assert hotkey.display == "Cmd+F12"
    elif contracts.IS_WINDOWS:
        assert hotkey.modifiers == frozenset({contracts.Modifier.WIN})
        assert hotkey.display == "Win+F12"


def test_parse_hotkey_rejects_what_the_contract_rejects():
    for text in ("", "Q", "Ctrl", "Alt+NoSuchKey"):
        with pytest.raises(HotkeyParseError):
            platform_mod.parse_hotkey(text)


@pytest.mark.parametrize("call", [
    lambda: platform_mod.parse_hotkey("Alt+Q"),
    lambda: platform_mod.format_hotkey(contracts.Hotkey(frozenset(), "Q", 0x51)),
    lambda: platform_mod.hotkey_parser(),
], ids=["parse_hotkey", "format_hotkey", "hotkey_parser"])
def test_hotkey_helpers_raise_import_error_on_an_unsupported_platform(monkeypatch, call):
    """app.core.config treats ImportError as "no parser here, keep the string unvalidated"."""
    _as_platform(monkeypatch, "linux")
    with pytest.raises(ImportError):
        call()


# ---------------------------------------------------------------------------
# make_* on this host: every factory returns its Protocol
# ---------------------------------------------------------------------------


def test_make_probe():
    assert isinstance(platform_mod.make_probe(), CudaProbe)


def test_make_engine():
    probe = platform_mod.make_probe()
    assert isinstance(platform_mod.make_engine(probe), SpeechEngine)


def test_make_downloader():
    assert isinstance(platform_mod.make_downloader(), ModelDownloader)


def test_make_api_is_the_hosts_api_object(api):
    assert api is not None
    if contracts.IS_MACOS:
        from app.macos._mac_api import RealMacApi

        assert isinstance(api, RealMacApi)
    elif contracts.IS_WINDOWS:
        from app.win32._winapi import RealWinApi

        assert isinstance(api, RealWinApi)


def test_make_clipboard(api):
    assert isinstance(platform_mod.make_clipboard(api), ClipboardBackend)


@pytest.mark.parametrize("mode", list(InjectionMode), ids=[m.value for m in InjectionMode])
def test_make_injector(api, mode):
    clipboard = platform_mod.make_clipboard(api)
    assert isinstance(platform_mod.make_injector(api, clipboard, mode), TextInjector)


def test_make_autostart():
    assert isinstance(platform_mod.make_autostart(), Autostart)


def test_make_single_instance(api):
    assert isinstance(platform_mod.make_single_instance(api), SingleInstance)


def test_make_hotkey_factory_returns_a_callable_building_a_manager(api):
    factory = platform_mod.make_hotkey_factory(api)
    assert callable(factory)
    manager = factory(lambda: None)
    assert isinstance(manager, HotkeyManager)


# ---------------------------------------------------------------------------
# make_* on an unsupported host
# ---------------------------------------------------------------------------

UNSUPPORTED_CALLS = {
    "make_probe": lambda: platform_mod.make_probe(),
    "make_engine": lambda: platform_mod.make_engine(None),
    "make_downloader": lambda: platform_mod.make_downloader(),
    "make_api": lambda: platform_mod.make_api(),
    "make_clipboard": lambda: platform_mod.make_clipboard(None),
    "make_injector": lambda: platform_mod.make_injector(None, None, InjectionMode.SENDINPUT),
    "make_autostart": lambda: platform_mod.make_autostart(),
    "make_hotkey_factory": lambda: platform_mod.make_hotkey_factory(None),
    "make_single_instance": lambda: platform_mod.make_single_instance(None),
}


@pytest.mark.parametrize("name", sorted(UNSUPPORTED_CALLS))
@pytest.mark.parametrize("platform", ["linux", "freebsd", "sunos5"])
def test_every_factory_raises_app_error_on_an_unsupported_platform(monkeypatch, name, platform):
    _as_platform(monkeypatch, platform)
    with pytest.raises(AppError) as excinfo:
        UNSUPPORTED_CALLS[name]()
    assert excinfo.value.user_message == "RoEnSpeak supports Windows and macOS."


def test_every_factory_is_covered_by_the_unsupported_platform_test():
    factories = {n for n in platform_mod.__all__ if n.startswith("make_")}
    assert factories == set(UNSUPPORTED_CALLS)


# ---------------------------------------------------------------------------
# open_folder
# ---------------------------------------------------------------------------


@pytest.fixture
def no_real_launch(monkeypatch):
    """Record what open_folder would launch; never open a real file manager window."""
    calls = {"popen": [], "startfile": []}
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, *a, **kw: calls["popen"].append(list(cmd)))
    monkeypatch.setattr(os, "startfile", lambda target: calls["startfile"].append(target),
                        raising=False)
    return calls


def test_open_folder_uses_open_on_macos(monkeypatch, no_real_launch, tmp_path):
    _as_platform(monkeypatch, "darwin")
    platform_mod.open_folder(tmp_path)
    assert no_real_launch["popen"] == [["open", str(tmp_path)]]
    assert no_real_launch["startfile"] == []


def test_open_folder_uses_startfile_on_windows(monkeypatch, no_real_launch, tmp_path):
    _as_platform(monkeypatch, "win32")
    platform_mod.open_folder(tmp_path)
    assert no_real_launch["startfile"] == [str(tmp_path)]
    assert no_real_launch["popen"] == []


def test_open_folder_uses_xdg_open_elsewhere(monkeypatch, no_real_launch, tmp_path):
    _as_platform(monkeypatch, "linux")
    platform_mod.open_folder(tmp_path)
    assert no_real_launch["popen"] == [["xdg-open", str(tmp_path)]]
    assert no_real_launch["startfile"] == []


def test_open_folder_passes_the_path_as_a_string(monkeypatch, no_real_launch, tmp_path):
    _as_platform(monkeypatch, "darwin")
    target = tmp_path / "logs"
    platform_mod.open_folder(target)
    (cmd,) = no_real_launch["popen"]
    assert all(isinstance(part, str) for part in cmd)
    assert cmd[-1] == str(target)


# ---------------------------------------------------------------------------
# User-facing wording
# ---------------------------------------------------------------------------


def test_autostart_label_on_this_host():
    label = platform_mod.autostart_label()
    assert isinstance(label, str) and label.strip()
    if contracts.IS_MACOS:
        assert label == "Open at login"
        assert "Windows" not in label
    elif contracts.IS_WINDOWS:
        assert label == "Start with Windows"


@pytest.mark.parametrize("platform,expected", [("win32", "Start with Windows"),
                                               ("darwin", "Open at login")])
def test_autostart_label_per_platform(monkeypatch, platform, expected):
    _as_platform(monkeypatch, platform)
    assert platform_mod.autostart_label() == expected


def test_autostart_label_elsewhere_is_still_usable(monkeypatch):
    _as_platform(monkeypatch, "linux")
    assert platform_mod.autostart_label().strip()


@pytest.mark.skipif(not contracts.IS_MACOS, reason="macOS wording")
@pytest.mark.parametrize("reason", [None, "", "Metal device not found"])
def test_gpu_requirement_message_on_macos(reason):
    msg = platform_mod.gpu_requirement_message(reason)
    assert msg.strip()
    assert "macOS" in msg and contracts.MIN_MACOS_VERSION in msg
    assert "CPU mode" in msg
    assert "NVIDIA" not in msg and "CUDA" not in msg
    if reason:
        assert reason.rstrip(". ") in msg


@pytest.mark.skipif(not contracts.IS_WINDOWS, reason="Windows wording")
@pytest.mark.parametrize("reason", [None, "cudnn64_9.dll not found"])
def test_gpu_requirement_message_on_windows(reason):
    msg = platform_mod.gpu_requirement_message(reason)
    assert msg.strip()
    assert "NVIDIA" in msg
    if reason:
        assert reason.rstrip(". ") in msg


def test_gpu_requirement_message_elsewhere(monkeypatch):
    _as_platform(monkeypatch, "linux")
    msg = platform_mod.gpu_requirement_message("no GPU backend")
    assert msg.strip()
    assert "CPU mode" in msg
    assert "no GPU backend" in msg
