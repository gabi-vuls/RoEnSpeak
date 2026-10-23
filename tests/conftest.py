"""Shared pytest setup (orchestrator-owned).

Every test runs with config/data dirs redirected to a temp dir, so the user's real state
(%APPDATA%/%LOCALAPPDATA% on Windows, ~/Library/Application Support and ~/Library/Logs on
macOS) is never touched. Tests that need the real downloaded models (markers gpu/metal/
desktop) use ROENSPEAK_TEST_MODELS_DIR, or default to the user's models dir.

Hardware gate (pytest_collection_modifyitems). pyproject deselects gpu/metal/mic/desktop by
default, so a bare `pytest` is hardware-free on both platforms; `-m <marker>` opts in. The
gate below then skips what the *host* cannot run:

    gpu      the Windows marker: NVIDIA GPU + CUDA (faster-whisper/CTranslate2).
    metal    the macOS marker: Apple silicon + Metal (whisper.cpp).
    desktop  both platforms (SendInput on Windows, CGEvent on macOS), each on its own host;
    mic      both platforms. A test written for one OS pairs its hardware marker with that
             OS's engine marker (gpu / metal), which is what picks the host.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from app import contracts

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
_REAL_MODELS_DIR = Path(os.environ.get("ROENSPEAK_TEST_MODELS_DIR") or contracts.models_dir())


@pytest.fixture(autouse=True)
def isolated_app_dirs(tmp_path, monkeypatch):
    cfg = tmp_path / "config"
    data = tmp_path / "data"
    monkeypatch.setenv(contracts.ENV_CONFIG_DIR, str(cfg))
    monkeypatch.setenv(contracts.ENV_DATA_DIR, str(data))
    monkeypatch.setenv(contracts.ENV_MODELS_DIR, str(_REAL_MODELS_DIR))
    return {"config": cfg, "data": data, "models": _REAL_MODELS_DIR}


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


def _skip_reason(item) -> str | None:
    """Why this host cannot run `item`, or None if it can."""
    if item.get_closest_marker("gpu") and not IS_WINDOWS:
        return "needs Windows (NVIDIA GPU + CUDA)"
    if item.get_closest_marker("metal") and not IS_MACOS:
        return "needs macOS (Apple silicon + Metal)"
    if not (IS_WINDOWS or IS_MACOS):
        for marker in ("desktop", "mic"):
            if item.get_closest_marker(marker):
                return f"needs Windows or macOS (got {sys.platform})"
    return None


def pytest_collection_modifyitems(config, items):
    for item in items:
        reason = _skip_reason(item)
        if reason:
            item.add_marker(pytest.mark.skip(reason=reason))
