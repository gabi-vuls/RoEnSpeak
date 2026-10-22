"""Shared pytest setup (orchestrator-owned).

Every test runs with config/data dirs redirected to a temp dir, so the user's real
%APPDATA%/%LOCALAPPDATA% state is never touched. Tests that need the real downloaded
models (marker gpu/desktop) use ROENSPEAK_TEST_MODELS_DIR, or default to the user's models dir.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from app import contracts

IS_WINDOWS = sys.platform == "win32"
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


def pytest_collection_modifyitems(config, items):
    if IS_WINDOWS:
        return
    skip = pytest.mark.skip(reason="needs Windows")
    for item in items:
        if any(item.get_closest_marker(m) for m in ("gpu", "mic", "desktop")):
            item.add_marker(skip)
