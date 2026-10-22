"""Contract tests for the path / name / model-file helpers in app.contracts.

Covers the "Paths" block (config_dir, config_path, data_dir, models_dir, logs_dir,
single_instance_name, single_instance_lock_path) and the ggml model metadata
(GGML_REPO, ggml_model_file, ModelSpec.ggml_file / ggml_mb), as documented in
app/contracts.py and in the macOS port contract:

* the three ROENSPEAK_* env overrides win on every host,
* macOS puts config and data under ~/Library/Application Support/RoEnSpeak and logs under
  ~/Library/Logs/RoEnSpeak -- except that logs follow data_dir() whenever ROENSPEAK_DATA_DIR
  is set, which is what keeps an isolated instance's logs next to its own data,
* the Windows and generic-POSIX branches keep behaving as documented.

Both platform branches are exercised on any host: contracts reads sys.platform exactly once,
into the module-level IS_WINDOWS / IS_MACOS flags, so those flags (and sys.platform, for
consistency) are what the tests monkeypatch. The autouse conftest fixture sets all three env
overrides, so every branch test clears them first.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

from app import contracts
from app.contracts import (
    APP_NAME,
    ENV_CONFIG_DIR,
    ENV_DATA_DIR,
    ENV_MODELS_DIR,
    GGML_REPO,
    MODELS,
    ComputeType,
    ggml_model_file,
)

PLATFORMS = ["win32", "darwin", "linux"]


def _as_platform(monkeypatch, platform: str) -> None:
    """Pretend contracts was imported on `platform`."""
    monkeypatch.setattr(sys, "platform", platform, raising=False)
    monkeypatch.setattr(contracts, "IS_WINDOWS", platform == "win32")
    monkeypatch.setattr(contracts, "IS_MACOS", platform == "darwin")


def _no_env_overrides(monkeypatch) -> None:
    for name in (ENV_CONFIG_DIR, ENV_DATA_DIR, ENV_MODELS_DIR):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Env overrides win everywhere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", PLATFORMS)
def test_env_overrides_win_on_every_platform(tmp_path, monkeypatch, platform):
    _as_platform(monkeypatch, platform)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    monkeypatch.setenv(ENV_CONFIG_DIR, str(tmp_path / "cfg"))
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "data"))
    monkeypatch.setenv(ENV_MODELS_DIR, str(tmp_path / "models"))

    assert contracts.config_dir() == tmp_path / "cfg"
    assert contracts.config_path() == tmp_path / "cfg" / "config.json"
    assert contracts.data_dir() == tmp_path / "data"
    assert contracts.models_dir() == tmp_path / "models"
    assert contracts.logs_dir() == tmp_path / "data" / "logs"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_conftest_isolation_is_in_effect(isolated_app_dirs, platform, monkeypatch):
    """The autouse fixture's dirs are what the app sees (no real user state is touched)."""
    _as_platform(monkeypatch, platform)
    assert contracts.config_dir() == isolated_app_dirs["config"]
    assert contracts.data_dir() == isolated_app_dirs["data"]
    assert contracts.models_dir() == Path(isolated_app_dirs["models"])
    assert contracts.logs_dir() == isolated_app_dirs["data"] / "logs"


# ---------------------------------------------------------------------------
# macOS branch
# ---------------------------------------------------------------------------


def test_macos_dirs(monkeypatch):
    _as_platform(monkeypatch, "darwin")
    _no_env_overrides(monkeypatch)
    support = Path.home() / "Library" / "Application Support" / APP_NAME
    assert contracts.config_dir() == support
    assert contracts.config_path() == support / "config.json"
    assert contracts.data_dir() == support
    assert contracts.models_dir() == support / "models"
    assert contracts.logs_dir() == Path.home() / "Library" / "Logs" / APP_NAME


def test_macos_ignores_appdata(monkeypatch, tmp_path):
    """%APPDATA%/%LOCALAPPDATA% in the environment must not leak into the macOS branch."""
    _as_platform(monkeypatch, "darwin")
    _no_env_overrides(monkeypatch)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    support = Path.home() / "Library" / "Application Support" / APP_NAME
    assert contracts.config_dir() == support
    assert contracts.data_dir() == support


def test_macos_logs_follow_data_dir_override(monkeypatch, tmp_path):
    """The rule test isolation depends on: ROENSPEAK_DATA_DIR moves the logs with the data."""
    _as_platform(monkeypatch, "darwin")
    _no_env_overrides(monkeypatch)
    assert contracts.logs_dir() == Path.home() / "Library" / "Logs" / APP_NAME
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "isolated"))
    assert contracts.logs_dir() == tmp_path / "isolated" / "logs"
    assert contracts.logs_dir().parent == contracts.data_dir()


def test_macos_models_dir_override_does_not_move_the_logs(monkeypatch, tmp_path):
    _as_platform(monkeypatch, "darwin")
    _no_env_overrides(monkeypatch)
    monkeypatch.setenv(ENV_MODELS_DIR, str(tmp_path / "models"))
    assert contracts.models_dir() == tmp_path / "models"
    assert contracts.logs_dir() == Path.home() / "Library" / "Logs" / APP_NAME


# ---------------------------------------------------------------------------
# Windows branch
# ---------------------------------------------------------------------------


def test_windows_dirs(monkeypatch, tmp_path):
    _as_platform(monkeypatch, "win32")
    _no_env_overrides(monkeypatch)
    appdata, local = tmp_path / "Roaming", tmp_path / "Local"
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    assert contracts.config_dir() == appdata / APP_NAME
    assert contracts.config_path() == appdata / APP_NAME / "config.json"
    assert contracts.data_dir() == local / APP_NAME
    assert contracts.models_dir() == local / APP_NAME / "models"
    assert contracts.logs_dir() == local / APP_NAME / "logs"


def test_windows_without_appdata_falls_back_to_the_posix_layout(monkeypatch):
    """APPDATA/LOCALAPPDATA always exist in a real session; this only pins the fallback."""
    _as_platform(monkeypatch, "win32")
    _no_env_overrides(monkeypatch)
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert contracts.config_dir() == Path.home() / ".config" / APP_NAME
    assert contracts.data_dir() == Path.home() / ".local" / "share" / APP_NAME
    assert contracts.logs_dir() == Path.home() / ".local" / "share" / APP_NAME / "logs"


# ---------------------------------------------------------------------------
# Generic POSIX branch
# ---------------------------------------------------------------------------


def test_generic_posix_dirs(monkeypatch, tmp_path):
    _as_platform(monkeypatch, "linux")
    _no_env_overrides(monkeypatch)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))  # must be ignored off Windows
    assert contracts.config_dir() == Path.home() / ".config" / APP_NAME
    assert contracts.data_dir() == Path.home() / ".local" / "share" / APP_NAME
    assert contracts.models_dir() == Path.home() / ".local" / "share" / APP_NAME / "models"
    assert contracts.logs_dir() == Path.home() / ".local" / "share" / APP_NAME / "logs"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_models_dir_derives_from_data_dir_unless_overridden(monkeypatch, tmp_path, platform):
    _as_platform(monkeypatch, platform)
    _no_env_overrides(monkeypatch)
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "data"))
    assert contracts.models_dir() == tmp_path / "data" / "models"
    monkeypatch.setenv(ENV_MODELS_DIR, str(tmp_path / "elsewhere"))
    assert contracts.models_dir() == tmp_path / "elsewhere"
    assert contracts.data_dir() == tmp_path / "data"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_path_helpers_return_absolute_paths(monkeypatch, tmp_path, platform):
    _as_platform(monkeypatch, platform)
    _no_env_overrides(monkeypatch)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    for fn in (contracts.config_dir, contracts.config_path, contracts.data_dir,
               contracts.models_dir, contracts.logs_dir, contracts.single_instance_lock_path):
        path = fn()
        assert isinstance(path, Path)
        assert path.is_absolute(), f"{fn.__name__} -> {path}"


# ---------------------------------------------------------------------------
# single_instance_name / single_instance_lock_path
# ---------------------------------------------------------------------------


def test_single_instance_name_windows(monkeypatch):
    _as_platform(monkeypatch, "win32")
    _no_env_overrides(monkeypatch)
    assert contracts.single_instance_name() == f"Local\\{APP_NAME}.SingleInstance"


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_single_instance_name_has_no_local_prefix_off_windows(monkeypatch, platform):
    _as_platform(monkeypatch, platform)
    _no_env_overrides(monkeypatch)
    name = contracts.single_instance_name()
    assert name == f"{APP_NAME}.SingleInstance"
    assert "\\" not in name and not name.startswith("Local")


@pytest.mark.parametrize("platform", PLATFORMS)
def test_single_instance_name_suffixed_per_config_dir(monkeypatch, tmp_path, platform):
    _as_platform(monkeypatch, platform)
    _no_env_overrides(monkeypatch)
    base = contracts.single_instance_name()

    override = str(tmp_path / "cfg")
    monkeypatch.setenv(ENV_CONFIG_DIR, override)
    suffixed = contracts.single_instance_name()
    expected = hashlib.sha1(override.encode("utf-8")).hexdigest()[:12]
    assert suffixed == f"{base}.{expected}"
    assert contracts.single_instance_name() == suffixed, "must be stable"

    monkeypatch.setenv(ENV_CONFIG_DIR, str(tmp_path / "other"))
    assert contracts.single_instance_name() != suffixed


@pytest.mark.parametrize("platform", PLATFORMS)
def test_lock_path_sits_under_data_dir_with_no_separator_in_the_name(monkeypatch, tmp_path, platform):
    _as_platform(monkeypatch, platform)
    _no_env_overrides(monkeypatch)
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "data"))
    monkeypatch.setenv(ENV_CONFIG_DIR, str(tmp_path / "cfg"))

    lock = contracts.single_instance_lock_path()
    assert lock.parent == contracts.data_dir() == tmp_path / "data"
    assert lock.suffix == ".lock"
    assert lock.name == contracts.single_instance_name().replace("\\", ".") + ".lock"
    for separator in ("\\", "/", os.sep, os.altsep or "/"):
        assert separator not in lock.name


# ---------------------------------------------------------------------------
# ggml model metadata (macOS engine)
# ---------------------------------------------------------------------------


def test_ggml_repo_is_set():
    assert isinstance(GGML_REPO, str)
    assert GGML_REPO.strip()
    assert GGML_REPO.count("/") == 1, "a Hugging Face repo id, <owner>/<name>"


@pytest.mark.parametrize("compute_type", list(ComputeType), ids=[c.value for c in ComputeType])
@pytest.mark.parametrize("model", sorted(MODELS))
def test_ggml_model_file_for_every_model_and_compute_type(model, compute_type):
    name = ggml_model_file(model, compute_type)
    assert name == MODELS[model].ggml_file[compute_type]
    assert name.endswith(".bin")
    assert "/" not in name and "\\" not in name, "a file name inside GGML_REPO, not a path"
    assert model in name, f"{name} should name its model"
    if compute_type is ComputeType.INT8_FLOAT16:
        assert "q5_0" in name, "int8_float16 maps onto the ggml q5_0 file"
    else:
        assert "q5_0" not in name


@pytest.mark.parametrize("model", sorted(MODELS))
def test_ggml_model_file_accepts_the_raw_enum_value(model):
    for compute_type in ComputeType:
        assert ggml_model_file(model, compute_type.value) == ggml_model_file(model, compute_type)


def test_ggml_model_files_are_unique_across_models_and_types():
    names = [ggml_model_file(m, c) for m in MODELS for c in ComputeType]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("model", ["huge", "", "large_v3", "distil-large-v3"])
def test_ggml_model_file_rejects_unknown_models(model):
    with pytest.raises(ValueError):
        ggml_model_file(model, ComputeType.FLOAT16)


def test_ggml_model_file_rejects_an_unknown_compute_type():
    with pytest.raises(ValueError):
        ggml_model_file(next(iter(MODELS)), "fp32")


@pytest.mark.parametrize("model", sorted(MODELS))
def test_every_model_has_ggml_files_and_sizes(model):
    spec = MODELS[model]
    for compute_type in ComputeType:
        assert spec.ggml_file.get(compute_type), f"{model}/{compute_type.value}: no ggml file"
        size = spec.ggml_mb.get(compute_type)
        assert isinstance(size, int) and size > 0, f"{model}/{compute_type.value}: no ggml size"
    assert spec.ggml_mb[ComputeType.FLOAT16] > spec.ggml_mb[ComputeType.INT8_FLOAT16], \
        "the quantized file must be the smaller download"
    assert spec.ggml_mb[ComputeType.FLOAT16] <= spec.vram_mb[ComputeType.FLOAT16], \
        "the model file cannot be bigger than the memory budget for loading it"
