"""Config persistence: JSON file at config_path(), schema migration, per-field validation.

Pure helpers (config_to_dict / config_from_dict / migrate) never touch the disk.
JsonConfigStore.load() never raises because of file content; see ConfigStore in contracts.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from app.contracts import (
    CONFIG_SCHEMA_VERSION,
    MODELS,
    ComputeType,
    Config,
    ConfigError,
    DecodeConfig,
    FilterConfig,
    InjectionMode,
    Task,
    UiConfig,
    VadConfig,
    config_path,
)

log = logging.getLogger("roenspeak.core.config")

_MISSING = object()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


def config_to_dict(cfg: Config) -> dict:
    """Config -> JSON-ready dict. Enums as values, tuples as lists, nested dataclasses as
    objects. "schema_version" is always the current CONFIG_SCHEMA_VERSION."""
    out = _to_jsonable(cfg)
    out["schema_version"] = CONFIG_SCHEMA_VERSION
    return out


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _schema_version_of(raw: Mapping) -> Any:
    return raw["schema_version"] if "schema_version" in raw else 0


def _migrate_0_to_1(old: Mapping) -> dict:
    """Flat legacy schema 0 -> nested schema 1. Values are copied as-is; validation happens
    later in config_from_dict, so an invalid legacy value still yields a per-field warning."""
    new: dict[str, Any] = {}
    for key in ("armed", "hotkey", "model", "compute_type"):
        if key in old:
            new[key] = old[key]
    if "mic" in old:
        new["mic_device"] = old["mic"]
    if "beam_size" in old:
        new["decode"] = {"beam_size": old["beam_size"]}
    if "translate" in old:
        v = old["translate"]
        new["task"] = (Task.TRANSLATE.value if v else Task.TRANSCRIBE.value) if isinstance(v, bool) else v
    if "clipboard_mode" in old:
        v = old["clipboard_mode"]
        new["injection_mode"] = (
            (InjectionMode.CLIPBOARD.value if v else InjectionMode.SENDINPUT.value) if isinstance(v, bool) else v
        )
    vad: dict[str, Any] = {}
    if "vad_silence_ms" in old:
        vad["min_silence_ms"] = old["vad_silence_ms"]
    if "vad_max_s" in old:
        vad["max_segment_s"] = old["vad_max_s"]
    if vad:
        new["vad"] = vad
    ui: dict[str, Any] = {}
    for old_key, new_key in (("overlay", "overlay"), ("sounds", "sounds"), ("autostart", "start_with_windows")):
        if old_key in old:
            ui[new_key] = old[old_key]
    if ui:
        new["ui"] = ui
    return new


_MIGRATIONS: dict[int, Callable[[Mapping], dict]] = {0: _migrate_0_to_1}


def migrate(raw: dict) -> dict:
    """Any supported schema -> current-schema dict. Pure (the input is not modified).

    Raises ConfigError when "schema_version" is not a non-negative integer or is newer
    than CONFIG_SCHEMA_VERSION.
    """
    if not isinstance(raw, Mapping):
        raise ConfigError("Config is not a JSON object.")
    version = _schema_version_of(raw)
    if not _is_int(version) or version < 0:
        raise ConfigError(f"Config has an invalid schema_version {version!r}.")
    if version > CONFIG_SCHEMA_VERSION:
        raise ConfigError(
            f"Config schema_version {version} is newer than this app supports ({CONFIG_SCHEMA_VERSION})."
        )
    data: dict = copy.deepcopy(dict(raw))
    while version < CONFIG_SCHEMA_VERSION:
        data = _MIGRATIONS[version](data)
        version += 1
        log.info("Config migrated to schema %d", version)
    data["schema_version"] = CONFIG_SCHEMA_VERSION
    return data


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class _Invalid(Exception):
    pass


def _v_bool(v: Any) -> bool:
    if not isinstance(v, bool):
        raise _Invalid(f"expected true/false, got {v!r}")
    return v


def _v_int(lo: int, hi: int) -> Callable[[Any], int]:
    def check(v: Any) -> int:
        if not _is_int(v):
            raise _Invalid(f"expected an integer, got {v!r}")
        if not lo <= v <= hi:
            raise _Invalid(f"{v} is outside {lo}..{hi}")
        return int(v)

    return check


def _v_float(lo: float, hi: float) -> Callable[[Any], float]:
    def check(v: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise _Invalid(f"expected a number, got {v!r}")
        if not lo <= v <= hi:
            raise _Invalid(f"{v} is outside {lo}..{hi}")
        return float(v)

    return check


def _v_enum(cls: type[enum.Enum]) -> Callable[[Any], enum.Enum]:
    def check(v: Any) -> enum.Enum:
        if isinstance(v, cls):
            return v
        try:
            return cls(v)
        except (ValueError, TypeError):
            allowed = ", ".join(str(m.value) for m in cls)
            raise _Invalid(f"unknown value {v!r} (allowed: {allowed})") from None

    return check


def _v_model(v: Any) -> str:
    if not isinstance(v, str) or v not in MODELS:
        raise _Invalid(f"unknown model {v!r} (allowed: {', '.join(MODELS)})")
    return v


def _v_mic(v: Any) -> Optional[str]:
    if v is None:
        return None
    if not isinstance(v, str):
        raise _Invalid(f"expected a device name or null, got {v!r}")
    return v if v.strip() else None


def _v_blocklist(v: Any) -> tuple[str, ...]:
    if not isinstance(v, (list, tuple)) or not all(isinstance(x, str) for x in v):
        raise _Invalid("expected a list of strings")
    return tuple(v)


def _v_hotkey(v: Any) -> str:
    if not isinstance(v, str) or not v.strip():
        raise _Invalid(f"expected a hotkey string, got {v!r}")
    try:
        from app.win32.hotkey_parse import parse_hotkey
    except ImportError as e:  # win32 package unavailable (non-Windows tooling); keep as typed
        log.debug("hotkey parser unavailable (%s); hotkey kept unvalidated", e)
        return v.strip()
    try:
        return parse_hotkey(v).display
    except Exception as e:  # HotkeyParseError, or anything the parser chokes on
        raise _Invalid(f"invalid hotkey {v!r}: {getattr(e, 'user_message', e)}") from None


_VAD_FIELDS: dict[str, Callable[[Any], Any]] = {
    "threshold": _v_float(0.05, 0.95),
    "neg_threshold_offset": _v_float(0.0, 0.5),
    "min_silence_ms": _v_int(100, 5000),
    "max_segment_s": _v_float(2.0, 30.0),
    "speech_pad_ms": _v_int(0, 1000),
    "min_speech_ms": _v_int(0, 2000),
}
_DECODE_FIELDS: dict[str, Callable[[Any], Any]] = {"beam_size": _v_int(1, 10)}
_FILTER_FIELDS: dict[str, Callable[[Any], Any]] = {
    "no_speech_prob_threshold": _v_float(0.0, 1.0),
    "avg_logprob_threshold": _v_float(-10.0, 0.0),
    "compression_ratio_threshold": _v_float(1.0, 10.0),
    "blocklist": _v_blocklist,
}
_UI_FIELDS: dict[str, Callable[[Any], Any]] = {
    "overlay": _v_bool,
    "sounds": _v_bool,
    "start_with_windows": _v_bool,
    "first_run_done": _v_bool,
}
_TOP_FIELDS: dict[str, Callable[[Any], Any]] = {
    "armed": _v_bool,
    "hotkey": _v_hotkey,
    "mic_device": _v_mic,
    "model": _v_model,
    "compute_type": _v_enum(ComputeType),
    "task": _v_enum(Task),
    "injection_mode": _v_enum(InjectionMode),
    "cpu_mode": _v_bool,
    "idle_unload_minutes": _v_int(0, 1440),
}
_SECTIONS: dict[str, tuple[type, dict[str, Callable[[Any], Any]]]] = {
    "decode": (DecodeConfig, _DECODE_FIELDS),
    "vad": (VadConfig, _VAD_FIELDS),
    "filter": (FilterConfig, _FILTER_FIELDS),
    "ui": (UiConfig, _UI_FIELDS),
}


def _validate_fields(
    data: Mapping, fields: Mapping[str, Callable[[Any], Any]], defaults: Any, prefix: str, warnings: list[str]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, check in fields.items():
        value = data.get(name, _MISSING)
        if value is _MISSING:
            continue
        try:
            out[name] = check(value)
        except _Invalid as e:
            default = _to_jsonable(getattr(defaults, name))
            warnings.append(f"Setting '{prefix}{name}' is invalid ({e}); using default {default!r}.")
    return out


def config_from_dict(raw: Mapping) -> tuple[Config, list[str]]:
    """Dict (any supported schema) -> (Config, warnings). Pure.

    Invalid fields fall back to their defaults with one warning each; unknown keys are
    ignored. A dict that cannot be migrated (not an object, bad/too-new schema_version)
    yields Config() and a single warning.
    """
    warnings: list[str] = []
    try:
        data = migrate(raw)  # type: ignore[arg-type]
    except ConfigError as e:
        return Config(), [f"{e.user_message} Using default settings."]

    defaults = Config()
    kwargs = _validate_fields(data, _TOP_FIELDS, defaults, "", warnings)

    for section, (cls, fields) in _SECTIONS.items():
        sec_default = getattr(defaults, section)
        value = data.get(section, _MISSING)
        if value is _MISSING:
            continue
        if not isinstance(value, Mapping):
            warnings.append(f"Setting '{section}' is invalid (expected an object); using defaults.")
            continue
        sec_kwargs = _validate_fields(value, fields, sec_default, section + ".", warnings)
        kwargs[section] = dataclasses.replace(sec_default, **sec_kwargs) if sec_kwargs else cls()

    # Cross-field rule: transcription-only model cannot translate -> model falls back.
    model = kwargs.get("model", defaults.model)
    task = kwargs.get("task", defaults.task)
    if task == Task.TRANSLATE and not MODELS[model].translate_capable:
        warnings.append(
            f"Setting 'model' is invalid ({model!r} is transcription-only and cannot translate); "
            f"using default {defaults.model!r}."
        )
        kwargs["model"] = defaults.model

    kwargs["schema_version"] = CONFIG_SCHEMA_VERSION
    return Config(**kwargs), warnings


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class JsonConfigStore:
    """ConfigStore backed by a JSON file (default: config_path(), resolved at use time)."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._explicit_path = Path(path) if path is not None else None
        self._warnings: list[str] = []

    @property
    def path(self) -> Path:
        return self._explicit_path if self._explicit_path is not None else config_path()

    @property
    def last_warnings(self) -> Sequence[str]:
        return tuple(self._warnings)

    # -- load -------------------------------------------------------------

    def load(self) -> Config:
        self._warnings = []
        path = self.path
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            log.info("No config file at %s; using defaults", path)
            return Config()
        except OSError as e:
            return self._reject(path, f"could not be read ({e})")

        try:
            raw = json.loads(data.decode("utf-8-sig"))
        except (UnicodeDecodeError, ValueError) as e:
            return self._reject(path, f"is not valid JSON ({e})")
        if not isinstance(raw, dict):
            return self._reject(path, "does not contain a JSON object")
        version = _schema_version_of(raw)
        if not _is_int(version) or version < 0:
            return self._reject(path, f"has an invalid schema_version {version!r}")
        if version > CONFIG_SCHEMA_VERSION:
            return self._reject(
                path, f"has schema_version {version}, newer than supported ({CONFIG_SCHEMA_VERSION})"
            )

        cfg, warnings = config_from_dict(raw)
        for w in warnings:
            log.warning("config: %s", w)
        self._warnings = list(warnings)
        log.info("Config loaded from %s (schema %s)", path, version)
        return cfg

    def _reject(self, path: Path, reason: str) -> Config:
        backup = self._backup(path)
        if backup is not None:
            msg = f"The settings file {reason}. It was renamed to {backup.name} and default settings are used."
        else:
            msg = f"The settings file {reason}. Default settings are used."
        log.warning("config: %s", msg)
        self._warnings = [msg]
        return Config()

    @staticmethod
    def _backup(path: Path) -> Optional[Path]:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        candidate = path.with_name(f"{path.name}.bad-{stamp}")
        n = 1
        while candidate.exists():
            n += 1
            candidate = path.with_name(f"{path.name}.bad-{stamp}-{n}")
        try:
            os.replace(path, candidate)
            return candidate
        except OSError as e:
            log.error("Could not back up invalid config %s: %s", path, e)
            return None

    # -- save -------------------------------------------------------------

    def save(self, cfg: Config) -> None:
        path = self.path
        text = json.dumps(config_to_dict(cfg), indent=2, ensure_ascii=False) + "\n"
        tmp_name: Optional[str] = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=".config-", suffix=".tmp", dir=str(path.parent))
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            self._replace_with_retry(tmp_name, path)
            tmp_name = None
        except OSError as e:
            log.error("Could not save config to %s: %s", path, e)
            raise ConfigError("Could not save settings.", details=f"{path}: {e}") from e
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
        log.debug("Config saved to %s", path)

    @staticmethod
    def _replace_with_retry(src: str, dst: Path, attempts: int = 10, delay_s: float = 0.05) -> None:
        # On Windows, os.replace fails with PermissionError while another process (AV, an
        # editor) briefly holds the destination open.
        for i in range(attempts):
            try:
                os.replace(src, dst)
                return
            except PermissionError:
                if i == attempts - 1:
                    raise
                time.sleep(delay_s)
