"""Spec/contract tests for app.core.config (JsonConfigStore, config_to_dict, config_from_dict, migrate).

Derived from SPEC.md (defaults) and contracts.Config / ConfigStore docstrings plus the
schema-0 comment. Every test runs with ROENSPEAK_CONFIG_DIR pointed at a temp dir (conftest).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
from pathlib import Path

import pytest

from app import contracts
from app.contracts import (
    CONFIG_SCHEMA_VERSION,
    DEFAULT_BLOCKLIST,
    ComputeType,
    Config,
    ConfigError,
    ConfigStore,
    DecodeConfig,
    FilterConfig,
    InjectionMode,
    Task,
    UiConfig,
    VadConfig,
)
from app.core.config import JsonConfigStore, config_from_dict, config_to_dict, migrate

BAD_NAME_RE = re.compile(r"^config\.json\.bad-\d{8}-\d{6}")

DEFAULT_DICT = {
    "armed": True,
    "hotkey": "Alt+Q",
    "mic_device": None,
    "model": "large-v3",
    "compute_type": "float16",
    "task": "translate",
    "injection_mode": "sendinput",
    "cpu_mode": False,
    "idle_unload_minutes": 0,
    "decode": {"beam_size": 5},
    "vad": {
        "threshold": 0.5,
        "neg_threshold_offset": 0.15,
        "min_silence_ms": 500,
        "max_segment_s": 15.0,
        "speech_pad_ms": 200,
        "min_speech_ms": 250,
    },
    "filter": {
        "no_speech_prob_threshold": 0.6,
        "avg_logprob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
        "blocklist": list(DEFAULT_BLOCKLIST),
    },
    "ui": {"overlay": True, "sounds": True, "start_with_windows": False, "first_run_done": False},
    "schema_version": CONFIG_SCHEMA_VERSION,
}

#: Every field differs from its default, so per-field fallbacks are observable.
CUSTOM = Config(
    armed=False,
    hotkey="Ctrl+Alt+F9",
    mic_device="Microfon (Realtek® Audio)",
    model="medium",
    compute_type=ComputeType.INT8_FLOAT16,
    task=Task.TRANSCRIBE,
    injection_mode=InjectionMode.CLIPBOARD,
    cpu_mode=True,
    idle_unload_minutes=30,
    decode=DecodeConfig(beam_size=3),
    vad=VadConfig(threshold=0.6, neg_threshold_offset=0.1, min_silence_ms=700, max_segment_s=10.0,
                  speech_pad_ms=300, min_speech_ms=100),
    filter=FilterConfig(no_speech_prob_threshold=0.7, avg_logprob_threshold=-2.0,
                        compression_ratio_threshold=3.0, blocklist=("foo bar", "baz*")),
    ui=UiConfig(overlay=False, sounds=False, start_with_windows=True, first_run_done=True),
)


def _assert_no_tmp_files(directory: Path) -> None:
    assert sorted(p.name for p in directory.iterdir()) == ["config.json"]


def _with_default(cfg: Config, path: tuple[str, ...]) -> Config:
    """cfg with the field at `path` reset to its default."""
    if len(path) == 1:
        return dataclasses.replace(cfg, **{path[0]: getattr(Config(), path[0])})
    sub_name, leaf = path
    sub = getattr(cfg, sub_name)
    default_sub = type(sub)()
    return dataclasses.replace(cfg, **{sub_name: dataclasses.replace(sub, **{leaf: getattr(default_sub, leaf)})})


def _with_value(raw: dict, path: tuple[str, ...], value) -> dict:
    raw = copy.deepcopy(raw)
    d = raw
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    return raw


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_spec_defaults():
    c = Config()
    assert c.hotkey == "Alt+Q"
    assert c.model == "large-v3"
    assert c.compute_type == ComputeType.FLOAT16
    assert c.decode.beam_size == 5
    assert c.task == Task.TRANSLATE
    assert c.injection_mode == InjectionMode.SENDINPUT
    assert c.cpu_mode is False
    assert c.idle_unload_minutes == 0
    assert c.vad.min_silence_ms == 500
    assert c.vad.max_segment_s == 15.0
    assert c.armed is True
    assert c.mic_device is None
    assert c.schema_version == CONFIG_SCHEMA_VERSION


def test_store_is_config_store_and_default_path(isolated_app_dirs):
    store = JsonConfigStore()
    assert isinstance(store, ConfigStore)
    assert Path(store.path) == contracts.config_path()
    assert Path(store.path).parent == isolated_app_dirs["config"]


def test_missing_file_gives_defaults_no_warnings_nothing_written(tmp_path):
    path = tmp_path / "cfgdir" / "config.json"
    store = JsonConfigStore(path)
    cfg = store.load()
    assert cfg == Config()
    assert list(store.last_warnings) == []
    assert not path.exists()
    assert not path.parent.exists() or list(path.parent.iterdir()) == []


def test_default_store_missing_file(isolated_app_dirs):
    store = JsonConfigStore()
    assert store.load() == Config()
    assert list(store.last_warnings) == []
    assert not contracts.config_path().exists()


# ---------------------------------------------------------------------------
# Serialization shape and round trip
# ---------------------------------------------------------------------------


def test_config_to_dict_shape():
    d = config_to_dict(Config())
    assert d == DEFAULT_DICT
    assert isinstance(d["filter"]["blocklist"], list)
    assert json.loads(json.dumps(d)) == d


def test_config_to_dict_enums_as_values():
    d = config_to_dict(CUSTOM)
    assert d["compute_type"] == "int8_float16"
    assert d["task"] == "transcribe"
    assert d["injection_mode"] == "clipboard"
    assert d["filter"]["blocklist"] == ["foo bar", "baz*"]
    assert d["schema_version"] == CONFIG_SCHEMA_VERSION
    json.dumps(d)  # plain JSON types only


@pytest.mark.parametrize("cfg", [Config(), CUSTOM,
                                 dataclasses.replace(CUSTOM, model="large-v3-turbo", task=Task.TRANSCRIBE),
                                 dataclasses.replace(CUSTOM, mic_device=None)])
def test_dict_round_trip(cfg):
    back, warnings = config_from_dict(config_to_dict(cfg))
    assert back == cfg
    assert list(warnings) == []


@pytest.mark.parametrize("cfg", [Config(), CUSTOM])
def test_save_load_round_trip(tmp_path, cfg):
    path = tmp_path / "a" / "b" / "config.json"  # dirs created by save
    store = JsonConfigStore(path)
    store.save(cfg)
    assert path.exists()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == config_to_dict(cfg)
    assert on_disk["schema_version"] == CONFIG_SCHEMA_VERSION
    loaded = JsonConfigStore(path).load()
    assert loaded == cfg
    _assert_no_tmp_files(path.parent)


def test_save_overwrites_atomically_without_leftovers(tmp_path):
    path = tmp_path / "config.json"
    store = JsonConfigStore(path)
    store.save(Config())
    store.save(CUSTOM)
    store.save(dataclasses.replace(CUSTOM, hotkey="Win+F12"))
    _assert_no_tmp_files(tmp_path)
    assert store.load().hotkey == "Win+F12"
    assert list(store.last_warnings) == []


def test_save_to_path_under_a_file_raises_config_error(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    store = JsonConfigStore(blocker / "config.json")
    with pytest.raises(ConfigError):
        store.save(Config())


def test_save_onto_a_directory_raises_config_error(tmp_path):
    target = tmp_path / "config.json"
    target.mkdir()
    (target / "keep").write_text("x", encoding="utf-8")
    store = JsonConfigStore(target)
    with pytest.raises(ConfigError):
        store.save(Config())
    assert (target / "keep").exists()


# ---------------------------------------------------------------------------
# Corrupt files -> backup + defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        b"{not json",
        b"",
        b"[1, 2, 3]",
        b"42",
        b"null",
        b'"a string"',
        b"\xff\xfe\xfa\x00{garbage",
        json.dumps({**DEFAULT_DICT, "schema_version": CONFIG_SCHEMA_VERSION + 1}).encode("utf-8"),
    ],
    ids=["invalid-json", "empty", "list", "number", "null", "string", "binary", "schema-too-new"],
)
def test_corrupt_file_backed_up_and_defaults_used(tmp_path, content):
    path = tmp_path / "config.json"
    path.write_bytes(content)
    store = JsonConfigStore(path)
    cfg = store.load()
    assert cfg == Config()
    assert len(store.last_warnings) == 1
    assert not path.exists(), "the corrupt file must be renamed away"
    backups = [p for p in tmp_path.iterdir() if BAD_NAME_RE.match(p.name)]
    assert len(backups) == 1, [p.name for p in tmp_path.iterdir()]
    assert backups[0].read_bytes() == content


def test_last_warnings_reflect_most_recent_load(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{oops", encoding="utf-8")
    store = JsonConfigStore(path)
    store.load()
    assert len(store.last_warnings) == 1
    store.save(CUSTOM)
    assert store.load() == CUSTOM
    assert list(store.last_warnings) == []


# ---------------------------------------------------------------------------
# Per-field validation
# ---------------------------------------------------------------------------

INVALID_FIELDS = [
    (("armed",), "yes"),
    (("armed",), 1),
    (("armed",), None),
    (("hotkey",), 5),
    (("hotkey",), "Q"),  # no modifier
    (("hotkey",), "Shift+Q"),  # Shift-only on a printable key
    (("hotkey",), ""),
    (("hotkey",), "Alt+NoSuchKey"),
    (("mic_device",), 5),
    (("mic_device",), ["a"]),
    (("model",), "huge"),
    (("model",), "distil-large-v3"),
    (("model",), 3),
    (("compute_type",), "fp32"),
    (("compute_type",), "int8"),
    (("compute_type",), 16),
    (("task",), "summarize"),
    (("injection_mode",), "paste"),
    (("cpu_mode",), "true"),
    (("idle_unload_minutes",), "10"),
    (("idle_unload_minutes",), -1),
    (("idle_unload_minutes",), 1441),
    (("decode", "beam_size"), 0),
    (("decode", "beam_size"), 11),
    (("decode", "beam_size"), "5"),
    (("vad", "threshold"), 0.04),
    (("vad", "threshold"), 0.96),
    (("vad", "threshold"), "0.5"),
    (("vad", "neg_threshold_offset"), -0.01),
    (("vad", "neg_threshold_offset"), 0.51),
    (("vad", "min_silence_ms"), 99),
    (("vad", "min_silence_ms"), 5001),
    (("vad", "min_silence_ms"), "500"),
    (("vad", "max_segment_s"), 1.9),
    (("vad", "max_segment_s"), 30.1),
    (("vad", "speech_pad_ms"), -1),
    (("vad", "speech_pad_ms"), 1001),
    (("vad", "min_speech_ms"), -1),
    (("vad", "min_speech_ms"), 2001),
    (("filter", "no_speech_prob_threshold"), -0.1),
    (("filter", "no_speech_prob_threshold"), 1.1),
    (("filter", "avg_logprob_threshold"), -10.1),
    (("filter", "avg_logprob_threshold"), 0.1),
    (("filter", "compression_ratio_threshold"), 0.9),
    (("filter", "compression_ratio_threshold"), 10.1),
    (("filter", "blocklist"), "foo"),
    (("ui", "overlay"), "no"),
    (("ui", "sounds"), None),
    (("ui", "start_with_windows"), 0),
    (("ui", "first_run_done"), "yes"),
]


@pytest.mark.parametrize("path,value", INVALID_FIELDS, ids=[f"{'.'.join(p)}={v!r}" for p, v in INVALID_FIELDS])
def test_invalid_field_falls_back_to_default_others_survive(path, value):
    raw = _with_value(config_to_dict(CUSTOM), path, value)
    cfg, warnings = config_from_dict(raw)
    assert cfg == _with_default(CUSTOM, path)
    assert len(warnings) == 1, warnings


VALID_EDGES = [
    (("idle_unload_minutes",), 0),
    (("idle_unload_minutes",), 1440),
    (("decode", "beam_size"), 1),
    (("decode", "beam_size"), 10),
    (("vad", "threshold"), 0.05),
    (("vad", "threshold"), 0.95),
    (("vad", "neg_threshold_offset"), 0.0),
    (("vad", "neg_threshold_offset"), 0.5),
    (("vad", "min_silence_ms"), 100),
    (("vad", "min_silence_ms"), 5000),
    (("vad", "max_segment_s"), 2.0),
    (("vad", "max_segment_s"), 30.0),
    (("vad", "speech_pad_ms"), 0),
    (("vad", "speech_pad_ms"), 1000),
    (("vad", "min_speech_ms"), 0),
    (("vad", "min_speech_ms"), 2000),
    (("filter", "no_speech_prob_threshold"), 0.0),
    (("filter", "no_speech_prob_threshold"), 1.0),
    (("filter", "avg_logprob_threshold"), -10.0),
    (("filter", "avg_logprob_threshold"), 0.0),
    (("filter", "compression_ratio_threshold"), 1.0),
    (("filter", "compression_ratio_threshold"), 10.0),
    (("filter", "blocklist"), []),
]


@pytest.mark.parametrize("path,value", VALID_EDGES, ids=[f"{'.'.join(p)}={v!r}" for p, v in VALID_EDGES])
def test_range_edges_are_valid(path, value):
    raw = _with_value(config_to_dict(CUSTOM), path, value)
    cfg, warnings = config_from_dict(raw)
    assert list(warnings) == []
    obj = cfg
    for k in path:
        obj = getattr(obj, k)
    expected = tuple(value) if isinstance(value, list) else value
    assert obj == expected


def test_int_accepted_for_float_field():
    # JSON has no int/float distinction a user would notice: "max_segment_s": 10 is valid.
    raw = _with_value(config_to_dict(Config()), ("vad", "max_segment_s"), 10)
    cfg, warnings = config_from_dict(raw)
    assert cfg.vad.max_segment_s == 10.0
    assert list(warnings) == []


def test_turbo_with_translate_falls_back_to_default_model():
    raw = config_to_dict(dataclasses.replace(CUSTOM, task=Task.TRANSLATE))
    raw["model"] = "large-v3-turbo"
    cfg, warnings = config_from_dict(raw)
    assert cfg.model == "large-v3"
    assert cfg.task == Task.TRANSLATE
    assert cfg == dataclasses.replace(CUSTOM, task=Task.TRANSLATE, model="large-v3")
    assert len(warnings) == 1


def test_turbo_with_transcribe_is_valid():
    raw = config_to_dict(dataclasses.replace(CUSTOM, model="large-v3-turbo", task=Task.TRANSCRIBE))
    cfg, warnings = config_from_dict(raw)
    assert cfg.model == "large-v3-turbo"
    assert list(warnings) == []


def test_one_warning_per_invalid_field():
    raw = config_to_dict(CUSTOM)
    raw["model"] = "huge"
    raw["vad"]["min_silence_ms"] = 1
    raw["ui"]["sounds"] = "loud"
    cfg, warnings = config_from_dict(raw)
    assert len(warnings) == 3
    expected = _with_default(_with_default(_with_default(CUSTOM, ("model",)), ("vad", "min_silence_ms")),
                             ("ui", "sounds"))
    assert cfg == expected


def test_unknown_keys_ignored():
    raw = config_to_dict(CUSTOM)
    raw["future_option"] = {"x": 1}
    raw["vad"]["new_knob"] = 3
    raw["ui"]["theme"] = "dark"
    cfg, warnings = config_from_dict(raw)
    assert cfg == CUSTOM
    assert list(warnings) == []


def test_missing_keys_default_without_warnings():
    cfg, warnings = config_from_dict({"schema_version": CONFIG_SCHEMA_VERSION, "model": "medium",
                                      "vad": {"min_silence_ms": 300}})
    assert cfg == dataclasses.replace(Config(), model="medium", vad=VadConfig(min_silence_ms=300))
    assert list(warnings) == []


@pytest.mark.parametrize("given,canonical", [("alt+q", "Alt+Q"), (" Alt + Q ", "Alt+Q"),
                                             ("ctrl + alt + f9", "Ctrl+Alt+F9"), ("alt+ctrl+d", "Ctrl+Alt+D")])
def test_hotkey_normalized(given, canonical):
    cfg, warnings = config_from_dict({**DEFAULT_DICT, "hotkey": given})
    assert cfg.hotkey == canonical
    assert list(warnings) == []


def test_invalid_field_in_file_is_not_backed_up(tmp_path):
    path = tmp_path / "config.json"
    raw = config_to_dict(CUSTOM)
    raw["decode"]["beam_size"] = 99
    path.write_text(json.dumps(raw), encoding="utf-8")
    store = JsonConfigStore(path)
    cfg = store.load()
    assert cfg == dataclasses.replace(CUSTOM, decode=DecodeConfig())
    assert len(store.last_warnings) == 1
    assert path.exists()
    assert not [p for p in tmp_path.iterdir() if BAD_NAME_RE.match(p.name)]


# ---------------------------------------------------------------------------
# Schema 0 migration
# ---------------------------------------------------------------------------

SCHEMA0_FULL = {
    "hotkey": "ctrl+alt+d",
    "model": "medium",
    "compute_type": "int8_float16",
    "beam_size": 2,
    "translate": False,
    "clipboard_mode": True,
    "vad_silence_ms": 800,
    "vad_max_s": 10.0,
    "overlay": False,
    "sounds": False,
    "autostart": True,
    "armed": False,
    "mic": "USB Mic",
}

SCHEMA0_EXPECTED = dataclasses.replace(
    Config(),
    hotkey="Ctrl+Alt+D",
    model="medium",
    compute_type=ComputeType.INT8_FLOAT16,
    decode=DecodeConfig(beam_size=2),
    task=Task.TRANSCRIBE,
    injection_mode=InjectionMode.CLIPBOARD,
    vad=VadConfig(min_silence_ms=800, max_segment_s=10.0),
    ui=UiConfig(overlay=False, sounds=False, start_with_windows=True),
    armed=False,
    mic_device="USB Mic",
)


def test_migrate_schema0_full():
    original = copy.deepcopy(SCHEMA0_FULL)
    migrated = migrate(SCHEMA0_FULL)
    assert SCHEMA0_FULL == original, "migrate must be pure (input not mutated)"
    assert migrated["schema_version"] == CONFIG_SCHEMA_VERSION
    cfg, warnings = config_from_dict(migrated)
    assert cfg == SCHEMA0_EXPECTED
    assert list(warnings) == []


def test_migrate_schema0_opposite_flags():
    raw = {"translate": True, "clipboard_mode": False, "overlay": True, "sounds": True, "autostart": False,
           "armed": True, "mic": None}
    cfg, warnings = config_from_dict(migrate(raw))
    assert cfg == Config()
    assert list(warnings) == []


def test_migrate_empty_schema0_is_defaults():
    cfg, warnings = config_from_dict(migrate({}))
    assert cfg == Config()
    assert list(warnings) == []


@pytest.mark.parametrize(
    "raw,check",
    [
        ({"translate": False}, lambda c: c.task == Task.TRANSCRIBE),
        ({"clipboard_mode": True}, lambda c: c.injection_mode == InjectionMode.CLIPBOARD),
        ({"vad_silence_ms": 300}, lambda c: c.vad.min_silence_ms == 300),
        ({"vad_max_s": 5.0}, lambda c: c.vad.max_segment_s == 5.0),
        ({"overlay": False}, lambda c: c.ui.overlay is False),
        ({"sounds": False}, lambda c: c.ui.sounds is False),
        ({"autostart": True}, lambda c: c.ui.start_with_windows is True),
        ({"mic": "Headset"}, lambda c: c.mic_device == "Headset"),
        ({"armed": False}, lambda c: c.armed is False),
        ({"beam_size": 1}, lambda c: c.decode.beam_size == 1),
        ({"hotkey": "alt+w"}, lambda c: c.hotkey == "Alt+W"),
        ({"model": "large-v2"}, lambda c: c.model == "large-v2"),
        ({"compute_type": "int8_float16"}, lambda c: c.compute_type == ComputeType.INT8_FLOAT16),
    ],
    ids=lambda v: next(iter(v)) if isinstance(v, dict) else "",
)
def test_migrate_schema0_single_key(raw, check):
    cfg, warnings = config_from_dict(migrate(raw))
    assert check(cfg)
    assert list(warnings) == []


def test_migrate_schema0_turbo_transcription_is_valid():
    cfg, warnings = config_from_dict(migrate({"model": "large-v3-turbo", "translate": False}))
    assert cfg.model == "large-v3-turbo" and cfg.task == Task.TRANSCRIBE
    assert list(warnings) == []


def test_migrate_schema0_invalid_value_falls_back_with_warning():
    raw = dict(SCHEMA0_FULL, beam_size=99)
    cfg, warnings = config_from_dict(migrate(raw))
    assert cfg == dataclasses.replace(SCHEMA0_EXPECTED, decode=DecodeConfig())
    assert len(warnings) == 1


def test_migrate_current_schema_is_identity_semantically():
    raw = config_to_dict(CUSTOM)
    cfg, warnings = config_from_dict(migrate(raw))
    assert cfg == CUSTOM
    assert list(warnings) == []


def test_store_loads_schema0_in_memory_then_saves_current(tmp_path):
    path = tmp_path / "config.json"
    text = json.dumps(SCHEMA0_FULL)
    path.write_text(text, encoding="utf-8")
    store = JsonConfigStore(path)
    cfg = store.load()
    assert cfg == SCHEMA0_EXPECTED
    assert list(store.last_warnings) == []
    assert not [p for p in tmp_path.iterdir() if BAD_NAME_RE.match(p.name)]
    store.save(cfg)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == CONFIG_SCHEMA_VERSION
    assert on_disk == config_to_dict(SCHEMA0_EXPECTED)
    assert JsonConfigStore(path).load() == SCHEMA0_EXPECTED


def test_store_load_does_not_rewrite_schema0_file(tmp_path):
    path = tmp_path / "config.json"
    text = json.dumps(SCHEMA0_FULL)
    path.write_text(text, encoding="utf-8")
    JsonConfigStore(path).load()
    assert path.read_text(encoding="utf-8") == text
