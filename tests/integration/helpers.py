"""Helpers shared by the integration tests (TEST-INTEGRATION-owned; plain module, not a conftest).

These tests drive the real app as a subprocess (``python -m app.cli`` / ``python -m app.main``)
and check only the observable contract in app/contracts.py: exit codes, stdout format,
the JSON shape, and LOG_* lines in <DATA_DIR>/logs/roenspeak.log.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from app import contracts

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
RO_SAMPLE_WAV = FIXTURES / "ro_sample.wav"
RO_SAMPLE_TXT = FIXTURES / "ro_sample.txt"
REAL_MODELS_DIR = Path(os.environ.get("ROENSPEAK_TEST_MODELS_DIR") or contracts.models_dir())

#: Characters that only appear in Romanian text (both comma-below and cedilla forms).
ROMANIAN_DIACRITICS = set("ăâîșțşţĂÂÎȘȚŞŢ")
DEFAULT_TIMEOUT_S = 600


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def safe_print(*parts: object) -> None:
    """print() that never raises on a non-UTF-8 console (e.g. cp1250 with `pytest -s`):
    characters the current stdout can't encode are written as backslash escapes."""
    text = " ".join(str(p) for p in parts)
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(enc, errors="backslashreplace").decode(enc, errors="replace"))


# ---------------------------------------------------------------------------
# Fixture text
# ---------------------------------------------------------------------------


def _sections(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    cur: Optional[str] = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"\[(\w+)\]", line)
        if m:
            cur = m.group(1)
            out[cur] = []
        elif cur:
            out[cur].append(line)
    return out


SAMPLE = _sections(RO_SAMPLE_TXT)
EN_KEYWORDS = tuple(SAMPLE["keywords"])
RO_KEYWORDS = tuple(SAMPLE["romanian_keywords"])


def fold(text: str) -> str:
    """casefold + strip diacritics (ș -> s, ă -> a ...)."""
    nfkd = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def found_keywords(text: str, keywords) -> list[str]:
    t = fold(text)
    return [k for k in keywords if re.search(r"\b" + re.escape(fold(k)), t)]


def assert_english(text: str) -> None:
    """Loose but meaningful 'this is the English translation of the fixture' check."""
    assert text.strip(), "translated text is empty"
    letters = [c for c in text if c.isalpha()]
    assert letters, f"no letters in {text!r}"
    ascii_ratio = sum(c.isascii() for c in letters) / len(letters)
    assert ascii_ratio >= 0.97, f"text is not ASCII-dominant ({ascii_ratio:.2f}): {text!r}"
    dia = sorted(set(text) & ROMANIAN_DIACRITICS)
    assert not dia, f"Romanian diacritics {dia} in translated text: {text!r}"
    assert found_keywords(text, EN_KEYWORDS), f"none of {EN_KEYWORDS} in {text!r}"


def assert_romanian(text: str) -> None:
    assert text.strip(), "transcribed text is empty"
    assert found_keywords(text, RO_KEYWORDS), f"none of the Romanian keywords {RO_KEYWORDS} in {text!r}"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def model_downloaded(model: str, root: Path = REAL_MODELS_DIR) -> bool:
    d = root / model
    return all((d / f).is_file() for f in contracts.MODEL_REQUIRED_FILES) and (
        d / contracts.MODEL_COMPLETE_MARKER
    ).exists()


def require_model(model: str = contracts.DEFAULT_MODEL) -> None:
    if not model_downloaded(model):
        pytest.skip(
            f"model {model!r} is not fully downloaded in {REAL_MODELS_DIR} "
            f"(needs {contracts.MODEL_REQUIRED_FILES} + {contracts.MODEL_COMPLETE_MARKER}); "
            "tests never download models"
        )


# ---------------------------------------------------------------------------
# Subprocess env + CLI runner
# ---------------------------------------------------------------------------


def app_env(
    data_dir: Path,
    config_dir: Path,
    models_dir: Path = REAL_MODELS_DIR,
    *,
    utf8: bool = True,
    extra: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    env = dict(os.environ)
    env[contracts.ENV_DATA_DIR] = str(data_dir)
    env[contracts.ENV_CONFIG_DIR] = str(config_dir)
    env[contracts.ENV_MODELS_DIR] = str(models_dir)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if utf8:
        env["PYTHONIOENCODING"] = "utf-8"
    else:
        env.pop("PYTHONIOENCODING", None)
        env.pop("PYTHONUTF8", None)
    if extra:
        env.update(extra)
    return env


@dataclass
class CliRun:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str
    data_dir: Path

    @property
    def log_path(self) -> Path:
        return self.data_dir / "logs" / "roenspeak.log"

    def log_text(self) -> str:
        p = self.log_path
        return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

    def json(self) -> dict:
        try:
            return json.loads(self.stdout)
        except json.JSONDecodeError as e:  # pragma: no cover - diagnostic
            raise AssertionError(f"--json stdout is not exactly one JSON object ({e}):\n{self.describe()}")

    def describe(self) -> str:
        return (
            f"args={self.args}\nexit={self.returncode}\n--- stdout ---\n{self.stdout[-4000:]}"
            f"\n--- stderr ---\n{self.stderr[-4000:]}"
        )


def run_cli(
    args: list[str],
    base_dir: Path,
    *,
    models_dir: Path = REAL_MODELS_DIR,
    utf8: bool = True,
    extra_env: Optional[dict[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> CliRun:
    """Run ``python -m app.cli ARGS`` with isolated data/config dirs under base_dir."""
    data_dir = base_dir / "data"
    config_dir = base_dir / "config"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "app.cli", *[str(a) for a in args]]
    p = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=app_env(data_dir, config_dir, models_dir, utf8=utf8, extra=extra_env),
        capture_output=True,
        timeout=timeout,
    )
    return CliRun(
        [str(a) for a in args],
        p.returncode,
        p.stdout.decode("utf-8", errors="replace"),
        p.stderr.decode("utf-8", errors="replace"),
        data_dir,
    )


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------


def log_records(log_text: str, key: str) -> list[dict[str, str]]:
    """Every '<key> k=v k=v ...' line (after any logging prefix) parsed into a dict.
    Values run until the next ' k=' token, so values with spaces (gpu names) survive."""
    out = []
    for line in log_text.splitlines():
        m = re.search(r"(?:^|\s)" + re.escape(key) + r"\s+(\w+=.*)$", line)
        if not m:
            continue
        body = m.group(1)
        pairs = re.findall(r"(\w+)=(.*?)(?=\s+\w+=|$)", body)
        out.append({k: v.strip() for k, v in pairs})
    return out


# ---------------------------------------------------------------------------
# Generated audio
# ---------------------------------------------------------------------------


def write_wav(path: Path, samples: np.ndarray, rate: int = contracts.SAMPLE_RATE) -> Path:
    import wave

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def wav_duration_s(path: Path) -> float:
    import wave

    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()
