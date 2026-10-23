"""Helpers shared by the integration tests (TEST-INTEGRATION-owned; plain module, not a conftest).

These tests drive the real app as a subprocess (``python -m app.cli`` / ``python -m app.main``)
and check only the observable contract in app/contracts.py: exit codes, stdout format,
the JSON shape, and LOG_* lines in <DATA_DIR>/logs/roenspeak.log.

One toolbox, two platforms. Everything here is platform-neutral except where the *contract*
differs, and those differences are named rather than hidden:

* ``EXPECTED_DEVICE`` / ``expected_device()`` -- EngineInfo.device is "cuda" on Windows and
  "metal" on macOS (and "cpu" in --cpu mode on both).
* ``expected_compute_type()`` -- EngineInfo.compute_type is the ctranslate2 compute type
  verbatim on Windows ("float16"), but the *ggml* type baked into the model file on macOS
  ("f16"/"q5_0"): whisper.cpp has no runtime compute type. See the ComputeType docstring in
  app/contracts.py.
* ``require_model()`` checks a CTranslate2 model *directory* (Windows); ``require_ggml_model()``
  checks the single ggml file whisper.cpp loads (macOS). ``require_platform_model()`` picks the
  right one. None of them ever downloads anything.
* ``broken_gpu_env()`` -- the "make the GPU backend unavailable" lever for the exit-2 test,
  which is a different mechanism on each platform.
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
# Platform
# ---------------------------------------------------------------------------

IS_WINDOWS = contracts.IS_WINDOWS
IS_MACOS = contracts.IS_MACOS

#: The env var that overrides where app/engine/whispercpp.py looks for libroenspeak_whisper.dylib.
#: Kept as a literal so importing this module never pulls the macOS engine in on Windows;
#: broken_gpu_env() cross-checks it against the real constant when it is actually used.
ENV_WHISPER_LIB = "ROENSPEAK_WHISPER_LIB"

#: EngineInfo.device for a healthy GPU run on this platform.
EXPECTED_DEVICE = "metal" if IS_MACOS else "cuda"

#: ComputeType -> the EngineInfo.compute_type macOS reports for it (the ggml file's type).
_MAC_GGML_COMPUTE_TYPE = {
    contracts.ComputeType.FLOAT16: "f16",
    contracts.ComputeType.INT8_FLOAT16: "q5_0",
}


def expected_device(cpu_mode: bool = False) -> str:
    """EngineInfo.device / the ENGINE line's device= for this platform."""
    return "cpu" if cpu_mode else EXPECTED_DEVICE


def expected_compute_type(compute_type=contracts.ComputeType.FLOAT16) -> str:
    """EngineInfo.compute_type for this platform and the requested ComputeType."""
    ct = contracts.ComputeType(compute_type)
    return _MAC_GGML_COMPUTE_TYPE[ct] if IS_MACOS else ct.value


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


def ggml_model_file_path(
    model: str = contracts.DEFAULT_MODEL,
    compute_type=contracts.ComputeType.FLOAT16,
    root: Path = REAL_MODELS_DIR,
) -> Path:
    """The single ggml file the macOS whisper.cpp engine loads for this model/compute type."""
    name = contracts.ggml_model_file(model, contracts.ComputeType(compute_type))
    return root / model / name


def ggml_model_downloaded(
    model: str = contracts.DEFAULT_MODEL,
    compute_type=contracts.ComputeType.FLOAT16,
    root: Path = REAL_MODELS_DIR,
) -> bool:
    """macOS counterpart of model_downloaded(): one ggml file + MODEL_COMPLETE_MARKER."""
    try:
        path = ggml_model_file_path(model, compute_type, root)
    except ValueError:
        return False
    return path.is_file() and (path.parent / contracts.MODEL_COMPLETE_MARKER).is_file()


def require_ggml_model(
    model: str = contracts.DEFAULT_MODEL,
    compute_type=contracts.ComputeType.FLOAT16,
) -> None:
    """Skip (never download) unless the ggml model file for this quantization is on disk."""
    ct = contracts.ComputeType(compute_type)
    try:
        path = ggml_model_file_path(model, ct)
    except ValueError as e:
        pytest.skip(f"no ggml model file for {model!r}/{ct.value}: {e}")
        return
    if not ggml_model_downloaded(model, ct):
        pytest.skip(
            f"ggml model {model!r} ({ct.value} -> {path.name}) is not downloaded: expected "
            f"{path} + {contracts.MODEL_COMPLETE_MARKER} in {path.parent}. Download it by hand "
            f"(the app's Settings, or `python -m app.cli <wav> --compute-type {ct.value} "
            "--download`); tests never download models."
        )


def require_platform_model(
    model: str = contracts.DEFAULT_MODEL,
    compute_type=contracts.ComputeType.FLOAT16,
) -> None:
    """require_ggml_model() on macOS, require_model() on Windows."""
    if IS_MACOS:
        require_ggml_model(model, compute_type)
    else:
        require_model(model)


# ---------------------------------------------------------------------------
# "No usable GPU backend" lever (exit code 2)
# ---------------------------------------------------------------------------


def broken_gpu_env(tmp_dir: Path) -> dict[str, str]:
    """Env vars that make this platform's GPU backend unavailable, for the exit-2 test.

    **Windows**: ``CUDA_VISIBLE_DEVICES=-1`` makes the CUDA runtime ctranslate2 uses see zero
    devices while the driver and NVML still exist -- the closest reproducible stand-in for
    "no usable CUDA device".

    **macOS**: there is no switch that hides the GPU; Metal is part of the OS and every Apple
    Silicon Mac has it. So the honest lever is the thing that actually decides the answer:
    ``MetalProbe._query_metal_device()`` loads ``libroenspeak_whisper.dylib`` and asks *it*
    whether Metal is usable, and ``whispercpp.library_candidates()`` puts
    ``$ROENSPEAK_WHISPER_LIB`` first in the search order. Pointing that at a file which is not
    a Mach-O object makes ``dlopen`` fail, so ``probe().ok`` is False with a real dlopen error.

    Note the var must name an **existing file**: ``library_path()`` returns the first candidate
    that exists, so a missing path simply falls through to ``build/native`` and the probe
    succeeds. That is why this helper writes the broken dylib instead of inventing a path.
    """
    if not IS_MACOS:
        return {"CUDA_VISIBLE_DEVICES": "-1"}
    from app.engine.whispercpp import ENV_WHISPER_LIB as REAL_ENV, WHISPER_LIB_NAME

    assert REAL_ENV == ENV_WHISPER_LIB, (
        f"whispercpp.ENV_WHISPER_LIB is {REAL_ENV!r} but helpers.app_env() pops {ENV_WHISPER_LIB!r}"
    )
    lib_dir = tmp_dir / "broken_engine_lib"
    lib_dir.mkdir(parents=True, exist_ok=True)
    fake = lib_dir / WHISPER_LIB_NAME
    # Deliberately not a Mach-O file: dlopen reports "slice is not valid mach-o file".
    fake.write_bytes(b"this is not a Mach-O dynamic library\n")
    return {REAL_ENV: str(fake)}


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
    # Both platforms' "GPU backend" levers are removed so an inherited value from the
    # developer's shell (or from native/whispercpp/build.sh's smoke test, which exports
    # ROENSPEAK_WHISPER_LIB) can never change what a test measures. broken_gpu_env() puts
    # the right one back on purpose.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop(ENV_WHISPER_LIB, None)
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
