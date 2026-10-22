"""Spec/contract tests for app.engine.metal_probe.MetalProbe (contracts.CudaProbe).

Expectations come from contracts.CudaProbe / CudaReport / GpuInfo / LOG_CUDA_PROBE /
MIN_MACOS_VERSION and from ARCHITECTURE.md section 12.1: setup_dll_paths() is a no-op
returning [], probe() reports the Apple chip, the unified-memory budget and Metal
availability, never raises, and emits the same `cuda_probe` line as Windows with
`cublas=none cudnn=none` so the log greps are shared.

The engine dylib and the sysctl/version helpers are faked, so nothing here needs a Mac,
a GPU or a built dylib; the module-level IS_MACOS flag is patched instead of sys.platform
where the code reads the flag. Real-hardware checks belong to the `metal` integration
tests.
"""

from __future__ import annotations

import logging
import re
import sys

import pytest

from app.contracts import (
    LOG_CUDA_PROBE,
    MIN_MACOS_VERSION,
    CudaProbe,
    CudaReport,
    GpuInfo,
)
from app.engine import metal_probe as mp
from app.engine import whispercpp
from app.engine.metal_probe import MetalProbe

DEVICES_M1_MAX = "MTL0:Apple M1 Max(type=gpu,free=25558MiB,total=25559MiB)"
HEALTHY_MACOS = "15.1"

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeShim:
    """The two shim entry points the probe uses."""

    def __init__(self, metal: bool = True, devices: str = DEVICES_M1_MAX) -> None:
        self._metal = metal
        self._devices = devices
        self.metal_calls = 0
        self.devices_calls = 0

    def metal_available(self) -> bool:
        self.metal_calls += 1
        if isinstance(self._metal, BaseException):
            raise self._metal
        return self._metal

    def devices(self) -> str:
        self.devices_calls += 1
        if isinstance(self._devices, BaseException):
            raise self._devices
        return self._devices


_KV_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=("(?:[^"\\]|\\.)*"|\S+)')


def kv_pairs(line: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2).strip('"')) for m in _KV_RE.finditer(line)]


def kv(line: str) -> dict[str, str]:
    return dict(kv_pairs(line))


def probe_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(LOG_CUDA_PROBE + " ")]


@pytest.fixture
def mac(monkeypatch):
    """A healthy Apple-silicon Mac, fully faked. Returns the fake shim so tests can break it."""
    shim = FakeShim()
    monkeypatch.setattr(mp, "IS_MACOS", True)
    monkeypatch.setattr(mp, "macos_version", lambda: HEALTHY_MACOS)
    monkeypatch.setattr(mp, "chip_name", lambda: "Apple M1 Max")
    monkeypatch.setattr(mp, "unified_memory_mb", lambda: 32768)
    monkeypatch.setattr(whispercpp, "load_shim", lambda: shim)
    return shim


# ---------------------------------------------------------------------------
# Contract shape
# ---------------------------------------------------------------------------


def test_is_a_cuda_probe():
    assert isinstance(MetalProbe(), CudaProbe)


def test_module_imports_on_any_platform():
    assert mp.MetalProbe is MetalProbe


# ---------------------------------------------------------------------------
# setup_dll_paths
# ---------------------------------------------------------------------------


def test_setup_dll_paths_returns_empty_and_is_idempotent():
    """Nothing to register: the shim's @loader_path rpath resolves its siblings."""
    p = MetalProbe()
    assert p.setup_dll_paths() == []
    assert p.setup_dll_paths() == []
    assert p.setup_dll_paths() == []


def test_setup_dll_paths_does_not_touch_the_environment(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/nowhere")
    MetalProbe().setup_dll_paths()
    import os

    assert os.environ["PATH"] == "/usr/bin"
    assert os.environ["DYLD_LIBRARY_PATH"] == "/nowhere"


# ---------------------------------------------------------------------------
# probe(): healthy Mac
# ---------------------------------------------------------------------------


def test_healthy_mac_is_ok_with_a_populated_gpu_info(mac):
    report = MetalProbe().probe()
    assert isinstance(report, CudaReport)
    assert report.ok is True
    assert report.error is None
    assert isinstance(report.gpu, GpuInfo)
    assert report.gpu.name == "Apple M1 Max"
    assert report.gpu.total_vram_mb == 25559  # the Metal device's working-set budget
    assert report.gpu.free_vram_mb == 25558
    assert report.gpu.driver_version == HEALTHY_MACOS  # macOS version: what a user updates
    # CUDA-only concepts; always None on a Mac.
    assert report.cublas_version is None and report.cudnn_version is None


def test_unified_memory_is_the_fallback_when_the_device_list_does_not_parse(mac):
    mac._devices = "MTL0:Apple M1 Max"  # metal_available() still says yes
    report = MetalProbe().probe()
    assert report.ok is True
    assert report.gpu.name == "Apple M1 Max"  # from chip_name()
    assert report.gpu.total_vram_mb == 32768  # from hw.memsize
    assert report.gpu.free_vram_mb == 32768


def test_free_memory_never_exceeds_total(mac):
    mac._devices = "MTL0:Apple M9(type=gpu,free=99999MiB,total=8192MiB)"
    report = MetalProbe().probe()
    assert report.gpu.free_vram_mb <= report.gpu.total_vram_mb


def test_minimum_supported_macos_is_ok(mac, monkeypatch):
    monkeypatch.setattr(mp, "macos_version", lambda: MIN_MACOS_VERSION)
    assert MetalProbe().probe().ok is True


def test_an_unknown_macos_version_does_not_fail_the_probe(mac, monkeypatch):
    """"" means "could not be determined"; that is not evidence of an unsupported Mac."""
    monkeypatch.setattr(mp, "macos_version", lambda: "")
    report = MetalProbe().probe()
    assert report.ok is True
    assert report.gpu.driver_version == "unknown"


# ---------------------------------------------------------------------------
# probe(): not ok
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["13.7", "12.0", "10.15.7", "9.1"])
def test_too_old_macos_is_not_ok(mac, monkeypatch, version):
    monkeypatch.setattr(mp, "macos_version", lambda: version)
    report = MetalProbe().probe()
    assert report.ok is False
    assert report.error and version in report.error
    assert MIN_MACOS_VERSION in report.error


def test_metal_unavailable_is_not_ok(mac):
    mac._metal = False
    report = MetalProbe().probe()
    assert report.ok is False
    assert report.error and "Metal" in report.error
    assert mac.devices_calls == 0  # no point asking further


def test_a_missing_dylib_is_not_ok_and_says_so(mac, monkeypatch):
    def boom():
        raise FileNotFoundError("libroenspeak_whisper.dylib was not found (looked in: /nowhere)")

    monkeypatch.setattr(whispercpp, "load_shim", boom)
    report = MetalProbe().probe()
    assert report.ok is False
    assert report.error and "missing" in report.error
    # The probe still describes the machine, so the UI can still show the chip/VRAM.
    assert report.gpu is not None and report.gpu.name == "Apple M1 Max"


def test_an_unloadable_dylib_is_not_ok(mac, monkeypatch):
    def boom():
        raise OSError("has ABI version 1, but this build of RoEnSpeak needs 2")

    monkeypatch.setattr(whispercpp, "load_shim", boom)
    report = MetalProbe().probe()
    assert report.ok is False
    assert report.error and "could not be loaded" in report.error


def test_several_problems_are_all_reported(mac, monkeypatch):
    monkeypatch.setattr(mp, "macos_version", lambda: "12.0")
    mac._metal = False
    report = MetalProbe().probe()
    assert report.ok is False
    assert "12.0" in report.error and "Metal" in report.error


def test_off_macos_is_not_ok_with_an_explanatory_error(monkeypatch):
    monkeypatch.setattr(mp, "IS_MACOS", False)
    monkeypatch.setattr(sys, "platform", "win32")
    report = MetalProbe().probe()
    assert report == CudaReport(
        False, None, None, None,
        "The Metal speech engine is only supported on macOS (this is win32)",
    )


def test_off_macos_does_not_load_the_dylib(monkeypatch):
    monkeypatch.setattr(mp, "IS_MACOS", False)

    def boom():  # pragma: no cover - must not be called
        raise AssertionError("the dylib must not be touched off macOS")

    monkeypatch.setattr(whispercpp, "load_shim", boom)
    assert MetalProbe().probe().ok is False


# ---------------------------------------------------------------------------
# probe(): never raises
# ---------------------------------------------------------------------------


def _break(monkeypatch, what: str, shim: FakeShim) -> None:
    if what == "macos_version_raises":
        monkeypatch.setattr(mp, "macos_version", lambda: (_ for _ in ()).throw(OSError("sysctl gone")))
    elif what == "macos_version_garbage":
        monkeypatch.setattr(mp, "macos_version", lambda: "banana")
    elif what == "chip_name_raises":
        monkeypatch.setattr(mp, "chip_name", lambda: (_ for _ in ()).throw(RuntimeError("no libc")))
    elif what == "memory_raises":
        monkeypatch.setattr(mp, "unified_memory_mb", lambda: (_ for _ in ()).throw(ValueError("nope")))
    elif what == "memory_negative":
        monkeypatch.setattr(mp, "unified_memory_mb", lambda: -1)
    elif what == "metal_available_raises":
        shim._metal = RuntimeError("the dylib crashed")
    elif what == "devices_raises":
        shim._devices = RuntimeError("the dylib crashed")
    elif what == "devices_garbage":
        shim._devices = "\x00\x01 not a device list at all (((("
    elif what == "devices_empty":
        shim._devices = ""
    elif what == "load_shim_raises":
        monkeypatch.setattr(
            whispercpp, "load_shim", lambda: (_ for _ in ()).throw(MemoryError("out of memory"))
        )
    elif what == "load_shim_returns_none":
        monkeypatch.setattr(whispercpp, "load_shim", lambda: None)
    else:  # pragma: no cover - guards the parametrize list
        raise AssertionError(what)


@pytest.mark.parametrize(
    "what",
    [
        "macos_version_raises",
        "macos_version_garbage",
        "chip_name_raises",
        "memory_raises",
        "memory_negative",
        "metal_available_raises",
        "devices_raises",
        "devices_garbage",
        "devices_empty",
        "load_shim_raises",
        "load_shim_returns_none",
    ],
)
def test_probe_never_raises(mac, monkeypatch, what, caplog):
    """contracts.CudaProbe: "probe(): ... Never raises; failures go in the report"."""
    caplog.set_level(logging.INFO)
    _break(monkeypatch, what, mac)
    report = MetalProbe().probe()
    assert isinstance(report, CudaReport)
    assert isinstance(report.ok, bool)
    if not report.ok:
        assert report.error  # a non-ok report always explains itself
    assert report.cublas_version is None and report.cudnn_version is None
    assert len(probe_lines(caplog)) == 1  # and it still logs its one line


def test_a_crashing_probe_is_reported_as_not_ok(mac, monkeypatch):
    monkeypatch.setattr(mp, "chip_name", lambda: (_ for _ in ()).throw(RuntimeError("no libc")))
    report = MetalProbe().probe()
    assert report.ok is False
    assert report.gpu is None
    assert "no libc" in report.error


# ---------------------------------------------------------------------------
# probe(): the cuda_probe log line
# ---------------------------------------------------------------------------


def test_exactly_one_cuda_probe_line_with_the_documented_keys(mac, caplog):
    caplog.set_level(logging.INFO)
    MetalProbe().probe()
    lines = probe_lines(caplog)
    assert len(lines) == 1
    assert [k for k, _ in kv_pairs(lines[0])] == ["ok", "gpu", "driver", "cublas", "cudnn", "error"]
    fields = kv(lines[0])
    assert fields["ok"] == "1"
    assert fields["gpu"] == "Apple M1 Max"
    assert fields["driver"] == HEALTHY_MACOS
    # This is what keeps the log greps shared with Windows.
    assert fields["cublas"] == "none" and fields["cudnn"] == "none"
    assert fields["error"] == "none"


def test_the_cuda_probe_line_reports_failures_too(mac, caplog):
    caplog.set_level(logging.INFO)
    mac._metal = False
    MetalProbe().probe()
    fields = kv(probe_lines(caplog)[0])
    assert fields["ok"] == "0"
    assert fields["error"] != "none"
    assert fields["cublas"] == "none" and fields["cudnn"] == "none"


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------


def test_the_result_is_cached(mac):
    p = MetalProbe()
    first = p.probe()
    second = p.probe()
    assert second is first
    assert mac.metal_calls == 1  # the dylib was asked exactly once


def test_refresh_re_probes(mac, monkeypatch):
    p = MetalProbe()
    assert p.probe().ok is True
    mac._metal = False
    assert p.probe().ok is True  # still the cached answer
    refreshed = p.probe(refresh=True)
    assert refreshed.ok is False
    assert mac.metal_calls == 2
    assert p.probe() is refreshed  # the new result is now the cached one


def test_a_cached_probe_does_not_log_again(mac, caplog):
    """One cuda_probe line per real probe, so the line count tracks actual probes."""
    caplog.set_level(logging.INFO)
    p = MetalProbe()
    p.probe()
    p.probe()
    assert len(probe_lines(caplog)) == 1
    p.probe(refresh=True)
    assert len(probe_lines(caplog)) == 2


def test_two_probes_do_not_share_a_cache(mac):
    a, b = MetalProbe(), MetalProbe()
    a.probe()
    b.probe()
    assert mac.metal_calls == 2


# ---------------------------------------------------------------------------
# parse_devices (pure)
# ---------------------------------------------------------------------------


def test_parse_devices_parses_the_documented_shape():
    [dev] = mp.parse_devices(DEVICES_M1_MAX)
    assert dev == {
        "id": "MTL0",
        "desc": "Apple M1 Max",
        "type": "gpu",
        "free_mb": 25558,
        "total_mb": 25559,
    }


def test_parse_devices_parses_several_devices():
    text = (
        "MTL0:Apple M1 Max(type=gpu,free=25558MiB,total=25559MiB),"
        "CPU0:Apple M1 Max(type=cpu,free=0MiB,total=32768MiB)"
    )
    devs = mp.parse_devices(text)
    assert [d["type"] for d in devs] == ["gpu", "cpu"]


def test_the_probe_picks_the_gpu_device(mac):
    mac._devices = (
        "CPU0:Apple M1 Max(type=cpu,free=0MiB,total=32768MiB),"
        "MTL0:Apple M1 Max GPU(type=gpu,free=100MiB,total=200MiB)"
    )
    report = MetalProbe().probe()
    assert report.gpu.name == "Apple M1 Max GPU"
    assert (report.gpu.total_vram_mb, report.gpu.free_vram_mb) == (200, 100)


@pytest.mark.parametrize("text", ["", "   ", "garbage", "MTL0:Apple(type=gpu)", None])
def test_parse_devices_tolerates_anything(text):
    assert mp.parse_devices(text) == []


# ---------------------------------------------------------------------------
# sysctl / version helpers
# ---------------------------------------------------------------------------


def test_sysctl_for_an_unknown_key_is_none():
    assert mp.sysctl_str("roenspeak.no.such.key") is None
    assert mp.sysctl_u64("roenspeak.no.such.key") is None


def test_macos_version_never_raises():
    assert isinstance(mp.macos_version(), str)


def test_chip_name_always_returns_something():
    assert mp.chip_name()


def test_unified_memory_is_never_negative():
    assert mp.unified_memory_mb() >= 0


@pytest.mark.skipif(sys.platform != "darwin", reason="reads real sysctl keys")
def test_real_sysctl_describes_this_mac():
    assert mp.chip_name() != "Apple Silicon" or mp.sysctl_str("hw.model")
    assert mp.unified_memory_mb() > 0
    assert mp.macos_version()
