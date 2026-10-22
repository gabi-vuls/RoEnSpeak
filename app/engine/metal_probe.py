"""Metal / whisper.cpp availability probe (macOS). The macOS CudaProbe (see contracts).

There is no CUDA, no cuBLAS and no cuDNN on Apple Silicon: the GPU work runs through ggml's
Metal backend inside libroenspeak_whisper.dylib. The CudaProbe contract is kept verbatim
anyway, because app/core and app/ui are written against it and every log key and UI string
must stay identical on both platforms. So:

* ``setup_dll_paths()`` is a documented no-op returning ``[]``. There is nothing to
  register: each dylib's install name is ``@rpath/<basename>`` and the shim is linked with
  ``-rpath @loader_path``, so loading the shim resolves its six siblings from its own
  directory. ``DYLD_LIBRARY_PATH`` is also ignored by hardened/signed binaries on modern
  macOS, which is exactly why the rpath design is the one that survives a signed .app.
* ``probe()`` reports the Apple chip, the Metal device's memory budget and the macOS
  version in the GpuInfo fields (``driver_version`` carries the macOS version -- it is the
  thing a user can update to fix a "too old" report), and leaves
  ``cublas_version``/``cudnn_version`` None. It never raises.
* ``ok`` is False only when Metal genuinely cannot be used: not macOS, macOS older than
  MIN_MACOS_VERSION, the engine dylib missing/unloadable, or ggml registering no GPU
  device.
"""

from __future__ import annotations

import ctypes
import logging
import platform
import re
import sys
import threading
from typing import Optional

from app.contracts import (
    IS_MACOS,
    LOG_CUDA_PROBE,
    MIN_MACOS_VERSION,
    CudaReport,
    GpuInfo,
)
from app.engine.cuda_probe import driver_at_least, format_kv

logger = logging.getLogger("roenspeak.engine.metal_probe")

#: rs_whisper_devices() entries: "MTL0:Apple M1 Max(type=gpu,free=25558MiB,total=25559MiB)"
_DEVICE_RE = re.compile(
    r"(?P<id>[^:,]+):(?P<desc>[^(]*)\(type=(?P<type>[a-z]+),free=(?P<free>\d+)MiB,total=(?P<total>\d+)MiB\)"
)
_GPU_TYPES = ("gpu", "igpu")


def macos_version() -> str:
    """The macOS product version, e.g. "26.6.2". "" when it cannot be determined."""
    try:
        v = platform.mac_ver()[0]
    except Exception:  # pragma: no cover - platform never raises in practice
        return ""
    return v or ""


def _libc() -> Optional[ctypes.CDLL]:
    try:
        return ctypes.CDLL(None, use_errno=True)
    except OSError:  # pragma: no cover - libSystem is always there on macOS
        return None


def sysctl_str(name: str) -> Optional[str]:
    """sysctlbyname(name) as a string, or None. Used instead of running /usr/sbin/sysctl so
    the probe does not fork a process on the UI thread (and works in a sandboxed .app)."""
    lib = _libc()
    if lib is None:
        return None
    try:
        fn = lib.sysctlbyname
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p, ctypes.c_size_t,
        ]
        size = ctypes.c_size_t(0)
        key = name.encode("ascii")
        if fn(key, None, ctypes.byref(size), None, 0) != 0 or size.value == 0:
            return None
        buf = ctypes.create_string_buffer(size.value)
        if fn(key, buf, ctypes.byref(size), None, 0) != 0:
            return None
        return buf.value.decode("utf-8", "replace").strip() or None
    except Exception:
        return None


def sysctl_u64(name: str) -> Optional[int]:
    """sysctlbyname(name) as an unsigned 64-bit integer, or None."""
    lib = _libc()
    if lib is None:
        return None
    try:
        fn = lib.sysctlbyname
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p, ctypes.c_size_t,
        ]
        val = ctypes.c_uint64(0)
        size = ctypes.c_size_t(ctypes.sizeof(val))
        if fn(name.encode("ascii"), ctypes.byref(val), ctypes.byref(size), None, 0) != 0:
            return None
        return int(val.value)
    except Exception:
        return None


def chip_name() -> str:
    """The Apple chip, e.g. "Apple M1 Max". machdep.cpu.brand_string is what Apple's own
    tools report; hw.model ("Mac13,2") is the fallback when it is unavailable."""
    return sysctl_str("machdep.cpu.brand_string") or sysctl_str("hw.model") or "Apple Silicon"


def unified_memory_mb() -> int:
    """Installed (unified) memory in MB; 0 when unknown."""
    total = sysctl_u64("hw.memsize")
    return int(total // (1024 * 1024)) if total else 0


def parse_devices(text: str) -> list[dict[str, object]]:
    """Parse rs_whisper_devices() into dicts with id/desc/type/free_mb/total_mb."""
    out: list[dict[str, object]] = []
    for m in _DEVICE_RE.finditer(text or ""):
        out.append(
            {
                "id": m.group("id").strip(),
                "desc": m.group("desc").strip(),
                "type": m.group("type"),
                "free_mb": int(m.group("free")),
                "total_mb": int(m.group("total")),
            }
        )
    return out


class MetalProbe:
    """CudaProbe implementation for macOS (see contracts.CudaProbe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cached: Optional[CudaReport] = None

    # -- DLL paths ----------------------------------------------------------------

    def setup_dll_paths(self) -> list[str]:
        """No-op by design: see the module docstring. The engine dylib's @loader_path rpath
        resolves its siblings, so there is no directory to register anywhere."""
        return []

    # -- probe --------------------------------------------------------------------

    def probe(self, refresh: bool = False) -> CudaReport:
        """Never raises. Cached after the first call unless refresh=True."""
        with self._lock:
            if self._cached is not None and not refresh:
                return self._cached
            try:
                report = self._probe_uncached()
            except Exception as e:  # defensive: probe() must never raise
                logger.exception("Metal probe crashed")
                report = CudaReport(False, None, None, None, f"Metal probe failed: {e}")
            self._cached = report
        logger.info(
            "%s ok=%s gpu=%s driver=%s cublas=%s cudnn=%s error=%s",
            LOG_CUDA_PROBE,
            format_kv(report.ok),
            format_kv(report.gpu.name if report.gpu else None),
            format_kv(report.gpu.driver_version if report.gpu else None),
            format_kv(report.cublas_version),
            format_kv(report.cudnn_version),
            format_kv(report.error),
        )
        return report

    def _probe_uncached(self) -> CudaReport:
        if not IS_MACOS:
            # Mirrors WindowsCudaProbe off Windows: no GpuInfo, one clear reason.
            return CudaReport(
                False, None, None, None,
                f"The Metal speech engine is only supported on macOS (this is {sys.platform})",
            )

        errors: list[str] = []
        mac_ver = macos_version()
        if mac_ver and not driver_at_least(mac_ver, MIN_MACOS_VERSION):
            errors.append(
                f"macOS {mac_ver} is too old; version {MIN_MACOS_VERSION} or newer is required"
            )

        name = chip_name()
        total_mb = unified_memory_mb()
        free_mb = total_mb

        gpu_desc, gpu_total, gpu_free, lib_err = self._query_metal_device()
        if lib_err:
            errors.append(lib_err)
        else:
            if gpu_desc:
                name = gpu_desc
            if gpu_total:
                # The Metal device's recommendedMaxWorkingSetSize is the budget that
                # actually limits how big a model can be loaded, so it is the honest
                # "total VRAM" for app.engine.gpu.vram_warning. hw.memsize is the fallback.
                total_mb = gpu_total
            free_mb = gpu_free if gpu_free else total_mb

        gpu = GpuInfo(
            name=name,
            total_vram_mb=int(total_mb),
            free_vram_mb=int(min(free_mb, total_mb) if total_mb else free_mb),
            driver_version=mac_ver or "unknown",
        )
        ok = not errors
        return CudaReport(ok, gpu, None, None, None if ok else "; ".join(errors))

    @staticmethod
    def _query_metal_device() -> tuple[Optional[str], int, int, Optional[str]]:
        """(description, total_mb, free_mb, error) from the engine dylib's ggml device list.

        The dylib is the only thing that can answer "will Metal actually work here", so a
        missing or broken dylib is reported as the (actionable) error it is.
        """
        try:
            from app.engine.whispercpp import load_shim  # lazy: keeps the import cycle open

            shim = load_shim()
        except FileNotFoundError as e:
            # Kept short and factual: whispercpp.library_missing_message() turns this into
            # the user-facing "reinstall / run the build script" wording.
            return None, 0, 0, f"the speech engine library is missing ({e})"
        except Exception as e:
            return None, 0, 0, f"the speech engine library could not be loaded: {e}"

        try:
            if not shim.metal_available():
                return None, 0, 0, "no Metal GPU device is available to the speech engine"
            devices = parse_devices(shim.devices())
        except Exception as e:  # pragma: no cover - the shim's queries cannot throw
            return None, 0, 0, f"the Metal device could not be queried: {e}"

        for dev in devices:
            if dev["type"] in _GPU_TYPES:
                return str(dev["desc"]), int(dev["total_mb"]), int(dev["free_mb"]), None
        # metal_available() said yes but the list did not parse: usable, just undescribed.
        return None, 0, 0, None
