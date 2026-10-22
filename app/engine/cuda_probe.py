"""CUDA / cuBLAS / cuDNN availability probe and DLL search-path setup (Windows).

Verified facts (ARCHITECTURE.md section 5): ctranslate2 4.8.3 links cudart statically, loads cublas64_12.dll
at runtime and ships a cuDNN 9 dispatcher (cudnn64_9.dll) that loads cudnn_*64_9.dll by bare
name through the standard search order. os.add_dll_directory alone is therefore not enough:
the nvidia/*/bin directories must also be prepended to PATH.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional

from app.contracts import LOG_CUDA_PROBE, MIN_DRIVER_VERSION, CudaReport, GpuInfo

logger = logging.getLogger("roenspeak.engine.cuda_probe")

#: Sub-packages of the nvidia-*-cu12 wheels whose bin/ dirs hold the DLLs we need.
NVIDIA_PACKAGES = ("cublas", "cudnn", "cuda_nvrtc")
CUBLAS_DLL = "cublas64_12.dll"
CUBLASLT_DLL = "cublasLt64_12.dll"
CUDNN_DLL = "cudnn64_9.dll"
#: cuDNN sub-libraries the dispatcher loads by name (via PATH); checked with the same search order.
CUDNN_SUBLIBS = ("cudnn_graph64_9.dll", "cudnn_ops64_9.dll")

_setup_lock = threading.Lock()
_added_dirs: list[str] = []
_dll_dir_handles: list[object] = []  # keep os.add_dll_directory handles alive


def format_kv(value: object) -> str:
    """Render a value for a k=v log line; values containing whitespace are double-quoted."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "1" if value else "0"
    s = str(value)
    if not s or any(ch.isspace() for ch in s) or '"' in s:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


def _version_tuple(v: str) -> tuple[int, ...]:
    parts = []
    for p in v.strip().split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def driver_at_least(driver: str, minimum: str = MIN_DRIVER_VERSION) -> bool:
    a, b = _version_tuple(driver), _version_tuple(minimum)
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)) >= b + (0,) * (n - len(b))


def cudnn_version_string(v: int) -> str:
    """cudnnGetVersion() -> "major.minor.patch" (cuDNN >= 9 encodes major*10000 + minor*100 + patch)."""
    if v >= 90000:
        return f"{v // 10000}.{(v % 10000) // 100}.{v % 100}"
    return f"{v // 1000}.{(v % 1000) // 100}.{v % 100}"


def candidate_nvidia_roots() -> list[Path]:
    """Directories that may contain an nvidia/ package tree with */bin DLL folders."""
    roots: list[Path] = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            roots.append(Path(meipass) / "nvidia")
        roots.append(Path(sys.executable).resolve().parent / "_internal" / "nvidia")
    for entry in sys.path:
        if not entry:
            continue
        p = Path(entry)
        if p.name.lower() == "site-packages":
            roots.append(p / "nvidia")
    try:
        import nvidia  # namespace package from the nvidia-*-cu12 wheels

        roots.extend(Path(p) for p in getattr(nvidia, "__path__", []))
    except ImportError:
        pass
    seen: set[str] = set()
    out: list[Path] = []
    for r in roots:
        key = os.path.normcase(str(r))
        if key not in seen and r.is_dir():
            seen.add(key)
            out.append(r)
    return out


def find_nvidia_bin_dirs() -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()
    for root in candidate_nvidia_roots():
        for pkg in NVIDIA_PACKAGES:
            d = root / pkg / "bin"
            key = os.path.normcase(str(d))
            if d.is_dir() and key not in seen:
                seen.add(key)
                dirs.append(d)
    return dirs


def _find_dll(name: str) -> Optional[Path]:
    for d in find_nvidia_bin_dirs():
        p = d / name
        if p.is_file():
            return p
    return None


class WindowsCudaProbe:
    """CudaProbe implementation (see contracts.CudaProbe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cached: Optional[CudaReport] = None
        self._dlls: list[object] = []  # loaded DLL handles, kept for the process lifetime

    # -- DLL paths ----------------------------------------------------------------

    def setup_dll_paths(self) -> list[str]:
        """Add nvidia/*/bin dirs via os.add_dll_directory and prepend them to PATH.
        Idempotent; returns every directory set up so far. No-op off Windows."""
        if sys.platform != "win32":
            return []
        with _setup_lock:
            dirs = [str(d) for d in find_nvidia_bin_dirs()]
            path_parts = os.environ.get("PATH", "").split(os.pathsep)
            norm_path = {os.path.normcase(os.path.normpath(p)) for p in path_parts if p}
            to_prepend: list[str] = []
            for d in dirs:
                if d not in _added_dirs:
                    try:
                        _dll_dir_handles.append(os.add_dll_directory(d))
                    except OSError as e:
                        logger.warning("os.add_dll_directory(%s) failed: %s", d, e)
                        continue
                    _added_dirs.append(d)
                if os.path.normcase(os.path.normpath(d)) not in norm_path:
                    to_prepend.append(d)
            if to_prepend:
                os.environ["PATH"] = os.pathsep.join(to_prepend + path_parts)
            if dirs:
                logger.debug("CUDA DLL dirs: %s", dirs)
            else:
                logger.warning("No nvidia/*/bin DLL directories found (cuBLAS/cuDNN wheels missing?)")
            return list(_added_dirs)

    # -- probe --------------------------------------------------------------------

    def probe(self, refresh: bool = False) -> CudaReport:
        """Never raises. Cached after the first call unless refresh=True."""
        with self._lock:
            if self._cached is not None and not refresh:
                return self._cached
            try:
                report = self._probe_uncached()
            except Exception as e:  # defensive: probe() must never raise
                logger.exception("CUDA probe crashed")
                report = CudaReport(False, None, None, None, f"CUDA probe failed: {e}")
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
        if sys.platform != "win32":
            return CudaReport(False, None, None, None, "CUDA probe is only supported on Windows")

        self.setup_dll_paths()
        errors: list[str] = []

        gpu, nvml_err = self._query_nvml()
        if nvml_err:
            errors.append(nvml_err)
        elif gpu is not None and not driver_at_least(gpu.driver_version):
            errors.append(
                f"NVIDIA driver {gpu.driver_version} is too old; version {MIN_DRIVER_VERSION} or newer is required"
            )

        cublas_version, cublas_err = self._load_cublas()
        if cublas_err:
            errors.append(cublas_err)
        cudnn_version, cudnn_err = self._load_cudnn()
        if cudnn_err:
            errors.append(cudnn_err)

        if not errors:
            dev_err = self._check_ctranslate2_devices()
            if dev_err:
                errors.append(dev_err)

        ok = not errors
        return CudaReport(ok, gpu, cublas_version, cudnn_version, None if ok else "; ".join(errors))

    def _query_nvml(self) -> tuple[Optional[GpuInfo], Optional[str]]:
        try:
            import pynvml
        except ImportError as e:
            return None, f"nvidia-ml-py is not installed ({e})"
        try:
            pynvml.nvmlInit()
        except Exception as e:
            return None, f"NVIDIA driver not found or not working (NVML init failed: {e})"
        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode("ascii", "replace")
            count = pynvml.nvmlDeviceGetCount()
            if count < 1:
                return None, "No NVIDIA GPU found"
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu = GpuInfo(
                name=str(name),
                total_vram_mb=int(mem.total // (1024 * 1024)),
                free_vram_mb=int(mem.free // (1024 * 1024)),
                driver_version=str(driver),
            )
            return gpu, None
        except Exception as e:
            return None, f"Could not query the NVIDIA GPU via NVML: {e}"
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _load(self, name: str) -> ctypes.CDLL:
        path = _find_dll(name)
        dll = ctypes.WinDLL(str(path) if path else name)  # type: ignore[attr-defined]
        self._dlls.append(dll)
        return dll

    def _load_cublas(self) -> tuple[Optional[str], Optional[str]]:
        try:
            self._load(CUBLASLT_DLL)
            cublas = self._load(CUBLAS_DLL)
        except OSError as e:
            return None, f"cuBLAS could not be loaded ({CUBLAS_DLL}: {e})"
        try:
            fn = cublas.cublasGetProperty
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
            parts = []
            for prop in (0, 1, 2):  # MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL
                val = ctypes.c_int(0)
                status = fn(prop, ctypes.byref(val))
                if status != 0:
                    return None, f"cublasGetProperty failed with status {status}"
                parts.append(str(val.value))
            return ".".join(parts), None
        except (AttributeError, OSError) as e:
            return None, f"cuBLAS version query failed: {e}"

    def _load_cudnn(self) -> tuple[Optional[str], Optional[str]]:
        try:
            cudnn = self._load(CUDNN_DLL)
        except OSError as e:
            return None, f"cuDNN could not be loaded ({CUDNN_DLL}: {e})"
        try:
            fn = cudnn.cudnnGetVersion
            fn.restype = ctypes.c_size_t
            fn.argtypes = []
            version = cudnn_version_string(int(fn()))
        except (AttributeError, OSError) as e:
            return None, f"cuDNN version query failed: {e}"
        # The cuDNN dispatcher loads its sub-libraries by bare name with the standard search
        # order (which includes PATH, unlike ctypes' default). Verify the same way.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        load_library = kernel32.LoadLibraryW
        load_library.restype = ctypes.c_void_p
        load_library.argtypes = [ctypes.c_wchar_p]
        for sub in CUDNN_SUBLIBS:
            if not load_library(sub):
                err = ctypes.get_last_error()  # type: ignore[attr-defined]
                return version, f"cuDNN sub-library {sub} could not be loaded (Win32 error {err})"
        return version, None

    @staticmethod
    def _check_ctranslate2_devices() -> Optional[str]:
        try:
            import ctranslate2
        except Exception as e:
            return f"ctranslate2 could not be imported: {e}"
        try:
            n = ctranslate2.get_cuda_device_count()
        except Exception as e:
            return f"ctranslate2 could not query CUDA devices: {e}"
        if n < 1:
            return "ctranslate2 found no usable CUDA device (driver too old or GPU not supported)"
        return None
