# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for RoEnSpeak: ONEDIR bundle with two executables sharing one _internal folder.

    RoEnSpeak.exe       windowed tray app   (build/launch_app.py -> app.main.main)
    RoEnSpeak-cli.exe   console .wav tester (build/launch_cli.py -> app.cli.main)

Build (from the repo root, Windows venv):
    python -m PyInstaller build/roenspeak.spec --noconfirm --workpath build/work --distpath build/out

Output: build/out/RoEnSpeak/{RoEnSpeak.exe, RoEnSpeak-cli.exe, _internal/...}

CUDA DLL layout (must match app/engine/cuda_probe.py candidate_nvidia_roots()/find_nvidia_bin_dirs()):
    _internal/nvidia/cublas/bin/*.dll, _internal/nvidia/cudnn/bin/*.dll, _internal/nvidia/cuda_nvrtc/bin/*.dll
They are appended to a.datas AFTER Analysis, so PyInstaller copies them verbatim (no binary
dependency analysis, no duplicate copies in the top-level folder). See BUILD.md "DLL bundling".

Environment knobs:
    ROENSPEAK_NO_TRIM=1   bundle every nvidia DLL (ignore NVIDIA_TRIM below) - for re-validating trimming.
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

SPEC_DIR = Path(SPECPATH).resolve()
ROOT = SPEC_DIR.parent
sys.path.insert(0, str(SPEC_DIR))
import gen_version_info  # noqa: E402  (build/gen_version_info.py)

APP_NAME = gen_version_info.app_name()
ICON = str(ROOT / "assets" / "icons" / "app.ico")

# ---------------------------------------------------------------------------------------------
# nvidia-*-cu12 wheels: bin/ directories copied verbatim to _internal/nvidia/<pkg>/bin.
# ---------------------------------------------------------------------------------------------
NVIDIA_PACKAGES = ("cublas", "cudnn", "cuda_nvrtc")  # == cuda_probe.NVIDIA_PACKAGES

#: DLLs left out of the bundle (~985 MB). Evidence and re-validation procedure: BUILD.md "Trimming".
#:  * nvblas64_12.dll: drop-in BLAS interposer for CPU BLAS callers; nothing in the bundle uses it.
#:  * cuDNN engine/sub-libraries: ctranslate2 4.8.3's Windows ctranslate2.dll neither imports nor
#:    references cuDNN (it only loads cublas64_12.dll at runtime), so inference never calls cuDNN.
#:    cudnn64_9.dll + cudnn_graph64_9.dll + cudnn_ops64_9.dll stay because cuda_probe loads them
#:    (CUDNN_DLL / CUDNN_SUBLIBS) to verify and report the cuDNN version at startup.
#:  * nvrtc64_120_0.alt.dll: alternate NVRTC build that no bundled DLL references by name.
#:    nvrtc64_120_0.dll + nvrtc-builtins64_129.dll stay: cublasLt64_12.dll may JIT kernels via NVRTC.
#: Validated with the frozen RoEnSpeak-cli.exe: large-v3 + medium x float16 + int8_float16 on CUDA,
#: text identical to the untrimmed bundle (plus beam 1 and --task transcribe).
NVIDIA_TRIM = {
    "nvblas64_12.dll",
    "cudnn_adv64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_engines_tensor_ir64_9.dll",
    "cudnn_ext64_9.dll",
    "cudnn_heuristic64_9.dll",
    "nvrtc64_120_0.alt.dll",
}
if os.environ.get("ROENSPEAK_NO_TRIM") == "1":
    NVIDIA_TRIM = set()


def _site_packages() -> Path:
    import ctranslate2

    return Path(ctranslate2.__file__).resolve().parents[1]


def nvidia_dll_datas() -> list[tuple[str, str, str]]:
    sp = _site_packages()
    out = []
    for pkg in NVIDIA_PACKAGES:
        bin_dir = sp / "nvidia" / pkg / "bin"
        if not bin_dir.is_dir():
            raise SystemExit(f"missing {bin_dir}: install requirements.txt (nvidia-*-cu12 wheels) first")
        for dll in sorted(bin_dir.glob("*.dll")):
            if dll.name.lower() in NVIDIA_TRIM:
                continue
            out.append((f"nvidia/{pkg}/bin/{dll.name}", str(dll), "DATA"))
    return out


# ---------------------------------------------------------------------------------------------
# Analysis (one graph for both entry scripts; each EXE gets only its own script, same PYZ).
# ---------------------------------------------------------------------------------------------
datas = [
    (str(ROOT / "assets"), "assets"),  # -> _internal/assets (app/ui/assets.py assets_dir())
]
datas += collect_data_files("faster_whisper")  # assets/silero_vad_v6.onnx

binaries = []
binaries += collect_dynamic_libs("ctranslate2")  # ctranslate2.dll, libiomp5md.dll, cudnn64_9.dll -> ctranslate2/

hiddenimports = [
    "app.main",
    "app.cli",
    "pynvml",
    "hf_xet",
    "soxr",
    "sounddevice",
    "_sounddevice_data",
    "tokenizers",
    "onnxruntime",
]
# huggingface_hub 1.x/2.x resolves top-level names (HfApi, ...) lazily, which static analysis can't follow.
hiddenimports += collect_submodules(
    "huggingface_hub",
    filter=lambda n: not n.startswith(("huggingface_hub.cli", "huggingface_hub.commands",
                                       "huggingface_hub._webhooks", "huggingface_hub.fastai_utils",
                                       "huggingface_hub.serialization._torch",
                                       "huggingface_hub.serialization._tensorflow",
                                       "huggingface_hub._tensorboard_logger")),
)

# Qt modules the app doesn't use (it imports only QtCore/QtGui/QtWidgets), plus heavy optional deps.
QT_UNUSED = [
    "Qt3DAnimation", "Qt3DCore", "Qt3DExtras", "Qt3DInput", "Qt3DLogic", "Qt3DRender", "QtAxContainer",
    "QtBluetooth", "QtCharts", "QtConcurrent", "QtDBus", "QtDataVisualization", "QtDesigner", "QtGraphs",
    "QtGraphsWidgets", "QtHelp", "QtHttpServer", "QtLocation", "QtMultimedia", "QtMultimediaWidgets",
    "QtNetwork", "QtNetworkAuth", "QtNfc", "QtOpenGL", "QtOpenGLWidgets", "QtPdf", "QtPdfWidgets",
    "QtPositioning", "QtPrintSupport", "QtQml", "QtQuick", "QtQuick3D", "QtQuickControls2", "QtQuickWidgets",
    "QtRemoteObjects", "QtScxml", "QtSensors", "QtSerialBus", "QtSerialPort", "QtSpatialAudio", "QtSql",
    "QtStateMachine", "QtSvg", "QtSvgWidgets", "QtTest", "QtTextToSpeech", "QtUiTools", "QtWebChannel",
    "QtWebEngineCore", "QtWebEngineWidgets", "QtWebEngineQuick", "QtWebSockets", "QtWebView", "QtXml",
]
excludes = [f"PySide6.{m}" for m in QT_UNUSED] + [
    "tkinter", "_tkinter", "turtle", "turtledemo", "idlelib", "lib2to3", "pydoc_data",
    "pytest", "_pytest", "py", "pluggy", "iniconfig",
    "PyInstaller", "setuptools", "pkg_resources", "distutils", "pip",
    "IPython", "matplotlib", "pandas", "scipy", "torch", "tensorflow", "transformers", "PIL",
    "onnxruntime.transformers", "onnxruntime.quantization", "onnxruntime.tools", "onnxruntime.datasets",
    "onnxruntime.backend",
    "ctranslate2.converters",  # model conversion only (needs transformers/torch)
    "numpy.tests", "numpy.testing.tests", "numpy.f2py",
]

a = Analysis(
    [str(SPEC_DIR / "launch_app.py"), str(SPEC_DIR / "launch_cli.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

# ---------------------------------------------------------------------------------------------
# Post-analysis filtering.
# ---------------------------------------------------------------------------------------------
#: Never bundle driver-provided DLLs (the ctypes scanner may pick them up from System32).
DRIVER_DLLS = {"nvml.dll", "nvcuda.dll", "nvapi64.dll"}
#: CUDA libraries are only allowed under nvidia/*/bin (added below) or the ctranslate2 dispatcher.
CUDA_PREFIXES = ("cublas", "cudnn", "nvrtc", "nvblas", "cudart")
#: Qt payload the UI does not need: translations (UI is English), software OpenGL fallback
#: (QPainter only, no QtQuick/OpenGL), and plugins for unused subsystems.
QT_DROP_PREFIXES = (
    "PySide6/translations/",
    "PySide6/plugins/qmltooling/", "PySide6/plugins/sqldrivers/", "PySide6/plugins/designer/",
    "PySide6/plugins/tls/", "PySide6/plugins/networkinformation/", "PySide6/plugins/generic/",
    "PySide6/plugins/vectorimageformats/",
)
QT_DROP_FILES = {"PySide6/opengl32sw.dll", "PySide6/d3dcompiler_47.dll"}


def _norm(dest: str) -> str:
    return dest.replace("\\", "/")


def _keep(entry) -> bool:
    dest = _norm(entry[0])
    base = dest.rsplit("/", 1)[-1].lower()
    if base in DRIVER_DLLS:
        return False
    if base.startswith(CUDA_PREFIXES) and base.endswith(".dll"):
        # The only CUDA DLL PyInstaller itself should collect is ctranslate2's bundled cuDNN dispatcher.
        return dest.lower() == "ctranslate2/cudnn64_9.dll"
    if dest.startswith(QT_DROP_PREFIXES) or dest in QT_DROP_FILES:
        return False
    if "/tests/" in dest.lower() and dest.lower().endswith(".py"):
        return False
    return True


a.binaries = [e for e in a.binaries if _keep(e)]
a.datas = [e for e in a.datas if _keep(e)]
a.datas += nvidia_dll_datas()

pyz = PYZ(a.pure)

app_scripts = [s for s in a.scripts if s[0] != "launch_cli"]
cli_scripts = [s for s in a.scripts if s[0] != "launch_app"]

exe_app = EXE(
    pyz,
    app_scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON,
    version=gen_version_info.version_info(f"{APP_NAME}.exe", f"{APP_NAME}: speak Romanian, type English"),
)

exe_cli = EXE(
    pyz,
    cli_scripts,
    [],
    exclude_binaries=True,
    name=f"{APP_NAME}-cli",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON,
    version=gen_version_info.version_info(f"{APP_NAME}-cli.exe", f"{APP_NAME} command-line .wav test mode"),
)

coll = COLLECT(
    exe_app,
    exe_cli,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
