# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for RoEnSpeak on macOS: ONEDIR + BUNDLE -> build/out/RoEnSpeak.app.

Two executables share one PYZ and one set of collected files (the same script-splitting trick as
build/roenspeak.spec):

    Contents/MacOS/RoEnSpeak        windowed menu-bar app  (build/launch_app.py -> app.main.main)
    Contents/MacOS/RoEnSpeak-cli    console .wav tester    (build/launch_cli.py -> app.cli.main)

Build (from the repo root, macOS arm64 venv):
    .venv/bin/python -m PyInstaller build/roenspeak_macos.spec --noconfirm \
        --workpath build/work --distpath build/out

build/build_macos.sh does this plus tests, sanity checks, signing and the .dmg.

Why a separate spec instead of branching build/roenspeak.spec: the two platforms share no payload.
Windows ships ctranslate2 + 1 GB of CUDA DLLs and a VSVersionInfo resource; macOS ships 7 whisper.cpp
dylibs, an .icns, an Info.plist and no faster-whisper at all. Keeping them apart means a change here
can never regress the Windows installer. The two specs share build/launch_{app,cli}.py and
build/gen_version_info.py verbatim, so the version still has exactly one source (APP_VERSION).

whisper.cpp dylib layout
------------------------
The 7 redistributable dylibs (see native/whispercpp/build.sh and the port notes) are collected FLAT
into Contents/Frameworks, which is where sys._MEIPASS points for a PyInstaller 6.x .app bundle:

    Contents/Frameworks/libroenspeak_whisper.dylib   the ctypes shim
    Contents/Frameworks/libwhisper.1.dylib
    Contents/Frameworks/libggml.0.dylib
    Contents/Frameworks/libggml-base.0.dylib
    Contents/Frameworks/libggml-cpu.0.dylib
    Contents/Frameworks/libggml-blas.0.dylib
    Contents/Frameworks/libggml-metal.0.dylib        Metal shaders are EMBEDDED, no .metallib ships

Flat and top-level is the only placement that works. PyInstaller rewrites every collected binary's
dependent-library paths to `@rpath/<basename>` and replaces its LC_RPATH with
`@loader_path/<..>*` relative to the entry's destination directory (PyInstaller/building/utils.py,
process_collected_binary). For a top-level destination that rpath is exactly `@loader_path`, which
is what the shim was linked with, so all six siblings resolve with no DYLD_LIBRARY_PATH and no
preloading. Putting them in a subdirectory would set the rpath to `@loader_path/..` and the
siblings would be looked up one level too high.

This matches app/engine/whispercpp.py's documented lookup order:
    1. $ROENSPEAK_WHISPER_LIB          explicit override (file or directory)
    2. sys._MEIPASS                    frozen app  -> Contents/Frameworks (here)
    3. <repo>/build/native            dev checkout -> what native/whispercpp/build.sh stages

Environment knobs:
    ROENSPEAK_NATIVE_DIR   directory holding the 7 dylibs (default: <repo>/build/native)
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

SPEC_DIR = Path(SPECPATH).resolve()
ROOT = SPEC_DIR.parent
sys.path.insert(0, str(SPEC_DIR))
sys.path.insert(0, str(ROOT))
import gen_version_info  # noqa: E402  (build/gen_version_info.py -- shared with the Windows spec)

from app import contracts  # noqa: E402  (APP_VERSION / MIN_MACOS_VERSION single source of truth)

if sys.platform != "darwin":
    raise SystemExit("build/roenspeak_macos.spec only builds on macOS; on Windows use build/roenspeak.spec")

APP_NAME = gen_version_info.app_name()
APP_VERSION = gen_version_info.app_version()

# ---------------------------------------------------------------------------------------------
# target_arch: arm64 only, never universal2.
#
# The decision is forced by the wheels, not by preference. onnxruntime 1.31.0
# (macosx_14_0_arm64), soxr 1.1.0 (macosx_11_0_arm64) and numpy 2.5.4 (macosx_14_0_arm64) publish
# no x86_64 or universal2 macOS wheels for CPython 3.12, and the venv interpreter itself is an
# arm64-thin build, so there is no x86_64 slice to put in a fat binary. PySide6-Essentials and
# shiboken6 ARE universal2 and would be thinned to arm64 here.
# whisper.cpp pushes the same way: the Metal backend is the whole point of this port, and ggml's
# Metal path is only interesting on Apple Silicon.
# Consequence: the .app does not run on Intel Macs. Documented in BUILD.md and README.md.
# ---------------------------------------------------------------------------------------------
TARGET_ARCH = "arm64"

# ---------------------------------------------------------------------------------------------
# Inputs produced by other parts of the build. Fail loudly, naming the command to run.
# ---------------------------------------------------------------------------------------------
ICON = ROOT / "assets" / "icons" / "app.icns"
if not ICON.is_file():
    raise SystemExit(
        f"missing {ICON}\n"
        f"    the .app icon is generated from the committed PNGs; run:\n"
        f"        .venv/bin/python -m app.ui.gen_icons"
    )

NATIVE_DIR = Path(os.environ.get("ROENSPEAK_NATIVE_DIR") or (ROOT / "build" / "native")).resolve()

#: whisper.cpp release the native build is pinned to (see native/whispercpp/build.sh and BUILD.md).
WHISPER_TAG = "v1.9.5"

#: The 7 files to redistribute, named after their install-name basenames (NOT symlinks).
#: libparakeet is deliberately absent: nothing in the whisper path links it.
WHISPER_DYLIBS = (
    "libroenspeak_whisper.dylib",
    "libwhisper.1.dylib",
    "libggml.0.dylib",
    "libggml-base.0.dylib",
    "libggml-cpu.0.dylib",
    "libggml-blas.0.dylib",
    "libggml-metal.0.dylib",
)

_missing = [name for name in WHISPER_DYLIBS if not (NATIVE_DIR / name).is_file()]
if _missing:
    raise SystemExit(
        f"missing whisper.cpp dylibs in {NATIVE_DIR}: {', '.join(_missing)}\n"
        f"    build them (clones whisper.cpp {WHISPER_TAG} and compiles the shim with Metal):\n"
        f"        bash native/whispercpp/build.sh\n"
        f"    or point the spec at an existing staging directory:\n"
        f"        ROENSPEAK_NATIVE_DIR=/path/to/dylibs .venv/bin/python -m PyInstaller build/roenspeak_macos.spec"
    )

# ---------------------------------------------------------------------------------------------
# Analysis (one graph for both entry scripts; each EXE gets only its own script, same PYZ).
# ---------------------------------------------------------------------------------------------
datas = [
    (str(ROOT / "assets"), "assets"),  # -> Contents/Resources/assets (app/ui/assets.py assets_dir())
]
# No collect_data_files("faster_whisper"): faster-whisper is not installed on macOS and never will be
# (see requirements-macos.in). The Silero VAD model it used to supply is vendored instead, and is
# part of the assets/ tree above as assets/vad/silero_vad_v6.onnx.

binaries = [(str(NATIVE_DIR / name), ".") for name in WHISPER_DYLIBS]

hiddenimports = [
    "app.main",
    "app.cli",
    "sounddevice",
    "_sounddevice_data",   # portaudio-binaries/libportaudio.dylib
    "onnxruntime",         # Silero VAD on CPU
    "soxr",                # resampling to 16 kHz
]
# No pynvml / tokenizers / ctranslate2: Windows-only dependencies, not installed here.
try:
    import hf_xet  # noqa: F401  (optional accelerated HF downloads)

    hiddenimports.append("hf_xet")
except ImportError:
    pass

# app/core/platform.py imports every platform module lazily, inside functions. PyInstaller's
# modulegraph does follow function-level imports, but these are the modules the app cannot start
# without, so name them explicitly: a missing one becomes a build-time warning instead of a
# runtime ImportError inside a frozen .app with no console.
hiddenimports += [
    "app.core.platform",
    "app.engine.whispercpp",
    "app.engine.metal_probe",
    "app.engine.ggml_downloader",
    "app.macos._mac_api",
    "app.macos.autostart",
    "app.macos.clipboard",
    "app.macos.hotkey",
    "app.macos.hotkey_parse",
    "app.macos.injector",
    "app.macos.instance",
    "app.macos.permissions",
    "app.ui._mac",
    "app.ui._platform",
]

# huggingface_hub 1.x/2.x resolves top-level names (HfApi, ...) lazily, which static analysis can't
# follow. Same filter as the Windows spec.
hiddenimports += collect_submodules(
    "huggingface_hub",
    filter=lambda n: not n.startswith(("huggingface_hub.cli", "huggingface_hub.commands",
                                       "huggingface_hub._webhooks", "huggingface_hub.fastai_utils",
                                       "huggingface_hub.serialization._torch",
                                       "huggingface_hub.serialization._tensorflow",
                                       "huggingface_hub._tensorboard_logger")),
)

# Qt modules the app doesn't use (it imports only QtCore/QtGui/QtWidgets), plus heavy optional deps.
# Same list as the Windows spec.
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
    "numpy.tests", "numpy.testing.tests", "numpy.f2py",
    # Windows-only engine stack. Not installed here; excluded so that a dev machine which happens to
    # have them can never produce a macOS bundle that silently carries them.
    "faster_whisper", "ctranslate2", "tokenizers", "av", "pynvml", "nvidia",
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
# Post-analysis filtering (the macOS counterpart of the Windows CUDA/Qt filter).
# ---------------------------------------------------------------------------------------------
#: Qt payload the UI does not need: translations (the UI is English) and plugins for unused
#: subsystems. The Windows spec's opengl32sw.dll / d3dcompiler_47.dll entries have no macOS
#: equivalent and are dropped from this list. On macOS the hook's prefix is PySide6/Qt/... .
QT_DROP_PREFIXES = (
    "PySide6/Qt/translations/",
    "PySide6/Qt/plugins/qmltooling/", "PySide6/Qt/plugins/sqldrivers/", "PySide6/Qt/plugins/designer/",
    "PySide6/Qt/plugins/tls/", "PySide6/Qt/plugins/networkinformation/", "PySide6/Qt/plugins/generic/",
    "PySide6/Qt/plugins/vectorimageformats/",
)
#: The whisper.cpp dylibs are collected by the explicit `binaries` list above, flat at the top level.
#: Any copy at another destination is either a duplicate found through @rpath resolution or a stray
#: pick-up from a dev machine's Homebrew/whisper.cpp tree; dropping them keeps one authoritative copy
#: next to the shim (and keeps 5.5 MB of libggml-metal from being bundled twice).
WHISPER_DYLIB_NAMES = set(WHISPER_DYLIBS) | {
    # Versioned real names, in case a staging dir was not renamed, plus the unneeded Parakeet backend.
    "libwhisper.1.9.5.dylib", "libggml.0.27.0.dylib", "libggml-base.0.27.0.dylib",
    "libggml-cpu.0.27.0.dylib", "libggml-blas.0.27.0.dylib", "libggml-metal.0.27.0.dylib",
    "libparakeet.1.9.5.dylib", "libparakeet.1.dylib",
}


def _norm(dest: str) -> str:
    return dest.replace("\\", "/")


def _keep(entry) -> bool:
    dest = _norm(entry[0])
    base = dest.rsplit("/", 1)[-1]
    if base in WHISPER_DYLIB_NAMES and dest != base:
        return False  # not at the top level -> duplicate or stray
    if base in ("libparakeet.1.9.5.dylib", "libparakeet.1.dylib"):
        return False  # never needed, even at the top level
    if dest.startswith(QT_DROP_PREFIXES):
        return False
    if "/tests/" in dest.lower() and dest.lower().endswith(".py"):
        return False
    return True


a.binaries = [e for e in a.binaries if _keep(e)]
a.datas = [e for e in a.datas if _keep(e)]

pyz = PYZ(a.pure)

app_scripts = [s for s in a.scripts if s[0] != "launch_cli"]
cli_scripts = [s for s in a.scripts if s[0] != "launch_app"]

# The windowed app MUST be the first EXE in COLLECT: BUNDLE picks CFBundleExecutable from the first
# EXECUTABLE entry in the (alphabetically sorted) TOC, and "RoEnSpeak" sorts before "RoEnSpeak-cli".
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
    argv_emulation=False,  # would need Apple Events; an LSUIElement app has no Dock icon to drop on
    target_arch=TARGET_ARCH,
    codesign_identity=None,  # build_macos.sh signs the finished bundle (ad-hoc by default)
    entitlements_file=None,
    icon=str(ICON),
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
    target_arch=TARGET_ARCH,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON),
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

# ---------------------------------------------------------------------------------------------
# Info.plist
#
# PyInstaller pre-fills CFBundleDisplayName/Name/Identifier/Executable/IconFile/PackageType/
# ShortVersionString and then updates the dict with everything below, so these keys win.
#
# LSBackgroundOnly MUST be overridden: BUNDLE inherits `console` from COLLECT (which inherits it
# from an EXE) and sets LSBackgroundOnly=True whenever that is True. A background-only app cannot
# be activated and cannot show the main window, so the value is pinned to False here and the
# menu-bar-only behaviour comes from LSUIElement instead.
# ---------------------------------------------------------------------------------------------
MIC_USAGE = (
    f"{APP_NAME} records from your microphone while you dictate, and translates the speech on this "
    f"Mac. Audio is never uploaded."
)

app = BUNDLE(
    coll,
    name=f"{APP_NAME}.app",
    icon=str(ICON),
    bundle_identifier="com.roenspeak.RoEnSpeak",
    version=APP_VERSION,
    info_plist={
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundleVersion": APP_VERSION,
        # Menu-bar / tray-only app: no Dock icon, no app switcher entry, windows still allowed.
        # This is the macOS equivalent of the Windows tray-only app.
        "LSUIElement": True,
        "LSBackgroundOnly": False,
        # Microphone TCC prompt. Without this key the prompt never appears and the mic silently
        # returns silence (or CoreAudio fails), with no way for the user to grant access.
        "NSMicrophoneUsageDescription": MIC_USAGE,
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": contracts.MIN_MACOS_VERSION,
        # The app owns a resident speech model, a Carbon hotkey registration and an audio stream.
        # Automatic termination would let macOS kill it while it looks idle in the menu bar; sudden
        # termination would SIGKILL it without releasing the Metal working set or the hotkey.
        "NSSupportsAutomaticTermination": False,
        "NSSupportsSuddenTermination": False,
        # An LSUIElement app with no visible window is a prime App Nap candidate, and App Nap
        # throttles timers and lowers thread priority -- which would show up as missed hotkeys and
        # stalled inference. Opt out.
        "NSAppSleepDisabled": True,
        # Only one instance may own the global hotkey; app/macos/instance.py enforces it with an
        # flock, and this stops Launch Services from starting a second copy from Finder at all.
        "LSMultipleInstancesProhibited": True,
        # Follow the system appearance (Qt handles dark mode).
        "NSRequiresAquaSystemAppearance": False,
        "LSApplicationCategoryType": "public.app-category.productivity",
        "NSHumanReadableCopyright": f"{gen_version_info.COPYRIGHT}. Speak Romanian, type English.",
        "NSPrincipalClass": "NSApplication",
    },
)
