#!/usr/bin/env bash
# Build RoEnSpeak for macOS from a clean checkout: venv -> requirements -> unit tests ->
# whisper.cpp/Metal native build -> PyInstaller onedir .app -> sanity checks -> codesign ->
# dist/RoEnSpeak-<version>.dmg
#
# The macOS counterpart of build/build.ps1; same step order, same strictness, same flag names.
# Run from anywhere (paths come from this script's location):
#
#     bash build/build_macos.sh [options]
#
# Options (the analogues of build.ps1's switches):
#   --skip-install   don't run pip (use the venv as is)
#   --skip-tests     don't run the unit tests
#   --skip-native    don't rebuild whisper.cpp; require build/native to be populated already
#   --skip-dmg       stop after the .app (no disk image)
#   --clean          PyInstaller --clean, and a clean native build
#   --gpu-smoke      run the frozen CLI on tests/fixtures/ro_sample.wav and assert Metal
#   --venv DIR       virtual environment (default: <repo>/.venv)
#   --python PATH    interpreter used to create the venv (default: python3.12)
#
# Signing (the SIGNTOOL_CERT analogue). Default is AD-HOC, on purpose -- see BUILD.md:
#   ROENSPEAK_CODESIGN_IDENTITY   Developer ID Application identity (name or SHA-1 hash) from
#                                 `security find-identity -v -p codesigning`. Unset => ad-hoc ("-").
#   ROENSPEAK_NOTARY_PROFILE      `xcrun notarytool store-credentials` profile name. Only used when
#                                 a real identity is set; submits and staples the .dmg.
#   ROENSPEAK_NATIVE_DIR          where the 7 whisper.cpp dylibs live (default: build/native)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

SKIP_INSTALL=0
SKIP_TESTS=0
SKIP_NATIVE=0
SKIP_DMG=0
CLEAN=0
GPU_SMOKE=0
VENV=""
BOOTSTRAP_PY="python3.12"

START_TS=$SECONDS

BOLD=""; CYAN=""; RED=""; GREEN=""; DIM=""; RESET=""
if [ -t 1 ]; then
    BOLD="$(printf '\033[1m')"; CYAN="$(printf '\033[36m')"; RED="$(printf '\033[31m')"
    GREEN="$(printf '\033[32m')"; DIM="$(printf '\033[2m')"; RESET="$(printf '\033[0m')"
fi

step() { printf '\n%s==> %s%s\n' "$CYAN$BOLD" "$*" "$RESET"; }
fail() { printf '%sBUILD FAILED: %s%s\n' "$RED$BOLD" "$*" "$RESET" >&2; exit 1; }
note() { printf '   %s\n' "$*"; }
run()  { printf '%s   > %s%s\n' "$DIM" "$*" "$RESET"; "$@" || fail "$1 exited with code $?"; }

size_mb() {  # size_mb PATH -> integer MB (file or directory)
    if [ -f "$1" ]; then
        printf '%s' "$(( $(stat -f%z "$1") / 1048576 ))"
    else
        printf '%s' "$(du -sk "$1" | awk '{printf "%d", $1/1024}')"
    fi
}

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; $d'; exit 0; }

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-install) SKIP_INSTALL=1; shift ;;
        --skip-tests)   SKIP_TESTS=1;   shift ;;
        --skip-native)  SKIP_NATIVE=1;  shift ;;
        --skip-dmg)     SKIP_DMG=1;     shift ;;
        --clean)        CLEAN=1;        shift ;;
        --gpu-smoke)    GPU_SMOKE=1;    shift ;;
        --venv)   [ $# -ge 2 ] || fail "--venv needs a directory";  VENV="$2";        shift 2 ;;
        --python) [ $# -ge 2 ] || fail "--python needs a path";     BOOTSTRAP_PY="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) fail "unknown option: $1 (try --help)" ;;
    esac
done

[ "$(uname -s)" = "Darwin" ] || fail "this is the macOS build; on Windows run build\\build.ps1"

# ---------------------------------------------------------------------------------------------
step "Python virtual environment"
[ -n "$VENV" ] || VENV="$ROOT/.venv"
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
    command -v "$BOOTSTRAP_PY" >/dev/null 2>&1 \
        || fail "'$BOOTSTRAP_PY' not found. Install CPython 3.12 for arm64 (python.org or 'brew install python@3.12'), or pass --python <path>."
    run "$BOOTSTRAP_PY" -m venv "$VENV"
fi
PYINFO="$("$PY" -c 'import sys,platform;print("%d.%d %s" % (sys.version_info[0], sys.version_info[1], platform.machine()))')" \
    || fail "venv python is broken: $PY"
note "venv: $VENV (Python $PYINFO)"
[ "$PYINFO" = "3.12 arm64" ] \
    || fail "need CPython 3.12 arm64 in the venv, got '$PYINFO' (delete $VENV and retry). The macOS build is arm64-only: onnxruntime, soxr and numpy publish no x86_64/universal2 wheels for 3.12."

# ---------------------------------------------------------------------------------------------
if [ "$SKIP_INSTALL" -eq 0 ]; then
    step "Install requirements (requirements-macos-dev.txt)"
    [ -f "$ROOT/requirements-macos-dev.txt" ] || fail "missing requirements-macos-dev.txt"
    run "$PY" -m pip install --disable-pip-version-check -r "$ROOT/requirements-macos-dev.txt"
    run "$PY" -m pip check
    if "$PY" -c 'import faster_whisper' >/dev/null 2>&1 || "$PY" -c 'import ctranslate2' >/dev/null 2>&1; then
        fail "faster-whisper / ctranslate2 are installed in $VENV. The macOS engine is whisper.cpp; those packages have no macOS GPU path and the spec excludes them. Recreate the venv."
    fi
else
    note "pip skipped (--skip-install)"
fi

# ---------------------------------------------------------------------------------------------
VERSION="$("$PY" "$ROOT/build/gen_version_info.py" --version | tr -d '[:space:]')" \
    || fail "could not read APP_VERSION from app/contracts.py"
[ -n "$VERSION" ] || fail "could not read APP_VERSION from app/contracts.py"
printf '\n   RoEnSpeak version %s (app/contracts.py APP_VERSION)\n' "$VERSION"

if [ "$SKIP_TESTS" -eq 0 ]; then
    step "Unit tests (hardware-free; gpu/mic/desktop/metal markers are excluded by pyproject.toml)"
    run "$PY" -m pytest -q -p no:cacheprovider
else
    note "tests skipped (--skip-tests)"
fi

# ---------------------------------------------------------------------------------------------
NATIVE_DIR="${ROENSPEAK_NATIVE_DIR:-$ROOT/build/native}"
WHISPER_DYLIBS="libroenspeak_whisper.dylib libwhisper.1.dylib libggml.0.dylib libggml-base.0.dylib libggml-cpu.0.dylib libggml-blas.0.dylib libggml-metal.0.dylib"

native_complete() {
    local name
    for name in $WHISPER_DYLIBS; do
        [ -f "$NATIVE_DIR/$name" ] || return 1
    done
    return 0
}

if [ "$SKIP_NATIVE" -eq 1 ]; then
    native_complete \
        || fail "--skip-native was given but $NATIVE_DIR does not hold all 7 whisper.cpp dylibs. Run: bash native/whispercpp/build.sh"
    note "native build skipped (--skip-native); using $NATIVE_DIR"
else
    step "whisper.cpp + Metal native build -> $NATIVE_DIR"
    [ -f "$ROOT/native/whispercpp/build.sh" ] \
        || fail "missing native/whispercpp/build.sh (it clones whisper.cpp v1.9.5, builds it with Metal and compiles the ctypes shim)"
    if [ "$CLEAN" -eq 1 ]; then
        run bash "$ROOT/native/whispercpp/build.sh" --out "$NATIVE_DIR" --clean
    else
        run bash "$ROOT/native/whispercpp/build.sh" --out "$NATIVE_DIR"
    fi
    native_complete || fail "native/whispercpp/build.sh did not stage all 7 dylibs into $NATIVE_DIR"
fi

# ---------------------------------------------------------------------------------------------
step "PyInstaller onedir BUNDLE -> build/out/RoEnSpeak.app"
ICNS="$ROOT/assets/icons/app.icns"
[ -f "$ICNS" ] || fail "missing $ICNS -- generate the icons first: $PY -m app.ui.gen_icons"
PYI_START=$SECONDS
PYI_ARGS=("$ROOT/build/roenspeak_macos.spec" --noconfirm --workpath "$ROOT/build/work" --distpath "$ROOT/build/out")
[ "$CLEAN" -eq 1 ] && PYI_ARGS+=(--clean)
ROENSPEAK_NATIVE_DIR="$NATIVE_DIR" run "$PY" -m PyInstaller "${PYI_ARGS[@]}"
APP="$ROOT/build/out/RoEnSpeak.app"
[ -d "$APP" ] || fail "PyInstaller did not produce $APP"
note "PyInstaller took $(( SECONDS - PYI_START )) s"

# ---------------------------------------------------------------------------------------------
step "Bundle sanity checks"
# Mirrors build.ps1's $required list. sys._MEIPASS is Contents/Frameworks for a PyInstaller 6.x
# .app, and data files live in Contents/Resources cross-linked into Contents/Frameworks, so both
# spellings must resolve.
REQUIRED=(
    "Contents/MacOS/RoEnSpeak"
    "Contents/MacOS/RoEnSpeak-cli"
    "Contents/Info.plist"
    "Contents/Resources/app.icns"
    "Contents/Frameworks/assets/icons"
    "Contents/Frameworks/assets/sounds"
    "Contents/Frameworks/assets/vad/silero_vad_v6.onnx"
    "Contents/Frameworks/_sounddevice_data/portaudio-binaries/libportaudio.dylib"
    "Contents/Frameworks/PySide6/Qt/plugins/platforms/libqcocoa.dylib"
)
for name in $WHISPER_DYLIBS; do
    REQUIRED+=("Contents/Frameworks/$name")
done
for rel in "${REQUIRED[@]}"; do
    [ -e "$APP/$rel" ] || fail "missing from bundle: $rel"
done
# onnxruntime's dylib carries its version in the file name, so match it by pattern.
ORT_DYLIB="$(find "$APP/Contents/Frameworks/onnxruntime" -name 'libonnxruntime*.dylib' -print -quit 2>/dev/null || true)"
[ -n "$ORT_DYLIB" ] || fail "missing from bundle: Contents/Frameworks/onnxruntime/libonnxruntime*.dylib (the Silero VAD runtime)"

# The whisper dylibs must sit FLAT next to each other, because the shim resolves its siblings
# through an @loader_path rpath. A stray copy in a subdirectory means the spec's filter let a
# duplicate through and 5.5 MB of libggml-metal is in the bundle twice.
STRAY="$(find "$APP/Contents/Frameworks" -mindepth 2 \( -name 'libggml*.dylib' -o -name 'libwhisper*.dylib' -o -name 'libroenspeak_whisper.dylib' -o -name 'libparakeet*' \) 2>/dev/null || true)"
[ -z "$STRAY" ] && [ ! -e "$APP/Contents/Frameworks/libparakeet.1.dylib" ] \
    || fail "whisper.cpp dylibs outside the top of Contents/Frameworks (should be flat): $STRAY"

# Every @rpath/... dependency of the 7 must resolve to a sibling in the same directory, and any
# dylib that has one must carry an @loader_path LC_RPATH. (libggml-base has no dependencies of its
# own beyond its install name, so PyInstaller correctly gives it no rpath -- hence the condition.)
for name in $WHISPER_DYLIBS; do
    LIB="$APP/Contents/Frameworks/$name"
    # tail -n +3 skips otool's filename line and the LC_ID_DYLIB line.
    DEPS="$(otool -L "$LIB" | tail -n +3 | awk '{print $1}' | grep '^@rpath/' || true)"
    if [ -n "$DEPS" ]; then
        otool -l "$LIB" | grep -A2 LC_RPATH | grep -qE '^ *path @loader_path( |$)' \
            || fail "$name loads $(printf '%s' "$DEPS" | tr '\n' ' ')but has no '@loader_path' LC_RPATH; the dylibs will not resolve"
        for dep in $DEPS; do
            [ -f "$APP/Contents/Frameworks/${dep#@rpath/}" ] \
                || fail "$name depends on $dep, which is not next to it in Contents/Frameworks"
        done
    fi
done

# Info.plist keys that the app cannot work without.
plist() { /usr/libexec/PlistBuddy -c "Print :$1" "$APP/Contents/Info.plist" 2>/dev/null; }
[ "$(plist CFBundleIdentifier)" = "com.roenspeak.RoEnSpeak" ] || fail "Info.plist CFBundleIdentifier is '$(plist CFBundleIdentifier)'"
[ "$(plist CFBundleExecutable)" = "RoEnSpeak" ] || fail "Info.plist CFBundleExecutable is '$(plist CFBundleExecutable)', expected RoEnSpeak"
[ "$(plist CFBundleShortVersionString)" = "$VERSION" ] || fail "Info.plist CFBundleShortVersionString is '$(plist CFBundleShortVersionString)', expected $VERSION"
[ "$(plist CFBundleVersion)" = "$VERSION" ] || fail "Info.plist CFBundleVersion is '$(plist CFBundleVersion)', expected $VERSION"
[ "$(plist LSUIElement)" = "true" ] || fail "Info.plist LSUIElement is not true (the app would get a Dock icon)"
[ "$(plist LSBackgroundOnly)" = "false" ] || fail "Info.plist LSBackgroundOnly is not false (the main window could never be shown)"
[ -n "$(plist NSMicrophoneUsageDescription)" ] || fail "Info.plist has no NSMicrophoneUsageDescription (the mic prompt would never appear)"
[ -n "$(plist LSMinimumSystemVersion)" ] || fail "Info.plist has no LSMinimumSystemVersion"

CLI="$APP/Contents/MacOS/RoEnSpeak-cli"
"$CLI" --help >/dev/null || fail "$CLI --help exited with $?"
note "layout OK, RoEnSpeak-cli starts; bundle size $(size_mb "$APP") MB"

# ---------------------------------------------------------------------------------------------
if [ "$GPU_SMOKE" -eq 1 ]; then
    step "Metal smoke test (frozen CLI, isolated config/logs, real models dir)"
    FIXTURE="$ROOT/tests/fixtures/ro_sample.wav"
    [ -f "$FIXTURE" ] || fail "missing $FIXTURE"
    SMOKE="$(mktemp -d "${TMPDIR:-/tmp}/roenspeak-smoke.XXXXXX")"
    mkdir -p "$SMOKE/data" "$SMOKE/config"
    : "${ROENSPEAK_MODELS_DIR:=$HOME/Library/Application Support/RoEnSpeak/models}"
    set +e
    SMOKE_JSON="$(ROENSPEAK_DATA_DIR="$SMOKE/data" ROENSPEAK_CONFIG_DIR="$SMOKE/config" \
        ROENSPEAK_MODELS_DIR="$ROENSPEAK_MODELS_DIR" "$CLI" "$FIXTURE" --json)"
    SMOKE_RC=$?
    set -e
    [ "$SMOKE_RC" -eq 0 ] || { printf '%s\n' "$SMOKE_JSON"; fail "Metal smoke test exited with $SMOKE_RC (logs: $SMOKE/data/logs)"; }
    DEVICE="$(printf '%s' "$SMOKE_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["engine"]["device"])')"
    [ "$DEVICE" = "metal" ] || fail "Metal smoke test ran on '$DEVICE', expected metal"
    printf '%s' "$SMOKE_JSON" | "$PY" -c '
import json, sys
o = json.load(sys.stdin)
e = o["engine"]
print("   %s %s on %s: %s" % (e.get("model"), e.get("compute_type"), e.get("gpu_name"),
                              " ".join(s.get("text", "") for s in o.get("segments", []))))'
    rm -rf "$SMOKE"
fi

# ---------------------------------------------------------------------------------------------
step "Code signing"
# WHY the default is to sign at all (build.ps1 leaves the Windows exes unsigned by default):
# on macOS a code signature is not cosmetic. An arm64 Mach-O will not even be loaded without one,
# and TCC -- the database behind the Accessibility and Microphone permissions this app cannot work
# without -- records its grants against the code signature, not the path. So the bundle is ALWAYS
# signed; the only question is with what. See BUILD.md "Signing and notarization (macOS)".
ENTITLEMENTS="$ROOT/build/entitlements.plist"
[ -f "$ENTITLEMENTS" ] || fail "missing $ENTITLEMENTS"
IDENTITY="${ROENSPEAK_CODESIGN_IDENTITY:-}"
if [ -n "$IDENTITY" ]; then
    note "signing with identity: $IDENTITY (hardened runtime + $ENTITLEMENTS)"
else
    IDENTITY="-"
    note "signing ad-hoc ('-'); set ROENSPEAK_CODESIGN_IDENTITY to use a Developer ID"
fi
# --deep is the right tool here and only here: PyInstaller already signed every nested Mach-O
# individually, and this re-seals them under one identity with one set of entitlements after the
# Info.plist and the dylib load commands were rewritten. Apple discourages --deep for *distribution*
# signing (each nested binary should be signed with its own entitlements); we have exactly one
# entitlement set for the whole bundle, so there is nothing to lose.
run codesign --force --deep --sign "$IDENTITY" --options runtime --entitlements "$ENTITLEMENTS" \
    --timestamp=none "$APP"
run codesign --verify --strict --verbose=2 "$APP"
# Prove the re-signed bundle still runs: re-signing rewrites every nested signature, and a broken
# one shows up as "killed: 9" rather than as an error from codesign.
"$CLI" --help >/dev/null || fail "RoEnSpeak-cli stopped working after codesign (exit $?)"
if [ "$IDENTITY" = "-" ]; then
    note "ad-hoc signature verified. 'spctl --assess' will reject it (no Developer ID); that is expected."
else
    printf '%s   > spctl --assess --type execute%s\n' "$DIM" "$RESET"
    spctl --assess --type execute --verbose=2 "$APP" || note "spctl rejected the bundle (notarization pending)"
fi

# ---------------------------------------------------------------------------------------------
if [ "$SKIP_DMG" -eq 1 ]; then
    printf '\n%sDone (no dmg) in %s s: %s%s\n' "$GREEN$BOLD" "$(( SECONDS - START_TS ))" "$APP" "$RESET"
    exit 0
fi

step "Disk image -> dist/RoEnSpeak-$VERSION.dmg"
DMG="$ROOT/dist/RoEnSpeak-$VERSION.dmg"
run bash "$ROOT/build/make_dmg.sh" --app "$APP" --out "$DMG" --version "$VERSION"
[ -f "$DMG" ] || fail "build/make_dmg.sh did not produce $DMG"

# ---------------------------------------------------------------------------------------------
if [ -n "${ROENSPEAK_NOTARY_PROFILE:-}" ]; then
    if [ "${ROENSPEAK_CODESIGN_IDENTITY:-}" = "" ]; then
        note "notarization skipped: ROENSPEAK_NOTARY_PROFILE is set but ROENSPEAK_CODESIGN_IDENTITY is not. Apple only notarizes Developer ID signatures, never ad-hoc ones."
    else
        step "Notarize + staple"
        run xcrun notarytool submit "$DMG" --keychain-profile "$ROENSPEAK_NOTARY_PROFILE" --wait
        run xcrun stapler staple "$DMG"
        run spctl --assess --type open --context context:primary-signature --verbose=2 "$DMG"
    fi
else
    note "notarization skipped (ROENSPEAK_NOTARY_PROFILE not set): first launch needs right-click > Open, or 'xattr -dr com.apple.quarantine'"
fi

printf '\n%sDone in %s s%s\n' "$GREEN$BOLD" "$(( SECONDS - START_TS ))" "$RESET"
printf '   app : %s (%s MB)\n' "$APP" "$(size_mb "$APP")"
printf '   dmg : %s (%s MB, version %s)\n' "$DMG" "$(size_mb "$DMG")" "$VERSION"
