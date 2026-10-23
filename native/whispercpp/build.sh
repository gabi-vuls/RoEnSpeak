#!/bin/bash
# Builds the macOS speech engine: whisper.cpp (Metal, shared libs) + the RoEnSpeak C shim,
# and stages the 7 redistributable dylibs flat into one output directory.
#
#   ./native/whispercpp/build.sh [--out DIR] [--work DIR] [--jobs N] [--clean]
#                               [--no-native | --native]
#
# Defaults: --work build/work/whispercpp   (gitignored via build/work/)
#           --out  build/native            (what app/engine/whispercpp.py looks for)
#           --jobs $(sysctl -n hw.ncpu)
#
# Idempotent: re-running reuses the clone and the cmake cache, so a second run is seconds.
# --clean removes the work dir and the output dir first.
#
# Requirements: git, cmake (brew install cmake), Xcode Command Line Tools (clang++).
# The Metal *Toolchain* Xcode component is deliberately NOT required: the build uses
# -DGGML_METAL_EMBED_LIBRARY=ON, which embeds the 36 .metal shader *sources* into
# libggml-metal.dylib and compiles them with the Metal framework at runtime
# ("loaded 36 libraries from embedded data in 0.018 sec"). There is no default.metallib
# to place next to the executable, which is what makes this work under PyInstaller.
#
# ---------------------------------------------------------------------------------------
# DECISION: -DGGML_NATIVE=OFF is the DEFAULT here.
#
# With GGML_NATIVE=ON (upstream's default) cmake picks -mcpu=native+<features> for
# libggml-cpu, i.e. it bakes the *build* machine's ISA into the binary. RoEnSpeak ships
# these dylibs inside a .dmg to Macs we never see, so a build made on an M4 (SME/i8mm)
# could SIGILL on an M1. Upstream hit exactly this class of bug for its own release
# images (whisper.cpp "devops : add -DGGML_NATIVE=OFF to CMAKE_ARGS to fix crashes",
# #4111). The cost is nil for us: inference runs on Metal, and the CPU backend only does
# the mel/front-end work and the opt-in "CPU mode (slow)" fallback, where the generic
# arm64 NEON+Accelerate baseline is what every Apple Silicon Mac has anyway.
#
#   --no-native  explicit form of the default (-DGGML_NATIVE=OFF) - redistributable
#   --native     opt in to -mcpu=native (-DGGML_NATIVE=ON) - local benchmarking only,
#                never for a build that goes into a .dmg
# ---------------------------------------------------------------------------------------

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

WHISPER_URL="https://github.com/ggml-org/whisper.cpp"
# Pinned release: tag v1.9.5 == commit d1be6fde11ac6e0407606b4e42fe72d34add8037.
WHISPER_TAG="v1.9.5"
WHISPER_COMMIT="d1be6fde11ac6e0407606b4e42fe72d34add8037"

WORK_DIR="$REPO_ROOT/build/work/whispercpp"
OUT_DIR="$REPO_ROOT/build/native"
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
GGML_NATIVE="OFF"
CLEAN=0

die() { printf 'build.sh: error: %s\n' "$*" >&2; exit 1; }
step() { printf '\n==> %s\n' "$*"; }

usage() {
    cat <<USAGE
usage: build.sh [--out DIR] [--work DIR] [--jobs N] [--clean] [--no-native|--native]

Builds whisper.cpp ($WHISPER_TAG) with Metal + embedded shaders and the RoEnSpeak C shim,
then stages the 7 redistributable dylibs flat into the output directory.

  --out DIR     where the 7 dylibs go            (default: $OUT_DIR)
  --work DIR    clone + cmake build tree         (default: $WORK_DIR)
  --jobs N      parallel compile jobs            (default: $JOBS)
  --clean       delete the work and output dirs first
  --no-native   -DGGML_NATIVE=OFF (the default): portable across Apple Silicon
  --native      -DGGML_NATIVE=ON: -mcpu=native, local benchmarking only, never for a .dmg
USAGE
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --out)   [ $# -ge 2 ] || die "--out needs a directory";  OUT_DIR="$2";  shift 2 ;;
        --work)  [ $# -ge 2 ] || die "--work needs a directory"; WORK_DIR="$2"; shift 2 ;;
        --jobs)  [ $# -ge 2 ] || die "--jobs needs a number";    JOBS="$2";     shift 2 ;;
        --out=*)  OUT_DIR="${1#*=}";  shift ;;
        --work=*) WORK_DIR="${1#*=}"; shift ;;
        --jobs=*) JOBS="${1#*=}";     shift ;;
        --clean)     CLEAN=1;          shift ;;
        --no-native) GGML_NATIVE="OFF"; shift ;;
        --native)    GGML_NATIVE="ON";  shift ;;
        -h|--help)   usage 0 ;;
        *) printf 'build.sh: error: unknown option %s\n' "$1" >&2; usage 1 ;;
    esac
done

case "$JOBS" in ''|*[!0-9]*) die "--jobs must be a positive integer, got '$JOBS'" ;; esac
[ "$JOBS" -ge 1 ] || die "--jobs must be >= 1"

# --- 0. preconditions ------------------------------------------------------------------

[ "$(uname -s)" = "Darwin" ] || die "this script builds the macOS engine and only runs on macOS (uname -s = $(uname -s))"

for tool in git cmake clang++ otool install_name_tool; do
    command -v "$tool" >/dev/null 2>&1 || case "$tool" in
        cmake) die "cmake was not found on PATH. Install it with:  brew install cmake" ;;
        git)   die "git was not found on PATH. Install the Xcode Command Line Tools with:  xcode-select --install" ;;
        *)     die "$tool was not found on PATH. Install the Xcode Command Line Tools with:  xcode-select --install" ;;
    esac
done

# --- 1. clone / check out the pinned commit --------------------------------------------

if [ "$CLEAN" = "1" ]; then
    step "cleaning $WORK_DIR and $OUT_DIR"
    rm -rf "$WORK_DIR" "$OUT_DIR"
fi

mkdir -p "$WORK_DIR"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
SRC="$WORK_DIR/whisper.cpp"
BUILD="$WORK_DIR/build"

if [ ! -d "$SRC/.git" ]; then
    step "cloning $WHISPER_URL @ $WHISPER_TAG into $SRC"
    rm -rf "$SRC"
    # --depth 1 on the tag, then fetch the exact commit so the pin is verifiable even if
    # the tag is ever moved.
    git clone --depth 1 --branch "$WHISPER_TAG" "$WHISPER_URL" "$SRC" \
        || die "git clone of $WHISPER_URL failed (no network?)"
fi

HAVE="$(git -C "$SRC" rev-parse HEAD)"
if [ "$HAVE" != "$WHISPER_COMMIT" ]; then
    step "checking out pinned commit $WHISPER_COMMIT (have $HAVE)"
    git -C "$SRC" fetch --depth 1 origin "$WHISPER_COMMIT" \
        || die "could not fetch the pinned commit $WHISPER_COMMIT from $WHISPER_URL"
    git -C "$SRC" checkout --detach --force "$WHISPER_COMMIT" \
        || die "could not check out the pinned commit $WHISPER_COMMIT"
    HAVE="$(git -C "$SRC" rev-parse HEAD)"
    [ "$HAVE" = "$WHISPER_COMMIT" ] || die "whisper.cpp is at $HAVE, expected $WHISPER_COMMIT"
fi
SHORT_COMMIT="$(git -C "$SRC" rev-parse --short HEAD)"
printf 'whisper.cpp %s @ %s (%s)\n' "$WHISPER_TAG" "$SHORT_COMMIT" "$WHISPER_COMMIT"

# --- 2. configure + build whisper.cpp with Metal ---------------------------------------

# GGML_METAL / GGML_METAL_EMBED_LIBRARY are already the defaults on Apple; they are passed
# explicitly so a future default flip upstream cannot silently change what we ship.
step "cmake configure (GGML_NATIVE=$GGML_NATIVE)"
cmake -S "$SRC" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DWHISPER_BUILD_EXAMPLES=OFF \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_SERVER=OFF \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_NATIVE="$GGML_NATIVE" \
    || die "cmake configure failed"

step "cmake build -j$JOBS"
cmake --build "$BUILD" -j"$JOBS" --config Release || die "cmake build failed"

BIN="$BUILD/bin"
[ -d "$BIN" ] || die "expected the built libraries in $BIN, but that directory does not exist"

# --- 3. the shim -----------------------------------------------------------------------

# Built as C++ (-x c++) although the source is valid C11: that enables the try/catch
# guards around whisper_init_* / whisper_full / result collection. Upstream only guards
# model load, so a std::bad_alloc out of whisper_full would unwind through an extern "C"
# frame into Python and abort the process.
# -install_name @rpath/... + -rpath @loader_path is the whole rpath design: a plain
# ctypes.CDLL(".../libroenspeak_whisper.dylib") then resolves the six siblings sitting
# next to it, with no DYLD_LIBRARY_PATH and no preloading.
step "compiling libroenspeak_whisper.dylib"
clang++ -x c++ -std=c++17 -O2 -fPIC -fvisibility=hidden -dynamiclib \
    -I "$SRC/include" -I "$SRC/ggml/include" \
    -DRS_WHISPER_BUILD_COMMIT="\"$SHORT_COMMIT\"" \
    -o "$WORK_DIR/libroenspeak_whisper.dylib" "$HERE/roenspeak_whisper.c" \
    -L "$BIN" -lwhisper -lggml -lggml-base \
    -Wl,-rpath,@loader_path \
    -install_name @rpath/libroenspeak_whisper.dylib \
    || die "compiling the shim failed"

# --- 4. stage the output dir -----------------------------------------------------------

# Staged under the *install-name* basenames (libwhisper.1.dylib, libggml.0.dylib, ...),
# which is what the loader asks for, and as real files rather than cmake's version
# symlinks (PyInstaller flattens/resolves symlinks inconsistently). cp -L follows the
# symlink, so the actual SO version (1.9.5 / 0.27.0 / ...) never has to be hardcoded.
EXPECTED="libroenspeak_whisper.dylib libwhisper.1.dylib libggml.0.dylib libggml-base.0.dylib libggml-cpu.0.dylib libggml-blas.0.dylib libggml-metal.0.dylib"

step "staging into $OUT_DIR"
for name in $EXPECTED; do
    if [ "$name" = "libroenspeak_whisper.dylib" ]; then
        src="$WORK_DIR/libroenspeak_whisper.dylib"
    else
        src="$BIN/$name"
    fi
    [ -e "$src" ] || die "expected $src to exist after the build but it does not"
    cp -L "$src" "$OUT_DIR/$name.new" || die "could not copy $src"
    mv -f "$OUT_DIR/$name.new" "$OUT_DIR/$name"
    chmod 0755 "$OUT_DIR/$name"
done

# --- 5. verify -------------------------------------------------------------------------

step "verifying $OUT_DIR"
fail=0
for name in $EXPECTED; do
    p="$OUT_DIR/$name"
    if [ ! -f "$p" ]; then
        printf '  MISSING  %s\n' "$name"; fail=1; continue
    fi
    [ -L "$p" ] && { printf '  SYMLINK  %s (must be a real file)\n' "$name"; fail=1; }
    printf '  %10s  %s\n' "$(stat -f %z "$p")" "$name"
done
[ "$fail" = "0" ] || die "the output directory is incomplete; see MISSING/SYMLINK above"

# Every @rpath dependency must be satisfied by a sibling in OUT_DIR: that is the property
# that makes a bare ctypes.CDLL() of the shim work, and the one a bad stage would break.
for name in $EXPECTED; do
    otool -L "$OUT_DIR/$name" | awk '/@rpath\//{print $1}' | sed 's|@rpath/||' | while read -r dep; do
        [ -f "$OUT_DIR/$dep" ] || { printf '  UNRESOLVED  %s needs %s\n' "$name" "$dep"; exit 1; }
    done || die "an @rpath dependency is missing from $OUT_DIR"
done

# Embedded Metal shaders: there must be no .metallib anywhere in the output, and the
# ggml_metallib_* symbols must be inside libggml-metal.
if find "$OUT_DIR" -name '*.metallib' | grep -q .; then
    die "unexpected .metallib in $OUT_DIR - was this configured with GGML_METAL_EMBED_LIBRARY=OFF?"
fi
nm -gU "$OUT_DIR/libggml-metal.0.dylib" 2>/dev/null | grep -q ggml_metallib \
    || printf '  note: no ggml_metallib_* symbol visible in libggml-metal (hidden visibility is fine)\n'

printf '\nOK  7/7 dylibs in %s  (whisper.cpp %s @ %s, GGML_NATIVE=%s, total %s)\n' \
    "$OUT_DIR" "$WHISPER_TAG" "$SHORT_COMMIT" "$GGML_NATIVE" \
    "$(du -sh "$OUT_DIR" | awk '{print $1}')"
printf 'Load it with:  ROENSPEAK_WHISPER_LIB=%s .venv/bin/python -c "..."\n' "$OUT_DIR"
