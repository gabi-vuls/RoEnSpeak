# `native/` — the macOS speech engine (whisper.cpp on Metal)

The Windows build runs faster-whisper / CTranslate2 on CUDA. There is no CUDA on Apple
Silicon and no CTranslate2 Metal backend, so the macOS build runs
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) on Metal behind a small C shim that
`app/engine/whispercpp.py` loads with `ctypes`. Nothing here is used on Windows.

```
native/whispercpp/roenspeak_whisper.h   the C ABI the Python side binds
native/whispercpp/roenspeak_whisper.c   the shim (valid C11, compiled as C++)
native/whispercpp/build.sh              clone + build + stage the 7 dylibs
```

## Build

```sh
./native/whispercpp/build.sh                 # -> build/native/*.dylib (7 files, 7.5 MB)
./native/whispercpp/build.sh --clean         # from scratch
./native/whispercpp/build.sh --out /tmp/x --work /tmp/w --jobs 4
```

Requirements: `git`, `cmake` (`brew install cmake`), Xcode Command Line Tools. **No Metal
Toolchain Xcode component** — see "Embedded Metal shaders" below. ~75 s from scratch on an
M1 Max, ~2 s when the clone and the cmake cache are warm (the script is idempotent).

The script fails loudly with an actionable message when a tool is missing, when the pinned
commit cannot be fetched, or when any of the 7 outputs is absent; at the end it prints
every file with its size and checks that every `@rpath` dependency resolves to a sibling in
the output directory.

`build/native` is where `app/engine/whispercpp.py` looks by default (after
`$ROENSPEAK_WHISPER_LIB` and `sys._MEIPASS`).

## The pin

| | |
|---|---|
| Repo | `https://github.com/ggml-org/whisper.cpp` |
| Tag | **`v1.9.5`** |
| Commit | **`d1be6fde11ac6e0407606b4e42fe72d34add8037`** |
| `whisper_version()` | `"1.9.5-dev"` (upstream's own string at that tag) |
| ggml version / commit | `0.27.0` / `d1be6fd` |

The script clones the tag shallowly and then verifies `HEAD == d1be6fde…`, fetching and
checking out that exact commit if the tag ever moves. Bumping the pin means changing both
`WHISPER_TAG` and `WHISPER_COMMIT` in `build.sh`.

## The exact commands

```sh
git clone --depth 1 --branch v1.9.5 https://github.com/ggml-org/whisper.cpp $WORK/whisper.cpp

cmake -S $WORK/whisper.cpp -B $WORK/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DWHISPER_BUILD_EXAMPLES=OFF \
  -DWHISPER_BUILD_TESTS=OFF \
  -DWHISPER_BUILD_SERVER=OFF \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_NATIVE=OFF
cmake --build $WORK/build -j$(sysctl -n hw.ncpu) --config Release

clang++ -x c++ -std=c++17 -O2 -fPIC -fvisibility=hidden -dynamiclib \
  -I $WORK/whisper.cpp/include -I $WORK/whisper.cpp/ggml/include \
  -DRS_WHISPER_BUILD_COMMIT="\"d1be6fd\"" \
  -o $WORK/libroenspeak_whisper.dylib native/whispercpp/roenspeak_whisper.c \
  -L $WORK/build/bin -lwhisper -lggml -lggml-base \
  -Wl,-rpath,@loader_path \
  -install_name @rpath/libroenspeak_whisper.dylib
```

`GGML_METAL` and `GGML_METAL_EMBED_LIBRARY` are already the Apple defaults; they are passed
explicitly so a future upstream default flip cannot silently change what ships.

The shim source is valid C11 but is compiled **as C++** (`-x c++`) on purpose: that enables
the `try`/`catch` guards around `whisper_init_*`, `whisper_full` and result collection.
Upstream only guards model load, so a `std::bad_alloc` out of `whisper_full` would unwind
through an `extern "C"` frame into the Python interpreter and abort the process.

### `-DGGML_NATIVE=OFF` is the default — decision

With upstream's default (`GGML_NATIVE=ON`) cmake selects
`-mcpu=native+dotprod+noi8mm+nosve+nosme` for `libggml-cpu`, i.e. it bakes the **build**
machine's ISA into a binary we ship in a `.dmg` to Macs we never see. A build made on an
M4 (SME/i8mm) can `SIGILL` on an M1. Upstream shipped exactly this bug in its own release
images (*"devops : add `-DGGML_NATIVE=OFF` to CMAKE_ARGS to fix crashes"*, #4111). The cost
of turning it off is nil for RoEnSpeak: inference runs on Metal, and the CPU backend only
handles the front end plus the opt-in "CPU mode (slow)" fallback, where the generic arm64
NEON + Accelerate baseline is what every Apple Silicon Mac has anyway. Verified: with
`GGML_NATIVE=OFF` no `-mcpu=`/`-march=` flag appears in `ggml-cpu.dir/flags.make`, and
`whisper_print_system_info()` still reports `NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 |
DOTPROD = 1 | ACCELERATE = 1 | REPACK = 1`.

`--native` opts back in for local benchmarking. Never use it for a release build.

## Embedded Metal shaders — nothing extra to redistribute

With `GGML_METAL_EMBED_LIBRARY=ON`, cmake concatenates each
`ggml/src/ggml-metal/kernels/*.metal` with its headers and assembles the result straight
into `libggml-metal.dylib` as `ggml_metallib_<name>_{start,end}` symbols (hence its 5.5 MB).
At runtime `ggml-metal-device.m` takes the embedded branch, logs *"using embedded metal
library"* and compiles the 36 MSL sources with the Metal framework in ~18 ms at context
init. `whisper_print_system_info()` reports `MTL : EMBED_LIBRARY = 1`.

There is **no `default.metallib`** in the output and none is looked up at runtime (the build
script fails if one appears). This matters for two reasons:

1. The non-embedded path needs the Metal Toolchain Xcode component
   (`xcodebuild -downloadComponent MetalToolchain`), which Xcode 26 does not install by
   default — that build simply fails with *"cannot execute tool 'metal'"*.
2. Even with it, `ggml_metal_find_metallib()` looks for `default.metallib` as a bundle
   resource or next to `argv[0]` — i.e. next to the *executable*, not next to the dylibs,
   which is the wrong place under PyInstaller.

Compiling the shaders at every context init costs ~18 ms. It is worth it.

## The 7 files and the rpath design

`build.sh` stages exactly these, flat, in one directory:

| file | what it is |
|---|---|
| `libroenspeak_whisper.dylib` | the shim; `-rpath @loader_path` |
| `libwhisper.1.dylib` | whisper.cpp |
| `libggml.0.dylib` | ggml front end |
| `libggml-base.0.dylib` | ggml core |
| `libggml-cpu.0.dylib` | CPU backend (NEON + Accelerate) |
| `libggml-blas.0.dylib` | BLAS backend (Accelerate) |
| `libggml-metal.0.dylib` | Metal backend + embedded shaders |

- Each library's **install name** is `@rpath/<basename>`, and the shim is linked with
  `-Wl,-rpath,@loader_path`. So a plain `ctypes.CDLL(".../libroenspeak_whisper.dylib")`
  resolves all six siblings from the directory the shim itself lives in. **No
  `DYLD_LIBRARY_PATH`, no preloading, no `ggml_backend_load_all()`** — `GGML_BACKEND_DL` is
  off, so the backends are ordinary link-time dependencies and register themselves through
  static initializers as the images load.
- Files are staged under the install-name basenames (`libwhisper.1.dylib`, not
  `libwhisper.1.9.5.dylib`) and as **real files, not cmake's version symlinks**:
  PyInstaller flattens and resolves symlinks inconsistently. `cp -L` does that without
  hardcoding the SO version.
- `libparakeet.<ver>.dylib` is also built (Parakeet ASR model support) and is deliberately
  **not** staged: nothing on the whisper path links it.
- Only OS-provided frameworks are used — `Metal`, `MetalKit`, `Foundation`,
  `CoreFoundation`, `Accelerate`, `libobjc`, `libc++`. Nothing else to bundle, and Metal
  needs no entitlement.
- PyInstaller should carry all 7 together as `binaries` (packaging owns the spec). The
  model `.bin` is data and is downloaded at first run, never bundled.

## The shim API

`roenspeak_whisper.h` is the contract. Rules it keeps so `ctypes` stays safe:

- Only `void*`, `int`, `float`, `double`, `int64_t` and `const char*` (UTF-8) cross the
  boundary. **No struct by value** — `whisper_full_default_params`,
  `whisper_full_get_token_data` and friends are never called from Python.
- `whisper_get_timings()` is never called: it `new`s a struct with no matching free in the
  C API. The shim measures wall-clock itself (`rs_whisper_last_run_ms`,
  `rs_whisper_last_prep_ms` via `encoder_begin_callback`).
- Returned strings are library-owned and valid until the next `rs_whisper_run` /
  `rs_whisper_free` on that handle. `rs_whisper_last_error()` is thread-local.
- `rs_whisper_run`, the `set_opt_*` setters, `reset_opts` and `free` take the handle's
  mutex. The **result accessors do not lock**; they must be called from the thread that ran
  the inference, between runs. RoEnSpeak drives one handle from one inference thread.
- Defaults are dictation-friendly: every `print_*` off, `no_context = true`,
  `temperature = 0`, `temperature_inc = 0` (temperature fallback off, so one bad segment
  cannot silently cost 5x latency), `suppress_blank = true`, greedy `best_of = 1`.
- A `__attribute__((constructor))` installs the `whisper_log_set` / `ggml_log_set` callback
  when the dylib loads, so nothing — not even a backend-registration line from a sibling's
  static initializer — can reach stdout/stderr unfiltered inside a bundled `.app`.
  `rs_whisper_set_log_sink()` (ABI 2) redirects those lines to a caller-supplied function;
  `app/engine/whispercpp.py` points it at the Python logger at DEBUG.
- `rs_whisper_request_abort()` is callable from any thread and makes the in-flight
  `rs_whisper_run` return `-9`; the handle stays usable afterwards.

Return codes: `0` ok, `-1` null handle, `-2` empty PCM, `-3` bad language code, `-9`
`whisper_full` failed or aborted, `-101`…`-104` a contained C++ exception
(`-101` = out of memory).

PCM is **float32, mono, 16000 Hz** in `[-1, 1]`. whisper.cpp has no resampler; the capture
path (`app/audio/capture.py`, soxr) must deliver 16 kHz.
