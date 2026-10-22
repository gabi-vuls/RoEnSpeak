# Building RoEnSpeak

RoEnSpeak builds two packages from one checkout. Both use a PyInstaller **onedir** bundle; everything
after that differs, so this file has two halves and each heading says which platform it belongs to.

| | Windows | macOS |
|---|---|---|
| Output | `dist\RoEnSpeak-Setup.exe` (Inno Setup, per-user) | `dist/RoEnSpeak-<version>.dmg` (drag-install) |
| Engine | faster-whisper / CTranslate2 on CUDA | whisper.cpp on Metal |
| Build command | `powershell -File build\build.ps1` | `bash build/build_macos.sh` |
| Spec | `build/roenspeak.spec` | `build/roenspeak_macos.spec` |
| Lock | `requirements.txt` | `requirements-macos.txt` |
| Package size | 587 MB setup / 1319 MB installed | 71 MB dmg / 185 MB installed |
| What the user needs | an NVIDIA driver | an Apple Silicon Mac |

The two halves share only `build/launch_app.py`, `build/launch_cli.py` and `build/gen_version_info.py`,
so `APP_VERSION` in `app/contracts.py` stays the single source of the version on both.

Windows sections come first (`## … `), then the macOS ones (`## … (macOS)`), in the same order.

## Prerequisites (Windows)

| What | Version / notes |
|---|---|
| Windows 10/11 x64 | The build runs on Windows. WSL is optional (see "WSL mirror workflow"). |
| Python 3.12 x64 | From python.org, with the `py` launcher. 3.12.10 was used. |
| Git | Must be on `PATH`, because pip installs faster-whisper from a local `git+file` URL. |
| faster-whisper clone | `C:\dev\faster-whisper`, containing commit `9e472ae9de8bb374552f531f752654554b76b2d6` (v1.2.1). |
| Inno Setup 6 | Tested with 6.7 (6.5+ recommended: it lifts the 2 GB single-file limit, in case trimming is ever undone). Per-user (`%LOCALAPPDATA%\Programs\Inno Setup 6`) or Program Files install. |
| NVIDIA GPU + driver ≥ 528.33 | Only for the GPU smoke test (`-GpuSmoke`) and for running the app. Not needed to build. |
| signtool.exe (optional) | Windows SDK. Only used when you configure signing. |

```powershell
git clone https://github.com/SYSTRAN/faster-whisper C:\dev\faster-whisper
git -C C:\dev\faster-whisper checkout 9e472ae9de8bb374552f531f752654554b76b2d6
```

`requirements.txt` pins the commit (`faster-whisper @ git+file:///C:/dev/faster-whisper@9e472ae…`). If your
clone is somewhere else, pass `-FasterWhisperRepo <path>`. The build script then rewrites the URL in a
temporary requirements file and keeps the same commit.

## One command (Windows)

From the repo root on Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File build\build.ps1
```

The script runs these steps:

1. Creates `.venv` with `py -3.12` if it is missing, and checks that it is CPython 3.12 64-bit.
2. Runs `pip install -r requirements-dev.txt` (which includes `requirements.txt` and faster-whisper from the clone), then `pip check`.
3. Reads `APP_VERSION` from `app\contracts.py` (via `build\gen_version_info.py --version`).
4. Runs the unit tests: `python -m pytest -q`. pyproject excludes the `gpu`/`mic`/`desktop` markers.
5. Runs PyInstaller: `python -m PyInstaller build\roenspeak.spec --noconfirm --workpath build\work --distpath build\out`.
6. Runs bundle sanity checks: required files exist at the expected paths, no CUDA/driver DLLs sit loose in
   `_internal\`, and `RoEnSpeak-cli.exe --help` starts.
7. Runs the optional GPU smoke test (`-GpuSmoke`): the frozen CLI translates `tests\fixtures\ro_sample.wav` on CUDA
   with isolated config and log dirs, using the real models dir.
8. Runs the optional signing hook (see "Signing").
9. Runs `ISCC /DAppVersion=<APP_VERSION> installer\RoEnSpeak.iss`, which writes `dist\RoEnSpeak-Setup.exe`.
10. Prints the output paths, sizes and timings.

Options: `-SkipInstall`, `-SkipTests`, `-SkipInstaller`, `-Clean` (PyInstaller `--clean`), `-GpuSmoke`,
`-Venv <dir>`, `-Python <python.exe>`, `-FasterWhisperRepo <dir>`, `-Iscc <ISCC.exe>`.

Reference timings (the dev machine, NVMe, warm pip cache):

| Run | Time |
|---|---|
| New venv + install + tests + PyInstaller `--clean` | 95 s |
| Existing venv: whole pipeline | ≈ 2 min |
| PyInstaller step | 22–35 s |
| ISCC step | 76–80 s |

### WSL mirror workflow (optional)

The source of truth is the WSL repo. `scripts/win_sync.sh` rsyncs it to `C:\dev\ro-en-speak`. The rsync uses
`--delete` but leaves `.venv`, `dist`, `build\work` and `build\out` alone. All generated output goes to those
folders.

```bash
build/build_from_wsl.sh                  # sync + build.ps1 (accepts the same options, e.g. -GpuSmoke)
scripts/win.sh -m PyInstaller build/roenspeak.spec --noconfirm --workpath build/work --distpath build/out
"/mnt/c/Users/$USER/AppData/Local/Programs/Inno Setup 6/ISCC.exe" /DAppVersion=0.1.0 'C:\dev\ro-en-speak\installer\RoEnSpeak.iss'
```

Output: `C:\dev\ro-en-speak\dist\RoEnSpeak-Setup.exe` (`/mnt/c/dev/ro-en-speak/dist/…`).

## Files (Windows)

| File | Purpose |
|---|---|
| `build/roenspeak.spec` | PyInstaller spec: one Analysis, two EXEs, one COLLECT (onedir), UPX off |
| `build/launch_app.py` | Entry for the windowed `RoEnSpeak.exe` → `app.main.main()` |
| `build/launch_cli.py` | Entry for the console `RoEnSpeak-cli.exe` → `app.cli.main()` |
| `build/gen_version_info.py` | Builds the Windows version resource from `APP_VERSION`. `--version` prints it; `--text [OUT]` writes the PyInstaller text form (default `build/work/version_info.txt`) for review |
| `build/build.ps1` | The one-command build |
| `build/build_from_wsl.sh` | WSL wrapper: sync, then `build.ps1` |
| `installer/RoEnSpeak.iss` | Inno Setup script |

**Version.** `app/contracts.py` `APP_VERSION` is the only place the version is set. The spec embeds it in both EXEs
(FileVersion/ProductVersion, plus OriginalFilename/Description per EXE). `build.ps1` passes it to ISCC as
`/DAppVersion`, which sets AppVersion and the Setup.exe VersionInfo. There is no checked-in version file to keep
in sync. Both EXEs use `assets/icons/app.ico` (16–256 px).

## Bundle layout and DLL bundling (Windows)

```
RoEnSpeak\
  RoEnSpeak.exe            windowed tray app
  RoEnSpeak-cli.exe        console .wav test mode (same _internal)
  _internal\
    assets\                icons + sounds          (app/ui/assets.py: sys._MEIPASS/assets)
    faster_whisper\assets\silero_vad_v6.onnx
    ctranslate2\           ctranslate2.dll, libiomp5md.dll, cudnn64_9.dll (CT2's dispatcher), _ext.pyd
    nvidia\cublas\bin\     cublas64_12.dll, cublasLt64_12.dll
    nvidia\cudnn\bin\      cudnn64_9.dll, cudnn_graph64_9.dll, cudnn_ops64_9.dll
    nvidia\cuda_nvrtc\bin\ nvrtc64_120_0.dll, nvrtc-builtins64_129.dll
    onnxruntime\capi\      onnxruntime.dll (Silero VAD on CPU)
    _sounddevice_data\portaudio-binaries\libportaudio64bit.dll
    PySide6\               Qt6Core/Gui/Widgets (+Svg/Network pulled in by plugins), platforms/styles/imageformats
    av\, av.libs\, tokenizers\, soxr\, hf_xet\, numpy\, …
```

- **CUDA DLL paths match `app/engine/cuda_probe.py`.** In a frozen build it looks in `sys._MEIPASS\nvidia\<pkg>\bin`
  (= `_internal\nvidia\…`) for `cublas`, `cudnn` and `cuda_nvrtc`. It adds those dirs with `os.add_dll_directory`
  and also prepends them to `PATH`, because the cuDNN dispatcher loads its sub-libraries by bare name.
- The spec appends the nvidia DLLs to `a.datas` **after** Analysis. PyInstaller then copies them verbatim: no
  binary-dependency analysis, and no duplicate copies of the 100–650 MB DLLs in `_internal\`. A post-Analysis
  filter drops any `cublas*/cudnn*/nvrtc*` DLL that PyInstaller collected anywhere else, except
  `ctranslate2\cudnn64_9.dll`. The same filter drops driver DLLs (`nvml.dll`, `nvcuda.dll`), which the ctypes scanner could
  otherwise copy from the build machine's System32. The driver provides those at runtime.
- **Qt.** The app imports only QtCore/QtGui/QtWidgets. The spec excludes every other PySide6 module, plus Qt
  translations (the UI is English), `opengl32sw.dll`, and the qml/sql/designer/tls/networkinformation plugins.
  Bundled PySide6 totals ≈ 46 MB.
- **Also excluded:** tkinter, pytest, PyInstaller/setuptools/pip, `ctranslate2.converters`, and the onnxruntime
  transformers/quantization/tools subpackages.
- `huggingface_hub` resolves names lazily, so its submodules are hidden imports. The cli, webhooks, torch and
  tensorflow parts are left out.

## Trimming (CUDA DLLs, Windows)

| | Untrimmed | Shipped |
|---|---|---|
| `_internal\nvidia` | 2010 MB | **1025 MB** |
| Whole onedir bundle | ≈ 2.3 GB | **1319 MB** |
| `RoEnSpeak-Setup.exe` (LZMA2 max, solid) | — | **587 MB** |
| Installed size | — | 1323 MB |

The spec's `NVIDIA_TRIM` set removes about 985 MB:

| Removed | Size | Why it is safe |
|---|---|---|
| `nvblas64_12.dll` | 0.3 MB | BLAS interposer for CPU BLAS callers. Nothing in the bundle uses it. |
| `cudnn_adv64_9`, `cudnn_cnn64_9`, `cudnn_engines_precompiled64_9`, `cudnn_engines_runtime_compiled64_9`, `cudnn_engines_tensor_ir64_9`, `cudnn_ext64_9`, `cudnn_heuristic64_9` | 895 MB | Inference never calls cuDNN, see below. |
| `nvrtc64_120_0.alt.dll` | 90 MB | Alternate NVRTC build. No bundled DLL references it by name. |

**Evidence**

1. `ctranslate2.dll` (CT2 4.8.3, Windows wheel) imports only `libiomp5md`, the MSVC runtime and kernel32 (checked with
   pefile, including delay-imports). It contains **no** `cudnn` string at all (case-insensitive byte scan). It loads
   `cublas64_12.dll` dynamically, by name. CT2's Whisper conv1d uses its own CUDA kernels, so no code path depends on
   the GPU architecture.
2. A module trace of the frozen CLI during a GPU run showed these CUDA DLLs loaded: `cublas64_12`, `cublasLt64_12`,
   `cudnn64_9` (both copies), `cudnn_graph64_9` and `cudnn_ops64_9`. The last two come from `cuda_probe`'s explicit
   load check. No other cuDNN DLL and no NVRTC DLL was loaded.
3. The A/B test below ran the frozen `RoEnSpeak-cli.exe` on `tests\fixtures\ro_sample.wav` (RTX 4060 Ti, driver
   610.88), first with the full set and then with the trimmed files physically removed. The text was **identical**
   in all four cases, with exit 0 and `device=cuda` in each. Beam 1 and `--task transcribe` also passed on the trimmed bundle.

| model / compute | text (both runs) | infer ms full → trimmed |
|---|---|---|
| large-v3 float16 | Hello! \| Today the weather is beautiful and I want to go to the park. \| Thank you for your help! | 858 → 854 |
| large-v3 int8_float16 | (same as above) | 763 → 798 |
| medium float16 | Good evening. \| Today the weather is beautiful and I want to go to the park. \| Thanks for your help! | 498 → 504 |
| medium int8_float16 | Good evening. \| Today the weather is beautiful and I want to go to the park. \| Thank you for your help | 448 → 522 |

**Kept on purpose**

- `cudnn64_9.dll`, `cudnn_graph64_9.dll` and `cudnn_ops64_9.dll` (≈ 206 MB). `cuda_probe` loads them to verify cuDNN
  and report its version at startup, as the SPEC requires.
- `nvrtc64_120_0.dll` and `nvrtc-builtins64_129.dll` (93 MB). `cublasLt64_12.dll` references NVRTC by name and may
  JIT-compile kernels on GPUs or paths we have not tested.
- The CUDA DLLs themselves: cuBLAS and cuBLASLt (736 MB) are required.

**Re-validate after any ctranslate2 / nvidia-\* upgrade.**

1. Build with `ROENSPEAK_NO_TRIM=1` and run the four-way matrix:
   `RoEnSpeak-cli.exe tests\fixtures\ro_sample.wav --model {large-v3,medium} --compute-type {float16,int8_float16} --json`.
2. Delete the `NVIDIA_TRIM` files from `build\out\RoEnSpeak\_internal\nvidia` and rerun the matrix.
3. Compare the texts. Re-check `ctranslate2.dll`'s imports and strings for `cudnn`.

## Installer behaviour (`installer/RoEnSpeak.iss`, Windows)

- **Install mode.** Per-user by default (`PrivilegesRequired=lowest`), so no UAC prompt. Files go to
  `%LOCALAPPDATA%\Programs\RoEnSpeak`. The wizard offers an "all users" choice (`/ALLUSERS`), which needs admin
  and installs to Program Files.
- **Shortcuts and launch.** Setup creates a Start Menu shortcut and an optional desktop shortcut (unchecked task
  `desktopicon`). The finish page has a "Launch RoEnSpeak" checkbox, which runs `RoEnSpeak.exe --minimized`.
  The checkbox is skipped in silent mode.
- **Compression.** `lzma2/max`, solid, in a separate process with 6 block threads and a 64 MB dictionary.
  Compiling takes about 80 s. Installing takes about 30 s.
- **Upgrades** delete `{app}\_internal` first, so files dropped from a newer build don't linger.
- **Running app.** `AppMutex=Local\RoEnSpeak.SingleInstance` matches `contracts.single_instance_name()`. `Local\` is
  the per-session namespace, so this detects the user's instance in the current session. Isolated test instances
  started with `ROENSPEAK_CONFIG_DIR` use a suffixed name and are not detected; Restart Manager
  (`CloseApplications=yes`) still covers their locked files.

  `InitializeSetup` and `InitializeUninstall` run before Inno's own mutex check (verified). They ask "RoEnSpeak is
  running. It will be closed…" (OK/Cancel; `/SUPPRESSMSGBOXES` answers OK). They then run `taskkill /IM RoEnSpeak.exe`,
  wait 3 s, and finish with `taskkill /F /T`.
- **Uninstall.**
  1. Removes the `HKCU\…\Run\RoEnSpeak` autostart value, but only if it points into this install dir.
  2. Asks **once** whether to delete the downloaded models **and logs** in `%LOCALAPPDATA%\RoEnSpeak` (it shows the
     size). The default is **No**, and `/SUPPRESSMSGBOXES` also answers No.
  3. Always keeps the settings in `%APPDATA%\RoEnSpeak\config.json` (a few KB), so a reinstall keeps them. Delete
     them by hand to reset.
- **Silent install and uninstall:**
  ```
  RoEnSpeak-Setup.exe /VERYSILENT /SUPPRESSMSGBOXES /CURRENTUSER /NORESTART [/DIR=<dir>] [/TASKS="desktopicon"] [/LOG=<file>]
  "<dir>\unins000.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART [/LOG=<file>]
  ```
  `unins000.exe` returns immediately (it relaunches itself from %TEMP%). Wait for `<dir>` to disappear.

> **macOS has no installer script.** A `.dmg` is a drag-install, so there is nothing to run on
> install; the only thing it cannot do is uninstall, which `installer/macos/uninstall.sh` covers
> step for step against the list above. See `installer/macos/README.md`.

## Signing (optional hook, Windows)

When `SIGNTOOL_CERT` is not set, the build prints `signing skipped …` and produces unsigned files. Windows
SmartScreen may then warn ("Windows protected your PC" → More info → Run anyway). To sign, set:

| Variable | Meaning |
|---|---|
| `SIGNTOOL_CERT` | Path to a `.pfx`, **or** the 40-hex SHA1 thumbprint of a cert in the Windows cert store |
| `SIGNTOOL_PASSWORD` | `.pfx` password (optional). It is visible in process command lines, so prefer a cert-store thumbprint. |
| `SIGNTOOL_PATH` | `signtool.exe`. Default: `PATH`, then the newest `Windows Kits\10\bin\*\x64\signtool.exe` |
| `SIGNTOOL_TIMESTAMP_URL` | RFC 3161 server. Default `http://timestamp.digicert.com` |

The build signs `RoEnSpeak.exe` and `RoEnSpeak-cli.exe` before packaging. It then hands ISCC a sign tool
(`/Sroensign=… $f` + `/DSignToolName=roensign`), which signs `RoEnSpeak-Setup.exe` and the uninstaller
(`SignedUninstaller=yes`). All signatures use sha256 with an RFC 3161 timestamp.

## Verifying a build (Windows, what was run for 0.1.0)

1. **Silent install** to a temp dir: `/VERYSILENT /SUPPRESSMSGBOXES /CURRENTUSER /DIR=C:\dev\pkgtest\RoEnSpeak /TASKS=""`.
   It finished with exit 0 in 29.5 s, 1323 MB on disk, with a Start Menu shortcut, no desktop shortcut, and an HKCU
   Uninstall key showing DisplayVersion 0.1.0.
2. **Installed CLI on GPU.** Run with `ROENSPEAK_DATA_DIR`/`ROENSPEAK_CONFIG_DIR` pointing at temp dirs, so the
   user's logs and config stay untouched, and `ROENSPEAK_MODELS_DIR=%LOCALAPPDATA%\RoEnSpeak\models`. Command:
   `RoEnSpeak-cli.exe tests\fixtures\ro_sample.wav --json`.
   - Result: exit 0, `device=cuda compute_type=float16`, large-v3 load 2.5 s, all 3 segments translated.
   - Infer time per segment: 231/375/273 ms. With `--realtime`, VAD-close→text latency was 244/564/479 ms.
   - The log contains `cuda_probe ok=1 … cublas=12.9.2 cudnn=9.27.0` and
     `engine_loaded device=cuda compute_type=float16 model=large-v3 …`.
3. **Installed GUI app.** `RoEnSpeak.exe` ran with `ROENSPEAK_CONFIG_DIR` pointing at a config containing
   `{"armed": false, "ui": {"first_run_done": true}}`.
   - It stayed up for 20 s and logged `starting (… frozen=True)`, `ui started minimized=True tray=True` and
     `cuda_probe ok=1`. It loaded and warmed up large-v3 on CUDA.
   - No hotkey was registered. After it was terminated, no child or orphan processes remained.
4. **Version resource and icon.** `(Get-Item RoEnSpeak.exe).VersionInfo` shows FileVersion/ProductVersion 0.1.0 and
   ProductName RoEnSpeak. Both EXEs have an RT_GROUP_ICON with the same 7 sizes as `app.ico`.
5. **Upgrade while running.** The app ran with the real mutex name (`APPDATA` redirected to a temp dir), and a silent
   setup was run over it. Setup closed the app and exited 0.
6. **Silent uninstall while running.** Setup closed the app (exit 0). Afterwards the install dir, Start Menu shortcut,
   Uninstall key and test HKCU Run value were all gone. `Defaulting to No … delete the downloaded speech models` was
   logged, and the real models dir was byte-for-byte unchanged (same file list, sizes and mtimes).

## Troubleshooting (Windows)

- **`ModuleNotFoundError: No module named 'pefile'` when running PyInstaller.** The venv was populated with
  `--no-deps`. Run `pip install -r requirements-dev.txt` normally (build.ps1 does), or `pip install pefile`.
- **`faster-whisper clone not found` / pip can't check out the commit.** Clone it to `C:\dev\faster-whisper` and make
  sure commit `9e472ae…` exists (`git -C C:\dev\faster-whisper fetch`), or pass `-FasterWhisperRepo`.
- **Bundle check fails with "CUDA/driver DLLs at the top of _internal".** A hook started collecting nvidia DLLs as
  binaries. Extend the post-Analysis filter in the spec, rather than shipping duplicates.
- **`RoEnSpeak-cli.exe` exits 2 (CUDA unavailable) on a test machine.** Check `cuda_probe … error=` in
  `%LOCALAPPDATA%\RoEnSpeak\logs\roenspeak.log`. Typical causes: the driver is older than 528.33, there is no NVIDIA
  GPU, or a trimmed DLL turned out to be needed. Rebuild with `ROENSPEAK_NO_TRIM=1` to rule out the last one.
- **ISCC "Preprocessing error … 'BEGIN' expected".** A `{ … }` Pascal comment in `[Code]` contains `{app}` or another
  brace. Use `//` comments there.
- **Setup reports that RoEnSpeak is running and can't close it.** Quit it from the tray icon, then press Retry.
  Instances in other user sessions are not touched.
- **SmartScreen warning.** Expected for unsigned builds. Configure signing (above).
- **Slow ISCC.** Lower `LZMANumBlockThreads` if RAM is short, or switch to `Compression=lzma2/normal` for faster test
  builds (the setup gets a bit larger).

---

# macOS

## Prerequisites (macOS)

| What | Version / notes |
|---|---|
| **Apple Silicon** Mac | M1 or newer. The build is **arm64-only** (see "target_arch" below); it does not run on Intel Macs. Verified on an M1 Max, macOS 26.6.2 (25G83). |
| macOS 14 or newer to build | `LSMinimumSystemVersion` comes from `contracts.MIN_MACOS_VERSION` (14.0). The APIs the macOS layer uses (Metal 3, Carbon hot keys, CGEvent) would run on 13.0, but the PySide6-Essentials 6.12, onnxruntime 1.31 and numpy 2.5 wheels in `requirements-macos.txt` are all tagged `macosx_14_0`, so 14.0 is the real floor for a built `.app`. |
| Python 3.12 **arm64** | From python.org or `brew install python@3.12`. `build_macos.sh` refuses anything else: `platform.machine()` must be `arm64`. 3.12.9 was used. |
| Xcode Command Line Tools | `xcode-select --install`. Provides `clang++`, `codesign`, `otool`, `install_name_tool`, `hdiutil`. Apple clang 17.0.0 (Xcode 26.2) was used. |
| CMake | `brew install cmake`. 4.4.4 was used. Needed only by `native/whispercpp/build.sh`. |
| Git | Needed by `native/whispercpp/build.sh` to clone whisper.cpp. |
| Developer ID + `notarytool` (optional) | Only to produce a `.dmg` that Gatekeeper opens without a right-click. See "Signing and notarization (macOS)". |

**The Metal Toolchain Xcode component is deliberately not required.** The native build uses
`-DGGML_METAL_EMBED_LIBRARY=ON`, which embeds the 36 `.metal` shader *sources* into
`libggml-metal.dylib` and compiles them with the Metal framework at runtime. There is no
`default.metallib` to ship, which is exactly what makes the Metal engine work under PyInstaller —
the non-embedded build looks for `default.metallib` next to `argv[0]`, which is `Contents/MacOS`,
not where the dylibs live. The non-embedded build also fails outright on Xcode 26 without
`xcodebuild -downloadComponent MetalToolchain`.

**whisper.cpp pin.** There is no faster-whisper clone to prepare, because the engine is not a
Python package on macOS:

| | |
|---|---|
| Repo | `https://github.com/ggml-org/whisper.cpp` |
| Tag | **`v1.9.5`** (tag object `d2a764c648779feb36d44363f212a29b00aad1af`) |
| Commit | **`d1be6fde11ac6e0407606b4e42fe72d34add8037`** |
| cmake flags | `-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON -DWHISPER_BUILD_{EXAMPLES,TESTS,SERVER}=OFF` |
| Shim | `native/whispercpp/roenspeak_whisper.c` → `libroenspeak_whisper.dylib`, linked `-Wl,-rpath,@loader_path` |

`native/whispercpp/build.sh` owns the pin and the flags; the spec only consumes the 7 dylibs it
stages. `-DGGML_NATIVE=OFF` is the default there on purpose: with `ON`, cmake bakes the *build*
machine's ISA into `libggml-cpu` (`-mcpu=native`), and a binary built on an M4 could fault on an M1.

## One command (macOS)

From the repo root:

```sh
bash build/build_macos.sh
```

The script runs these steps, mirroring `build.ps1`:

1. Creates `.venv` with `python3.12` if missing, and checks that it is **CPython 3.12 arm64**.
2. Runs `pip install -r requirements-macos-dev.txt`, then `pip check`, then **fails** if
   `faster_whisper` or `ctranslate2` turn out to be importable (they must never be on macOS).
3. Reads `APP_VERSION` from `app/contracts.py` (via `build/gen_version_info.py --version`).
4. Runs the unit tests: `python -m pytest -q`. pyproject excludes the
   `gpu`/`mic`/`desktop`/`metal` markers.
5. Runs `bash native/whispercpp/build.sh --out build/native` (idempotent: it reuses the clone and
   the cmake cache, so a second run takes seconds), then checks that all 7 dylibs are there.
6. Runs PyInstaller:
   `python -m PyInstaller build/roenspeak_macos.spec --noconfirm --workpath build/work --distpath build/out`.
7. Runs bundle sanity checks (below).
8. Runs the optional Metal smoke test (`--gpu-smoke`).
9. **Code-signs the bundle — always**, ad-hoc by default. See "Signing and notarization (macOS)".
10. Runs `bash build/make_dmg.sh`, which writes `dist/RoEnSpeak-<version>.dmg`.
11. Optionally notarizes and staples the `.dmg` (`ROENSPEAK_NOTARY_PROFILE`).
12. Prints the output paths and sizes.

Options, named after `build.ps1`'s switches: `--skip-install`, `--skip-tests`, `--skip-native`,
`--skip-dmg`, `--clean` (PyInstaller `--clean` plus a clean native build), `--gpu-smoke`,
`--venv <dir>`, `--python <path>`.

The **bundle sanity checks** are the counterpart of `build.ps1`'s `$required` list, plus three
macOS-specific ones:

- both executables, `Contents/Info.plist`, `Contents/Resources/app.icns`,
  `assets/icons`, `assets/sounds`, `assets/vad/silero_vad_v6.onnx`, all 7 whisper.cpp dylibs,
  `_sounddevice_data/portaudio-binaries/libportaudio.dylib`,
  `onnxruntime/libonnxruntime*.dylib`, `PySide6/Qt/plugins/platforms/libqcocoa.dylib`;
- **no whisper/ggml dylib below the top of `Contents/Frameworks`** (the analogue of "no CUDA DLLs
  loose in `_internal\`"), and every `@rpath/…` dependency of the 7 resolves to a sibling, and each
  of them that has such a dependency carries an `@loader_path` `LC_RPATH`;
- `Info.plist` really contains `CFBundleIdentifier=com.roenspeak.RoEnSpeak`,
  `CFBundleExecutable=RoEnSpeak`, the right version, `LSUIElement=true`, `LSBackgroundOnly=false`,
  a non-empty `NSMicrophoneUsageDescription` and an `LSMinimumSystemVersion`;
- `RoEnSpeak-cli --help` exits 0, both before and **after** re-signing.

Reference timings (M1 Max, warm caches):

| Run | Time |
|---|---|
| `--skip-install --skip-tests --skip-native` (PyInstaller + checks + sign + dmg) | 31 s |
| PyInstaller step | 22 s |
| `native/whispercpp/build.sh` from scratch (clone + cmake + shim) | ≈ 30 s |
| `hdiutil` step | 7 s |

## Files (macOS)

| File | Purpose |
|---|---|
| `build/roenspeak_macos.spec` | PyInstaller spec: one Analysis, two EXEs, one COLLECT, one `BUNDLE` → `RoEnSpeak.app`. UPX off, `target_arch="arm64"` |
| `build/launch_app.py` | Shared with Windows. Entry for the windowed `RoEnSpeak` → `app.main.main()` |
| `build/launch_cli.py` | Shared with Windows. Entry for the console `RoEnSpeak-cli` → `app.cli.main()` |
| `build/gen_version_info.py` | Shared with Windows. `--version` is what feeds `CFBundleShortVersionString`/`CFBundleVersion`; the `VSVersionInfo` half is simply unused here |
| `build/build_macos.sh` | The one-command build |
| `build/make_dmg.sh` | `hdiutil` packaging, callable on its own |
| `build/entitlements.plist` | The two entitlements the app needs, with the rationale inline |
| `native/whispercpp/build.sh` | Clones + builds whisper.cpp with Metal and the C shim; stages the 7 dylibs into `build/native` |
| `installer/macos/uninstall.sh` | The uninstall half of `installer/RoEnSpeak.iss`, which a `.dmg` cannot provide |
| `installer/macos/README.md` | What the drag-install covers and what the script covers |
| `requirements-macos.in` / `.txt` / `-dev.txt` | Provenance, the pinned lock, and the build/test extras |

**Version.** Unchanged from Windows: `app/contracts.py` `APP_VERSION` is the only place it is set.
`build/gen_version_info.py` is imported by both specs, so there is no second (or third) copy. The
macOS spec puts it in `CFBundleShortVersionString` *and* `CFBundleVersion`, and `build_macos.sh`
asserts both in the finished `Info.plist` and uses it in the `.dmg` file name and volume name.
Both executables use `assets/icons/app.icns`.

**`target_arch = "arm64"`, not `universal2`.** This is forced by the wheels, not chosen:

| Package | macOS wheel tag |
|---|---|
| `onnxruntime` 1.31.0 | `macosx_14_0_arm64` — no x86_64, no universal2 |
| `soxr` 1.1.0 | `macosx_11_0_arm64` — no x86_64, no universal2 |
| `numpy` 2.5.4 | `macosx_14_0_arm64` — no x86_64, no universal2 |
| `PySide6-Essentials` / `shiboken6` 6.12.0 | `macosx_14_0_universal2` — thinned to arm64 at build time |
| the venv interpreter | arm64-thin (`lipo -archs .venv/bin/python` → `arm64`) |

There is no x86_64 slice to put in a fat binary for three of the dependencies, and no arm64 Python
would run it anyway. whisper.cpp points the same way: Metal is the whole point of this port.
**Consequence: the `.app` does not run on Intel Macs**, and the README says so.

## Bundle layout (macOS)

```
RoEnSpeak.app/
  Contents/
    Info.plist                     22 keys; see the table below
    _CodeSignature/                ad-hoc (or Developer ID) seal over everything
    MacOS/
      RoEnSpeak                    windowed menu-bar app   (CFBundleExecutable)
      RoEnSpeak-cli                console .wav test mode  (same PYZ, same Frameworks)
    Resources/
      app.icns
      assets/                      icons + sounds + vad/silero_vad_v6.onnx   (1.9 MB, the real files)
      base_library.zip
      … every data file, cross-linked from Contents/Frameworks …
    Frameworks/                    == sys._MEIPASS
      libroenspeak_whisper.dylib   the ctypes shim           56 KB
      libwhisper.1.dylib                                    428 KB
      libggml.0.dylib                                        60 KB
      libggml-base.0.dylib                                  774 KB
      libggml-cpu.0.dylib                                   902 KB
      libggml-blas.0.dylib                                   59 KB
      libggml-metal.0.dylib        Metal shaders embedded    5.5 MB
      assets -> ../Resources/assets
      onnxruntime/libonnxruntime.1.31.0.dylib   Silero VAD on CPU   70 MB
      _sounddevice_data/portaudio-binaries/libportaudio.dylib
      PySide6/Qt/lib/Qt{Core,Gui,Widgets,DBus,Network,Svg}.framework   48 MB total
      PySide6/Qt/plugins/{platforms,styles,imageformats,iconengines,platforminputcontexts}/
      Python.framework/, python3.12/, numpy/, soxr/, hf_xet/, yaml/, …
```

185 MB installed, 71 MB as a `.dmg`. The three big items are onnxruntime (70 MB), PySide6 (48 MB)
and the Python framework + stdlib; the speech engine is only 7.5 MB because the model is downloaded
at first run, not bundled.

**Why the 7 dylibs are flat at the top of `Contents/Frameworks`.** `sys._MEIPASS` is
`Contents/Frameworks` for a PyInstaller 6.x `.app`, and that is candidate #2 in
`app/engine/whispercpp.py`'s lookup order (`$ROENSPEAK_WHISPER_LIB` → `sys._MEIPASS` →
`<repo>/build/native`). More importantly, PyInstaller rewrites every collected binary's dependent
paths to `@rpath/<basename>`, drops its existing `LC_RPATH`s and adds one pointing at the bundle
root relative to the file (`PyInstaller/building/utils.py`, `process_collected_binary`). For a
top-level destination that rpath is exactly `@loader_path`, which is what the shim was linked with,
so all six siblings resolve with **no `DYLD_LIBRARY_PATH`, no preloading and no
`ggml_backend_load_all()`**. A subdirectory would get `@loader_path/..` instead and the siblings
would be looked for one level too high. Verified on the built bundle:

```
$ otool -l Contents/Frameworks/libroenspeak_whisper.dylib | grep -A2 LC_RPATH | grep path
         path @loader_path (offset 12)
$ otool -L Contents/Frameworks/libroenspeak_whisper.dylib | grep @rpath
	@rpath/libroenspeak_whisper.dylib
	@rpath/libwhisper.1.dylib
	@rpath/libggml.0.dylib
	@rpath/libggml-base.0.dylib
```

`libggml-base.0.dylib` is the one of the seven with **no** `LC_RPATH`, and that is correct: it has
no dependencies beyond its own install name, so there is nothing for an rpath to resolve. The
sanity check only demands an `@loader_path` rpath from dylibs that actually load something through
`@rpath`.

There is **no `.metallib`** anywhere in the bundle, and none is looked up at runtime
(`find RoEnSpeak.app -name '*.metallib'` → nothing). `libggml-metal.dylib` is 5.5 MB because the
shaders are inside it.

**No faster-whisper data.** The Windows spec collects
`faster_whisper/assets/silero_vad_v6.onnx`; here the package is not installed at all, and the same
ONNX model is vendored at `assets/vad/silero_vad_v6.onnx`, so it arrives as part of the one
`assets/` tree. `app/ui/assets.py`'s `assets_dir()` is unchanged on both platforms.

**Qt trimming** is the same aggressive list as Windows (every PySide6 module except
QtCore/QtGui/QtWidgets, plus Qt translations and the qmltooling/sqldrivers/designer/tls/
networkinformation/generic/vectorimageformats plugin dirs), adapted in two ways: the prefix is
`PySide6/Qt/plugins/…` on macOS rather than `PySide6/plugins/…`, and the Windows-only
`opengl32sw.dll` / `d3dcompiler_47.dll` entries are gone. The `/tests/*.py` drop is unchanged.
`faster_whisper`, `ctranslate2`, `tokenizers`, `av`, `pynvml` and `nvidia` are in `excludes` so a
dev machine that happens to have them can never leak them into a macOS bundle.

**`Info.plist`** keys that are not boilerplate:

| Key | Value | Why |
|---|---|---|
| `LSUIElement` | `true` | Menu-bar-only app: no Dock icon, no app-switcher entry. The macOS equivalent of the Windows tray-only app. Windows may still be shown. |
| `LSBackgroundOnly` | `false` | **Must be set explicitly.** `BUNDLE` inherits `console` from `COLLECT` and sets `LSBackgroundOnly=True` when it is true, and a background-only app can never be activated or show the main window. |
| `NSMicrophoneUsageDescription` | one sentence | Without it the TCC prompt never appears and the microphone silently fails, with no way for the user to grant access. |
| `LSMinimumSystemVersion` | `contracts.MIN_MACOS_VERSION` (14.0) | Set by the pinned wheels, not by the APIs the app uses. See "Prerequisites (macOS)". |
| `NSHighResolutionCapable` | `true` | Retina. |
| `NSSupportsAutomaticTermination` | `false` | The app owns a resident model, a Carbon hotkey registration and an audio stream; macOS must not terminate it because it looks idle. |
| `NSSupportsSuddenTermination` | `false` | Same, for `SIGKILL` at logout: the Metal working set and the hotkey need releasing. |
| `NSAppSleepDisabled` | `true` | An `LSUIElement` app with no visible window is a prime App Nap candidate, and App Nap throttles timers and lowers thread priority — missed hotkeys, stalled inference. |
| `LSMultipleInstancesProhibited` | `true` | Only one process may own the global hotkey. `app/macos/instance.py` enforces it with an `flock`; this stops Launch Services starting a second copy from Finder in the first place. |
| `NSRequiresAquaSystemAppearance` | `false` | Follow the system appearance (Qt handles dark mode). |
| `LSApplicationCategoryType` | `public.app-category.productivity` | Required metadata if the bundle is ever notarized. |
| `NSHumanReadableCopyright` | from `gen_version_info.COPYRIGHT` | Shown in Finder's Get Info. |

Keys deliberately **absent**: `NSAppleEventsUsageDescription` (the app sends no Apple Events —
keystrokes go through `CGEventCreateKeyboardEvent`, which is also why the spec sets
`argv_emulation=False`), and anything App-Sandbox-related.

**There is no Info.plist key for Accessibility.** The Accessibility grant is pure TCC: the app calls
`AXIsProcessTrusted()` / `AXIsProcessTrustedWithOptions()` and macOS lists it in
System Settings → Privacy & Security → Accessibility. What it needs from the build is a stable
bundle identifier and a code signature — hence the next section.

## Signing and notarization (macOS)

`build.ps1` leaves the Windows executables unsigned unless you configure `SIGNTOOL_CERT`.
**`build_macos.sh` always signs**, because on macOS a signature is not cosmetic:

1. An arm64 Mach-O that has been modified and left unsigned **will not load at all** — the kernel
   kills it. PyInstaller already ad-hoc signs each binary it processes for exactly this reason, and
   the final `codesign` re-seals them after the `Info.plist` and the dylib load commands are
   rewritten.
2. **TCC records its grants against the code signature, not the path.** Accessibility and
   Microphone are what this app lives on, and a bundle with no signature has nothing for TCC to
   remember.

So the only question is *with what*:

| | Default | `ROENSPEAK_CODESIGN_IDENTITY` set |
|---|---|---|
| Identity | ad-hoc (`-`) | your Developer ID Application certificate |
| Command | `codesign --force --deep --sign - --options runtime --entitlements build/entitlements.plist --timestamp=none` | the same with `--sign "$ROENSPEAK_CODESIGN_IDENTITY"` and a real `--timestamp` |
| `codesign --verify --strict` | passes | passes |
| `spctl --assess` | **rejected** | rejected until notarized, then accepted |
| Gatekeeper on another Mac | right-click → Open, or `xattr -dr com.apple.quarantine` | clean double-click once notarized |
| TCC grants across rebuilds | **lost on every rebuild** | kept |

Environment variables (the `SIGNTOOL_*` analogues):

| Variable | Meaning |
|---|---|
| `ROENSPEAK_CODESIGN_IDENTITY` | Identity name or SHA-1 hash from `security find-identity -v -p codesigning`. Unset ⇒ ad-hoc. |
| `ROENSPEAK_NOTARY_PROFILE` | Keychain profile created with `xcrun notarytool store-credentials`. The build then runs `notarytool submit --wait` on the `.dmg` and `stapler staple`. Ignored (with a message) when no real identity is set, because Apple does not notarize ad-hoc signatures. |
| `ROENSPEAK_NATIVE_DIR` | Where the 7 dylibs live. Default `build/native`. |

**Be honest about ad-hoc and TCC.** An ad-hoc signature's designated requirement is a *cdhash*,
which changes every time anything in the bundle changes. So a rebuild still invalidates the
Accessibility and Microphone grants and macOS asks again — ad-hoc is strictly better than unsigned
(the bundle loads, the identifier is stable, `codesign --verify` passes, and the grant survives
*moving* the app), but only a Developer ID signature, whose requirement is
`identifier + team id`, makes grants survive rebuilds. If you are iterating on the app and tired of
re-granting Accessibility, sign with a real identity.

**`--deep` is the right tool here and only here.** Apple discourages it for distribution signing,
because each nested binary should normally get its own entitlements. We have exactly one
entitlement set for the whole bundle and several hundred nested Mach-O files that PyInstaller
already signed individually, so `--deep` re-seals them in one pass with nothing to lose.

**Entitlements** (`build/entitlements.plist`), and the hardened runtime:

- `com.apple.security.device.audio-input` — microphone under the hardened runtime.
- `com.apple.security.cs.disable-library-validation` — without it, the hardened runtime refuses to
  load code that is not signed by the app's own Team ID (or Apple's). RoEnSpeak `dlopen`s the seven
  whisper/ggml dylibs, Qt's plugins, onnxruntime and PortAudio; in an ad-hoc build none of them
  carry a Team ID at all, and the Metal engine would fail to load.
- **No `com.apple.security.app-sandbox`, ever.** A sandboxed app cannot hold an Accessibility
  grant, and Accessibility is what lets RoEnSpeak post `CGEvent`s into other applications. The
  sandbox would break text injection outright, which is also why RoEnSpeak can never ship on the
  Mac App Store.
- No `allow-jit` / `allow-unsigned-executable-memory`: CPython 3.12 has no JIT and ggml's Metal
  shader compilation goes through the framework's out-of-process compiler service. Verified: the
  engine runs on Metal with the hardened runtime on and neither entitlement present.
- `--options runtime` does **not** interfere with TCC. A hardened-runtime app appears in
  System Settings → Privacy & Security → Accessibility and behaves like any other.

**Editing `build/entitlements.plist`:** `codesign` parses it with `AMFIUnserializeXML`, which is
much stricter than `plutil -lint`. It rejects a comment before the root element, angle brackets
inside a comment, and a doubled hyphen inside a comment. `plutil -lint` passes all three; `codesign`
fails with `AMFIUnserializeXML: syntax error near line N`. Always re-run the build after editing it.

## Verifying a build (macOS)

What was actually run for 0.1.0 on an M1 Max (macOS 26.6.2, Xcode 26.2, cmake 4.4.4, CPython
3.12.9 arm64):

1. **`bash build/build_macos.sh --skip-install --skip-tests --skip-native`** → 31 s, exit 0.
   `build/out/RoEnSpeak.app` 185 MB, `dist/RoEnSpeak-0.1.0.dmg` 71 MB.
2. **Signature.**
   ```
   $ codesign --verify --strict --verbose=2 build/out/RoEnSpeak.app
   build/out/RoEnSpeak.app: valid on disk
   build/out/RoEnSpeak.app: satisfies its Designated Requirement

   $ codesign -dv --verbose=4 build/out/RoEnSpeak.app
   Identifier=com.roenspeak.RoEnSpeak
   Format=app bundle with Mach-O thin (arm64)
   CodeDirectory v=20500 ... flags=0x10002(adhoc,runtime) hashes=569+7
   Signature=adhoc
   Info.plist entries=22
   ```
   Note `thin (arm64)` and `flags=…(adhoc,runtime)`: arm64-only, hardened runtime on, ad-hoc.
3. **`spctl` rejects it, as expected and by design.**
   ```
   $ spctl --assess --type execute --verbose=4 "/Volumes/RoEnSpeak 0.1.0/RoEnSpeak.app"
   rejected                                      # exit 3
   $ spctl --assess --type open --context context:primary-signature dist/RoEnSpeak-0.1.0.dmg
   rejected
   source=no usable signature                    # exit 3
   ```
   An ad-hoc signature is not a Developer ID and the image is not notarized, so Gatekeeper refuses
   it. This is **not** a build failure. On the build machine the app runs normally (no quarantine
   attribute); a user who downloads this `.dmg` needs right-click → Open once, or
   `xattr -dr com.apple.quarantine /Applications/RoEnSpeak.app`. Set
   `ROENSPEAK_CODESIGN_IDENTITY` + `ROENSPEAK_NOTARY_PROFILE` for a `.dmg` that passes.
4. **Disk image contents.**
   ```
   $ hdiutil attach dist/RoEnSpeak-0.1.0.dmg -nobrowse -readonly
   /dev/disk6s1   Apple_HFS   /Volumes/RoEnSpeak 0.1.0
   $ ls -la "/Volumes/RoEnSpeak 0.1.0/"
   lrwxr-xr-x  Applications -> /Applications
   drwxr-xr-x  RoEnSpeak.app        (mtime 2020-01-01, from SOURCE_DATE_EPOCH)
   ```
   `codesign --verify` on the *mounted* copy also reports `valid on disk` /
   `satisfies its Designated Requirement`, i.e. the image round-trips the signature intact.
   The reproducibility is in the inputs, not the output: `make_dmg.sh` fixes the layout and sets
   every mtime to `SOURCE_DATE_EPOCH` (default `1577836800`), but `hdiutil` stamps its own UDIF
   header, so two runs are not guaranteed byte-identical.
5. **Frozen CLI, from inside the signed bundle, end to end on Metal.**
   ```
   $ ROENSPEAK_DATA_DIR=$t/data ROENSPEAK_CONFIG_DIR=$t/config \
       build/out/RoEnSpeak.app/Contents/MacOS/RoEnSpeak-cli \
       tests/fixtures/ro_sample.wav --model medium --download --json
   {"engine": {"model": "medium", "device": "metal", "compute_type": "f16",
               "gpu_name": "Apple M1 Max", "load_ms": 522.4},
    "segments": [{"id": 1, "infer_ms": 259.0, "text": "Good evening."},
                 {"id": 2, "infer_ms": 378.0, "text": "Today the weather is beautiful and I want to go to the park."},
                 {"id": 3, "infer_ms": 286.6, "text": "Thank you for your help."}], ...}
   ```
   and in the log:
   ```
   whispercpp_loaded lib=/…/RoEnSpeak.app/Contents/Frameworks/libroenspeak_whisper.dylib
       whisper=1.9.5-dev commit=d1be6fd metal=1
       devices="MTL0:Apple M1 Max(type=gpu,free=25558MiB,total=25559MiB), BLAS:Accelerate…, CPU:Apple M1 Max…"
   cuda_probe ok=1 gpu="Apple M1 Max" driver=26.6.2 cublas=none cudnn=none error=none
   engine_loaded device=metal compute_type=f16 model=medium gpu="Apple M1 Max" load_ms=522.4
   segment_latency session=1 id=1 … infer_ms=259.0 latency_ms=259.1 chars=13 filtered=0
   ```
   The `lib=…/Contents/Frameworks/…` line is the proof that the dylib placement and the
   `@loader_path` rpath work inside a signed `.app`: `sys._MEIPASS` resolved the shim and the shim
   resolved its six siblings with no environment help.
6. **`--gpu-smoke`** does the same thing with the *default* model and asserts
   `engine.device == "metal"`, exactly as `build.ps1 -GpuSmoke` asserts `cuda`. Like the Windows
   step it needs the default model already downloaded
   (`~/Library/Application Support/RoEnSpeak/models/large-v3`).
7. **Uninstaller**, against an isolated `HOME`: `--yes` removed the LaunchAgent, the bundle, the
   models and the logs, and **kept `config.json`**; a second run reported "not running / no login
   item installed / nothing to remove / none found" and exited 0; and with the LaunchAgent pointing
   at a *different* bundle it printed `kept: it points somewhere else` and left the plist in place.

## Troubleshooting (macOS)

- **`missing whisper.cpp dylibs in …/build/native`** from the spec. Run
  `bash native/whispercpp/build.sh`, or point the build at an existing staging directory with
  `ROENSPEAK_NATIVE_DIR=/path/to/dylibs`.
- **`missing …/assets/icons/app.icns`.** Run `.venv/bin/python -m app.ui.gen_icons`.
- **`need CPython 3.12 arm64 in the venv, got '3.12 x86_64'`.** You are in a Rosetta shell or an
  Intel Python. `arch -arm64 zsh`, then delete `.venv` and retry. There is no universal2 build; see
  "target_arch" above.
- **`faster-whisper / ctranslate2 are installed in .venv`.** The macOS engine is whisper.cpp. Those
  packages have no Metal path, and the spec excludes them, so a venv containing them is a venv built
  from the wrong requirements file. Recreate it from `requirements-macos-dev.txt`.
- **`Failed to parse entitlements: AMFIUnserializeXML: syntax error near line N`.** You edited
  `build/entitlements.plist` and added a comment with an angle bracket or a doubled hyphen in it, or
  put a comment before `<plist>`. `plutil -lint` will not catch it. See the end of
  "Signing and notarization (macOS)".
- **`RoEnSpeak-cli stopped working after codesign`.** The re-sign broke a nested signature. Check
  that `build/entitlements.plist` still parses and re-run with `--clean`.
- **`code signature … not valid for use in process` at startup, or the Metal engine fails to load
  in a signed build.** `com.apple.security.cs.disable-library-validation` is missing from the
  entitlements, or the signature was applied without `--entitlements`.
- **`spctl --assess` says `rejected`.** Expected for the default ad-hoc build; see "Verifying"
  step 3. Only a notarized Developer ID build passes.
- **The app launches but never appears.** It is an `LSUIElement` app: it has no Dock icon on
  purpose. Look in the menu bar at the top right. If the menu-bar icon is missing too, check
  `~/Library/Logs/RoEnSpeak/roenspeak.log`.
- **The app has a Dock icon / the window will not open.** `Info.plist` lost `LSUIElement=true` or
  gained `LSBackgroundOnly=true`. The sanity checks in `build_macos.sh` fail the build on both, so
  this only happens if the plist was edited after the build.
- **"RoEnSpeak" cannot be opened because the developer cannot be verified** on someone else's Mac.
  Unsigned-by-Apple download. Right-click → Open, or
  `xattr -dr com.apple.quarantine /Applications/RoEnSpeak.app`, or notarize the `.dmg`.
- **`xcrun: error: unable to find utility "notarytool"`.** Full Xcode is required (the Command Line
  Tools alone do not ship `notarytool`), and `ROENSPEAK_NOTARY_PROFILE` must name a profile created
  with `xcrun notarytool store-credentials`.
- **`error: cannot execute tool 'metal' due to missing Metal Toolchain`** during the native build.
  Something turned `GGML_METAL_EMBED_LIBRARY` off. The embedded build is the only supported one;
  do not install the Metal Toolchain to "fix" this.
- **`hdiutil: create failed - Resource busy`.** A previous image is still mounted:
  `hdiutil detach "/Volumes/RoEnSpeak <version>"`.
