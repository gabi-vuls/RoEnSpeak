# Building RoEnSpeak

These steps take a clean checkout to `dist\RoEnSpeak-Setup.exe`: a per-user Inno Setup installer that
contains a PyInstaller **onedir** bundle and the CUDA DLLs. The people you give it to only need an NVIDIA driver.

## Prerequisites

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

## One command

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

## Files

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

## Bundle layout and DLL bundling

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

## Trimming (CUDA DLLs)

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

## Installer behaviour (`installer/RoEnSpeak.iss`)

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

## Signing (optional hook)

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

## Verifying a build (what was run for 0.1.0)

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

## Troubleshooting

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
