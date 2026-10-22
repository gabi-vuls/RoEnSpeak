<#
.SYNOPSIS
  Build RoEnSpeak from a clean checkout: venv -> requirements -> unit tests -> PyInstaller onedir
  -> (optional) code signing -> Inno Setup -> dist\RoEnSpeak-Setup.exe.

.DESCRIPTION
  Run from anywhere (paths are resolved from this script's location), in Windows PowerShell 5.1+:
    powershell -NoProfile -ExecutionPolicy Bypass -File build\build.ps1
  From WSL (mirror workflow):  build/build_from_wsl.sh [same options]
  See BUILD.md for prerequisites and details.

  Signing hook (optional; skipped with a message when SIGNTOOL_CERT is not set):
    SIGNTOOL_CERT           path to a .pfx file, or the SHA1 thumbprint of a cert in the user/machine store
    SIGNTOOL_PASSWORD       .pfx password (optional; visible in the ISCC/signtool command lines)
    SIGNTOOL_PATH           signtool.exe (default: PATH, then the newest Windows 10/11 SDK x64 signtool)
    SIGNTOOL_TIMESTAMP_URL  RFC 3161 timestamp server (default: http://timestamp.digicert.com)
  When configured, RoEnSpeak.exe and RoEnSpeak-cli.exe are signed before packaging and ISCC signs
  RoEnSpeak-Setup.exe and the uninstaller.

.PARAMETER Python
  Interpreter used to create the venv (default: "py -3.12", i.e. the Python launcher).
.PARAMETER Venv
  Virtual environment directory (default: <repo>\.venv). Created if missing.
.PARAMETER FasterWhisperRepo
  Local faster-whisper clone. Default: the git+file URL pinned in requirements.txt (C:\dev\faster-whisper).
  The pinned commit is always used; this only changes where the clone lives.
.PARAMETER SkipInstall
  Don't run pip (use the venv as is).
.PARAMETER SkipTests
  Don't run the unit tests.
.PARAMETER SkipInstaller
  Stop after PyInstaller (no ISCC).
.PARAMETER Clean
  PyInstaller --clean (drop the analysis cache in build\work).
.PARAMETER GpuSmoke
  After PyInstaller, run the frozen RoEnSpeak-cli.exe on tests\fixtures\ro_sample.wav on the GPU
  (needs an NVIDIA GPU and the default model already in %LOCALAPPDATA%\RoEnSpeak\models).
.PARAMETER Iscc
  Path to ISCC.exe (default: Inno Setup 6 per-user or Program Files install, or PATH).
#>
[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$Venv = "",
    [string]$FasterWhisperRepo = "",
    [switch]$SkipInstall,
    [switch]$SkipTests,
    [switch]$SkipInstaller,
    [switch]$Clean,
    [switch]$GpuSmoke,
    [string]$Iscc = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Total = [Diagnostics.Stopwatch]::StartNew()

function Step([string]$msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail([string]$msg) { Write-Host "BUILD FAILED: $msg" -ForegroundColor Red; exit 1 }
function Run([string]$exe, [string[]]$argv) {
    Write-Host "   > $exe $($argv -join ' ')" -ForegroundColor DarkGray
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { Fail "$exe exited with code $LASTEXITCODE" }
}
function SizeMB([string]$path) {
    if (Test-Path $path -PathType Leaf) { return [math]::Round((Get-Item $path).Length / 1MB, 1) }
    $sum = (Get-ChildItem $path -Recurse -File | Measure-Object -Property Length -Sum).Sum
    return [math]::Round($sum / 1MB, 1)
}

# ---------------------------------------------------------------------------------------------
Step "Python virtual environment"
if (-not $Venv) { $Venv = Join-Path $Root ".venv" }
$Py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Py)) {
    if ($Python) { $launcher = $Python; $largs = @() } else { $launcher = "py"; $largs = @("-3.12") }
    if (-not (Get-Command $launcher -ErrorAction SilentlyContinue)) {
        Fail "'$launcher' not found. Install Python 3.12 x64 (with the py launcher) or pass -Python <python.exe>."
    }
    Run $launcher ($largs + @("-m", "venv", $Venv))
}
$pyinfo = & $Py -c "import sys,struct;print('%d.%d %d' % (sys.version_info[0], sys.version_info[1], struct.calcsize('P')*8))"
if ($LASTEXITCODE -ne 0) { Fail "venv python is broken: $Py" }
Write-Host "   venv: $Venv (Python $pyinfo-bit)"
if (-not $pyinfo.StartsWith("3.12 64")) { Fail "need CPython 3.12 x64 in the venv, got $pyinfo (delete $Venv and retry)" }

# ---------------------------------------------------------------------------------------------
if (-not $SkipInstall) {
    Step "Install requirements (requirements-dev.txt; faster-whisper from the local clone, pinned commit)"
    $reqFile = Join-Path $Root "requirements-dev.txt"
    $fwLine = (Get-Content (Join-Path $Root "requirements.txt") | Where-Object { $_ -match "^faster-whisper @ git\+file:///" })
    if (-not $fwLine) { Fail "requirements.txt has no 'faster-whisper @ git+file:///...' line" }
    if ($fwLine -notmatch "^faster-whisper @ git\+file:///(?<path>[^@]+)@(?<commit>[0-9a-f]+)") { Fail "can't parse: $fwLine" }
    $fwPath = $Matches["path"] -replace "/", "\"
    $fwCommit = $Matches["commit"]
    if ($FasterWhisperRepo) {
        $fwPath = (Resolve-Path $FasterWhisperRepo).Path
        $url = "git+file:///" + ($fwPath -replace "\\", "/") + "@" + $fwCommit
        $tmpReq = Join-Path $env:TEMP "roenspeak-requirements-build.txt"
        $lines = @()
        $lines += Get-Content (Join-Path $Root "requirements.txt") | ForEach-Object {
            if ($_ -match "^faster-whisper @ git\+file:///") { "faster-whisper @ $url" } else { $_ }
        }
        $lines += Get-Content $reqFile | Where-Object { $_ -notmatch "^-r " }
        Set-Content -Path $tmpReq -Value $lines -Encoding ASCII
        $reqFile = $tmpReq
    }
    if (-not (Test-Path (Join-Path $fwPath ".git"))) {
        Fail "faster-whisper clone not found at $fwPath. git clone https://github.com/SYSTRAN/faster-whisper $fwPath (commit $fwCommit) or pass -FasterWhisperRepo."
    }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Fail "git is not on PATH (pip needs it for the git+file requirement)" }
    Write-Host "   faster-whisper: $fwPath @ $fwCommit"
    Run $Py @("-m", "pip", "install", "--disable-pip-version-check", "-r", $reqFile)
    Run $Py @("-m", "pip", "check")
}

# ---------------------------------------------------------------------------------------------
$Version = (& $Py (Join-Path $Root "build\gen_version_info.py") --version).Trim()
if ($LASTEXITCODE -ne 0 -or -not $Version) { Fail "could not read APP_VERSION from app\contracts.py" }
Write-Host ""
Write-Host "   RoEnSpeak version $Version (app\contracts.py APP_VERSION)"

if (-not $SkipTests) {
    Step "Unit tests (hardware-free; gpu/mic/desktop markers are excluded by pyproject.toml)"
    Run $Py @("-m", "pytest", "-q", "-p", "no:cacheprovider")
}

# ---------------------------------------------------------------------------------------------
Step "PyInstaller onedir -> build\out\RoEnSpeak"
$sw = [Diagnostics.Stopwatch]::StartNew()
$pyiArgs = @("-m", "PyInstaller", "build\roenspeak.spec", "--noconfirm", "--workpath", "build\work", "--distpath", "build\out")
if ($Clean) { $pyiArgs += "--clean" }
Run $Py $pyiArgs
$Bundle = Join-Path $Root "build\out\RoEnSpeak"
Write-Host ("   PyInstaller took {0:n0} s" -f $sw.Elapsed.TotalSeconds)

Step "Bundle sanity checks"
$required = @(
    "RoEnSpeak.exe", "RoEnSpeak-cli.exe",
    "_internal\assets\icons\app.ico", "_internal\assets\sounds",
    "_internal\faster_whisper\assets\silero_vad_v6.onnx",
    "_internal\ctranslate2\ctranslate2.dll", "_internal\ctranslate2\cudnn64_9.dll",
    "_internal\nvidia\cublas\bin\cublas64_12.dll", "_internal\nvidia\cublas\bin\cublasLt64_12.dll",
    "_internal\nvidia\cudnn\bin\cudnn64_9.dll", "_internal\nvidia\cudnn\bin\cudnn_graph64_9.dll",
    "_internal\nvidia\cudnn\bin\cudnn_ops64_9.dll", "_internal\nvidia\cuda_nvrtc\bin\nvrtc64_120_0.dll",
    "_internal\_sounddevice_data\portaudio-binaries\libportaudio64bit.dll"
)
foreach ($r in $required) {
    if (-not (Test-Path (Join-Path $Bundle $r))) { Fail "missing from bundle: $r" }
}
$stray = Get-ChildItem (Join-Path $Bundle "_internal") -File -Filter "*.dll" |
    Where-Object { $_.Name -match "^(cublas|cudnn|nvrtc|nvml|nvcuda)" }
if ($stray) { Fail "CUDA/driver DLLs at the top of _internal (should only be under nvidia\*\bin): $($stray.Name -join ', ')" }
$cli = Join-Path $Bundle "RoEnSpeak-cli.exe"
& $cli --help | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "RoEnSpeak-cli.exe --help exited with $LASTEXITCODE" }
Write-Host "   layout OK, RoEnSpeak-cli.exe starts; bundle size $(SizeMB $Bundle) MB"

if ($GpuSmoke) {
    Step "GPU smoke test (frozen CLI, isolated config/logs, real models dir)"
    $tmp = Join-Path $env:TEMP ("roenspeak-smoke-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
    New-Item -ItemType Directory -Force -Path "$tmp\data", "$tmp\config" | Out-Null
    $env:ROENSPEAK_DATA_DIR = "$tmp\data"
    $env:ROENSPEAK_CONFIG_DIR = "$tmp\config"
    if (-not $env:ROENSPEAK_MODELS_DIR) { $env:ROENSPEAK_MODELS_DIR = Join-Path $env:LOCALAPPDATA "RoEnSpeak\models" }
    try {
        $json = & $cli (Join-Path $Root "tests\fixtures\ro_sample.wav") --json
        $rc = $LASTEXITCODE
    } finally {
        Remove-Item Env:\ROENSPEAK_DATA_DIR, Env:\ROENSPEAK_CONFIG_DIR -ErrorAction SilentlyContinue
    }
    if ($rc -ne 0) { Fail "GPU smoke test exited with $rc (logs: $tmp\data\logs)" }
    $o = $json | ConvertFrom-Json
    if ($o.engine.device -ne "cuda") { Fail "GPU smoke test ran on $($o.engine.device), expected cuda" }
    Write-Host ("   {0} {1} on {2}: {3}" -f $o.engine.model, $o.engine.compute_type, $o.engine.gpu_name,
        (($o.segments | ForEach-Object { $_.text }) -join " "))
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

# ---------------------------------------------------------------------------------------------
Step "Code signing hook"
$SignArgs = @()
if ($env:SIGNTOOL_CERT) {
    $signtool = $env:SIGNTOOL_PATH
    if (-not $signtool) {
        $cmd = Get-Command signtool.exe -ErrorAction SilentlyContinue
        if ($cmd) { $signtool = $cmd.Source }
    }
    if (-not $signtool) {
        $kits = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"
        if (Test-Path $kits) {
            $found = Get-ChildItem $kits -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -match "\\x64\\" } | Sort-Object FullName -Descending | Select-Object -First 1
            if ($found) { $signtool = $found.FullName }
        }
    }
    if (-not $signtool -or -not (Test-Path $signtool)) { Fail "SIGNTOOL_CERT is set but signtool.exe was not found (set SIGNTOOL_PATH)" }
    $ts = $env:SIGNTOOL_TIMESTAMP_URL
    if (-not $ts) { $ts = "http://timestamp.digicert.com" }
    if ($env:SIGNTOOL_CERT -match "^[0-9A-Fa-f]{40}$") {
        $certArgs = @("/sha1", $env:SIGNTOOL_CERT)
    } else {
        $certArgs = @("/f", (Resolve-Path $env:SIGNTOOL_CERT).Path)
        if ($env:SIGNTOOL_PASSWORD) { $certArgs += @("/p", $env:SIGNTOOL_PASSWORD) }
    }
    $signBase = @("sign", "/fd", "sha256", "/tr", $ts, "/td", "sha256") + $certArgs
    foreach ($exe in @("RoEnSpeak.exe", "RoEnSpeak-cli.exe")) {
        Run $signtool ($signBase + @((Join-Path $Bundle $exe)))
    }
    # ISCC sign tool command: $q = quote, $f = file to sign (Setup.exe and the uninstaller).
    $isccCert = ($certArgs | ForEach-Object { if ($_ -match "[\s\\]") { '$q' + $_ + '$q' } else { $_ } }) -join " "
    $SignArgs = @("/DSignToolName=roensign",
        ('/Sroensign=$q' + $signtool + '$q sign /fd sha256 /tr $q' + $ts + '$q /td sha256 ' + $isccCert + ' $f'))
    Write-Host "   signing with $signtool (exes signed; ISCC will sign Setup.exe + uninstaller)"
} else {
    Write-Host "   signing skipped (SIGNTOOL_CERT not set): exes and setup are unsigned, SmartScreen may warn"
}

if ($SkipInstaller) {
    Write-Host ""
    Write-Host ("Done (no installer) in {0:n0} s: {1}" -f $Total.Elapsed.TotalSeconds, $Bundle) -ForegroundColor Green
    exit 0
}

# ---------------------------------------------------------------------------------------------
Step "Inno Setup -> dist\RoEnSpeak-Setup.exe"
if (-not $Iscc) {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
    )
    $Iscc = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $Iscc) {
        $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
        if ($cmd) { $Iscc = $cmd.Source }
    }
}
if (-not $Iscc -or -not (Test-Path $Iscc)) { Fail "ISCC.exe (Inno Setup 6) not found; install it or pass -Iscc <path>" }
$Dist = Join-Path $Root "dist"
New-Item -ItemType Directory -Force -Path $Dist | Out-Null
$Setup = Join-Path $Dist "RoEnSpeak-Setup.exe"
if (Test-Path $Setup) { Remove-Item -Force $Setup }
$sw = [Diagnostics.Stopwatch]::StartNew()
Run $Iscc (@("/Qp", "/DAppVersion=$Version", "/DBundleDir=$Bundle", "/DOutputDir=$Dist") + $SignArgs + @((Join-Path $Root "installer\RoEnSpeak.iss")))
Write-Host ("   ISCC took {0:n0} s" -f $sw.Elapsed.TotalSeconds)
if (-not (Test-Path $Setup)) { Fail "ISCC did not produce $Setup" }

Write-Host ""
Write-Host ("Done in {0:n0} s" -f $Total.Elapsed.TotalSeconds) -ForegroundColor Green
Write-Host ("   bundle : {0} ({1} MB)" -f $Bundle, (SizeMB $Bundle))
Write-Host ("   setup  : {0} ({1} MB, version {2})" -f $Setup, (SizeMB $Setup), $Version)
