#!/usr/bin/env bash
# Build RoEnSpeak-Setup.exe from WSL: mirror the repo to C:\dev\ro-en-speak (scripts/win_sync.sh),
# then run build\build.ps1 there with Windows PowerShell. Options are passed through, e.g.
#   build/build_from_wsl.sh -SkipTests -GpuSmoke
# Output: C:\dev\ro-en-speak\dist\RoEnSpeak-Setup.exe (= /mnt/c/dev/ro-en-speak/dist/RoEnSpeak-Setup.exe)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
"$HERE/../scripts/win_sync.sh"
cd /mnt/c/dev/ro-en-speak
exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'C:\dev\ro-en-speak\build\build.ps1' "$@"
