#!/usr/bin/env bash
# Sync to C:\dev\ro-en-speak, then run the Windows venv python there with the given args.
#   scripts/win.sh -m pytest tests/unit -q
#   scripts/win.sh -c "import app.contracts"
# Set NO_SYNC=1 to skip the sync step.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
[ "${NO_SYNC:-0}" = "1" ] || "$HERE/win_sync.sh"
cd /mnt/c/dev/ro-en-speak
exec ./.venv/Scripts/python.exe "$@"
