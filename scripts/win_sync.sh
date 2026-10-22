#!/usr/bin/env bash
# Mirror the WSL source tree to C:\dev\ro-en-speak (Windows side runs tests/builds there).
# The Windows-only dirs (.venv, dist, build output, caches) are never touched.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)/"
DST="/mnt/c/dev/ro-en-speak/"
mkdir -p "$DST"
rsync -a --delete \
  --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' --exclude '.pytest_cache/' \
  --exclude 'dist/' --exclude 'build/work/' --exclude 'build/out/' --exclude '*.pyc' \
  "$SRC" "$DST"
