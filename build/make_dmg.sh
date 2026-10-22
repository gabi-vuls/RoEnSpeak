#!/usr/bin/env bash
# Package RoEnSpeak.app into a drag-install disk image.
#
#   build/make_dmg.sh [--app DIR] [--out FILE] [--version V] [--volname NAME] [--identity ID]
#
# Defaults:  --app      build/out/RoEnSpeak.app
#            --version  APP_VERSION from app/contracts.py (build/gen_version_info.py --version)
#            --out      dist/RoEnSpeak-<version>.dmg
#            --volname  "RoEnSpeak <version>"
#
# Uses only /usr/bin/hdiutil, which ships with macOS. There is deliberately NO dependency on
# `create-dmg` (a Homebrew formula that also wants AppleScript to pose the Finder window): a
# background picture and icon coordinates would be stored in a .DS_Store, which is exactly the
# kind of non-reproducible, locale-dependent junk this script avoids. The image gets the .app and
# an /Applications symlink, which is all a drag-install needs.
#
# Reproducibility: every input is made deterministic -- a fixed staging layout and all mtimes set
# to $SOURCE_DATE_EPOCH (default: 2020-01-01T00:00:00Z) -- and the image is a compressed read-only
# UDZO at a fixed zlib level. hdiutil still stamps its own UDIF header, so two runs are not
# guaranteed byte-identical; the *contents* are. See BUILD.md "Verifying a build (macOS)".
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

APP=""
OUT=""
VERSION=""
VOLNAME=""
IDENTITY="${ROENSPEAK_CODESIGN_IDENTITY:-}"

die() { printf 'make_dmg.sh: %s\n' "$*" >&2; exit 1; }

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; $d'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --app)      [ $# -ge 2 ] || die "--app needs a path";     APP="$2";     shift 2 ;;
        --out)      [ $# -ge 2 ] || die "--out needs a path";     OUT="$2";     shift 2 ;;
        --version)  [ $# -ge 2 ] || die "--version needs a value"; VERSION="$2"; shift 2 ;;
        --volname)  [ $# -ge 2 ] || die "--volname needs a value"; VOLNAME="$2"; shift 2 ;;
        --identity) [ $# -ge 2 ] || die "--identity needs a value"; IDENTITY="$2"; shift 2 ;;
        -h|--help)  usage ;;
        *)          die "unknown option: $1 (try --help)" ;;
    esac
done

[ -n "$APP" ] || APP="$ROOT/build/out/RoEnSpeak.app"
[ -d "$APP" ] || die "no app bundle at $APP -- run build/build_macos.sh first"

if [ -z "$VERSION" ]; then
    PY="$ROOT/.venv/bin/python"
    [ -x "$PY" ] || PY="$(command -v python3)" || die "no python to read APP_VERSION; pass --version"
    VERSION="$("$PY" "$ROOT/build/gen_version_info.py" --version)"
fi
[ -n "$VERSION" ] || die "could not determine the version; pass --version"

[ -n "$OUT" ]     || OUT="$ROOT/dist/RoEnSpeak-$VERSION.dmg"
[ -n "$VOLNAME" ] || VOLNAME="RoEnSpeak $VERSION"

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/roenspeak-dmg.XXXXXX")"
cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

printf '   staging %s -> %s\n' "$(basename "$APP")" "$STAGE"
# -R keeps symlinks as symlinks (Qt .framework bundles and PyInstaller's cross-links are symlinks;
# dereferencing them would multiply the image size and break the code signature).
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"

# Deterministic timestamps on everything that goes into the image.
EPOCH="${SOURCE_DATE_EPOCH:-1577836800}"   # 1577836800 = 2020-01-01T00:00:00Z
TOUCH_TS="$(date -u -r "$EPOCH" +%Y%m%d%H%M.%S)"
find "$STAGE" -exec touch -h -t "$TOUCH_TS" {} + 2>/dev/null || true

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

printf '   hdiutil create (UDZO, zlib-9, HFS+)\n'
hdiutil create \
    -volname "$VOLNAME" \
    -srcfolder "$STAGE" \
    -fs HFS+ \
    -format UDZO \
    -imagekey zlib-level=9 \
    -ov -quiet \
    "$OUT"

[ -f "$OUT" ] || die "hdiutil did not produce $OUT"

# A disk image can carry its own signature. Ad-hoc adds nothing a user can check, so only sign the
# .dmg when a real identity is configured -- that is the artefact notarytool is given and stapled.
if [ -n "$IDENTITY" ]; then
    printf '   codesign the disk image with %s\n' "$IDENTITY"
    codesign --force --sign "$IDENTITY" --timestamp "$OUT"
fi

SIZE_MB="$(( $(stat -f%z "$OUT") / 1048576 ))"
printf '   dmg: %s (%s MB, volume %s)\n' "$OUT" "$SIZE_MB" "$VOLNAME"
