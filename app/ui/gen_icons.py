"""Generate the committed icon assets.

    python -m app.ui.gen_icons [--out DIR]     (default DIR: <repo>/assets/icons)

Writes:
  tray_<state>_<size>.png            for every state in icon_art.TRAY_STATES and size in TRAY_SIZES
  tray_<state>_template_<size>.png   macOS status-item masks (monochrome silhouettes), for
                                     every size in TRAY_TEMPLATE_SIZES (16 = 1x, 32 = 2x, 64 = 4x)
  app_<size>.png                     for every size in ICO_SIZES
  app.ico                            multi-size icon (16..256) used by the window, PyInstaller and
                                     Inno Setup. Sizes below 256 are stored as 32-bit BMP/DIB
                                     entries with an AND mask (best compatibility with resource
                                     tools); 256 is PNG-compressed.
  app.icns                           macOS app-bundle icon (CFBundleIconFile), 16..1024 px. Built
                                     with `iconutil` when it is available (the Apple-blessed path),
                                     otherwise with the pure-Python writer below, which produces
                                     the same PNG-based ICNS container.

Everything is deterministic: re-running overwrites the files with identical content.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger("roenspeak.ui.gen_icons")

#: Sizes inside app.icns. 1024 exists only as the @2x variant of 512.
ICNS_SIZES = (16, 32, 64, 128, 256, 512, 1024)

#: ICNS element type per (size, scale). The @2x types (ic11..ic14) are what Retina Macs pick;
#: the icp4/icp5/ic07.. types are the 1x ladder. Every one of them holds a PNG.
ICNS_ELEMENTS: tuple[tuple[bytes, int], ...] = (
    (b"icp4", 16),    # 16x16
    (b"icp5", 32),    # 32x32
    (b"ic11", 32),    # 16x16@2x
    (b"ic12", 64),    # 32x32@2x
    (b"ic07", 128),   # 128x128
    (b"ic13", 256),   # 128x128@2x
    (b"ic08", 256),   # 256x256
    (b"ic14", 512),   # 256x256@2x
    (b"ic09", 512),   # 512x512
    (b"ic10", 1024),  # 512x512@2x
)

#: File names inside the .iconset directory `iconutil` consumes.
ICONSET_FILES: tuple[tuple[str, int], ...] = (
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
)


def _png_bytes(img) -> bytes:
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice

    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    return bytes(ba.data())


def _dib_bytes(img) -> bytes:
    """32-bit BGRA bottom-up DIB + 1-bit AND mask, as stored inside .ico files."""
    from PySide6.QtGui import QImage

    argb = img.convertToFormat(QImage.Format_ARGB32)  # non-premultiplied; bytes are B,G,R,A on LE
    w, h = argb.width(), argb.height()
    bpl = argb.bytesPerLine()
    raw = bytes(argb.constBits())[: bpl * h]
    rows = [raw[y * bpl: y * bpl + w * 4] for y in range(h)]
    xor = b"".join(reversed(rows))
    mask_stride = ((w + 31) // 32) * 4
    mask_rows = []
    for row in reversed(rows):
        bits = bytearray(mask_stride)
        for x in range(w):
            if row[x * 4 + 3] == 0:
                bits[x // 8] |= 0x80 >> (x % 8)
        mask_rows.append(bytes(bits))
    and_mask = b"".join(mask_rows)
    header = struct.pack("<IiiHHIIiiII", 40, w, h * 2, 1, 32, 0, len(xor) + len(and_mask), 0, 0, 0, 0)
    return header + xor + and_mask


def write_ico(images: list, path: Path) -> None:
    entries = []
    for img in images:
        size = img.width()
        data = _png_bytes(img) if size >= 256 else _dib_bytes(img)
        entries.append((size, data))
    out = bytearray(struct.pack("<HHH", 0, 1, len(entries)))
    offset = 6 + 16 * len(entries)
    for size, data in entries:
        dim = 0 if size >= 256 else size
        out += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    for _, data in entries:
        out += data
    path.write_bytes(bytes(out))


def write_icns(pngs: dict[int, bytes], path: Path) -> None:
    """Pure-Python ICNS writer (the fallback for hosts without `iconutil`).

    Container: "icns" + total length, then one element per entry: 4-byte OSType, 4-byte
    big-endian length *including* the 8-byte element header, then the PNG. Every type used
    here accepts a PNG payload (macOS 10.7+ / iconutil writes exactly this).
    """
    elements = bytearray()
    for ostype, size in ICNS_ELEMENTS:
        data = pngs[size]
        elements += ostype + struct.pack(">I", len(data) + 8) + data
    path.write_bytes(b"icns" + struct.pack(">I", len(elements) + 8) + bytes(elements))


def build_icns(pngs: dict[int, bytes], path: Path) -> str:
    """Write path from the rendered PNGs. Returns "iconutil" or "python" (which tool did it)."""
    iconutil = shutil.which("iconutil")
    if iconutil:
        with tempfile.TemporaryDirectory() as tmp:
            iconset = Path(tmp) / "app.iconset"
            iconset.mkdir()
            for name, size in ICONSET_FILES:
                (iconset / name).write_bytes(pngs[size])
            result = subprocess.run(
                [iconutil, "--convert", "icns", "--output", str(path), str(iconset)],
                capture_output=True, text=True,
            )
            if result.returncode == 0 and path.is_file():
                return "iconutil"
            log.warning(
                "iconutil failed (rc=%d): %s; falling back to the Python writer",
                result.returncode, (result.stderr or "").strip(),
            )
    write_icns(pngs, path)
    return "python"


def generate(out_dir: Path) -> list[Path]:
    from PySide6.QtGui import QGuiApplication

    from app.ui.icon_art import (
        ICO_SIZES,
        TRAY_SIZES,
        TRAY_STATES,
        TRAY_TEMPLATE_SIZES,
        render_app_icon,
        render_tray_icon,
        render_tray_template,
    )

    _app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])  # noqa: F841 (needed for QPainter)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for state in TRAY_STATES:
        for size in TRAY_SIZES:
            p = out_dir / f"tray_{state}_{size}.png"
            render_tray_icon(state, size).save(str(p), "PNG")
            written.append(p)
        for size in TRAY_TEMPLATE_SIZES:
            p = out_dir / f"tray_{state}_template_{size}.png"
            render_tray_template(state, size).save(str(p), "PNG")
            written.append(p)
    app_images = []
    for size in ICO_SIZES:
        img = render_app_icon(size)
        app_images.append(img)
        p = out_dir / f"app_{size}.png"
        img.save(str(p), "PNG")
        written.append(p)
    ico = out_dir / "app.ico"
    write_ico(app_images, ico)
    written.append(ico)
    icns = out_dir / "app.icns"
    tool = build_icns({size: _png_bytes(render_app_icon(size)) for size in ICNS_SIZES}, icns)
    log.info("app.icns built with %s", tool)
    written.append(icns)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_out = Path(__file__).resolve().parents[2] / "assets" / "icons"
    parser.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    for p in generate(args.out):
        log.info("wrote %s", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
