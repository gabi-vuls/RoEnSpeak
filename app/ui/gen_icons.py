"""Generate the committed icon assets.

    python -m app.ui.gen_icons [--out DIR]     (default DIR: <repo>/assets/icons)

Writes:
  tray_<state>_<size>.png   for every state in icon_art.TRAY_STATES and size in TRAY_SIZES
  app_<size>.png            for every size in ICO_SIZES
  app.ico                   multi-size icon (16..256) used by the window, PyInstaller and Inno Setup.
                            Sizes below 256 are stored as 32-bit BMP/DIB entries with an AND mask
                            (best compatibility with resource tools); 256 is PNG-compressed.
"""

from __future__ import annotations

import argparse
import logging
import os
import struct
import sys
from pathlib import Path

log = logging.getLogger("roenspeak.ui.gen_icons")


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


def generate(out_dir: Path) -> list[Path]:
    from PySide6.QtGui import QGuiApplication

    from app.ui.icon_art import ICO_SIZES, TRAY_SIZES, TRAY_STATES, render_app_icon, render_tray_icon

    _app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])  # noqa: F841 (needed for QPainter)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for state in TRAY_STATES:
        for size in TRAY_SIZES:
            p = out_dir / f"tray_{state}_{size}.png"
            render_tray_icon(state, size).save(str(p), "PNG")
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
