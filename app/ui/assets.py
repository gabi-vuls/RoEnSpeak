"""Asset path resolution (dev checkout and PyInstaller frozen build) and icon loading."""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path

from PySide6.QtGui import QIcon, QPixmap

log = logging.getLogger("roenspeak.ui.assets")


def assets_dir() -> Path:
    """<bundle>/assets when frozen (sys._MEIPASS, i.e. the onedir _internal folder),
    otherwise <repo>/assets."""
    meipass = getattr(sys, "_MEIPASS", None)
    if getattr(sys, "frozen", False) and meipass:
        return Path(meipass) / "assets"
    return Path(__file__).resolve().parents[2] / "assets"


def asset_path(*parts: str) -> Path:
    return assets_dir().joinpath(*parts)


def sound_path(name: str) -> Path:
    return asset_path("sounds", f"{name}.wav")


@lru_cache(maxsize=None)
def tray_icon(state: str) -> QIcon:
    """Multi-resolution tray icon for one of icon_art.TRAY_STATES (needs a QGuiApplication)."""
    from app.ui.icon_art import TRAY_SIZES, render_tray_icon

    icon = QIcon()
    for size in TRAY_SIZES:
        p = asset_path("icons", f"tray_{state}_{size}.png")
        if p.is_file():
            icon.addFile(str(p))
        else:
            log.warning("missing icon %s; rendering at runtime", p)
            icon.addPixmap(QPixmap.fromImage(render_tray_icon(state, size)))
    return icon


@lru_cache(maxsize=None)
def app_icon() -> QIcon:
    from app.ui.icon_art import ICO_SIZES, render_app_icon

    icon = QIcon()
    for size in ICO_SIZES:
        p = asset_path("icons", f"app_{size}.png")
        if p.is_file():
            icon.addFile(str(p))
        else:
            icon.addPixmap(QPixmap.fromImage(render_app_icon(size)))
    return icon
