"""Start/stop beeps played asynchronously with winsound (stdlib; no QtMultimedia needed)."""

from __future__ import annotations

import logging
import sys

from app.ui.assets import sound_path

log = logging.getLogger("roenspeak.ui.sounds")

try:
    import winsound
except ImportError:  # non-Windows: sounds are silently unavailable
    winsound = None  # type: ignore[assignment]


class SoundPlayer:
    """play("start" | "stop"). Never blocks: SND_ASYNC returns immediately."""

    def __init__(self) -> None:
        self._paths = {name: sound_path(name) for name in ("start", "stop")}
        for name, p in self._paths.items():
            if not p.is_file():
                log.warning("sound file missing: %s", p)
        self.last_played: str | None = None

    def play(self, name: str) -> None:
        path = self._paths.get(name)
        self.last_played = name
        if winsound is None or sys.platform != "win32" or path is None or not path.is_file():
            log.debug("sound %s not played (unavailable)", name)
            return
        try:
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        except RuntimeError as exc:  # no audio output device, etc.
            log.warning("PlaySound(%s) failed: %s", name, exc)
