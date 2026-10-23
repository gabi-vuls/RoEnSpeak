"""Start/stop beeps, played asynchronously with the host's own sound API.

No QtMultimedia: PySide6-**Essentials** is a deliberate choice (it keeps the installer and
the .app small), and QtMultimedia is not part of it.

* Windows: ``winsound.PlaySound(SND_ASYNC)`` (stdlib).
* macOS:   ``NSSound`` through ctypes (app.ui._platform.play_sound_async), with ``afplay``
  as a subprocess fallback for the rare case where NSSound cannot load the file.

Both paths return immediately, so the Qt thread is never blocked while dictation starts or
stops, and neither ever raises: a missing audio device must not break dictation.
"""

from __future__ import annotations

import logging
import subprocess
import sys

from app.contracts import IS_MACOS
from app.ui import _platform
from app.ui.assets import sound_path

log = logging.getLogger("roenspeak.ui.sounds")

try:
    import winsound
except ImportError:  # non-Windows: winsound does not exist
    winsound = None  # type: ignore[assignment]


class SoundPlayer:
    """play("start" | "stop"). Never blocks and never raises."""

    def __init__(self) -> None:
        self._paths = {name: sound_path(name) for name in ("start", "stop")}
        for name, p in self._paths.items():
            if not p.is_file():
                log.warning("sound file missing: %s", p)
        self.last_played: str | None = None
        self._procs: list[subprocess.Popen] = []

    def play(self, name: str) -> None:
        path = self._paths.get(name)
        self.last_played = name
        if path is None or not path.is_file():
            log.debug("sound %s not played (unavailable)", name)
            return
        if IS_MACOS:
            self._play_macos(name, str(path))
            return
        if winsound is None or sys.platform != "win32":
            log.debug("sound %s not played (unavailable)", name)
            return
        try:
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        except RuntimeError as exc:  # no audio output device, etc.
            log.warning("PlaySound(%s) failed: %s", name, exc)

    # -- macOS --------------------------------------------------------------
    def _play_macos(self, name: str, path: str) -> None:
        if _platform.play_sound_async(path):
            return
        self._reap()
        try:
            # Detached so it cannot inherit our run loop or block on our stdio.
            self._procs.append(
                subprocess.Popen(
                    ["/usr/bin/afplay", path],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            )
            log.debug("sound %s played with afplay (NSSound unavailable)", name)
        except OSError as exc:
            log.warning("afplay(%s) failed: %s", name, exc)

    def _reap(self) -> None:
        """Collect finished afplay children so they do not pile up as zombies."""
        self._procs = [p for p in self._procs if p.poll() is None]
