"""Prompt adjustment history: logs_dir()/prompt_history.jsonl, one JSON object per run.

Unlike roenspeak.log (which never contains user text), this file keeps the original
selection and the rewrite in plain text, so a replaced selection can always be recovered.
Size-capped: at max_bytes the file rotates to .1 .. .<backups> (oldest dropped).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from app.contracts import logs_dir

log = logging.getLogger("roenspeak.core.prompt_history")

HISTORY_FILE_NAME = "prompt_history.jsonl"


class PromptHistory:
    def __init__(self, path: Optional[Path] = None, max_bytes: int = 5_000_000, backups: int = 3) -> None:
        self._explicit_path = Path(path) if path is not None else None
        self._max_bytes = max_bytes
        self._backups = backups
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._explicit_path if self._explicit_path is not None else logs_dir() / HISTORY_FILE_NAME

    def append(self, record: Mapping[str, Any]) -> None:
        """Adds one record (a "ts" field is filled in when missing). Never raises: a history
        write failure must not lose the paste."""
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **record}
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._lock:
            path = self.path
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                self._rotate_if_needed(path, len(line.encode("utf-8")))
                with open(path, "a", encoding="utf-8", newline="\n") as f:
                    f.write(line)
            except OSError as e:
                log.error("could not write prompt history to %s: %s", path, e)

    def _rotate_if_needed(self, path: Path, incoming: int) -> None:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return
        if size + incoming <= self._max_bytes:
            return
        for i in range(self._backups, 0, -1):
            src = path if i == 1 else path.with_name(f"{path.name}.{i - 1}")
            dst = path.with_name(f"{path.name}.{i}")
            if src.exists():
                os.replace(src, dst)
