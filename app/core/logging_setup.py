"""Process-wide logging: rotating file in logs_dir(), stderr when a console exists, and
logging of uncaught exceptions on any thread."""

from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
from pathlib import Path
from typing import Optional

from app.contracts import logs_dir

LOG_FILE_NAME = "roenspeak.log"
LOG_MAX_BYTES = 2 * 1024 * 1024
LOG_BACKUP_COUNT = 4  # current file + 4 backups = 5 x 2 MB
LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

_HANDLER_TAG = "_roenspeak_handler"
_hooks_installed = False


def _has_console() -> bool:
    stream = sys.stderr
    if stream is None:  # pythonw.exe / frozen windowed build
        return False
    try:
        stream.fileno()
    except (AttributeError, OSError, ValueError):
        return False
    return True


def _install_excepthooks() -> None:
    global _hooks_installed
    if _hooks_installed:
        return
    _hooks_installed = True
    prev_sys_hook = sys.excepthook
    prev_thread_hook = threading.excepthook
    log = logging.getLogger("roenspeak.unhandled")

    def sys_hook(exc_type, exc, tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            log.critical("Unhandled exception", exc_info=(exc_type, exc, tb))
        if _has_console():
            prev_sys_hook(exc_type, exc, tb)

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread is not None else "?"
        log.critical(
            "Unhandled exception in thread %s", name, exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
        )
        if _has_console():
            prev_thread_hook(args)

    sys.excepthook = sys_hook
    threading.excepthook = thread_hook


def setup_logging(
    log_dir: Optional[Path] = None, level: int = logging.INFO, *, console_level: Optional[int] = None
) -> Path:
    """Configure the root logger. Returns the log file path. Idempotent: handlers installed
    by a previous call are replaced. console_level (default: level) only affects stderr."""
    directory = Path(log_dir) if log_dir is not None else logs_dir()
    log_file = directory / LOG_FILE_NAME
    formatter = logging.Formatter(LOG_FORMAT, LOG_DATEFMT)

    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, _HANDLER_TAG, False):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    file_error: Optional[BaseException] = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8", delay=False
        )
        fh.setFormatter(formatter)
        fh.setLevel(level)
        setattr(fh, _HANDLER_TAG, True)
        root.addHandler(fh)
    except OSError as e:
        file_error = e

    if _has_console():
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(formatter)
        sh.setLevel(console_level if console_level is not None else level)
        setattr(sh, _HANDLER_TAG, True)
        root.addHandler(sh)

    root.setLevel(min(level, console_level) if console_level is not None else level)
    # faster-whisper logs on its own "faster_whisper" logger; make sure it reaches our handlers.
    fw = logging.getLogger("faster_whisper")
    fw.setLevel(level)
    fw.propagate = True
    for noisy in ("urllib3", "httpx", "httpcore", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    _install_excepthooks()
    logging.captureWarnings(True)
    if file_error is not None:
        logging.getLogger("roenspeak.core.logging").error("Could not open log file %s: %s", log_file, file_error)
    return log_file
