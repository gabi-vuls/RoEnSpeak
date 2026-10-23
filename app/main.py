"""RoEnSpeak tray app entry point.

    python -m app.main [--minimized] [--debug-audio-wav PATH]

Order: GPU library paths -> logging -> single instance -> QApplication -> QtUiFacade ->
AppController -> run_ui. controller.shutdown() runs on every exit path (normal quit,
exception, WM_ENDSESSION via commitDataRequest/aboutToQuit, atexit backstop).
"""

from __future__ import annotations

import sys

# The probe is created and setup_dll_paths() runs at import time, before anything imports the
# engine: on Windows ctranslate2 resolves cuBLAS/cuDNN at import/load time, so the DLL
# directories must already be on the search path. (No-op on macOS, where it returns [].)
from app.core import platform

_PROBE = platform.make_probe()
_PROBE.setup_dll_paths()

import argparse  # noqa: E402
import atexit  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional, Sequence  # noqa: E402

from app.contracts import APP_NAME, APP_VERSION  # noqa: E402

log = logging.getLogger("roenspeak.main")


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=APP_NAME, description="Speak Romanian, type English.")
    p.add_argument("--minimized", action="store_true", help="start in the tray (default behavior; used by autostart)")
    p.add_argument("--debug-audio-wav", metavar="PATH", type=Path,
                   help="use this wav (real-time) instead of the microphone for each dictation session")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    from app.core.logging_setup import setup_logging

    log_file = setup_logging()
    log.info("%s %s starting (python %s, frozen=%s), log=%s", APP_NAME, APP_VERSION, sys.version.split()[0],
             getattr(sys, "frozen", False), log_file)

    api = platform.make_api()
    instance = platform.make_single_instance(api)
    if not instance.acquire():
        log.info("another instance is running; asking it to show its window")
        instance.notify_existing()
        instance.release()
        return 0
    atexit.register(instance.release)

    controller = None
    try:
        from PySide6.QtCore import QTimer

        from app.core.controller import AppController
        from app.core.events import SimpleEventBus
        from app.ui.app import create_qapplication, run_ui
        from app.ui.facade import QtUiFacade

        qapp = create_qapplication()
        facade = QtUiFacade()
        bus = SimpleEventBus()
        controller = AppController(
            ui=facade, bus=bus, probe=_PROBE, api=api, single_instance=instance,
            debug_audio_wav=args.debug_audio_wav,
        )
        atexit.register(controller.shutdown, 10.0)
        # Logoff/shutdown (WM_QUERYENDSESSION/WM_ENDSESSION) -> commitDataRequest; then aboutToQuit.
        # macOS never emits commitDataRequest, so the connection is simply never fired there.
        qapp.commitDataRequest.connect(lambda _mgr: controller.shutdown(8.0))
        qapp.aboutToQuit.connect(lambda: controller.shutdown(15.0))
        # Start once the event loop runs, i.e. after run_ui has built the window and subscribed
        # to the bus, so startup events (warnings, state) are not lost.
        QTimer.singleShot(0, controller.start)
        if args.debug_audio_wav is not None:
            log.warning("debug audio source: %s", args.debug_audio_wav)
        rc = run_ui(controller, bus, facade, True)
        log.info("UI loop exited with %s", rc)
        return int(rc or 0)
    except Exception:
        log.exception("fatal error in main")
        return 1
    finally:
        if controller is not None:
            if not controller.shutdown(75.0):  # covers waiting for an in-progress model load
                log.warning("core shutdown did not finish in time")
        instance.release()


if __name__ == "__main__":
    sys.exit(main())
