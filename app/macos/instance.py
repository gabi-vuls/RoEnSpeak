"""Single-instance guard via flock + a Unix socket (``contracts.SingleInstance``).

The macOS counterpart of ``app.win32.instance.NamedMutexSingleInstance``. The Windows version
uses a named mutex for "am I the first?" and a broadcast window message for "show your
window". macOS has neither, so:

* ``flock(LOCK_EX | LOCK_NB)`` on ``contracts.single_instance_lock_path()`` is the mutex. The
  kernel drops the lock when the process dies, so a crash cannot leave a stale lock behind
  (a stale *file* is harmless; the lock, not its content, is what counts). Our pid is written
  into the file for diagnostics.
* The sibling Unix socket ``<lock stem>.sock`` is the broadcast message. The first instance
  listens on it on a daemon thread; a second launch connects, sends
  ``contracts.SHOW_WINDOW_MESSAGE_NAME`` and exits.

**Wiring the "show window" callback.** ``acquire()`` starts the listener immediately, so the
instance answers even while it is still starting up. Assign the callback to the public
attribute ``on_show_window`` -- ``instance.on_show_window = ui.show_main_window`` -- at any
time, from any thread; until then a request is only logged. The callback runs on the listener
thread, so it must marshal onto the Qt thread itself (``QtUiFacade`` already does).

Like the Windows version this never refuses to run: if the lock file or the socket cannot be
used for any reason other than "another instance holds it", the failure is logged and
``acquire()`` returns True. That also covers the wrong platform: ``fcntl`` and
``AF_UNIX`` do not exist on Windows, so this module still *imports* there (the platform-guard
rule) and simply reports "we are the only instance".
"""

from __future__ import annotations

import errno
import logging
import os
import socket
import threading
from pathlib import Path
from typing import Callable, Optional

from app.contracts import SHOW_WINDOW_MESSAGE_NAME, data_dir, single_instance_lock_path, \
    single_instance_name

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl; import must still succeed
    fcntl = None  # type: ignore[assignment]

log = logging.getLogger("roenspeak.macos.instance")

#: The listener accepts tiny text messages only.
_MAX_MESSAGE_BYTES = 256
_ACCEPT_BACKLOG = 4
_CONNECT_TIMEOUT_S = 1.0
_JOIN_TIMEOUT_S = 2.0


def _default_on_show_window() -> None:
    log.info("show-window request ignored: no handler is wired yet")


class FileLockSingleInstance:
    """See ``contracts.SingleInstance`` and this module's docstring.

    ``api`` is accepted (and ignored) so that ``app.core.platform.make_single_instance(api)``
    can construct either platform's guard with the same call; nothing here needs the system
    API wrapper. ``name`` overrides the lock/socket base name, as it overrides the mutex name
    on Windows.
    """

    def __init__(self, api=None, name: Optional[str] = None) -> None:
        self._api = api
        self._name = name if name is not None else single_instance_name()
        if name is None:
            self._lock_path = single_instance_lock_path()
        else:
            self._lock_path = data_dir() / (name.replace("\\", ".") + ".lock")
        self._socket_path = self._lock_path.with_suffix(".sock")
        self._lock = threading.Lock()
        self._fd: Optional[int] = None
        self._server: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        #: Called (on the listener thread) when another launch asks us to show the window.
        self.on_show_window: Callable[[], None] = _default_on_show_window

    @property
    def name(self) -> str:
        return self._name

    @property
    def lock_path(self) -> Path:
        return self._lock_path

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    # --- acquire / release ---------------------------------------------------------------

    def acquire(self) -> bool:
        with self._lock:
            if self._fd is not None:
                return True
            if fcntl is None or not hasattr(socket, "AF_UNIX"):
                log.warning("flock/AF_UNIX are unavailable on this platform; "
                            "single-instance check skipped")
                return True
            try:
                self._lock_path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
            except OSError as exc:
                log.warning("could not open %s (%s); single-instance check skipped",
                            self._lock_path, exc)
                return True
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                os.close(fd)
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                    log.info("another instance holds %s", self._lock_path)
                    return False
                # Any other failure (a filesystem with no locking, for instance): don't
                # refuse to run.
                log.warning("flock(%s) failed (%s); single-instance check skipped",
                            self._lock_path, exc)
                return True
            self._fd = fd
            try:
                os.ftruncate(fd, 0)
                os.write(fd, f"{os.getpid()}\n".encode("ascii"))
                os.fsync(fd)
            except OSError as exc:
                log.warning("could not write the pid into %s: %s", self._lock_path, exc)
            log.info("single-instance lock acquired %s", self._lock_path)
            self._start_listener()
            return True

    def release(self) -> None:
        with self._lock:
            self._stop_listener()
            fd, self._fd = self._fd, None
            if fd is None:
                return
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError as exc:
                log.debug("flock(LOCK_UN) failed: %s", exc)
            try:
                os.close(fd)
            except OSError as exc:
                log.debug("closing the lock file failed: %s", exc)
            log.info("single-instance lock released")

    # --- "show window" -------------------------------------------------------------------

    def notify_existing(self) -> None:
        """Ask the running instance to show its window; stay quiet if it cannot be reached."""
        if not hasattr(socket, "AF_UNIX"):
            log.warning("AF_UNIX is unavailable on this platform; cannot reach the running instance")
            return
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.settimeout(_CONNECT_TIMEOUT_S)
            client.connect(str(self._socket_path))
            client.sendall(SHOW_WINDOW_MESSAGE_NAME.encode("utf-8"))
            log.info("asked the running instance to show its window")
        except OSError as exc:
            log.warning("could not reach the running instance on %s: %s", self._socket_path, exc)
        finally:
            try:
                client.close()
            except OSError:
                pass

    def _start_listener(self) -> None:
        """Bind and listen on the sibling socket. Never fatal (we hold the lock either way)."""
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            # We own the lock, so any socket file here is from a dead instance.
            try:
                self._socket_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                log.warning("could not remove the stale socket %s: %s", self._socket_path, exc)
            server.bind(str(self._socket_path))
            server.listen(_ACCEPT_BACKLOG)
        except OSError as exc:
            server.close()
            log.warning("could not listen on %s (%s); a second launch will not be able to "
                        "show this window", self._socket_path, exc)
            return
        self._server = server
        self._stopping.clear()
        thread = threading.Thread(target=self._serve, name="roenspeak-instance", daemon=True)
        self._thread = thread
        thread.start()
        log.debug("show-window listener on %s", self._socket_path)

    def _serve(self) -> None:
        server = self._server
        assert server is not None
        while not self._stopping.is_set():
            try:
                conn, _addr = server.accept()
            except OSError:
                break  # the socket was closed by release()
            try:
                conn.settimeout(_CONNECT_TIMEOUT_S)
                message = conn.recv(_MAX_MESSAGE_BYTES).decode("utf-8", "replace").strip()
            except OSError as exc:
                log.debug("show-window read failed: %s", exc)
                message = ""
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
            if message == SHOW_WINDOW_MESSAGE_NAME:
                log.info("another launch asked us to show the window")
                try:
                    self.on_show_window()
                except Exception:
                    log.exception("on_show_window callback raised")
            elif message:
                log.warning("ignoring unknown instance message %r", message[:64])
        log.debug("show-window listener stopped")

    def _stop_listener(self) -> None:
        self._stopping.set()
        server, self._server = self._server, None
        if server is not None:
            try:
                server.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                server.close()
            except OSError:
                pass
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(_JOIN_TIMEOUT_S)
            if thread.is_alive():
                log.warning("show-window listener did not stop within %.1f s", _JOIN_TIMEOUT_S)
        try:
            self._socket_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.debug("could not remove %s: %s", self._socket_path, exc)
