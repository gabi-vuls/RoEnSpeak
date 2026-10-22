"""Contract tests for app.macos.instance.FileLockSingleInstance (contracts.SingleInstance).

The contract: acquire() is True for the first instance and False while another one holds the
lock, in which case the caller calls notify_existing() (which asks the running instance to
show its window) and exits; release() closes everything and is idempotent. The Windows
counterpart is a named mutex plus a broadcast window message; on macOS it is flock on
contracts.single_instance_lock_path() plus a sibling Unix socket. Like the Windows one it
must never refuse to run: a lock that cannot be created for any reason other than "someone
else holds it" is logged and reported as "we are the only instance".

Isolation: ROENSPEAK_DATA_DIR points at a fresh temp dir, so no real lock or socket is
touched. The dir is created with tempfile (short path) and the instance gets a short name,
because an AF_UNIX socket path is capped at ~104 bytes -- pytest's own tmp_path is already
longer than that, which is exactly the degradation covered by
test_deep_data_dir_keeps_the_lock_but_loses_show_window.
"""

from __future__ import annotations

import errno
import fcntl
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from app import contracts
from app.contracts import SHOW_WINDOW_MESSAGE_NAME, SingleInstance
from app.macos.instance import FileLockSingleInstance

pytestmark = pytest.mark.skipif(not contracts.IS_MACOS, reason="the macOS single-instance guard")

REPO_ROOT = Path(__file__).resolve().parents[2]
#: Short on purpose: str(lock_path) + ".sock" has to stay under the AF_UNIX path limit.
NAME = "rs-test"
#: Generous; nothing here sleeps, every wait is on an event or a process exit.
TIMEOUT_S = 30.0
#: sockaddr_un.sun_path on macOS.
AF_UNIX_PATH_MAX = 104


@pytest.fixture
def instance_dir(monkeypatch):
    base = Path(tempfile.mkdtemp(prefix="rs-si-"))
    monkeypatch.setenv(contracts.ENV_DATA_DIR, str(base))
    yield base
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def instance(instance_dir):
    inst = FileLockSingleInstance(name=NAME)
    try:
        yield inst
    finally:
        inst.release()


def _run_second_instance(*, notify: bool = False) -> bool:
    """Start a second instance in its own process; True if *it* got the lock.

    A separate process is the only honest test of the lock: it inherits ROENSPEAK_DATA_DIR,
    so it computes the same lock path.
    """
    code = (
        "from app.macos.instance import FileLockSingleInstance;"
        f"inst = FileLockSingleInstance(name={NAME!r});"
        "first = inst.acquire();"
        f"(not first and {notify!r}) and inst.notify_existing();"
        "print(first)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                            env=os.environ.copy(), capture_output=True, text=True,
                            timeout=TIMEOUT_S)
    assert result.returncode == 0, result.stderr
    out = result.stdout.strip().splitlines()[-1]
    assert out in ("True", "False"), result.stdout
    return out == "True"


# ---------------------------------------------------------------------------
# Shape and placement
# ---------------------------------------------------------------------------


def test_satisfies_the_single_instance_contract(instance):
    assert isinstance(instance, SingleInstance)


def test_default_lock_path_comes_from_the_contract(instance_dir):
    inst = FileLockSingleInstance()
    assert inst.name == contracts.single_instance_name()
    assert inst.lock_path == contracts.single_instance_lock_path()
    assert inst.lock_path.parent == instance_dir
    if len(str(inst.lock_path.with_suffix(".sock")).encode()) <= AF_UNIX_PATH_MAX - 1:
        assert inst.socket_path == inst.lock_path.with_suffix(".sock")
    else:
        # Too deep a data dir falls back to a short path; see the test below.
        assert len(str(inst.socket_path).encode()) <= AF_UNIX_PATH_MAX - 1


def test_the_test_socket_path_fits_the_af_unix_limit(instance):
    assert len(str(instance.socket_path).encode()) <= AF_UNIX_PATH_MAX


# ---------------------------------------------------------------------------
# acquire / release
# ---------------------------------------------------------------------------


def test_first_instance_acquires_and_acquire_is_idempotent(instance):
    assert instance.acquire() is True
    assert instance.lock_path.exists()
    assert instance.socket_path.exists()
    assert instance.acquire() is True
    assert instance.acquire() is True


def test_second_instance_in_another_process_is_refused(instance):
    assert instance.acquire() is True
    assert _run_second_instance() is False
    assert instance.acquire() is True, "being asked must not cost us the lock"


def test_lock_is_free_again_after_release(instance):
    assert instance.acquire() is True
    assert _run_second_instance() is False
    instance.release()
    assert _run_second_instance() is True


def test_release_is_idempotent_removes_the_socket_and_allows_reacquire(instance):
    assert instance.acquire() is True
    socket_path = instance.socket_path
    assert socket_path.exists()

    instance.release()
    assert not socket_path.exists()
    instance.release()
    instance.release()

    assert instance.acquire() is True
    assert socket_path.exists()


def test_release_without_acquire_is_a_no_op(instance_dir):
    FileLockSingleInstance(name=NAME).release()


def test_two_names_do_not_collide(instance):
    """An isolated instance (suffixed name) runs next to the user's real one."""
    assert instance.acquire() is True
    other = FileLockSingleInstance(name=NAME + "-other")
    try:
        assert other.lock_path != instance.lock_path
        assert other.acquire() is True
    finally:
        other.release()


# ---------------------------------------------------------------------------
# notify_existing -> on_show_window
# ---------------------------------------------------------------------------


def test_notify_existing_reaches_the_holders_callback(instance):
    seen = threading.Event()
    instance.on_show_window = seen.set
    assert instance.acquire() is True

    FileLockSingleInstance(name=NAME).notify_existing()
    assert seen.wait(TIMEOUT_S), "the running instance never got the show-window request"


def test_second_process_asks_the_holder_to_show_its_window(instance):
    calls = []
    seen = threading.Event()

    def on_show_window() -> None:
        calls.append(1)
        seen.set()

    instance.on_show_window = on_show_window
    assert instance.acquire() is True

    assert _run_second_instance(notify=True) is False
    assert seen.wait(TIMEOUT_S), "the second launch did not reach the running instance"
    assert calls == [1]
    assert instance.acquire() is True, "the holder keeps the lock"


def test_callback_can_be_wired_after_acquire(instance):
    assert instance.acquire() is True
    seen = threading.Event()
    instance.on_show_window = seen.set  # listener already running
    FileLockSingleInstance(name=NAME).notify_existing()
    assert seen.wait(TIMEOUT_S)


def test_callback_is_not_called_without_a_request(instance):
    seen = threading.Event()
    instance.on_show_window = seen.set
    assert instance.acquire() is True
    assert not seen.is_set()
    instance.release()
    assert not seen.is_set()


def test_a_raising_callback_does_not_kill_the_listener(instance):
    seen = threading.Event()
    calls = []

    def on_show_window() -> None:
        calls.append(1)
        seen.set()
        raise RuntimeError("boom")

    instance.on_show_window = on_show_window
    assert instance.acquire() is True
    FileLockSingleInstance(name=NAME).notify_existing()
    assert seen.wait(TIMEOUT_S)

    seen.clear()
    FileLockSingleInstance(name=NAME).notify_existing()
    assert seen.wait(TIMEOUT_S), "the listener stopped after the callback raised"
    assert calls == [1, 1]


def test_notify_existing_without_a_running_instance_is_quiet(instance_dir):
    """The second-launch path must not blow up when the holder is already gone."""
    FileLockSingleInstance(name=NAME).notify_existing()


def test_notify_existing_after_release_is_quiet(instance):
    assert instance.acquire() is True
    instance.release()
    instance.notify_existing()


def test_show_window_message_name_is_the_contract_one(instance):
    """A message that is not SHOW_WINDOW_MESSAGE_NAME is not a show-window request."""
    import socket as socket_mod

    assert SHOW_WINDOW_MESSAGE_NAME == f"{contracts.APP_NAME}.ShowWindow"
    seen = threading.Event()
    instance.on_show_window = seen.set
    assert instance.acquire() is True

    client = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
    try:
        client.settimeout(TIMEOUT_S)
        client.connect(str(instance.socket_path))
        client.sendall(b"SomethingElse")
    finally:
        client.close()

    FileLockSingleInstance(name=NAME).notify_existing()
    assert seen.wait(TIMEOUT_S)  # the real message still gets through


# ---------------------------------------------------------------------------
# Never refuse to run
# ---------------------------------------------------------------------------


def test_contention_reports_another_instance(instance_dir, monkeypatch):
    inst = FileLockSingleInstance(name=NAME)

    def busy(fd, operation):
        raise OSError(errno.EAGAIN, "Resource temporarily unavailable")

    monkeypatch.setattr(fcntl, "flock", busy)
    assert inst.acquire() is False


@pytest.mark.parametrize("failure", ["open", "flock"])
def test_an_unrelated_lock_failure_is_logged_and_the_app_still_runs(instance_dir, monkeypatch,
                                                                   caplog, failure):
    """Contract: never refuse to start because the lock could not be created."""
    inst = FileLockSingleInstance(name=NAME)

    if failure == "open":
        real_open = os.open

        def fake_open(path, *args, **kwargs):
            if str(path) == str(inst.lock_path):
                raise PermissionError(errno.EACCES, "Permission denied")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(os, "open", fake_open)
    else:
        def broken(fd, operation):
            raise OSError(errno.EIO, "Input/output error")  # not "someone holds it"

        monkeypatch.setattr(fcntl, "flock", broken)

    with caplog.at_level(logging.WARNING, logger="roenspeak.macos.instance"):
        assert inst.acquire() is True
    assert caplog.records, "the failure must be logged"
    inst.release()


def test_deep_data_dir_still_delivers_show_window(tmp_path, monkeypatch):
    """A data dir too deep for AF_UNIX must not cost us the "show window" channel.

    sun_path is 104 bytes, and macOS's own $TMPDIR is already ~48 of them, so the sibling
    socket path is unbindable for any isolated or deeply nested data dir. That used to fail
    open and silently: the lock still worked, a second launch still exited 0, and the user's
    window just never came to the front. Both sides now derive the same short fallback path
    from the lock path, so the ping keeps working."""
    deep = tmp_path / ("d" * 60) / ("e" * 60)
    deep.mkdir(parents=True)
    monkeypatch.setenv(contracts.ENV_DATA_DIR, str(deep))

    inst = FileLockSingleInstance(name=NAME)
    assert len(str(inst.lock_path.with_suffix(".sock")).encode()) > AF_UNIX_PATH_MAX
    assert len(str(inst.socket_path).encode()) <= AF_UNIX_PATH_MAX - 1
    assert inst.socket_path.parent != inst.lock_path.parent
    seen = threading.Event()
    inst.on_show_window = seen.set
    try:
        assert inst.acquire() is True
        assert inst.lock_path.exists()
        assert inst.socket_path.exists(), "the fallback socket must be bound"

        second = FileLockSingleInstance(name=NAME)
        # Both instances derive the same fallback path from the same lock path.
        assert second.socket_path == inst.socket_path
        assert second.acquire() is False, "the lock itself must still hold"
        second.notify_existing()
        assert seen.wait(3.0), "the show-window ping must reach the running instance"
    finally:
        inst.release()
    assert not inst.socket_path.exists(), "release() must unlink the fallback socket too"

    after = FileLockSingleInstance(name=NAME)
    try:
        assert after.acquire() is True
    finally:
        after.release()
