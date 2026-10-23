"""Desktop end-to-end tests on macOS (markers ``desktop`` + ``metal``): real Carbon hotkey,
real CGEvent injection into a real window.

WARNING: test_dictation_types_english_into_focused_window sends real keystrokes
(Control+Option+Shift+F10) and types text into the frontmost app. Don't touch the keyboard,
the mouse or the trackpad while it runs (~1 min).

    .venv/bin/python -m pytest tests/integration -m desktop -q -s

**Accessibility is required and cannot be faked.** macOS ignores ``CGEventPost`` from an
untrusted process -- silently, with no error -- so both the app under test *and* this pytest
process must hold an Accessibility grant, and the grant is tied to the *interpreter binary*
(``.venv/bin/python``, or the Terminal/iTerm that launched it). There is no way to grant it
from code: ``AXIsProcessTrustedWithOptions`` only opens the system prompt. So the injection
test **skips with an actionable message** when ``app.macos.permissions.accessibility_granted()``
is False, rather than failing on a test-environment problem. To grant it:

    System Settings > Privacy & Security > Accessibility > + >
    /Users/<you>/Personal/projects/RoEnSpeak/.venv/bin/python3.12
    (add your terminal app too if you run pytest from one), then restart the terminal.

How test_dictation_types_english_into_focused_window works:
  1. Writes an isolated config (ROENSPEAK_CONFIG_DIR = temp): armed, hotkey
     Ctrl+Alt+Shift+F10 (which the macOS parser reads as Control+Option+Shift+F10, so it
     can't clash with a real instance on Alt+Q/Option+Q; the flock + socket single-instance
     files live under the temp ROENSPEAK_DATA_DIR, so a real running instance is untouched).
  2. Opens a tkinter target window in a subprocess (_mac_target.py), which reports its pid
     and mirrors its text to a file.
  3. Starts ``python -m app.main --debug-audio-wav tests/fixtures/ro_sample.wav`` and waits
     for LOG_ENGINE_LOADED in <DATA_DIR>/logs/roenspeak.log.
  4. Activates the target app, posts the hotkey (start; the wav plays in realtime), waits for
     the wav duration + margin, posts the hotkey again (stop; flush).
  5. Polls the target's text until English with an expected keyword appears and stays stable.
  6. Checks append-only output (no duplicate sentences, single spaces between segments) and
     the segment_latency log lines, then terminates the app and the target.

test_second_instance_exits_quickly_with_code_0 needs no Accessibility (it posts nothing): it
only checks FileLockSingleInstance's contract -- flock held by the first instance, the second
pings the sibling Unix socket and exits 0.

Both tests need a real, logged-in GUI session (Qt and the WindowServer), which is what the
``desktop`` marker means; ``metal`` pairs with it so tests/conftest.py routes this file to a
Mac and the Windows file to a Windows host.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from app import contracts

from .helpers import (
    REPO_ROOT,
    RO_SAMPLE_WAV,
    app_env,
    assert_english,
    log_records,
    require_ggml_model,
    safe_print,
    wav_duration_s,
)

pytestmark = [
    pytest.mark.desktop,
    pytest.mark.metal,
    pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only (Carbon hotkey + CGEvent)"),
]

#: Parsed by app.macos.hotkey_parse as Control+Option+Shift+F10. Same string as the Windows
#: test uses, so one config.json stays valid on both platforms (ARCHITECTURE.md 12.2).
TEST_HOTKEY = "Ctrl+Alt+Shift+F10"

#: Carbon kVK_* codes (HIToolbox/Events.h), the same table app/macos/hotkey_parse.py uses.
KVK_SHIFT, KVK_CONTROL, KVK_OPTION, KVK_F10 = 0x38, 0x3B, 0x3A, 0x6D
#: CGEventFlags.
CG_FLAG_SHIFT, CG_FLAG_CONTROL, CG_FLAG_OPTION = 0x00020000, 0x00040000, 0x00080000
#: kCGHIDEventTap: posts at the lowest level, where the WindowServer dispatches hotkeys.
KCG_HID_EVENT_TAP = 0

#: (keycode, flags held once this key is down), in press order. Release is the reverse.
HOTKEY_SEQUENCE = (
    (KVK_CONTROL, CG_FLAG_CONTROL),
    (KVK_OPTION, CG_FLAG_CONTROL | CG_FLAG_OPTION),
    (KVK_SHIFT, CG_FLAG_CONTROL | CG_FLAG_OPTION | CG_FLAG_SHIFT),
)
HOTKEY_FLAGS = CG_FLAG_CONTROL | CG_FLAG_OPTION | CG_FLAG_SHIFT

ENGINE_TIMEOUT_S = 180
TEXT_TIMEOUT_S = 30
MAC_TARGET = Path(__file__).with_name("_mac_target.py")

#: NSApplicationActivationOptions (see _mac_target.py).
NS_ACTIVATE_ALL_WINDOWS = 1 << 0
NS_ACTIVATE_IGNORING_OTHER_APPS = 1 << 1

ACCESSIBILITY_SKIP = (
    "Accessibility permission is not granted to this interpreter, so macOS silently drops "
    f"every synthetic keystroke. Grant it in System Settings > Privacy & Security > "
    f"Accessibility to {sys.executable} (and to the terminal app you run pytest from), "
    "restart the terminal, and run again. Nothing in code can grant it."
)


# ---------------------------------------------------------------------------
# CGEvent / activation via ctypes (deliberately not app.macos._mac_api: the harness must not
# be the thing under test)
# ---------------------------------------------------------------------------


class _Mac:
    """The handful of CoreGraphics/AppKit entry points this test needs."""

    def __init__(self) -> None:
        cg = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
        )
        cf = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AppKit.framework/AppKit")

        V, U32, U16, B = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint16, ctypes.c_bool
        cg.CGEventCreateKeyboardEvent.restype = V
        cg.CGEventCreateKeyboardEvent.argtypes = [V, U16, B]
        cg.CGEventSetFlags.restype = None
        cg.CGEventSetFlags.argtypes = [V, ctypes.c_uint64]
        cg.CGEventPost.restype = None
        cg.CGEventPost.argtypes = [U32, V]
        cf.CFRelease.restype = None
        cf.CFRelease.argtypes = [V]
        objc.objc_getClass.restype = V
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = V
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self.cg, self.cf, self.objc = cg, cf, objc

    # -- Objective-C helpers -------------------------------------------------------

    def _cls(self, name: bytes) -> int:
        return self.objc.objc_getClass(name)

    def _sel(self, name: bytes) -> int:
        return self.objc.sel_registerName(name)

    def _send(self, restype, argtypes, obj, selector: bytes, *args):
        fn = self.objc.objc_msgSend
        fn.restype = restype
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *argtypes]
        return fn(obj, self._sel(selector), *args)

    # -- keyboard -----------------------------------------------------------------

    def post_key(self, keycode: int, down: bool, flags: int) -> None:
        event = self.cg.CGEventCreateKeyboardEvent(None, keycode & 0xFFFF, down)
        assert event, f"CGEventCreateKeyboardEvent returned NULL (keycode={keycode:#x}, down={down})"
        try:
            self.cg.CGEventSetFlags(event, flags)
            self.cg.CGEventPost(KCG_HID_EVENT_TAP, event)
        finally:
            self.cf.CFRelease(event)

    def press_hotkey(self) -> None:
        """Control down, Option down, Shift down, F10 down/up, modifiers up in reverse order.

        Posting the modifier keys (and not only the flags on the F10 event) is what makes the
        Carbon hot key fire the way a real press does, and matches what the Windows test does
        with SendInput.
        """
        try:
            for keycode, flags in HOTKEY_SEQUENCE:
                self.post_key(keycode, True, flags)
                time.sleep(0.02)
            self.post_key(KVK_F10, True, HOTKEY_FLAGS)
            time.sleep(0.05)
            self.post_key(KVK_F10, False, HOTKEY_FLAGS)
        finally:
            self.release_all_modifiers()

    def release_all_modifiers(self) -> None:
        """Never leave a modifier stuck down, whatever failed (the analogue of the Windows
        test's release_all_modifiers; a stuck Control would wreck the user's session)."""
        for keycode, _ in reversed(HOTKEY_SEQUENCE):
            try:
                self.post_key(keycode, False, 0)
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    # -- activation ---------------------------------------------------------------

    def frontmost_pid(self) -> int:
        ws = self._send(ctypes.c_void_p, [], self._cls(b"NSWorkspace"), b"sharedWorkspace")
        app = self._send(ctypes.c_void_p, [], ws, b"frontmostApplication")
        if not app:
            return 0
        return int(self._send(ctypes.c_int32, [], app, b"processIdentifier"))

    def activate_pid(self, pid: int) -> bool:
        app = self._send(
            ctypes.c_void_p, [ctypes.c_int32],
            self._cls(b"NSRunningApplication"), b"runningApplicationWithProcessIdentifier:", pid,
        )
        if not app:
            return False
        return bool(self._send(
            ctypes.c_bool, [ctypes.c_ulong], app, b"activateWithOptions:",
            NS_ACTIVATE_ALL_WINDOWS | NS_ACTIVATE_IGNORING_OTHER_APPS,
        ))


@pytest.fixture(scope="module")
def mac() -> _Mac:
    return _Mac()


def focus_app(mac: _Mac, pid: int, out_dir: Path, timeout: float = 20.0) -> None:
    """Make the process `pid` the frontmost application.

    There is no HWND and no SetForegroundWindow on macOS: activation is per application, so
    "frontmost app == target pid" is the check. Getting there needs the target's cooperation:
    macOS only honours a *cross*-app activation from a process that is itself an active app,
    and pytest is a background CLI tool, so ``activateWithOptions:`` returns YES and does
    nothing (measured: 20 attempts over 6 s never took the front from Finder). Hence the
    ``raise`` sentinel, which asks _mac_target.py to activate *itself* -- that always works.
    The cross-app call is still made as a cheap belt-and-braces.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mac.frontmost_pid() == pid:
            return
        (out_dir / "raise").touch()
        mac.activate_pid(pid)
        time.sleep(0.3)
    pytest.fail(
        f"could not bring the test window (pid {pid}) to the front "
        f"(frontmost pid={mac.frontmost_pid()}); is another app holding activation?"
    )


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------


def write_config(config_dir: Path, *, armed: bool) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "schema_version": contracts.CONFIG_SCHEMA_VERSION,
        "armed": armed,
        "hotkey": TEST_HOTKEY,
        "model": contracts.DEFAULT_MODEL,
        "compute_type": contracts.ComputeType.FLOAT16.value,
        "task": contracts.Task.TRANSLATE.value,
        # SENDINPUT is the direct-typing mode on both platforms; on macOS it means
        # CGEventKeyboardSetUnicodeString (see contracts.InjectionMode).
        "injection_mode": contracts.InjectionMode.SENDINPUT.value,
        "cpu_mode": False,
        # start_with_windows keeps its name on macOS (no schema migration); only the UI label
        # changes to "Open at login". False here so no LaunchAgent plist is ever written.
        "ui": {"overlay": False, "sounds": False, "start_with_windows": False, "first_run_done": True},
    }
    path = config_dir / "config.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


class AppProcess:
    def __init__(self, base: Path, name: str, extra_args: list[str]):
        self.base = base
        self.data_dir = base / "data"
        self.config_dir = base / "config"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.out = open(base / f"{name}.stdout.txt", "wb")
        self.err = open(base / f"{name}.stderr.txt", "wb")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "app.main", *extra_args],
            cwd=str(REPO_ROOT),
            env=app_env(self.data_dir, self.config_dir),
            stdout=self.out,
            stderr=self.err,
            # Own process group, so the escalation in terminate() can take the whole tree
            # down even if Qt or the inference thread ignores SIGTERM.
            start_new_session=True,
        )
        self.name = name

    @property
    def log_path(self) -> Path:
        return self.data_dir / "logs" / "roenspeak.log"

    def log_text(self) -> str:
        p = self.log_path
        return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

    def diagnostics(self) -> str:
        self.out.flush()
        self.err.flush()
        err = (self.base / f"{self.name}.stderr.txt").read_text(encoding="utf-8", errors="replace")
        return (f"exit={self.proc.poll()}\n--- log tail ---\n{self.log_text()[-3000:]}"
                f"\n--- stderr tail ---\n{err[-3000:]}")

    def wait_for_log(self, key: str, timeout: float) -> dict[str, str]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            recs = log_records(self.log_text(), key)
            if recs:
                return recs[-1]
            if self.proc.poll() is not None:
                pytest.fail(f"app exited before logging {key!r}\n{self.diagnostics()}")
            time.sleep(0.5)
        pytest.fail(f"timed out after {timeout}s waiting for {key!r} in {self.log_path}\n{self.diagnostics()}")

    def terminate(self) -> int:
        """SIGTERM, then SIGKILL, then the whole process group. Never leaves anything behind."""
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
            try:
                code = self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                try:
                    code = self.proc.wait(timeout=15)
                except subprocess.TimeoutExpired:  # pragma: no cover - last resort
                    try:
                        os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    code = self.proc.wait(timeout=15)
            return code
        finally:
            self.out.close()
            self.err.close()


@pytest.fixture
def injection_possible():
    """Skip the injection test *before* anything is launched.

    Requested by mac_target, so the skip happens while pytest is still building fixtures: no
    window flashes up and no app is started for a test that cannot run.
    """
    from app.macos.permissions import accessibility_granted

    if not accessibility_granted():
        pytest.skip(ACCESSIBILITY_SKIP)
    require_ggml_model()


@pytest.fixture
def mac_target(tmp_path, injection_possible):
    """The tkinter target window, as (pid, text_file, out_dir). Always torn down."""
    out = tmp_path / "target"
    out.mkdir()
    proc = subprocess.Popen([sys.executable, str(MAC_TARGET), str(out)], cwd=str(REPO_ROOT),
                            start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        while not (out / "pid.txt").exists():
            if proc.poll() is not None:
                pytest.fail(f"the tkinter test window exited with {proc.returncode} before it was ready")
            if time.monotonic() > deadline:
                pytest.fail("the tkinter test window did not become ready within 30 s")
            time.sleep(0.1)
        pid = int((out / "pid.txt").read_text(encoding="utf-8").strip())
        assert pid == proc.pid, f"target reported pid {pid} but we spawned {proc.pid}"
        yield pid, out / "text.txt", out
    finally:
        (out / "quit").touch()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - last resort
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass


def read_target(text_file: Path) -> str:
    try:
        return text_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [re.sub(r"\W+", " ", p).strip().casefold() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dictation_types_english_into_focused_window(tmp_path, mac, mac_target):
    pid, text_file, target_dir = mac_target
    write_config(tmp_path / "config", armed=True)
    app = AppProcess(tmp_path, "app", ["--debug-audio-wav", str(RO_SAMPLE_WAV)])
    try:
        loaded = app.wait_for_log(contracts.LOG_ENGINE_LOADED, ENGINE_TIMEOUT_S)
        assert loaded.get("device") == "metal" and loaded.get("compute_type") == "f16", loaded
        time.sleep(3.0)  # warm-up inference + Carbon hotkey registration settle
        assert app.proc.poll() is None, app.diagnostics()

        focus_app(mac, pid, target_dir)
        assert read_target(text_file) == "", "target window not empty before dictation"
        mac.press_hotkey()  # start: the wav plays in realtime
        time.sleep(wav_duration_s(RO_SAMPLE_WAV) + 1.5)
        assert mac.frontmost_pid() == pid, "focus left the test window during dictation (did someone click?)"
        mac.press_hotkey()  # stop: flush

        deadline = time.monotonic() + TEXT_TIMEOUT_S
        text, stable_since = "", time.monotonic()
        while time.monotonic() < deadline:
            cur = read_target(text_file)
            if cur != text:
                text, stable_since = cur, time.monotonic()
            elif text.strip() and time.monotonic() - stable_since >= 4.0:
                break
            time.sleep(0.2)
        safe_print(f"\n[desktop] injected text: {text!r}")
        assert text.strip(), f"no text was injected within {TEXT_TIMEOUT_S}s\n{app.diagnostics()}"
        assert mac.frontmost_pid() == pid, "focus left the test window before the text was checked"

        assert_english(text)
        # Append-only, one space between segments, first segment without a leading space.
        assert text == text.strip(), f"leading/trailing whitespace: {text!r}"
        assert "  " not in text, f"double space between segments: {text!r}"
        assert "\n" not in text and "\t" not in text, f"newline/tab injected: {text!r}"
        assert not re.search(r"[.!?,][A-Za-z]", text), f"segments glued without a space: {text!r}"
        sents = sentences(text)
        dups = {s for s in sents if sents.count(s) > 1}
        assert not dups, f"duplicated sentences (not append-only / session restarted?): {dups} in {text!r}"

        lat = log_records(app.log_text(), contracts.LOG_SEGMENT_LATENCY)
        safe_print("[desktop] segment_latency:",
                   [(r.get("id"), r.get("reason"), r.get("infer_ms"), r.get("latency_ms"),
                     r.get("filtered")) for r in lat])
        assert len(lat) >= 2, f"expected >= 2 segment_latency lines\n{app.diagnostics()}"
        kept = [r for r in lat if r.get("filtered") in ("0", "False", "false")]
        assert kept, lat
        for r in kept:
            v = float(r["latency_ms"])
            assert 0 < v < 10_000, r
    finally:
        try:
            mac.release_all_modifiers()
        finally:
            app.terminate()
    assert app.proc.returncode is not None


@pytest.fixture
def short_base():
    """A base dir short enough for an AF_UNIX socket path, always cleaned up.

    ``sockaddr_un.sun_path`` is 104 bytes on macOS, and the single-instance socket sits at
    ``<DATA_DIR>/RoEnSpeak.SingleInstance.<12 hex>.sock`` -- 42 bytes on its own, plus
    "/data/". pytest's ``tmp_path`` lives under ``$TMPDIR``, which is already ~48 bytes of
    ``/var/folders/<2>/<28>/T/`` on macOS, so a pytest tmp dir cannot hold this socket at all
    and the app degrades to "lock only, no ping" (see the REPORT note: that is a real
    implementation limit, not a test artefact). Rooting this one test at /tmp is the only way
    to exercise the ping half of the contract on a Mac.
    """
    import shutil
    import tempfile

    base = Path(tempfile.mkdtemp(prefix="rs-si-", dir="/tmp"))
    sock_len = len(str(base / "data" / f"{contracts.single_instance_name()}.sock").encode())
    assert sock_len < 104, f"base {base} is already too long for AF_UNIX ({sock_len} bytes)"
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_second_instance_exits_quickly_with_code_0(short_base):
    """Spec: single instance. On macOS that is flock on single_instance_lock_path() plus a
    sibling Unix socket (ARCHITECTURE.md 12.2): a second launch with the same data dir asks
    the first to show its window and exits 0. Disarmed config, so no model load and no hotkey
    registration -- and no Accessibility needed, since nothing is injected."""
    tmp_path = short_base
    write_config(tmp_path / "config", armed=False)
    first = AppProcess(tmp_path, "first", [])
    try:
        deadline = time.monotonic() + 60
        while not first.log_text():
            if first.proc.poll() is not None:
                pytest.fail(f"first instance exited early\n{first.diagnostics()}")
            if time.monotonic() > deadline:
                pytest.fail(f"first instance never wrote its log\n{first.diagnostics()}")
            time.sleep(0.3)
        time.sleep(5.0)  # let it take the lock, bind the socket and build the UI
        assert first.proc.poll() is None, first.diagnostics()

        # single_instance_name() is suffixed with a hash of ROENSPEAK_CONFIG_DIR, so glob for
        # the pair rather than recomputing the name in this process.
        locks = sorted(first.data_dir.glob("*.lock"))
        socks = sorted(first.data_dir.glob("*.sock"))
        assert locks, f"no single-instance lock file in {first.data_dir}\n{first.diagnostics()}"
        assert socks, f"no show-window socket in {first.data_dir}\n{first.diagnostics()}"
        assert locks[0].stem == socks[0].stem, (locks, socks)
        assert contracts.APP_NAME in locks[0].name, locks

        t0 = time.monotonic()
        second = AppProcess(tmp_path, "second", [])
        try:
            code = second.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            second.terminate()
            pytest.fail("second instance did not exit within 30 s (flock single-instance not honoured)")
        elapsed = time.monotonic() - t0
        second.out.close()
        second.err.close()
        safe_print(f"\n[desktop] second instance exited with {code} after {elapsed:.1f}s")
        assert code == 0, second.diagnostics()
        assert elapsed < 30
        time.sleep(1.0)
        assert first.proc.poll() is None, "first instance died when the second launched"
        # The ping really reached the running instance (FileLockSingleInstance._serve logs
        # this exact line when it reads SHOW_WINDOW_MESSAGE_NAME off the socket).
        assert "another launch asked us to show the window" in first.log_text(), (
            f"the running instance never received the show-window ping\n{first.diagnostics()}"
        )
    finally:
        first.terminate()
