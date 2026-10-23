"""Desktop end-to-end tests (marker ``desktop``): real hotkey, real SendInput into a real window.

WARNING: these tests send real keystrokes (Ctrl+Alt+Shift+F10) and type text into the focused
window. Don't touch the keyboard or mouse while they run (~1 min).

    scripts/win.sh -m pytest tests/integration -m desktop -q -s

How test_dictation_types_english_into_focused_window works:
  1. Writes an isolated config (ROENSPEAK_CONFIG_DIR = temp): armed, hotkey Ctrl+Alt+Shift+F10
     (so it can't clash with a real instance on Alt+Q; the single-instance mutex is suffixed
     per config dir), overlay and sounds off, first_run_done so no first-run dialog.
  2. Opens a tkinter target window in a subprocess (_tk_target.py), which reports its HWND and
     mirrors its text to a file.
  3. Starts ``python -m app.main --debug-audio-wav tests/fixtures/ro_sample.wav`` and waits for
     LOG_ENGINE_LOADED in <DATA_DIR>/logs/roenspeak.log.
  4. Focuses the target, SendInput hotkey (start; the wav plays in realtime), waits for the wav
     duration + margin, SendInput hotkey (stop; flush).
  5. Polls the target's text until English with an expected keyword appears and stays stable.
  6. Checks append-only output (no duplicate sentences, single spaces between segments) and the
     segment_latency log lines, then terminates the app and the target.
"""

from __future__ import annotations

import ctypes
import json
import re
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

import pytest

from app import contracts

from .helpers import (
    safe_print,
    REPO_ROOT,
    RO_SAMPLE_WAV,
    app_env,
    assert_english,
    log_records,
    require_model,
    wav_duration_s,
)

pytestmark = [
    pytest.mark.desktop,
    # Windows-only harness (SendInput/HWND via ctypes.windll). tests/conftest.py gates by
    # marker, and `desktop` now runs on whichever host selects it, so this file needs its
    # own host guard -- otherwise `-m desktop` on a Mac would run it and die in windll.
    # The macOS counterpart is test_desktop_injection_mac.py.
    pytest.mark.skipif(sys.platform != "win32", reason="needs Windows (SendInput)"),
]

TEST_HOTKEY = "Ctrl+Alt+Shift+F10"
VK_SHIFT, VK_CONTROL, VK_MENU, VK_F10 = 0x10, 0x11, 0x12, 0x79
HOTKEY_VKS = (VK_CONTROL, VK_MENU, VK_SHIFT, VK_F10)
ENGINE_TIMEOUT_S = 180
TEXT_TIMEOUT_S = 30
TK_TARGET = Path(__file__).with_name("_tk_target.py")


# ---------------------------------------------------------------------------
# SendInput / focus via ctypes
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    ULONG_PTR = ctypes.c_size_t

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]

    INPUT_KEYBOARD, KEYEVENTF_KEYUP = 1, 0x0002
    user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
    user32.SendInput.restype = wintypes.UINT
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.AttachThreadInput.argtypes = (wintypes.DWORD, wintypes.DWORD, wintypes.BOOL)
    user32.BringWindowToTop.argtypes = (wintypes.HWND,)


def _send_keys(events: list[tuple[int, bool]]) -> None:
    arr = (INPUT * len(events))()
    for i, (vk, up) in enumerate(events):
        arr[i].type = INPUT_KEYBOARD
        arr[i].ki = KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP if up else 0, 0, 0)
    sent = user32.SendInput(len(events), arr, ctypes.sizeof(INPUT))
    assert sent == len(events), f"SendInput inserted {sent}/{len(events)} (error {ctypes.get_last_error()})"


def press_hotkey() -> None:
    """Ctrl down, Alt down, Shift down, F10 down/up, then modifiers up in reverse order."""
    downs = [(vk, False) for vk in HOTKEY_VKS]
    ups = [(vk, True) for vk in reversed(HOTKEY_VKS)]
    try:
        _send_keys(downs)
        time.sleep(0.05)
    finally:
        _send_keys(ups)


def release_all_modifiers() -> None:
    _send_keys([(vk, True) for vk in (VK_F10, VK_SHIFT, VK_MENU, VK_CONTROL)])


def foreground() -> int:
    return user32.GetForegroundWindow() or 0


def focus_window(hwnd: int, timeout: float = 5.0) -> None:
    """Make hwnd the foreground window despite the foreground lock: plain call, then the
    Alt-tap trick, then AttachThreadInput to the current foreground thread."""
    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        if foreground() == hwnd:
            return
        attempt += 1
        if attempt % 3 == 1:
            user32.SetForegroundWindow(hwnd)
        elif attempt % 3 == 2:
            _send_keys([(VK_MENU, False)])
            try:
                user32.SetForegroundWindow(hwnd)
            finally:
                _send_keys([(VK_MENU, True)])
        else:
            fg = foreground()
            fg_tid = user32.GetWindowThreadProcessId(fg, None) if fg else 0
            me = kernel32.GetCurrentThreadId()
            if fg_tid and fg_tid != me:
                user32.AttachThreadInput(me, fg_tid, True)
                try:
                    user32.BringWindowToTop(hwnd)
                    user32.SetForegroundWindow(hwnd)
                finally:
                    user32.AttachThreadInput(me, fg_tid, False)
        time.sleep(0.2)
    pytest.fail(f"could not bring the test window {hwnd} to the foreground (foreground={foreground()})")


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
        "injection_mode": contracts.InjectionMode.SENDINPUT.value,
        "cpu_mode": False,
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
        return f"exit={self.proc.poll()}\n--- log tail ---\n{self.log_text()[-3000:]}\n--- stderr tail ---\n{err[-3000:]}"

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
        if self.proc.poll() is None:
            self.proc.terminate()
        try:
            code = self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            code = self.proc.wait(timeout=15)
        self.out.close()
        self.err.close()
        return code


@pytest.fixture
def tk_target(tmp_path):
    out = tmp_path / "tk"
    out.mkdir()
    proc = subprocess.Popen([sys.executable, str(TK_TARGET), str(out)], cwd=str(REPO_ROOT))
    try:
        deadline = time.monotonic() + 20
        while not (out / "hwnd.txt").exists():
            if proc.poll() is not None or time.monotonic() > deadline:
                pytest.fail("tkinter test window did not start")
            time.sleep(0.1)
        hwnd = int((out / "hwnd.txt").read_text(encoding="utf-8"))
        yield hwnd, out / "text.txt"
    finally:
        (out / "quit").touch()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


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


def test_dictation_types_english_into_focused_window(tmp_path, tk_target):
    require_model()
    hwnd, text_file = tk_target
    write_config(tmp_path / "config", armed=True)
    app = AppProcess(tmp_path, "app", ["--debug-audio-wav", str(RO_SAMPLE_WAV)])
    try:
        loaded = app.wait_for_log(contracts.LOG_ENGINE_LOADED, ENGINE_TIMEOUT_S)
        assert loaded.get("device") == "cuda" and loaded.get("compute_type") == "float16", loaded
        time.sleep(3.0)  # warm-up inference + hotkey registration settle
        assert app.proc.poll() is None, app.diagnostics()

        focus_window(hwnd)
        assert read_target(text_file) == "", "target window not empty before dictation"
        press_hotkey()  # start: the wav plays in realtime
        time.sleep(wav_duration_s(RO_SAMPLE_WAV) + 1.5)
        assert foreground() == hwnd, "focus left the test window during dictation (did someone click?)"
        press_hotkey()  # stop: flush

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
        assert foreground() == hwnd, "focus left the test window before the text was checked"

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
        safe_print("[desktop] segment_latency:", [(r.get("id"), r.get("reason"), r.get("infer_ms"),
                                              r.get("latency_ms"), r.get("filtered")) for r in lat])
        assert len(lat) >= 2, f"expected >= 2 segment_latency lines\n{app.diagnostics()}"
        kept = [r for r in lat if r.get("filtered") in ("0", "False", "false")]
        assert kept, lat
        for r in kept:
            v = float(r["latency_ms"])
            assert 0 < v < 10_000, r
    finally:
        try:
            release_all_modifiers()
        finally:
            app.terminate()
    assert app.proc.returncode is not None


def test_second_instance_exits_quickly_with_code_0(tmp_path):
    """Spec: single instance (named mutex). A second launch with the same config dir asks the
    first to show its window and exits. Disarmed config, so no model load or hotkey."""
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
        time.sleep(5.0)  # let it acquire the mutex and build the UI
        assert first.proc.poll() is None, first.diagnostics()

        t0 = time.monotonic()
        second = AppProcess(tmp_path, "second", [])
        try:
            code = second.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            second.terminate()
            pytest.fail("second instance did not exit within 30 s (single-instance mutex not honoured)")
        elapsed = time.monotonic() - t0
        second.out.close()
        second.err.close()
        safe_print(f"\n[desktop] second instance exited with {code} after {elapsed:.1f}s")
        assert code == 0, second.diagnostics()
        assert elapsed < 30
        time.sleep(1.0)
        assert first.proc.poll() is None, "first instance died when the second launched"
    finally:
        first.terminate()
