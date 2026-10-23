"""Contract tests for app.macos.autostart.LaunchAgentAutostart (contracts.Autostart).

The contract is is_enabled()/set_enabled(bool) and ConfigError on failure. On macOS "Open at
login" is a LaunchAgent plist (~/Library/LaunchAgents/com.roenspeak.RoEnSpeak.plist with
RunAtLoad and --minimized) instead of the Windows HKCU Run value; the config field keeps its
name (ui.start_with_windows), only the label changes.

SAFETY: these tests must never touch the user's real ~/Library/LaunchAgents and must never
run launchctl -- loading a RunAtLoad agent would start the app. The autouse fixture below
redirects the module's LaunchAgents directory into tmp_path and replaces subprocess.run with
a recorder, so a real launchctl invocation is impossible for the duration of a test.
"""

from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

from app import contracts
from app.contracts import Autostart, ConfigError
from app.macos import autostart as autostart_mod
from app.macos.autostart import BUNDLE_ID, LaunchAgentAutostart, default_command

pytestmark = pytest.mark.skipif(not contracts.IS_MACOS, reason="the macOS LaunchAgent autostart")

#: A frozen-build command: deterministic argv, unlike the dev command (see
#: test_default_command_asks_for_a_minimized_start).
COMMAND = "/Applications/RoEnSpeak.app/Contents/MacOS/RoEnSpeak --minimized"


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """Redirect the LaunchAgents dir and make launchctl unreachable."""
    agents = tmp_path / "LaunchAgents"
    monkeypatch.setattr(autostart_mod, "LAUNCH_AGENTS_DIR", agents)
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append([str(part) for part in cmd])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    # Patched on the subprocess module itself: nothing in this test file can spawn launchctl.
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert autostart_mod.plist_path() == agents / f"{BUNDLE_ID}.plist"
    return {"dir": agents, "plist": agents / f"{BUNDLE_ID}.plist", "launchctl": calls}


@pytest.fixture
def agent():
    return LaunchAgentAutostart(COMMAND)


def _load(path: Path) -> dict:
    with path.open("rb") as handle:
        return plistlib.load(handle)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_satisfies_the_autostart_contract(agent):
    assert isinstance(agent, Autostart)


def test_bundle_id_names_the_plist_and_the_label():
    assert BUNDLE_ID == "com.roenspeak.RoEnSpeak"
    assert autostart_mod.plist_path().name == f"{BUNDLE_ID}.plist"
    assert autostart_mod.plist_path().parent == autostart_mod.LAUNCH_AGENTS_DIR


def test_program_arguments_split_the_command(agent):
    assert agent.program_arguments == [
        "/Applications/RoEnSpeak.app/Contents/MacOS/RoEnSpeak", "--minimized"]


def test_a_quoted_command_keeps_its_spaces():
    quoted = LaunchAgentAutostart('"/Applications/My App.app/Contents/MacOS/RoEnSpeak" --minimized')
    assert quoted.program_arguments == [
        "/Applications/My App.app/Contents/MacOS/RoEnSpeak", "--minimized"]


def test_default_command_asks_for_a_minimized_start():
    args = LaunchAgentAutostart().program_arguments
    assert args, "the default command must not be empty"
    # Dev (python -m app.main / -c ...) and frozen (the binary inside RoEnSpeak.app) both
    # start argv with sys.executable; only the dev forms carry --minimized inside -c code.
    assert "--minimized" in " ".join(args), args
    assert args[0] == sys.executable
    assert default_command().strip()


# ---------------------------------------------------------------------------
# Enabling
# ---------------------------------------------------------------------------


def test_is_enabled_is_false_without_a_plist(agent, sandbox):
    assert not sandbox["plist"].exists()
    assert agent.is_enabled() is False


def test_set_enabled_true_writes_a_valid_plist(agent, sandbox):
    agent.set_enabled(True)

    plist = sandbox["plist"]
    assert plist.exists()
    data = _load(plist)  # a parse failure here is a failed test
    assert data["Label"] == BUNDLE_ID
    assert data["ProgramArguments"] == agent.program_arguments
    assert "--minimized" in data["ProgramArguments"]
    assert data["RunAtLoad"] is True
    assert all(isinstance(part, str) for part in data["ProgramArguments"])
    assert sorted(p.name for p in sandbox["dir"].iterdir()) == [plist.name], "no temp leftovers"
    assert agent.is_enabled() is True


def test_set_enabled_true_creates_the_launch_agents_dir(agent, sandbox):
    assert not sandbox["dir"].exists()
    agent.set_enabled(True)
    assert sandbox["dir"].is_dir()


def test_set_enabled_true_is_idempotent(agent, sandbox):
    agent.set_enabled(True)
    first = sandbox["plist"].read_bytes()
    agent.set_enabled(True)
    assert sandbox["plist"].read_bytes() == first
    assert agent.is_enabled() is True
    assert sorted(p.name for p in sandbox["dir"].iterdir()) == [sandbox["plist"].name]


def test_launchctl_is_asked_to_load_the_agent(agent, sandbox):
    agent.set_enabled(True)
    calls = sandbox["launchctl"]
    assert calls, "the change should also apply to the running login session"
    assert [call[1] for call in calls][-1] == "load"
    for call in calls:
        assert call[0] == autostart_mod.LAUNCHCTL
        assert call[1] in ("load", "unload")
        assert call[-1] == str(sandbox["plist"])
        assert "-w" in call


def test_a_failing_launchctl_is_not_fatal(agent, sandbox, monkeypatch):
    """launchd reads ~/Library/LaunchAgents at the next login by itself."""
    def failing_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "Load failed: 5: Input/output error")

    monkeypatch.setattr(subprocess, "run", failing_run)
    agent.set_enabled(True)
    assert agent.is_enabled() is True


def test_a_missing_launchctl_is_not_fatal(agent, monkeypatch):
    def missing_run(cmd, *args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", str(cmd[0]))

    monkeypatch.setattr(subprocess, "run", missing_run)
    agent.set_enabled(True)
    assert agent.is_enabled() is True


# ---------------------------------------------------------------------------
# Reading someone else's plist
# ---------------------------------------------------------------------------


def _write_plist(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        plistlib.dump(payload, handle)


def test_a_plist_for_a_different_command_reads_as_disabled(agent, sandbox):
    """An older install (moved .app, other interpreter) would not start this build, so the
    checkbox must show "off" and rewrite the file when switched on."""
    _write_plist(sandbox["plist"], {"Label": BUNDLE_ID, "RunAtLoad": True,
                                    "ProgramArguments": ["/old/RoEnSpeak.app/Contents/MacOS/RoEnSpeak",
                                                         "--minimized"]})
    assert agent.is_enabled() is False

    agent.set_enabled(True)
    assert _load(sandbox["plist"])["ProgramArguments"] == agent.program_arguments
    assert agent.is_enabled() is True


@pytest.mark.parametrize("payload", [
    {"Label": BUNDLE_ID, "RunAtLoad": True},
    {"Label": BUNDLE_ID, "RunAtLoad": True, "ProgramArguments": []},
    {"Label": BUNDLE_ID, "RunAtLoad": True, "ProgramArguments": "not a list"},
], ids=["missing", "empty", "not-a-list"])
def test_a_plist_without_usable_program_arguments_reads_as_disabled(agent, sandbox, payload):
    _write_plist(sandbox["plist"], payload)
    assert agent.is_enabled() is False


def test_an_unreadable_plist_reads_as_disabled(agent, sandbox):
    sandbox["plist"].parent.mkdir(parents=True, exist_ok=True)
    sandbox["plist"].write_bytes(b"\x00not a plist at all")
    assert agent.is_enabled() is False  # must not raise


# ---------------------------------------------------------------------------
# Disabling
# ---------------------------------------------------------------------------


def test_set_enabled_false_removes_the_plist_and_is_idempotent(agent, sandbox):
    agent.set_enabled(True)
    assert sandbox["plist"].exists()

    agent.set_enabled(False)
    assert not sandbox["plist"].exists()
    assert agent.is_enabled() is False

    agent.set_enabled(False)
    agent.set_enabled(False)
    assert agent.is_enabled() is False


def test_set_enabled_false_without_a_plist_is_a_no_op(agent, sandbox):
    agent.set_enabled(False)
    assert not sandbox["plist"].exists()
    assert agent.is_enabled() is False


def test_enable_disable_enable_round_trip(agent, sandbox):
    for expected in (True, False, True, False):
        agent.set_enabled(expected)
        assert agent.is_enabled() is expected
        assert sandbox["plist"].exists() is expected


def test_disable_leaves_other_launch_agents_alone(agent, sandbox):
    other = sandbox["dir"] / "com.example.other.plist"
    _write_plist(other, {"Label": "com.example.other", "ProgramArguments": ["/bin/true"]})
    agent.set_enabled(True)
    agent.set_enabled(False)
    assert other.exists()
    assert _load(other)["Label"] == "com.example.other"


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_write_failure_raises_config_error(sandbox, monkeypatch, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(autostart_mod, "LAUNCH_AGENTS_DIR", blocker / "LaunchAgents")
    agent = LaunchAgentAutostart(COMMAND)
    with pytest.raises(ConfigError) as excinfo:
        agent.set_enabled(True)
    assert excinfo.value.user_message.strip()


def test_remove_failure_raises_config_error(agent, sandbox, monkeypatch):
    agent.set_enabled(True)

    def denied(self, missing_ok=False):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "unlink", denied)
    with pytest.raises(ConfigError) as excinfo:
        agent.set_enabled(False)
    assert excinfo.value.user_message.strip()


def test_an_empty_command_raises_config_error(sandbox):
    with pytest.raises(ConfigError):
        LaunchAgentAutostart("   ").set_enabled(True)


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_off_macos_every_method_raises_config_error(agent, monkeypatch, sandbox, platform):
    monkeypatch.setattr(sys, "platform", platform)
    for call in (agent.is_enabled, lambda: agent.set_enabled(True), lambda: agent.set_enabled(False)):
        with pytest.raises(ConfigError) as excinfo:
            call()
        assert excinfo.value.user_message.strip()
    assert not sandbox["plist"].exists()
