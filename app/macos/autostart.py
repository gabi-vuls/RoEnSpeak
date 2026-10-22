""""Open at login" via a LaunchAgent plist (``contracts.Autostart``).

The macOS counterpart of ``app.win32.autostart.RunKeyAutostart``: where Windows writes one
value under ``HKCU\\...\\Run``, macOS writes
``~/Library/LaunchAgents/com.roenspeak.RoEnSpeak.plist`` with ``RunAtLoad``, and removes the
file again when the setting is switched off.

The plist holds:

``Label``             the bundle id, which is also the launchd job name.
``ProgramArguments``  the same three command shapes the Windows version computes, split into
                      argv (see ``default_command``), always with ``--minimized``.
``RunAtLoad``         start the job when the agent is loaded, i.e. at login.
``ProcessType``       ``Interactive``: this is a UI app, so launchd must not throttle it the
                      way it throttles background jobs.

``launchctl load``/``unload`` is run as a best effort so the change also takes effect in the
*current* login session; launchd reads ``~/Library/LaunchAgents`` by itself at the next login,
so a missing or failing ``launchctl`` is logged and never fatal.

Off macOS every method raises ``ConfigError``, exactly as the Windows one does off Windows.
"""

from __future__ import annotations

import logging
import plistlib
import shlex
import site
import subprocess
import sys
from pathlib import Path
from typing import Optional

from app.contracts import ConfigError

log = logging.getLogger("roenspeak.macos.autostart")

#: Bundle id of the app, which is also the LaunchAgent label and plist file name.
BUNDLE_ID = "com.roenspeak.RoEnSpeak"
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
LAUNCHCTL = "/bin/launchctl"
_LAUNCHCTL_TIMEOUT_S = 10.0


def plist_path() -> Path:
    return LAUNCH_AGENTS_DIR / f"{BUNDLE_ID}.plist"


def _package_root() -> Path:
    """Directory that contains the ``app`` package."""
    import app

    return Path(app.__file__).resolve().parent.parent


def _installed_in_site_packages(root: Path) -> bool:
    dirs = list(site.getsitepackages())
    user = site.getusersitepackages()
    if isinstance(user, str):
        dirs.append(user)
    for d in dirs:
        try:
            if Path(d).resolve() == root:
                return True
        except OSError:
            continue
    return False


def default_command() -> str:
    """Command line stored in ProgramArguments, as a shell-quoted string.

    Frozen build: the executable inside ``RoEnSpeak.app/Contents/MacOS`` plus ``--minimized``.
    (The binary is launched directly rather than through ``open -a``: launchd then owns the
    process, so ``RunAtLoad`` and ``ProcessType`` apply to the app itself.)
    Dev, package installed in site-packages: ``<python> -m app.main --minimized``.
    Dev from a source checkout: launchd starts jobs with an arbitrary working directory, so a
    bare ``-m app.main`` would not find the package. The command puts the checkout on
    sys.path and runs ``app.main`` as ``__main__``, which is equivalent to
    ``-m app.main --minimized`` run from the checkout.

    ``LaunchAgentAutostart`` splits the result with ``shlex.split``, so a command passed to
    the constructor must be quoted the same way.
    """
    if getattr(sys, "frozen", False):
        return f"{shlex.quote(sys.executable)} --minimized"
    python = shlex.quote(sys.executable)
    root = _package_root()
    if _installed_in_site_packages(root):
        return f"{python} -m app.main --minimized"
    code = (
        "import sys,runpy;"
        f"sys.path.insert(0,{str(root)!r});"
        "sys.argv[1:]=['--minimized'];"
        "runpy.run_module('app.main',run_name='__main__',alter_sys=True)"
    )
    return f"{python} -c {shlex.quote(code)}"


class LaunchAgentAutostart:
    """``~/Library/LaunchAgents/com.roenspeak.RoEnSpeak.plist``."""

    def __init__(self, command: Optional[str] = None) -> None:
        self._command = command

    @property
    def command(self) -> str:
        return self._command if self._command is not None else default_command()

    @property
    def program_arguments(self) -> list[str]:
        return shlex.split(self.command)

    def is_enabled(self) -> bool:
        """True when the plist exists and still points at our command.

        A plist left behind by an older install (a different interpreter or a moved .app)
        would not start this build, so it reads as "off"; switching the setting on rewrites it.
        """
        _require_macos()
        path = plist_path()
        try:
            with path.open("rb") as handle:
                data = plistlib.load(handle)
        except FileNotFoundError:
            return False
        except (OSError, plistlib.InvalidFileException, ValueError) as exc:
            log.warning("could not read %s: %s", path, exc)
            return False
        args = data.get("ProgramArguments")
        if not isinstance(args, list) or not args:
            log.warning("%s has no ProgramArguments; treating autostart as off", path)
            return False
        if [str(a) for a in args] != self.program_arguments:
            log.info("%s points at a different command (%s); treating autostart as off",
                     path, " ".join(str(a) for a in args))
            return False
        return True

    def set_enabled(self, enabled: bool) -> None:
        _require_macos()
        path = plist_path()
        try:
            if enabled:
                args = self.program_arguments
                if not args:
                    raise ConfigError(
                        'Could not change the "Open at login" setting.',
                        details=f"empty autostart command {self.command!r}",
                    )
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "Label": BUNDLE_ID,
                    "ProgramArguments": args,
                    "RunAtLoad": True,
                    "ProcessType": "Interactive",
                }
                tmp = path.with_name(path.name + ".tmp")
                with tmp.open("wb") as handle:
                    plistlib.dump(payload, handle)
                tmp.replace(path)
                # Replacing the file under a loaded job needs a reload to take effect now.
                _launchctl("unload", path, expect_failure=True)
                _launchctl("load", path)
                log.info("autostart enabled: %s", " ".join(args))
            else:
                _launchctl("unload", path, expect_failure=not path.exists())
                try:
                    path.unlink()
                    log.info("autostart disabled")
                except FileNotFoundError:
                    pass
        except OSError as exc:
            raise ConfigError(
                'Could not change the "Open at login" setting.',
                details=f"{path}: {exc}",
            ) from exc


def _launchctl(verb: str, path: Path, expect_failure: bool = False) -> None:
    """``launchctl <verb> -w <plist>``, best effort.

    launchd picks the agent up at the next login on its own, so failures here only mean the
    change does not apply to the running session.
    """
    try:
        result = subprocess.run(
            [LAUNCHCTL, verb, "-w", str(path)],
            capture_output=True, text=True, timeout=_LAUNCHCTL_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("launchctl %s failed: %s", verb, exc)
        return
    if result.returncode != 0 and not expect_failure:
        log.warning("launchctl %s exited %d: %s", verb, result.returncode,
                    (result.stderr or result.stdout or "").strip())


def _require_macos() -> None:
    if sys.platform != "darwin":
        raise ConfigError('"Open at login" is only available on macOS.')
