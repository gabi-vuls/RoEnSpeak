""""Start with Windows" via the HKCU Run key (``contracts.Autostart``)."""

from __future__ import annotations

import logging
import site
import sys
from pathlib import Path
from typing import Optional

from app.contracts import APP_NAME, ConfigError

log = logging.getLogger("roenspeak.win32.autostart")

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def _pythonw_path() -> Path:
    exe = Path(sys.executable)
    candidate = exe.with_name("pythonw.exe")
    return candidate if candidate.exists() else exe


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
    """Command line stored in the Run value.

    Frozen build: ``"<exe>" --minimized``.
    Dev, package installed in site-packages: ``"<pythonw.exe>" -m app.main --minimized``.
    Dev from a source checkout: Run-key programs start with an arbitrary working directory
    (usually System32), so a bare ``-m app.main`` would not find the package. The command
    puts the checkout on sys.path and runs ``app.main`` as ``__main__``, which is equivalent
    to ``-m app.main --minimized`` run from the checkout.
    """
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}" --minimized'
    pyw = _pythonw_path()
    root = _package_root()
    if _installed_in_site_packages(root):
        return f'"{pyw}" -m app.main --minimized'
    code = (
        "import sys,runpy;"
        f"sys.path.insert(0,{str(root)!r});"
        "sys.argv[1:]=['--minimized'];"
        "runpy.run_module('app.main',run_name='__main__',alter_sys=True)"
    )
    return f'"{pyw}" -c "{code}"'


class RunKeyAutostart:
    """HKCU\\...\\Run value named APP_NAME."""

    def __init__(self, command: Optional[str] = None) -> None:
        self._command = command

    @property
    def command(self) -> str:
        return self._command if self._command is not None else default_command()

    def is_enabled(self) -> bool:
        winreg = _winreg()
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_QUERY_VALUE) as key:
                value, _type = winreg.QueryValueEx(key, APP_NAME)
        except FileNotFoundError:
            return False
        except OSError as exc:
            log.warning("could not read Run key: %s", exc)
            return False
        return bool(value)

    def set_enabled(self, enabled: bool) -> None:
        winreg = _winreg()
        try:
            with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, RUN_KEY, 0,
                                    winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE) as key:
                if enabled:
                    cmd = self.command
                    winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, cmd)
                    log.info("autostart enabled: %s", cmd)
                else:
                    try:
                        winreg.DeleteValue(key, APP_NAME)
                        log.info("autostart disabled")
                    except FileNotFoundError:
                        pass
        except OSError as exc:
            raise ConfigError(
                "Could not change the \"Start with Windows\" setting.",
                details=f"HKCU\\{RUN_KEY}\\{APP_NAME}: {exc}",
            ) from exc


def _winreg():
    if sys.platform != "win32":
        raise ConfigError('"Start with Windows" is only available on Windows.')
    import winreg

    return winreg
