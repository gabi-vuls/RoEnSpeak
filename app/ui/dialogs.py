"""Non-modal error dialogs and the model download dialog.

Titles and help texts are platform-aware: the Windows build talks about the NVIDIA driver,
VRAM and Windows privacy settings, the macOS build about Metal, unified memory, System
Settings and - the macOS-only failure mode - the Accessibility permission without which
every synthetic keystroke is silently dropped. Nothing else differs.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from app.contracts import (
    APP_NAME,
    GGML_REPO,
    IS_MACOS,
    MIN_DRIVER_VERSION,
    MIN_MACOS_VERSION,
    MODELS,
    AppError,
    Controller,
    DownloadFinished,
    DownloadProgress,
    models_dir,
)
from app.ui.assets import app_icon
from app.ui.status import format_bytes, format_eta, format_rate

log = logging.getLogger("roenspeak.ui.dialogs")

ERROR_COLOR = "#d13438"
OK_COLOR = "#1f9d55"
HINT_COLOR = "#7a7f87"

#: Friendly dialog titles per ErrorRaised.kind (AppError subclass name).
ERROR_TITLES: dict[str, str] = {
    "CudaUnavailableError": "GPU required" if IS_MACOS else "NVIDIA GPU required",
    "GpuOutOfMemoryError": "Not enough GPU memory",
    "ModelNotDownloadedError": "Speech model missing",
    "DownloadError": "Model download failed",
    "AudioDeviceError": "Microphone problem",
    "HotkeyConflictError": "Hotkey unavailable",
    "HotkeyParseError": "Invalid hotkey",
    "ConfigError": "Settings problem",
    "InjectionBlockedError": "Can't type into this window",
    "ModifiersHeldError": "Text not typed",
    "TaskNotSupportedError": "Model can't translate",
}

#: Extra guidance shown under the message for some error kinds.
if IS_MACOS:
    ERROR_HELP: dict[str, str] = {
        "CudaUnavailableError": (
            "RoEnSpeak runs speech recognition on the Mac's GPU through Metal and needs macOS "
            f"{MIN_MACOS_VERSION} or newer.\n\n"
            "• Update macOS, then restart RoEnSpeak.\n"
            "• Or enable Settings → Model → “CPU mode (slow)”. It works, but each phrase takes "
            "several seconds."
        ),
        "GpuOutOfMemoryError": (
            "The Mac ran out of memory for the speech model.\n\n"
            "• Close memory-heavy apps and try again.\n"
            "• Or choose Settings → Model → Compute type “int8_float16”, or the smaller “medium” model."
        ),
        "AudioDeviceError": (
            "Check that the microphone is plugged in and allowed in System Settings → Privacy & "
            "Security → Microphone, then choose it in Settings → General."
        ),
        "InjectionBlockedError": (
            "macOS only lets an app type into other apps once you allow it under Accessibility, "
            "and it silently drops the keystrokes until you do.\n\n"
            "• Open System Settings → Privacy & Security → Accessibility and switch RoEnSpeak on.\n"
            "• Already on after an update? Switch it off and on again.\n"
            "• Password fields block synthetic keystrokes even with permission; dictate into a "
            "normal text field."
        ),
        "DownloadError": "Check your internet connection, then try again. Partially downloaded files are kept.",
        "ModelNotDownloadedError": "Open RoEnSpeak and download the model when asked.",
    }
else:
    ERROR_HELP: dict[str, str] = {
        "CudaUnavailableError": (
            "RoEnSpeak runs speech recognition on an NVIDIA graphics card and needs an NVIDIA GPU with "
            f"driver {MIN_DRIVER_VERSION} or newer.\n\n"
            "• Install or update the NVIDIA driver (nvidia.com/drivers), then restart RoEnSpeak.\n"
            "• No NVIDIA GPU? Enable Settings → Model → “CPU mode (slow)”. It works, but each phrase takes several seconds."
        ),
        "GpuOutOfMemoryError": (
            "The graphics card ran out of memory for the speech model.\n\n"
            "• Close GPU-heavy apps (games, video editors, other AI tools) and try again.\n"
            "• Or choose Settings → Model → Compute type “int8_float16”, or the smaller “medium” model."
        ),
        "AudioDeviceError": (
            "Check that the microphone is plugged in and allowed in Windows Settings → Privacy → Microphone, "
            "then choose it in Settings → General."
        ),
        "DownloadError": "Check your internet connection, then try again. Partially downloaded files are kept.",
        "ModelNotDownloadedError": "Open RoEnSpeak and download the model when asked.",
    }

#: Title and text of the first-run Accessibility prompt (macOS only). The wording of the
#: failure itself lives in contracts.InjectionBlockedError.MESSAGES["no_accessibility"].
ACCESSIBILITY_TITLE = "Allow RoEnSpeak to type"
ACCESSIBILITY_BUTTON = "Open System Settings…"


def mac_permissions():
    """app.macos.permissions, or None when it is unavailable (non-macOS, or a build without
    it). Imported lazily so the UI never hard-depends on the platform package."""
    if not IS_MACOS:
        return None
    try:
        from app.macos import permissions
    except Exception:  # noqa: BLE001 (any import failure just means "no helper")
        log.debug("app.macos.permissions unavailable", exc_info=True)
        return None
    return permissions


def open_accessibility_settings() -> None:
    """Ask macOS for Accessibility access and open the right System Settings pane.

    request_accessibility() shows the system prompt (which has its own "Open System
    Settings" button) the first time; open_privacy_settings() is what actually gets the user
    to the switch on later attempts, so both are called.
    """
    permissions = mac_permissions()
    if permissions is None:
        log.warning("cannot open Accessibility settings: app.macos.permissions unavailable")
        return
    for name, args in (("request_accessibility", ()), ("open_privacy_settings", ("accessibility",))):
        func = getattr(permissions, name, None)
        if func is None:
            log.warning("app.macos.permissions.%s is missing", name)
            continue
        try:
            func(*args)
        except Exception:  # noqa: BLE001 (a permission helper must never break the dialog)
            log.exception("app.macos.permissions.%s failed", name)


def accessibility_granted() -> Optional[bool]:
    """True/False from app.macos.permissions, or None when it cannot be determined."""
    permissions = mac_permissions()
    func = getattr(permissions, "accessibility_granted", None) if permissions else None
    if func is None:
        return None
    try:
        return bool(func())
    except Exception:  # noqa: BLE001
        log.exception("accessibility_granted() failed")
        return None


def present(widget, activate: bool) -> None:
    """Show a top-level dialog. activate=False shows it without taking keyboard focus (used
    while dictating, so injected text never lands in our dialog)."""
    widget.setAttribute(Qt.WA_ShowWithoutActivating, not activate)
    widget.show()
    if activate:
        widget.raise_()
        widget.activateWindow()


class ErrorDialogs:
    """Keeps at most one open dialog per (title, message); repeats just raise it."""

    def __init__(self) -> None:
        self._open: dict[tuple[str, str], QMessageBox] = {}

    def show(
        self,
        title: str,
        message: str,
        details: str = "",
        kind: str = "",
        activate: bool = True,
        accessibility_help: bool = False,
    ) -> QMessageBox:
        """accessibility_help adds an "Open System Settings…" button (macOS only) that takes
        the user straight to Privacy & Security → Accessibility."""
        title = title or ERROR_TITLES.get(kind, f"{APP_NAME} error")
        key = (title, message)
        box = self._open.get(key)
        if box is not None:
            present(box, activate)
            return box
        box = QMessageBox(QMessageBox.Critical, f"{APP_NAME} - {title}", message, QMessageBox.Ok)
        box.setWindowIcon(app_icon())
        box.setWindowModality(Qt.NonModal)
        box.setAttribute(Qt.WA_DeleteOnClose, True)
        box.setTextInteractionFlags(Qt.TextSelectableByMouse)
        help_text = ERROR_HELP.get(kind, "")
        if help_text:
            box.setInformativeText(help_text)
        if details:
            box.setDetailedText(details)
        if accessibility_help and IS_MACOS:
            btn = box.addButton(ACCESSIBILITY_BUTTON, QMessageBox.ActionRole)
            btn.clicked.connect(open_accessibility_settings)
            box.setDefaultButton(btn)
            # With more than one button QMessageBox finds no escape button on its own, and
            # then refuses every close() - which also vetoes QApplication.quit(), because Qt
            # cancels a quit when a window will not close. Name it explicitly.
            box.setEscapeButton(box.button(QMessageBox.Ok))
        box.finished.connect(lambda _r, k=key: self._open.pop(k, None))
        self._open[key] = box
        log.info("error dialog: %s | %s (activate=%s)", title, message, activate)
        present(box, activate)
        return box

    def open_dialogs(self) -> list[QMessageBox]:
        return list(self._open.values())

    def close_all(self) -> None:
        for box in list(self._open.values()):
            box.close()


class DownloadDialog(QDialog):
    """First-run / model-change download. Tracks DownloadProgressEvent / DownloadFinished for
    its model. Closing it while a download runs only hides it (the download continues)."""

    IDLE, DOWNLOADING, DONE, FAILED, CANCELLED = range(5)

    def __init__(self, controller: Controller, model: str, size_mb: int, parent=None) -> None:
        super().__init__(parent)
        self._controller = controller
        self.model = model
        self.size_mb = size_mb
        self.phase = self.IDLE
        self._cancel_requested = False
        self._started_at = 0.0

        self.setWindowTitle(f"{APP_NAME} - Download speech model")
        self.setWindowIcon(app_icon())
        self.setWindowModality(Qt.NonModal)
        self.setFixedWidth(520)  # fixed width keeps word-wrapped label heights correct

        spec = MODELS.get(model)
        label = spec.label if spec else model
        # macOS downloads one ggml file from GGML_REPO; Windows a CTranslate2 model directory.
        repo = (GGML_REPO if IS_MACOS else spec.hf_repo) if spec else model
        heading = QLabel(f"<b>Download the speech model “{label}”</b>")
        explain = QLabel(
            f"RoEnSpeak needs this model to understand speech. It is about {size_mb / 1000:.1f} GB and is "
            f"downloaded once from Hugging Face ({repo}). Nothing you say is ever uploaded.<br>"
            f"<span style='color:{HINT_COLOR}'>Saved to {models_dir() / model}</span>"
        )
        explain.setWordWrap(True)
        explain.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("")
        self.detail = QLabel("")
        self.detail.setStyleSheet(f"color: {HINT_COLOR};")
        self.file_label = QLabel("")
        self.file_label.setStyleSheet(f"color: {HINT_COLOR};")
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.download_btn = QPushButton("Download")
        self.download_btn.setDefault(True)
        self.cancel_btn = QPushButton("Cancel download")
        self.close_btn = QPushButton("Later")
        self.download_btn.clicked.connect(self.start)
        self.cancel_btn.clicked.connect(self.cancel)
        self.close_btn.clicked.connect(self.close)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.close_btn)
        buttons.addWidget(self.download_btn)

        lay = QVBoxLayout(self)
        lay.setSpacing(8)
        lay.addWidget(heading)
        lay.addWidget(explain)
        lay.addSpacing(4)
        lay.addWidget(self.progress)
        lay.addWidget(self.detail)
        lay.addWidget(self.file_label)
        lay.addWidget(self.message)
        lay.addSpacing(4)
        lay.addLayout(buttons)

        self._auto_close = QTimer(self)
        self._auto_close.setSingleShot(True)
        self._auto_close.timeout.connect(self.accept)
        self._render()

    # -- actions ------------------------------------------------------------
    def start(self) -> None:
        if self.phase == self.DOWNLOADING:
            return
        self._cancel_requested = False
        self._started_at = time.monotonic()
        self.phase = self.DOWNLOADING
        self.message.setText("")
        self.progress.setRange(0, 0)  # indeterminate until the first progress event
        self._render()
        try:
            self._controller.start_model_download()
        except AppError as exc:
            log.warning("start_model_download failed: %s", exc.user_message)
            self._fail(exc.user_message)

    def cancel(self) -> None:
        if self.phase != self.DOWNLOADING:
            return
        self._cancel_requested = True
        self.cancel_btn.setEnabled(False)
        self.message.setStyleSheet(f"color: {HINT_COLOR};")
        self.message.setText("Cancelling…")
        try:
            self._controller.cancel_model_download()
        except AppError as exc:
            log.warning("cancel_model_download failed: %s", exc.user_message)

    # -- events -------------------------------------------------------------
    def on_progress(self, p: DownloadProgress) -> None:
        if p.model != self.model:
            return
        if self.phase in (self.IDLE, self.FAILED, self.CANCELLED) and not self._cancel_requested:
            # Core (re)started the download itself (e.g. arming); follow it.
            self.phase = self.DOWNLOADING
            self._started_at = time.monotonic()
            self.message.setText("")
            self._render()
        if self.phase != self.DOWNLOADING:
            return
        if p.total_bytes > 0:
            self.progress.setRange(0, 1000)
            frac = min(1.0, p.downloaded_bytes / p.total_bytes)
            self.progress.setValue(int(frac * 1000))
            self.progress.setFormat(f"{frac * 100:.0f}%")
            parts = [f"{format_bytes(p.downloaded_bytes)} of {format_bytes(p.total_bytes)}"]
        else:
            self.progress.setRange(0, 0)
            parts = [format_bytes(p.downloaded_bytes)]
        if p.bytes_per_s > 0:
            parts.append(format_rate(p.bytes_per_s))
            if p.total_bytes > 0:
                eta = format_eta(p.total_bytes - p.downloaded_bytes, p.bytes_per_s)
                if eta:
                    parts.append(eta)
        self.detail.setText(" · ".join(parts))
        self.file_label.setText(f"File: {p.current_file}" if p.current_file else "")
        self.file_label.setVisible(bool(p.current_file))
        self._fit()

    def on_finished(self, ev: DownloadFinished) -> None:
        if ev.model != self.model:
            return
        if ev.ok:
            self.phase = self.DONE
            self.progress.setRange(0, 1000)
            self.progress.setValue(1000)
            self.progress.setFormat("100%")
            self.message.setStyleSheet(f"color: {OK_COLOR};")
            self.message.setText("Download complete. Loading the model…")
            self._render()
            self._auto_close.start(1500)
            return
        if self._cancel_requested:
            self.phase = self.CANCELLED
            self._cancel_requested = False
            self.message.setStyleSheet(f"color: {HINT_COLOR};")
            self.message.setText("Download cancelled. Already downloaded data is kept; Download resumes from there.")
            self._render()
            return
        if ev.offline:
            self._fail(
                "No internet connection. RoEnSpeak could not reach Hugging Face to download the model. "
                "Connect to the internet and click Retry; the download resumes where it stopped."
                + (f"\n({ev.error})" if ev.error else "")
            )
        else:
            self._fail(f"Download failed: {ev.error or 'unknown error'}. Click Retry to resume.")

    def _fail(self, text: str) -> None:
        self.phase = self.FAILED
        self.message.setStyleSheet(f"color: {ERROR_COLOR};")
        self.message.setText(text)
        self._render()

    def _render(self) -> None:
        downloading = self.phase == self.DOWNLOADING
        self.download_btn.setVisible(self.phase != self.DONE)
        self.download_btn.setEnabled(not downloading)
        self.download_btn.setText("Retry" if self.phase in (self.FAILED,) else "Download")
        self.cancel_btn.setVisible(downloading)
        self.cancel_btn.setEnabled(downloading and not self._cancel_requested)
        self.close_btn.setText("Hide" if downloading else ("Close" if self.phase == self.DONE else "Later"))
        self.progress.setVisible(self.phase != self.IDLE)
        self.detail.setVisible(self.phase != self.IDLE)
        self.file_label.setVisible(downloading)
        self.message.setVisible(bool(self.message.text()))
        if self.phase in (self.FAILED, self.CANCELLED):
            self.progress.setRange(0, 1000)
        self._fit()

    def _fit(self) -> None:
        """Shrink/grow to the content (word-wrapped labels need heightForWidth)."""
        lay = self.layout()
        h = lay.totalHeightForWidth(self.width()) if lay.hasHeightForWidth() else lay.totalSizeHint().height()
        self.setFixedHeight(max(h, lay.totalMinimumSize().height()))

    def is_downloading(self) -> bool:
        return self.phase == self.DOWNLOADING

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.phase == self.DOWNLOADING:
            log.info("download dialog hidden; download continues")
        super().closeEvent(event)

