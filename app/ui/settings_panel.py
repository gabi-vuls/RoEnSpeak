"""Settings tabs (General / Model / Advanced) with an explicit Apply button.

Changes are staged in the widgets and applied together with Apply, because model, compute type
and CPU mode changes trigger a model reload/download; applying each spin-box click live would
thrash the GPU. Revert reloads the widgets from Controller.config. Dirty state is computed by
comparing the Config the widgets would produce with the current Config.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.contracts import (
    DEFAULT_BLOCKLIST,
    DEFAULT_MODEL,
    MODELS,
    AppError,
    ComputeType,
    Config,
    Controller,
    InjectionMode,
    Task,
)

log = logging.getLogger("roenspeak.ui.settings")

SYSTEM_DEFAULT_MIC = "System default"
ERROR_COLOR = "#d13438"
WARN_COLOR = "#b45309"
OK_COLOR = "#1f9d55"
HINT_COLOR = "#7a7f87"  # readable on both light and dark backgrounds

TASK_ITEMS = (("Translate Romanian → English", Task.TRANSLATE), ("Romanian transcription only", Task.TRANSCRIBE))
INJECTION_ITEMS = (
    ("Type keystrokes (SendInput)", InjectionMode.SENDINPUT),
    ("Paste via clipboard", InjectionMode.CLIPBOARD),
)
COMPUTE_ITEMS = (
    ("float16 (best, more VRAM)", ComputeType.FLOAT16),
    ("int8_float16 (less VRAM)", ComputeType.INT8_FLOAT16),
)


def models_for_task(task: Task) -> list[str]:
    """Translation offers only translate-capable models; transcription offers all of them."""
    return [name for name, spec in MODELS.items() if task == Task.TRANSCRIBE or spec.translate_capable]


def parse_blocklist(text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for line in text.splitlines():
        entry = line.strip()
        if entry and entry.casefold() not in seen:
            seen.add(entry.casefold())
            out.append(entry)
    return tuple(out)


def _select_data(combo: QComboBox, value) -> None:
    idx = combo.findData(value)
    if idx >= 0:
        combo.setCurrentIndex(idx)


def _hint(text: str = "", color: Optional[str] = None) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    lbl.setStyleSheet(f"font-size: 8.5pt; color: {color or HINT_COLOR};")
    return lbl


class SettingsPanel(QWidget):
    """Owns the settings widgets and the Apply/Revert flow."""

    applied = Signal(object)  # the Config passed to update_config successfully

    def __init__(self, controller: Controller, parent=None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._base: Config = controller.config
        self._loading = False

        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._build_general(), "General")
        self.tabs.addTab(self._build_model(), "Model")
        self.tabs.addTab(self._build_advanced(), "Advanced")

        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.open_logs_btn = QPushButton("Open log folder")
        self.revert_btn = QPushButton("Revert")
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setDefault(True)
        self.open_logs_btn.clicked.connect(self._open_logs)
        self.revert_btn.clicked.connect(self.revert)
        self.apply_btn.clicked.connect(self.apply)

        buttons = QHBoxLayout()
        buttons.addWidget(self.open_logs_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.revert_btn)
        buttons.addWidget(self.apply_btn)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.tabs)
        lay.addWidget(self.message)
        lay.addLayout(buttons)

        self._set_message("", None)
        self.refresh_devices()
        self.load(self._base)

    # -- construction -------------------------------------------------------
    def _build_general(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.task_combo = QComboBox()
        for text, task in TASK_ITEMS:
            self.task_combo.addItem(text, task)
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        form.addRow("Mode", self.task_combo)

        self.mic_combo = QComboBox()
        self.mic_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.mic_combo.setMinimumContentsLength(18)
        self.mic_refresh = QToolButton()
        self.mic_refresh.setText("↻")
        self.mic_refresh.setToolTip("Refresh the microphone list")
        self.mic_refresh.clicked.connect(self.refresh_devices)
        mic_row = QHBoxLayout()
        mic_row.setContentsMargins(0, 0, 0, 0)
        mic_row.addWidget(self.mic_combo, 1)
        mic_row.addWidget(self.mic_refresh)
        form.addRow("Microphone", mic_row)

        self.injection_combo = QComboBox()
        for text, mode in INJECTION_ITEMS:
            self.injection_combo.addItem(text, mode)
        self.injection_combo.setToolTip(
            "Keystrokes work almost everywhere and leave the clipboard alone.\n"
            "Clipboard paste is faster for long text; the previous clipboard is restored."
        )
        form.addRow("Typing method", self.injection_combo)

        self.overlay_check = QCheckBox("Show “● REC” overlay while listening")
        self.sounds_check = QCheckBox("Play start/stop sounds")
        self.autostart_check = QCheckBox("Start with Windows")
        form.addRow("", self.overlay_check)
        form.addRow("", self.sounds_check)
        form.addRow("", self.autostart_check)

        for sig in (self.mic_combo.currentIndexChanged, self.injection_combo.currentIndexChanged):
            sig.connect(self._changed)
        for cb in (self.overlay_check, self.sounds_check, self.autostart_check):
            cb.toggled.connect(self._changed)
        return w

    def _build_model(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.model_combo = QComboBox()
        self.model_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)  # labels change with the mode
        self.model_combo.currentIndexChanged.connect(self._changed)
        form.addRow("Model", self.model_combo)
        self.model_hint = _hint()
        self.model_hint.setWordWrap(False)
        form.addRow("", self.model_hint)

        self.compute_combo = QComboBox()
        for text, ct in COMPUTE_ITEMS:
            self.compute_combo.addItem(text, ct)
        self.compute_combo.currentIndexChanged.connect(self._changed)
        form.addRow("Compute type", self.compute_combo)

        self.beam_spin = QSpinBox()
        self.beam_spin.setRange(1, 10)
        self.beam_spin.setToolTip("5 = best quality (default). 1 = fastest.")
        self.beam_spin.valueChanged.connect(self._changed)
        form.addRow("Beam size", self.beam_spin)

        self.idle_spin = QSpinBox()
        self.idle_spin.setRange(0, 1440)
        self.idle_spin.setSuffix(" min")
        self.idle_spin.setSpecialValueText("Off (keep model loaded)")
        self.idle_spin.setToolTip("Free the GPU memory after this many idle minutes. Reloads on the next hotkey press.")
        self.idle_spin.valueChanged.connect(self._changed)
        form.addRow("Unload when idle", self.idle_spin)

        self.cpu_check = QCheckBox("CPU mode (slow)")
        self.cpu_check.toggled.connect(self._changed)
        form.addRow("", self.cpu_check)
        self.cpu_warning = _hint(
            "⚠ Runs the model on the CPU instead of an NVIDIA GPU. Expect several seconds per phrase. "
            "Use only if this PC has no supported NVIDIA GPU.",
            WARN_COLOR,
        )
        form.addRow(self.cpu_warning)  # spans both columns so it wraps at full width
        return w

    def _build_advanced(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.silence_spin = QSpinBox()
        self.silence_spin.setRange(100, 5000)
        self.silence_spin.setSingleStep(50)
        self.silence_spin.setSuffix(" ms")
        self.silence_spin.setToolTip("Pause length that ends a phrase (default 500 ms).")
        form.addRow("Silence ends phrase", self.silence_spin)

        self.maxseg_spin = QDoubleSpinBox()
        self.maxseg_spin.setRange(2.0, 30.0)
        self.maxseg_spin.setSingleStep(0.5)
        self.maxseg_spin.setDecimals(1)
        self.maxseg_spin.setSuffix(" s")
        self.maxseg_spin.setToolTip("Continuous speech is cut after this long (default 15 s).")
        form.addRow("Max phrase length", self.maxseg_spin)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.05, 0.95)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setToolTip("Voice detection sensitivity: lower = more sensitive (default 0.50).")
        form.addRow("Speech threshold", self.threshold_spin)
        lay.addLayout(form)

        bl_head = QHBoxLayout()
        bl_head.addWidget(QLabel("Hallucination blocklist (one phrase per line, * = prefix)"))
        bl_head.addStretch(1)
        self.blocklist_reset = QToolButton()
        self.blocklist_reset.setText("Defaults")
        self.blocklist_reset.setToolTip("Restore the built-in blocklist")
        self.blocklist_reset.clicked.connect(lambda: self.blocklist_edit.setPlainText("\n".join(DEFAULT_BLOCKLIST)))
        bl_head.addWidget(self.blocklist_reset)
        lay.addLayout(bl_head)
        self.blocklist_edit = QPlainTextEdit()
        self.blocklist_edit.setTabChangesFocus(True)
        self.blocklist_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.blocklist_edit.setFixedHeight(110)
        lay.addWidget(self.blocklist_edit)
        lay.addStretch(1)

        self.silence_spin.valueChanged.connect(self._changed)
        self.maxseg_spin.valueChanged.connect(self._changed)
        self.threshold_spin.valueChanged.connect(self._changed)
        self.blocklist_edit.textChanged.connect(self._changed)
        return w

    # -- data <-> widgets ---------------------------------------------------
    def refresh_devices(self) -> None:
        current = self.mic_combo.currentData() if self.mic_combo.count() else self._base.mic_device
        try:
            devices = list(self._controller.list_input_devices())
        except Exception as exc:  # noqa: BLE001 (device enumeration must never break the window)
            log.warning("list_input_devices failed: %s", exc)
            devices = []
        was_loading = self._loading
        self._loading = True
        self.mic_combo.clear()
        self.mic_combo.addItem(SYSTEM_DEFAULT_MIC, None)
        for name in devices:
            self.mic_combo.addItem(name, name)
        self._ensure_mic_item(current)
        _select_data(self.mic_combo, current)
        self._loading = was_loading
        self._changed()

    def _ensure_mic_item(self, name: Optional[str]) -> None:
        if name is not None and self.mic_combo.findData(name) < 0:
            self.mic_combo.addItem(f"{name} (not connected)", name)

    def _fill_models(self, task: Task, selected: str) -> Optional[str]:
        """Repopulate the model combo for task; returns a note if the selection had to change."""
        names = models_for_task(task)
        note = None
        if selected not in names:
            note = (
                f"{MODELS[selected].label if selected in MODELS else selected} can't translate; "
                f"switched to {MODELS[DEFAULT_MODEL].label}."
            )
            selected = DEFAULT_MODEL
        was_loading = self._loading
        self._loading = True
        self.model_combo.clear()
        for name in names:
            self.model_combo.addItem(MODELS[name].label, name)
        _select_data(self.model_combo, selected)
        self._loading = was_loading
        return note

    def load(self, cfg: Config) -> None:
        """Populate every widget from cfg (no dirty marking)."""
        self._loading = True
        try:
            self._base = cfg
            _select_data(self.task_combo, cfg.task)
            self._fill_models(cfg.task, cfg.model)
            self._ensure_mic_item(cfg.mic_device)
            _select_data(self.mic_combo, cfg.mic_device)
            _select_data(self.injection_combo, cfg.injection_mode)
            _select_data(self.compute_combo, cfg.compute_type)
            self.overlay_check.setChecked(cfg.ui.overlay)
            self.sounds_check.setChecked(cfg.ui.sounds)
            self.autostart_check.setChecked(cfg.ui.start_with_windows)
            self.beam_spin.setValue(cfg.decode.beam_size)
            self.idle_spin.setValue(cfg.idle_unload_minutes)
            self.cpu_check.setChecked(cfg.cpu_mode)
            self.silence_spin.setValue(cfg.vad.min_silence_ms)
            self.maxseg_spin.setValue(cfg.vad.max_segment_s)
            self.threshold_spin.setValue(cfg.vad.threshold)
            self.blocklist_edit.setPlainText("\n".join(cfg.filter.blocklist))
        finally:
            self._loading = False
        self._changed()

    def build_config(self, base: Config) -> Config:
        """The Config the widgets describe, layered onto base (armed/hotkey/etc. untouched)."""
        return dataclasses.replace(
            base,
            mic_device=self.mic_combo.currentData(),
            model=self.model_combo.currentData() or base.model,
            compute_type=self.compute_combo.currentData(),
            task=self.task_combo.currentData(),
            injection_mode=self.injection_combo.currentData(),
            cpu_mode=self.cpu_check.isChecked(),
            idle_unload_minutes=self.idle_spin.value(),
            decode=dataclasses.replace(base.decode, beam_size=self.beam_spin.value()),
            vad=dataclasses.replace(
                base.vad,
                min_silence_ms=self.silence_spin.value(),
                max_segment_s=round(self.maxseg_spin.value(), 1),
                threshold=round(self.threshold_spin.value(), 2),
            ),
            filter=dataclasses.replace(base.filter, blocklist=parse_blocklist(self.blocklist_edit.toPlainText())),
            ui=dataclasses.replace(
                base.ui,
                overlay=self.overlay_check.isChecked(),
                sounds=self.sounds_check.isChecked(),
                start_with_windows=self.autostart_check.isChecked(),
            ),
        )

    @property
    def dirty(self) -> bool:
        return self.build_config(self._base) != self._base

    # -- reactions ----------------------------------------------------------
    def on_config_changed(self, cfg: Config) -> None:
        """Config changed in core (Apply, hotkey, arm...). Keep the user's unsaved edits."""
        if self.dirty:
            self._base = cfg
            self._changed()
        else:
            self.load(cfg)

    def _on_task_changed(self) -> None:
        if self._loading:
            return
        note = self._fill_models(self.task_combo.currentData(), self.model_combo.currentData() or DEFAULT_MODEL)
        self._changed()
        if note:
            self._set_message(note, WARN_COLOR)

    def _changed(self, *_args) -> None:
        if self._loading:
            return
        model = self.model_combo.currentData()
        ct = self.compute_combo.currentData()
        cpu = self.cpu_check.isChecked()
        if model in MODELS and ct is not None:
            spec = MODELS[model]
            vram = "runs on the CPU" if cpu else f"needs ~{spec.vram_mb[ct] / 1000:.1f} GB VRAM"
            self.model_hint.setText(f"{vram} · {spec.download_mb / 1000:.1f} GB download")
        self.compute_combo.setEnabled(not cpu)
        self.compute_combo.setToolTip("Ignored in CPU mode (always int8)." if cpu else "")
        self.cpu_warning.setStyleSheet(
            f"font-size: 8.5pt; color: {WARN_COLOR};" + (" font-weight: bold;" if cpu else "")
        )
        dirty = self.dirty
        self.apply_btn.setEnabled(dirty)
        self.revert_btn.setEnabled(dirty)
        if dirty:
            self._set_message("Unsaved changes - click Apply.", None)
        elif self.message.text() == "Unsaved changes - click Apply.":
            self._set_message("", None)

    def _set_message(self, text: str, color: Optional[str]) -> None:
        self.message.setText(text)
        self.message.setStyleSheet(f"color: {color or HINT_COLOR};")
        self.message.setVisible(bool(text))

    def apply(self) -> bool:
        base = self._controller.config
        cfg = self.build_config(base)
        if cfg == base:
            self._changed()
            return True
        try:
            self._controller.update_config(cfg)
        except AppError as exc:
            log.warning("update_config rejected: %s (%s)", exc.user_message, exc.details)
            text = exc.user_message + (f"\n{exc.details}" if exc.details else "")
            self._set_message(f"Settings not saved: {text}", ERROR_COLOR)
            return False
        log.info("settings applied")
        self._base = cfg
        self._changed()
        self._set_message("Settings saved.", OK_COLOR)
        self.applied.emit(cfg)
        return True

    def revert(self) -> None:
        self.load(self._controller.config)
        self._set_message("", None)

    def _open_logs(self) -> None:
        try:
            self._controller.open_log_folder()
        except AppError as exc:
            self._set_message(exc.user_message, ERROR_COLOR)
