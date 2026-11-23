"""Modal dialog for configuring Faster-Whisper runtime and filtering parameters."""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rtt.config import AppConfig


class WhisperSettingsDialog(QDialog):
    """Exposes Faster-Whisper decode, compute type, prompt, and hallucination guard settings."""

    def __init__(self, config: AppConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Whisper Configuration")
        self.setMinimumWidth(420)

        root = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(8)

        self.compute_type_combo = QComboBox()
        for label, val in (
            ("Auto (float16 on CUDA / int8 on CPU)", "auto"),
            ("float16", "float16"),
            ("int8_float16", "int8_float16"),
            ("int8", "int8"),
            ("float32", "float32"),
        ):
            self.compute_type_combo.addItem(label, val)
        form.addRow("Compute Type:", self.compute_type_combo)

        self.task_combo = QComboBox()
        self.task_combo.addItem("Transcribe (source language)", "transcribe")
        self.task_combo.addItem("Translate to English", "translate")
        form.addRow("Task:", self.task_combo)

        self.beam_size_spin = QSpinBox()
        self.beam_size_spin.setRange(1, 10)
        form.addRow("Beam Size:", self.beam_size_spin)

        self.best_of_spin = QSpinBox()
        self.best_of_spin.setRange(1, 10)
        form.addRow("Best Of:", self.best_of_spin)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 1.0)
        self.temperature_spin.setSingleStep(0.05)
        self.temperature_spin.setDecimals(2)
        form.addRow("Temperature:", self.temperature_spin)

        self.vad_filter_chk = QCheckBox("Use Whisper internal VAD filter")
        form.addRow("Whisper VAD:", self.vad_filter_chk)

        self.condition_prev_chk = QCheckBox("Condition on previous text")
        form.addRow("Context Carry:", self.condition_prev_chk)

        self.initial_prompt_edit = QLineEdit()
        self.initial_prompt_edit.setPlaceholderText("Optional vocabulary, names, or style hint...")
        form.addRow("Initial Prompt:", self.initial_prompt_edit)

        self.no_speech_spin = QDoubleSpinBox()
        self.no_speech_spin.setRange(0.0, 1.0)
        self.no_speech_spin.setSingleStep(0.05)
        self.no_speech_spin.setDecimals(2)
        form.addRow("No-Speech Thr:", self.no_speech_spin)

        self.compression_spin = QDoubleSpinBox()
        self.compression_spin.setRange(1.0, 5.0)
        self.compression_spin.setSingleStep(0.10)
        self.compression_spin.setDecimals(2)
        form.addRow("Compression Thr:", self.compression_spin)

        root.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.reset_btn = QPushButton("Reset Defaults")
        self.reset_btn.clicked.connect(self._reset_defaults)
        buttons.addButton(self.reset_btn, QDialogButtonBox.ButtonRole.ResetRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._load_from_config(config)

    @staticmethod
    def _select_combo_data(combo: QComboBox, value: str) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _load_from_config(self, config: AppConfig) -> None:
        self._select_combo_data(self.compute_type_combo, config.whisper_compute_type_mode)
        self._select_combo_data(self.task_combo, config.whisper_task)
        self.beam_size_spin.setValue(int(config.whisper_beam_size))
        self.best_of_spin.setValue(int(config.whisper_best_of))
        self.temperature_spin.setValue(float(config.whisper_temperature))
        self.vad_filter_chk.setChecked(bool(config.whisper_vad_filter))
        self.condition_prev_chk.setChecked(bool(config.whisper_condition_on_previous_text))
        self.initial_prompt_edit.setText(config.whisper_initial_prompt)
        self.no_speech_spin.setValue(float(config.whisper_no_speech_threshold))
        self.compression_spin.setValue(float(config.whisper_compression_ratio_threshold))

    def _reset_defaults(self) -> None:
        self._load_from_config(AppConfig())

    def apply_to_config(self, config: AppConfig) -> None:
        config.whisper_compute_type_mode = str(self.compute_type_combo.currentData())
        config.whisper_task = str(self.task_combo.currentData())
        config.whisper_beam_size = int(self.beam_size_spin.value())
        config.whisper_best_of = int(self.best_of_spin.value())
        config.whisper_temperature = round(float(self.temperature_spin.value()), 2)
        config.whisper_vad_filter = bool(self.vad_filter_chk.isChecked())
        config.whisper_condition_on_previous_text = bool(self.condition_prev_chk.isChecked())
        config.whisper_initial_prompt = self.initial_prompt_edit.text().strip()
        config.whisper_no_speech_threshold = round(float(self.no_speech_spin.value()), 2)
        config.whisper_compression_ratio_threshold = round(float(self.compression_spin.value()), 2)
