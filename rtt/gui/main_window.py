"""Main PySide6 Control Window for RealtimeTranslator."""

from datetime import datetime
import math

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rtt import paths, secrets
from rtt.audio.devices import AudioDeviceInfo, default_loopback_index, list_wasapi_devices
from rtt.audio.ring_buffer import AudioRingBuffer
from rtt.config import AppConfig
from rtt.gui.overlay import SubtitleOverlay
from rtt.gui.workers import AudioCaptureThread, STTWorker, TranslationWorker
from rtt.logs import export as log_export
from rtt.logs.transcript_logger import DualTranscriptLogger
from rtt.translate.pipeline import UtteranceTask


SOURCE_LANGUAGES = [
    ("Auto Detect", "auto"),
    ("English (en)", "en"),
    ("Japanese (ja)", "ja"),
    ("Chinese (zh)", "zh"),
    ("Korean (ko)", "ko"),
    ("Vietnamese (vi)", "vi"),
    ("Spanish (es)", "es"),
    ("French (fr)", "fr"),
    ("German (de)", "de"),
    ("Russian (ru)", "ru"),
    ("Portuguese (pt)", "pt"),
]

TARGET_LANGUAGES = [
    ("English (en)", "en"),
    ("Vietnamese (vi)", "vi"),
    ("Japanese (ja)", "ja"),
    ("Chinese Simplified (zh-CN)", "zh-CN"),
    ("Korean (ko)", "ko"),
    ("Spanish (es)", "es"),
    ("French (fr)", "fr"),
    ("German (de)", "de"),
    ("Russian (ru)", "ru"),
    ("Portuguese (pt)", "pt"),
]


class MainWindow(QMainWindow):
    """Primary configuration dashboard, live feed viewer, and session manager."""

    def __init__(self, config: AppConfig | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.config = config or AppConfig.load()
        self.setWindowTitle("RealtimeTranslator")
        self.resize(1120, 760)

        self._devices: list[AudioDeviceInfo] = []
        self._running = False
        self._muted = False
        self._ring: AudioRingBuffer | None = None
        self._logger: DualTranscriptLogger | None = None
        self._capture: AudioCaptureThread | None = None
        self._stt_worker: STTWorker | None = None
        self._tr_worker: TranslationWorker | None = None
        self._seq_rows: dict[int, int] = {}

        self.overlay = SubtitleOverlay(self.config)
        self.overlay.geometry_changed.connect(self._on_overlay_geometry)

        self._build_ui()
        self._populate_from_config()
        self.refresh_devices()
        self._update_engine_warnings()
        self._update_translator_fields()
        self._update_history_availability()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        top_panels = QHBoxLayout()
        top_panels.setSpacing(10)
        top_panels.addWidget(self._build_audio_group(), 1)
        top_panels.addWidget(self._build_stt_group(), 1)
        top_panels.addWidget(self._build_translation_group(), 1)
        root.addLayout(top_panels)

        root.addWidget(self._build_controls_bar())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_live_feed_tab(), "Live Feed")
        self.tabs.addTab(self._build_history_tab(), "History Search")
        root.addWidget(self.tabs, 1)

        self.setCentralWidget(central)
        self.status_bar_label = QLabel("Idle")
        self.status_bar_label.setObjectName("statusLabel")
        self.statusBar().addPermanentWidget(self.status_bar_label, 1)

    def _build_audio_group(self) -> QGroupBox:
        box = QGroupBox("Audio Source (WASAPI)")
        form = QFormLayout(box)
        form.setSpacing(8)

        dev_row = QHBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.refresh_devices_btn = QPushButton("Refresh")
        self.refresh_devices_btn.clicked.connect(self.refresh_devices)
        dev_row.addWidget(self.device_combo, 1)
        dev_row.addWidget(self.refresh_devices_btn)
        form.addRow("Device:", dev_row)

        self.vu_meter = QProgressBar()
        self.vu_meter.setRange(0, 1000)
        self.vu_meter.setValue(0)
        self.vu_meter.setTextVisible(False)
        form.addRow("Input Level:", self.vu_meter)

        sil_row = QHBoxLayout()
        self.silence_slider = QSlider(Qt.Orientation.Horizontal)
        self.silence_slider.setRange(250, 800)
        self.silence_slider.setSingleStep(10)
        self.silence_value_label = QLabel("420 ms")
        self.silence_slider.valueChanged.connect(self._on_silence_changed)
        sil_row.addWidget(self.silence_slider, 1)
        sil_row.addWidget(self.silence_value_label)
        form.addRow("Pause Split:", sil_row)

        self.mute_btn = QPushButton("Mute Capture (Ctrl+Shift+M)")
        self.mute_btn.setCheckable(True)
        self.mute_btn.toggled.connect(self.set_muted)
        form.addRow("", self.mute_btn)

        self.audio_warning_label = QLabel("")
        self.audio_warning_label.setObjectName("warningLabel")
        self.audio_warning_label.setWordWrap(True)
        self.audio_warning_label.hide()
        form.addRow("", self.audio_warning_label)
        return box

    def _build_stt_group(self) -> QGroupBox:
        box = QGroupBox("Speech-to-Text")
        form = QFormLayout(box)
        form.setSpacing(8)

        self.stt_engine_combo = QComboBox()
        self.stt_engine_combo.addItem("Faster-Whisper", "whisper")
        self.stt_engine_combo.addItem("Windows Native STT", "windows_native")
        self.stt_engine_combo.addItem("Custom Neural Net", "custom")
        self.stt_engine_combo.currentIndexChanged.connect(self._on_stt_changed)
        form.addRow("Engine:", self.stt_engine_combo)

        self.whisper_model_combo = QComboBox()
        for name in ("tiny", "base", "small", "medium", "large-v3-turbo"):
            self.whisper_model_combo.addItem(name, name)
        self.whisper_model_combo.currentIndexChanged.connect(self._save_config)
        form.addRow("Whisper Model:", self.whisper_model_combo)

        self.compute_device_combo = QComboBox()
        self.compute_device_combo.addItem("CUDA (GPU float16)", "cuda")
        self.compute_device_combo.addItem("CPU (int8)", "cpu")
        self.compute_device_combo.currentIndexChanged.connect(self._save_config)
        form.addRow("Compute:", self.compute_device_combo)

        self.source_lang_combo = QComboBox()
        for label, code in SOURCE_LANGUAGES:
            self.source_lang_combo.addItem(label, code)
        self.source_lang_combo.currentIndexChanged.connect(self._save_config)
        form.addRow("Source Lang:", self.source_lang_combo)

        self.custom_model_path_edit = QLineEdit()
        self.custom_model_path_edit.setPlaceholderText("Path to custom model...")
        self.custom_model_path_edit.editingFinished.connect(self._save_config)
        self.custom_entrypoint_edit = QLineEdit()
        self.custom_entrypoint_edit.setPlaceholderText("pkg.mod:factory")
        self.custom_entrypoint_edit.editingFinished.connect(self._save_config)
        self.custom_rows_widget = QWidget()
        custom_form = QFormLayout(self.custom_rows_widget)
        custom_form.setContentsMargins(0, 0, 0, 0)
        custom_form.addRow("Model Path:", self.custom_model_path_edit)
        custom_form.addRow("Entrypoint:", self.custom_entrypoint_edit)
        form.addRow(self.custom_rows_widget)
        return box

    def _build_translation_group(self) -> QGroupBox:
        box = QGroupBox("Translation & Rate Limit")
        form = QFormLayout(box)
        form.setSpacing(8)

        self.primary_combo = QComboBox()
        for label, val in (
            ("LLM API (OpenAI / Ollama)", "llm"),
            ("Google Translate", "google"),
            ("DeepL API", "deepl"),
            ("None", "none"),
        ):
            self.primary_combo.addItem(label, val)
        self.primary_combo.currentIndexChanged.connect(self._on_translator_changed)
        form.addRow("Primary:", self.primary_combo)

        self.fallback_combo = QComboBox()
        for label, val in (
            ("Google Translate", "google"),
            ("LLM API", "llm"),
            ("DeepL API", "deepl"),
            ("None", "none"),
        ):
            self.fallback_combo.addItem(label, val)
        self.fallback_combo.currentIndexChanged.connect(self._on_translator_changed)
        form.addRow("Fallback:", self.fallback_combo)

        self.llm_url_edit = QLineEdit()
        self.llm_url_edit.setPlaceholderText("https://api.openai.com/v1 or http://localhost:11434/v1")
        self.llm_url_edit.editingFinished.connect(self._save_config)
        form.addRow("LLM Endpoint:", self.llm_url_edit)

        self.llm_model_edit = QLineEdit()
        self.llm_model_edit.setPlaceholderText("gpt-4o-mini")
        self.llm_model_edit.editingFinished.connect(self._save_config)
        form.addRow("LLM Model:", self.llm_model_edit)

        key_row = QHBoxLayout()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("Stored in Windows Credential Manager")
        self.save_key_btn = QPushButton("Save Key")
        self.save_key_btn.clicked.connect(self._save_api_key)
        key_row.addWidget(self.api_key_edit, 1)
        key_row.addWidget(self.save_key_btn)
        form.addRow("API Key:", key_row)

        self.target_lang_combo = QComboBox()
        for label, code in TARGET_LANGUAGES:
            self.target_lang_combo.addItem(label, code)
        self.target_lang_combo.currentIndexChanged.connect(self._save_config)
        form.addRow("Target Lang:", self.target_lang_combo)

        rate_row = QHBoxLayout()
        self.rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.rate_slider.setRange(1, 600)
        self.rate_value_label = QLabel("60 req/min")
        self.rate_slider.valueChanged.connect(self._on_rate_changed)
        rate_row.addWidget(self.rate_slider, 1)
        rate_row.addWidget(self.rate_value_label)
        form.addRow("Rate Limit:", rate_row)

        self.max_sentence_spin = QSpinBox()
        self.max_sentence_spin.setRange(20, 1000)
        self.max_sentence_spin.setSingleStep(10)
        self.max_sentence_spin.setSuffix(" chars")
        self.max_sentence_spin.setToolTip(
            "Maximum character length of each sentence sent for translation"
        )
        self.max_sentence_spin.valueChanged.connect(self._on_max_sentence_changed)
        form.addRow("Max Sentence:", self.max_sentence_spin)
        return box

    def _build_controls_bar(self) -> QGroupBox:
        box = QGroupBox("Session, Logging & Overlay Controls")
        outer = QHBoxLayout(box)
        outer.setSpacing(10)

        self.start_stop_btn = QPushButton("Start Translation")
        self.start_stop_btn.setObjectName("primaryBtn")
        self.start_stop_btn.clicked.connect(self.toggle_running)
        outer.addWidget(self.start_stop_btn)

        self.overlay_toggle_btn = QPushButton("Show Overlay (Ctrl+Shift+T)")
        self.overlay_toggle_btn.setCheckable(True)
        self.overlay_toggle_btn.toggled.connect(self.set_overlay_visible)
        outer.addWidget(self.overlay_toggle_btn)

        self.click_through_chk = QCheckBox("Click-Through")
        self.click_through_chk.toggled.connect(self._on_click_through_changed)
        outer.addWidget(self.click_through_chk)

        self.overlay_mode_combo = QComboBox()
        self.overlay_mode_combo.addItem("Bilingual", "bilingual")
        self.overlay_mode_combo.addItem("Translation Only", "translation")
        self.overlay_mode_combo.addItem("Original Only", "original")
        self.overlay_mode_combo.currentIndexChanged.connect(self._on_overlay_style_changed)
        outer.addWidget(self.overlay_mode_combo)

        self.overlay_font_spin = QSpinBox()
        self.overlay_font_spin.setRange(12, 48)
        self.overlay_font_spin.setSuffix(" pt")
        self.overlay_font_spin.valueChanged.connect(self._on_overlay_style_changed)
        outer.addWidget(self.overlay_font_spin)

        self.overlay_lines_spin = QSpinBox()
        self.overlay_lines_spin.setRange(1, 10)
        self.overlay_lines_spin.setSuffix(" lines")
        self.overlay_lines_spin.setToolTip(
            "Maximum number of subtitle lines displayed at once on the overlay"
        )
        self.overlay_lines_spin.valueChanged.connect(self._on_overlay_style_changed)
        outer.addWidget(self.overlay_lines_spin)

        outer.addStretch(1)

        outer.addWidget(QLabel("Log:"))
        self.logging_mode_combo = QComboBox()
        self.logging_mode_combo.addItem("Full Bilingual", "full")
        self.logging_mode_combo.addItem("Source Only", "source_only")
        self.logging_mode_combo.addItem("Ephemeral", "ephemeral")
        self.logging_mode_combo.currentIndexChanged.connect(self._on_logging_mode_changed)
        outer.addWidget(self.logging_mode_combo)

        self.retention_spin = QSpinBox()
        self.retention_spin.setRange(0, 365)
        self.retention_spin.setSuffix(" d")
        self.retention_spin.setToolTip("Auto-delete logs older than N days (0 = off)")
        self.retention_spin.valueChanged.connect(self._save_config)
        outer.addWidget(self.retention_spin)

        for fmt in ("srt", "txt", "jsonl", "csv"):
            btn = QPushButton(f".{fmt}")
            btn.clicked.connect(lambda _=False, f=fmt: self.export_current(f))
            outer.addWidget(btn)

        return box

    def _build_live_feed_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)

        self.feed_table = QTableWidget(0, 4)
        self.feed_table.setHorizontalHeaderLabels(["Time", "Source", "Translation", "Status"])
        self.feed_table.setAlternatingRowColors(True)
        self.feed_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.feed_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self.feed_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.feed_table)
        return widget

    def _build_history_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        search_row = QHBoxLayout()
        self.history_search_edit = QLineEdit()
        self.history_search_edit.setPlaceholderText("Search past sessions (source or translated text)...")
        self.history_search_edit.returnPressed.connect(self.run_history_search)
        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.run_history_search)
        search_row.addWidget(self.history_search_edit, 1)
        search_row.addWidget(self.search_btn)
        layout.addLayout(search_row)

        self.history_table = QTableWidget(0, 4)
        self.history_table.setHorizontalHeaderLabels(["Session", "Source", "Translation", "Status"])
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.history_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.history_table, 1)
        return widget

    def _set_combo_by_data(self, combo: QComboBox, value) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _populate_from_config(self) -> None:
        self.silence_slider.blockSignals(True)
        self.silence_slider.setValue(int(self.config.silence_ms))
        self.silence_slider.blockSignals(False)
        self.silence_value_label.setText(f"{self.config.silence_ms} ms")

        self._set_combo_by_data(self.stt_engine_combo, self.config.stt_engine)
        self._set_combo_by_data(self.whisper_model_combo, self.config.whisper_model)
        self._set_combo_by_data(self.compute_device_combo, self.config.compute_device)
        self._set_combo_by_data(self.source_lang_combo, self.config.source_lang)
        self.custom_model_path_edit.setText(self.config.custom_model_path)
        self.custom_entrypoint_edit.setText(self.config.custom_entrypoint)

        self._set_combo_by_data(self.primary_combo, self.config.primary_translator)
        self._set_combo_by_data(self.fallback_combo, self.config.fallback_translator)
        self.llm_url_edit.setText(self.config.llm_base_url)
        self.llm_model_edit.setText(self.config.llm_model)
        self._set_combo_by_data(self.target_lang_combo, self.config.target_lang)

        self.rate_slider.blockSignals(True)
        self.rate_slider.setValue(int(self.config.max_requests_per_min))
        self.rate_slider.blockSignals(False)
        self.rate_value_label.setText(f"{self.config.max_requests_per_min} req/min")

        self.max_sentence_spin.blockSignals(True)
        self.max_sentence_spin.setValue(int(self.config.max_sentence_chars))
        self.max_sentence_spin.blockSignals(False)

        self._set_combo_by_data(self.logging_mode_combo, self.config.logging_mode)
        self.retention_spin.blockSignals(True)
        self.retention_spin.setValue(int(self.config.retention_days))
        self.retention_spin.blockSignals(False)

        self.click_through_chk.blockSignals(True)
        self.click_through_chk.setChecked(bool(self.config.overlay_click_through))
        self.click_through_chk.blockSignals(False)
        self._set_combo_by_data(self.overlay_mode_combo, self.config.overlay_display_mode)
        self.overlay_font_spin.blockSignals(True)
        self.overlay_font_spin.setValue(int(self.config.overlay_font_size))
        self.overlay_font_spin.blockSignals(False)
        self.overlay_lines_spin.blockSignals(True)
        self.overlay_lines_spin.setValue(int(self.config.overlay_max_lines))
        self.overlay_lines_spin.blockSignals(False)
        self._load_masked_key()

    def _active_key_provider(self) -> str:
        p = self.primary_combo.currentData()
        if p in {"llm", "deepl"}:
            return p
        fb = self.fallback_combo.currentData()
        if fb in {"llm", "deepl"}:
            return fb
        return "llm"

    def _load_masked_key(self) -> None:
        provider = self._active_key_provider()
        self.api_key_edit.setText(secrets.get_api_key(provider))

    def _save_api_key(self) -> None:
        provider = self._active_key_provider()
        ok = secrets.set_api_key(provider, self.api_key_edit.text())
        if ok:
            self.set_status(f"Saved {provider.upper()} API key to Windows Credential Manager.")
        else:
            self.set_status("Could not store API key in keyring.")

    def refresh_devices(self) -> None:
        try:
            self._devices = list_wasapi_devices()
        except Exception as exc:
            self._devices = []
            self.set_status(f"Device scan failed: {exc}")

        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for dev in self._devices:
            tag = "[Loopback]" if dev.is_loopback else "[Mic]"
            self.device_combo.addItem(f"{tag} {dev.name}", dev.index)

        target_idx = self.config.device_index
        if target_idx is None:
            target_idx = default_loopback_index()
        pos = self.device_combo.findData(target_idx)
        if pos >= 0:
            self.device_combo.setCurrentIndex(pos)
        elif self.device_combo.count() > 0:
            self.device_combo.setCurrentIndex(0)
        self.device_combo.blockSignals(False)
        self._update_engine_warnings()

    def _selected_device(self) -> AudioDeviceInfo | None:
        idx = self.device_combo.currentData()
        return next((d for d in self._devices if d.index == idx), None)

    def _on_device_changed(self) -> None:
        self._save_config()
        self._update_engine_warnings()

    def _on_stt_changed(self) -> None:
        self._save_config()
        self._update_engine_warnings()

    def _update_engine_warnings(self) -> None:
        engine = self.stt_engine_combo.currentData()
        self.custom_rows_widget.setVisible(engine == "custom")
        is_whisper = engine == "whisper"
        self.whisper_model_combo.setEnabled(is_whisper)
        self.compute_device_combo.setEnabled(is_whisper)

        dev = self._selected_device()
        if engine == "windows_native" and dev is not None and dev.is_loopback:
            self.audio_warning_label.setText(
                "Note: Windows Native STT captures from the default recording microphone, not the selected loopback device."
            )
            self.audio_warning_label.show()
        else:
            self.audio_warning_label.hide()

    def _on_translator_changed(self) -> None:
        self._save_config()
        self._update_translator_fields()
        self._load_masked_key()

    def _update_translator_fields(self) -> None:
        uses_llm = "llm" in {self.primary_combo.currentData(), self.fallback_combo.currentData()}
        self.llm_url_edit.setEnabled(uses_llm)
        self.llm_model_edit.setEnabled(uses_llm)

    def _on_silence_changed(self, value: int) -> None:
        self.silence_value_label.setText(f"{value} ms")
        self._save_config()
        if self._stt_worker is not None:
            self._stt_worker.set_silence_ms(value)

    def _on_rate_changed(self, value: int) -> None:
        self.rate_value_label.setText(f"{value} req/min")
        self._save_config()
        if self._tr_worker is not None:
            self._tr_worker.update_rate(value)

    def _on_max_sentence_changed(self, value: int) -> None:
        self._save_config()
        if self._stt_worker is not None:
            self._stt_worker.set_max_sentence_chars(value)
        if self._tr_worker is not None:
            self._tr_worker.update_max_sentence_chars(value)

    def _on_logging_mode_changed(self) -> None:
        self._save_config()
        self._update_history_availability()

    def _update_history_availability(self) -> None:
        ephemeral = self.logging_mode_combo.currentData() == "ephemeral"
        self.history_search_edit.setEnabled(not ephemeral)
        self.search_btn.setEnabled(not ephemeral)
        if ephemeral:
            self.history_search_edit.setPlaceholderText("History search disabled in Ephemeral mode")
        else:
            self.history_search_edit.setPlaceholderText("Search past sessions (source or translated text)...")

    def _on_click_through_changed(self, checked: bool) -> None:
        self.overlay.set_click_through(checked)
        self._save_config()

    def _on_overlay_style_changed(self) -> None:
        self.overlay.set_display_mode(self.overlay_mode_combo.currentData())
        self.overlay.set_font(self.config.overlay_font_family, self.overlay_font_spin.value())
        self.overlay.set_max_lines(self.overlay_lines_spin.value())
        self._save_config()

    def _on_overlay_geometry(self, geom: list[int]) -> None:
        self.config.overlay_geometry = geom

    def _collect_config(self) -> AppConfig:
        self.config.device_index = self.device_combo.currentData()
        self.config.silence_ms = int(self.silence_slider.value())
        self.config.stt_engine = str(self.stt_engine_combo.currentData())
        self.config.whisper_model = str(self.whisper_model_combo.currentData())
        self.config.compute_device = str(self.compute_device_combo.currentData())
        self.config.source_lang = str(self.source_lang_combo.currentData())
        self.config.custom_model_path = self.custom_model_path_edit.text().strip()
        self.config.custom_entrypoint = self.custom_entrypoint_edit.text().strip()
        self.config.primary_translator = str(self.primary_combo.currentData())
        self.config.fallback_translator = str(self.fallback_combo.currentData())
        self.config.llm_base_url = self.llm_url_edit.text().strip() or "https://api.openai.com/v1"
        self.config.llm_model = self.llm_model_edit.text().strip() or "gpt-4o-mini"
        self.config.target_lang = str(self.target_lang_combo.currentData())
        self.config.max_requests_per_min = int(self.rate_slider.value())
        self.config.max_sentence_chars = int(self.max_sentence_spin.value())
        self.config.logging_mode = str(self.logging_mode_combo.currentData())
        self.config.retention_days = int(self.retention_spin.value())
        self.config.overlay_display_mode = str(self.overlay_mode_combo.currentData())
        self.config.overlay_font_size = int(self.overlay_font_spin.value())
        self.config.overlay_max_lines = int(self.overlay_lines_spin.value())
        self.config.overlay_click_through = bool(self.click_through_chk.isChecked())
        return self.config

    def _save_config(self) -> None:
        try:
            self._collect_config().save()
        except OSError:
            pass

    @Slot(bool)
    def set_overlay_visible(self, visible: bool) -> None:
        if self.overlay_toggle_btn.isChecked() != bool(visible):
            self.overlay_toggle_btn.blockSignals(True)
            self.overlay_toggle_btn.setChecked(bool(visible))
            self.overlay_toggle_btn.blockSignals(False)
        if visible:
            self.overlay.show()
            self.overlay.raise_()
        else:
            self.overlay.hide()

    def toggle_overlay(self) -> None:
        self.set_overlay_visible(not self.overlay.isVisible())

    @Slot(bool)
    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)
        if self.mute_btn.isChecked() != self._muted:
            self.mute_btn.blockSignals(True)
            self.mute_btn.setChecked(self._muted)
            self.mute_btn.blockSignals(False)
        if self._capture is not None:
            self._capture.set_muted(self._muted)
        self.set_status("Muted" if self._muted else ("Listening" if self._running else "Idle"))

    def toggle_mute(self) -> None:
        self.set_muted(not self._muted)

    def toggle_running(self) -> None:
        if self._running:
            self.stop_session()
        else:
            self.start_session()

    def start_session(self) -> None:
        if self._running:
            return
        cfg = self._collect_config()
        cfg.save()
        dev = self._selected_device()
        if dev is None and cfg.stt_engine != "windows_native":
            QMessageBox.warning(self, "No Audio Device", "Please select a valid WASAPI audio device.")
            return

        self._seq_rows.clear()
        self.feed_table.setRowCount(0)
        self.overlay.clear()
        self._ring = AudioRingBuffer(max_seconds=30.0)
        self._logger = DualTranscriptLogger(
            base_dir=paths.logs_dir(),
            target_lang=cfg.target_lang,
            mode=cfg.logging_mode,
        )

        self._tr_worker = TranslationWorker(cfg, self._logger, self)
        self._tr_worker.utterance_translated.connect(self.on_utterance_translated)
        self._tr_worker.status.connect(self.set_status)
        self._tr_worker.error.connect(self._on_worker_error)
        self._tr_worker.start()

        self._stt_worker = STTWorker(cfg, self._ring, self._logger, self)
        self._stt_worker.utterance_transcribed.connect(self.on_utterance_transcribed)
        self._stt_worker.status.connect(self.set_status)
        self._stt_worker.error.connect(self._on_worker_error)
        self._stt_worker.start()

        if dev is not None and cfg.stt_engine != "windows_native":
            self._capture = AudioCaptureThread(dev, self._ring, self)
            self._capture.level.connect(self._on_vu_level)
            self._capture.error.connect(self._on_worker_error)
            self._capture.set_muted(self._muted)
            try:
                self._capture.start()
            except Exception as exc:
                self._on_worker_error(str(exc))
                self.stop_session()
                return

        self._running = True
        self.start_stop_btn.setText("Stop Translation")
        self.start_stop_btn.setObjectName("stopBtn")
        self.start_stop_btn.style().unpolish(self.start_stop_btn)
        self.start_stop_btn.style().polish(self.start_stop_btn)

    def stop_session(self) -> None:
        if self._capture is not None:
            try:
                self._capture.stop()
            except Exception:
                pass
            self._capture = None

        if self._stt_worker is not None:
            self._stt_worker.stop()
            self._stt_worker.wait(5000)
            self._stt_worker = None

        if self._tr_worker is not None:
            self._tr_worker.stop()
            self._tr_worker.wait(5000)
            self._tr_worker = None

        if self._logger is not None:
            self._logger.shutdown()

        self._running = False
        self.vu_meter.setValue(0)
        self.start_stop_btn.setText("Start Translation")
        self.start_stop_btn.setObjectName("primaryBtn")
        self.start_stop_btn.style().unpolish(self.start_stop_btn)
        self.start_stop_btn.style().polish(self.start_stop_btn)
        self.set_status("Idle")

    @Slot(float)
    def _on_vu_level(self, level: float) -> None:
        if level <= 1e-6:
            scaled = 0
        else:
            db = 20.0 * math.log10(min(1.0, level))
            scaled = int(max(0.0, min(1.0, (db + 60.0) / 60.0)) * 1000)
        self.vu_meter.setValue(scaled)

    @Slot(str)
    def set_status(self, message: str) -> None:
        self.status_bar_label.setText(message)

    @Slot(str)
    def _on_worker_error(self, message: str) -> None:
        self.set_status(f"Error: {message}")

    @Slot(object)
    def on_utterance_transcribed(self, task: UtteranceTask) -> None:
        if self._tr_worker is not None:
            self._tr_worker.submit(task)

        row = self.feed_table.rowCount()
        self.feed_table.insertRow(row)
        ts_str = datetime.fromtimestamp(task.start_ts).strftime("%H:%M:%S")
        badge = "[Whisper]" if self.config.stt_engine == "whisper" else f"[{self.config.stt_engine}]"
        self.feed_table.setItem(row, 0, QTableWidgetItem(ts_str))
        self.feed_table.setItem(row, 1, QTableWidgetItem(task.source_text))
        self.feed_table.setItem(row, 2, QTableWidgetItem("..."))
        self.feed_table.setItem(row, 3, QTableWidgetItem(f"{badge} [Pending]"))
        self._seq_rows[task.seq_id] = row
        self.feed_table.scrollToBottom()
        self.overlay.show_pending(task.seq_id, task.source_text)

    @Slot(int, object, str, object, str, int, str)
    def on_utterance_translated(
        self,
        seq_id: int,
        merged_ids: list[int],
        combined_src: str,
        translated: str | None,
        provider: str,
        latency_ms: int,
        status: str,
    ) -> None:
        badge_map = {
            "TRANSLATED": f"[{provider}]",
            "FALLBACK_TRANSLATED": f"[Fallback: {provider}]",
            "FAILED": "[Failed]",
            "NOT_TRANSLATED": "[Source Only]",
        }
        badge = badge_map.get(status, f"[{status}]")

        row = self._seq_rows.get(seq_id)
        if row is not None and row < self.feed_table.rowCount():
            self.feed_table.setItem(row, 1, QTableWidgetItem(combined_src))
            self.feed_table.setItem(row, 2, QTableWidgetItem(translated or ""))
            self.feed_table.setItem(row, 3, QTableWidgetItem(f"{badge} ({latency_ms} ms)"))

        for sub_id in merged_ids:
            if sub_id != seq_id:
                sub_row = self._seq_rows.get(sub_id)
                if sub_row is not None and sub_row < self.feed_table.rowCount():
                    self.feed_table.setItem(sub_row, 2, QTableWidgetItem(f"↳ merged into #{seq_id}"))
                    self.feed_table.setItem(sub_row, 3, QTableWidgetItem("[Coalesced]"))

        self.feed_table.scrollToBottom()
        self.overlay.show_translation(seq_id, combined_src, translated, merged_ids=merged_ids)

    def run_history_search(self) -> None:
        if self.config.logging_mode == "ephemeral":
            return
        db_path = paths.logs_dir() / "sessions.db"
        rows = log_export.search(db_path, self.history_search_edit.text(), limit=200)
        self.history_table.setRowCount(0)
        for item in rows:
            r = self.history_table.rowCount()
            self.history_table.insertRow(r)
            self.history_table.setItem(r, 0, QTableWidgetItem(str(item.get("session_id", ""))))
            self.history_table.setItem(r, 1, QTableWidgetItem(str(item.get("source_text", ""))))
            self.history_table.setItem(r, 2, QTableWidgetItem(str(item.get("translated_text") or "")))
            self.history_table.setItem(r, 3, QTableWidgetItem(str(item.get("status", ""))))

    def export_current(self, fmt: str) -> None:
        if self._logger is None:
            QMessageBox.information(self, "Nothing to Export", "Start a translation session first.")
            return
        self._logger.flush()
        entries = self._logger.snapshot_entries()
        if not entries:
            QMessageBox.information(self, "Nothing to Export", "No transcript entries captured yet.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export .{fmt}",
            str(paths.logs_dir() / f"transcript.{fmt}"),
            f"{fmt.upper()} Files (*.{fmt})",
        )
        if not out_path:
            return
        log_export.export_entries(entries, fmt=fmt, out_path=out_path, session_start=self._logger.session_start)
        self.set_status(f"Exported session to {out_path}")

    def closeEvent(self, event: QCloseEvent) -> None:
        self.stop_session()
        self.overlay.close()
        self._save_config()
        super().closeEvent(event)
