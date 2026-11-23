"""Offscreen PySide6 GUI and SubtitleOverlay smoke tests."""

import ctypes
from ctypes import wintypes
import time
from PySide6.QtCore import Qt

from rtt.audio.devices import AudioDeviceInfo
from rtt.config import AppConfig
from rtt.gui.hotkeys import HOTKEY_TOGGLE_MUTE, HOTKEY_TOGGLE_OVERLAY, WM_HOTKEY, GlobalHotkeys
from rtt.gui.main_window import MainWindow
from rtt.gui.overlay import SubtitleOverlay
from rtt.gui.whisper_dialog import WhisperSettingsDialog
from rtt.translate.pipeline import UtteranceTask


def test_subtitle_overlay_modes_and_click_through(qapp):
    overlay = SubtitleOverlay(AppConfig())
    overlay.show()
    overlay.show_pending(1, "こんにちは")
    assert overlay.source_label.text() == "こんにちは"
    assert "Translating" in overlay.translation_label.text()

    overlay.show_translation(1, "こんにちは", "Hello")
    assert overlay.translation_label.text() == "Hello"

    overlay.set_display_mode("translation")
    assert not overlay.source_label.isVisible()
    assert overlay.translation_label.isVisible()

    overlay.set_display_mode("original")
    assert overlay.source_label.isVisible()
    assert not overlay.translation_label.isVisible()

    overlay.set_display_mode("bilingual")
    assert overlay.source_label.isVisible()
    assert overlay.translation_label.isVisible()

    overlay.set_click_through(True)
    assert bool(overlay.windowFlags() & Qt.WindowType.WindowTransparentForInput)
    overlay.set_click_through(False)
    assert not bool(overlay.windowFlags() & Qt.WindowType.WindowTransparentForInput)
    overlay.close()


def test_subtitle_overlay_max_lines_coalesce_and_clear(qapp):
    cfg = AppConfig(overlay_max_lines=2)
    overlay = SubtitleOverlay(cfg)

    overlay.show_pending(1, "s1")
    overlay.show_translation(1, "s1", "t1")
    overlay.show_pending(2, "s2")
    overlay.show_pending(3, "s3")
    # Exceeding max_lines=2 drops seq_id 1
    assert overlay.source_label.text() == "s2\ns3"
    assert overlay.translation_label.text() == "Translating...\nTranslating..."

    # Coalescing [2, 3] removes seq_id 3 and updates seq_id 2
    overlay.show_translation(2, "s2 s3", "t2 t3", merged_ids=[2, 3])
    assert overlay.source_label.text() == "s2 s3"
    assert overlay.translation_label.text() == "t2 t3"

    # Add two more lines and test live set_max_lines trimming
    overlay.show_translation(4, "s4", "t4")
    overlay.show_translation(5, "s5", "t5")
    assert overlay.source_label.text() == "s4\ns5"
    overlay.set_max_lines(1)
    assert overlay.source_label.text() == "s5"
    assert overlay.translation_label.text() == "t5"

    overlay.clear()
    assert overlay.source_label.text() == ""
    assert overlay.translation_label.text() == "Ready for live translation..."
    overlay.close()


def test_main_window_line_and_sentence_spinboxes(qapp, monkeypatch):
    monkeypatch.setattr("rtt.gui.main_window.list_wasapi_devices", lambda: [])
    monkeypatch.setattr("rtt.gui.main_window.default_loopback_index", lambda: None)
    monkeypatch.setattr("rtt.secrets.get_api_key", lambda prov: "")

    cfg = AppConfig(overlay_max_lines=5, max_sentence_chars=220)
    win = MainWindow(cfg)
    assert win.overlay_lines_spin.value() == 5
    assert win.max_sentence_spin.value() == 220

    win.overlay_lines_spin.setValue(4)
    win.max_sentence_spin.setValue(180)
    collected = win._collect_config()
    assert collected.overlay_max_lines == 4
    assert collected.max_sentence_chars == 180
    assert win.overlay._max_lines == 4
    win.close()


def test_main_window_updates_feed_and_overlay_slots(qapp, monkeypatch):
    fake_devices = [
        AudioDeviceInfo(index=1, name="Default Mic", sample_rate=48000, channels=2, is_loopback=False),
        AudioDeviceInfo(index=2, name="Speakers [Loopback]", sample_rate=48000, channels=2, is_loopback=True),
    ]
    mem_keys: dict[str, str] = {}

    monkeypatch.setattr("rtt.gui.main_window.list_wasapi_devices", lambda: fake_devices)
    monkeypatch.setattr("rtt.gui.main_window.default_loopback_index", lambda: 2)
    monkeypatch.setattr("rtt.secrets.get_api_key", lambda prov: mem_keys.get(prov, ""))
    monkeypatch.setattr("rtt.secrets.set_api_key", lambda prov, val: mem_keys.__setitem__(prov, val) or True)

    win = MainWindow(AppConfig())
    assert win.device_combo.count() == 2
    assert win.device_combo.currentData() == 2

    now = time.time()
    task = UtteranceTask(seq_id=1, start_ts=now, end_ts=now + 1.0, source_lang="ja", source_text="おはようございます")
    win.on_utterance_transcribed(task)
    assert win.feed_table.rowCount() == 1
    assert win.feed_table.item(0, 1).text() == "おはようございます"
    assert "Pending" in win.feed_table.item(0, 3).text()

    win.on_utterance_translated(1, [1], "おはようございます", "Good morning", "LLM(gpt-4o-mini)", 240, "TRANSLATED")
    assert win.feed_table.item(0, 2).text() == "Good morning"
    assert "LLM(gpt-4o-mini)" in win.feed_table.item(0, 3).text()
    assert win.overlay.translation_label.text() == "Good morning"

    win.api_key_edit.setText("sk-test-123")
    win._save_api_key()
    assert mem_keys["llm"] == "sk-test-123"
    win.close()


def test_global_hotkeys_dispatch_and_unregister(qapp, monkeypatch):
    toggled = []
    muted = []
    hk = GlobalHotkeys(
        on_toggle_overlay=lambda: toggled.append(True),
        on_toggle_mute=lambda: muted.append(True),
        register=False,
    )
    msg = wintypes.MSG()
    msg.message = WM_HOTKEY
    msg.wParam = HOTKEY_TOGGLE_OVERLAY
    handled, _ = hk.nativeEventFilter(b"windows_generic_MSG", ctypes.addressof(msg))
    qapp.processEvents()
    assert handled and len(toggled) == 1

    msg.wParam = HOTKEY_TOGGLE_MUTE
    handled2, _ = hk.nativeEventFilter(b"windows_generic_MSG", ctypes.addressof(msg))
    qapp.processEvents()
    assert handled2 and len(muted) == 1
    hk.unregister_all()


def test_whisper_settings_dialog_and_main_window_button(qapp, monkeypatch):
    cfg = AppConfig(
        whisper_compute_type_mode="int8_float16",
        whisper_task="translate",
        whisper_beam_size=6,
        whisper_best_of=4,
        whisper_temperature=0.25,
        whisper_vad_filter=True,
        whisper_condition_on_previous_text=True,
        whisper_initial_prompt="Technical terms",
        whisper_no_speech_threshold=0.75,
        whisper_compression_ratio_threshold=2.8,
    )
    dlg = WhisperSettingsDialog(cfg)
    assert dlg.compute_type_combo.currentData() == "int8_float16"
    assert dlg.task_combo.currentData() == "translate"
    assert dlg.beam_size_spin.value() == 6
    assert dlg.best_of_spin.value() == 4
    assert dlg.temperature_spin.value() == 0.25
    assert dlg.vad_filter_chk.isChecked() is True
    assert dlg.condition_prev_chk.isChecked() is True
    assert dlg.initial_prompt_edit.text() == "Technical terms"
    assert dlg.no_speech_spin.value() == 0.75
    assert dlg.compression_spin.value() == 2.8

    dlg.beam_size_spin.setValue(8)
    dlg.initial_prompt_edit.setText("  Custom prompt  ")
    target_cfg = AppConfig()
    dlg.apply_to_config(target_cfg)
    assert target_cfg.whisper_compute_type_mode == "int8_float16"
    assert target_cfg.whisper_compute_type == "int8_float16"
    assert target_cfg.whisper_beam_size == 8
    assert target_cfg.whisper_initial_prompt == "Custom prompt"

    dlg._reset_defaults()
    dlg.apply_to_config(target_cfg)
    assert target_cfg.whisper_compute_type_mode == "auto"
    assert target_cfg.whisper_compute_type == "float16"
    assert target_cfg.whisper_beam_size == 3
    assert target_cfg.whisper_task == "transcribe"
    assert target_cfg.whisper_initial_prompt == ""
    dlg.close()

    monkeypatch.setattr("rtt.gui.main_window.list_wasapi_devices", lambda: [])
    monkeypatch.setattr("rtt.gui.main_window.default_loopback_index", lambda: None)
    monkeypatch.setattr("rtt.secrets.get_api_key", lambda prov: "")

    win = MainWindow(AppConfig(stt_engine="whisper"))
    assert win.whisper_settings_btn.isEnabled() is True
    idx_native = win.stt_engine_combo.findData("windows_native")
    win.stt_engine_combo.setCurrentIndex(idx_native)
    assert win.whisper_settings_btn.isEnabled() is False
    win.close()
