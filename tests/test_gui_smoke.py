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
