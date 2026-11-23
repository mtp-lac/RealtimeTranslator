"""Application startup entrypoint for RealtimeTranslator."""

import sys

from rtt import paths
from rtt.config import AppConfig
from rtt.logs.transcript_logger import prune_older_than


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from rtt.gui.main_window import MainWindow
    from rtt.gui.styles import DARK_QSS

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("RealtimeTranslator")
    app.setStyleSheet(DARK_QSS)

    config = AppConfig.load()
    if config.retention_days > 0:
        try:
            prune_older_than(paths.logs_dir(), config.retention_days)
        except Exception:
            pass

    window = MainWindow(config=config)
    window.show()

    hotkey_filter = None
    try:
        from rtt.gui.hotkeys import GlobalHotkeys

        hotkey_filter = GlobalHotkeys(
            on_toggle_overlay=window.toggle_overlay,
            on_toggle_mute=window.toggle_mute,
            on_status=window.set_status,
        )
        app.installNativeEventFilter(hotkey_filter)
        app.aboutToQuit.connect(hotkey_filter.unregister_all)
    except ImportError:
        pass

    return app.exec()
