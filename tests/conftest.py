"""Shared pytest fixtures for offscreen Qt tests and isolated temp app_dir."""

import os
from pathlib import Path
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def isolated_app_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect rtt.paths.app_dir to a fresh temporary directory for every test."""
    target = tmp_path / "RealtimeTranslator"
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("rtt.paths.app_dir", lambda: target)
    return target


@pytest.fixture(scope="session")
def qapp():
    """Return a shared offscreen QApplication instance."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
