"""Application data directory and path helpers."""

import os
from pathlib import Path


APP_NAME = "RealtimeTranslator"


def app_dir() -> Path:
    """Return the writable per-user data directory, creating it if needed."""
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    root = base / APP_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def logs_dir() -> Path:
    """Return the transcript logs directory, creating it if needed."""
    path = app_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    """Return the local models directory, creating it if needed."""
    path = app_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path() -> Path:
    """Return the path to the JSON configuration file."""
    return app_dir() / "config.json"
