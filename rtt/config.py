"""Application configuration dataclass with safe JSON load/save."""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

from rtt import paths


@dataclass
class AppConfig:
    """Non-secret user configuration persisted under %APPDATA%/RealtimeTranslator/config.json."""

    device_index: int | None = None
    silence_ms: int = 420
    stt_engine: str = "whisper"
    whisper_model: str = "small"
    compute_device: str = "cuda"
    whisper_compute_type_mode: str = "auto"
    whisper_task: str = "transcribe"
    whisper_beam_size: int = 3
    whisper_best_of: int = 1
    whisper_temperature: float = 0.0
    whisper_vad_filter: bool = False
    whisper_condition_on_previous_text: bool = False
    whisper_initial_prompt: str = ""
    whisper_no_speech_threshold: float = 0.65
    whisper_compression_ratio_threshold: float = 2.4
    source_lang: str = "auto"
    primary_translator: str = "llm"
    fallback_translator: str = "google"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    target_lang: str = "en"
    max_requests_per_min: int = 60
    logging_mode: str = "full"
    retention_days: int = 30
    overlay_font_family: str = "Segoe UI"
    overlay_font_size: int = 22
    overlay_max_lines: int = 3
    overlay_display_mode: str = "bilingual"
    overlay_source_color: str = "#CBD5E1"
    overlay_translation_color: str = "#FFFFFF"
    overlay_geometry: list[int] | None = None
    overlay_click_through: bool = False
    max_sentence_chars: int = 140
    custom_model_path: str = ""
    custom_entrypoint: str = ""

    @property
    def whisper_compute_type(self) -> str:
        if self.whisper_compute_type_mode and self.whisper_compute_type_mode != "auto":
            return self.whisper_compute_type_mode
        return "float16" if self.compute_device == "cuda" else "int8"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        if not isinstance(data, dict):
            return cls()
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        target = path or paths.config_path()
        if not target.is_file():
            return cls()
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        return cls.from_dict(data)

    def save(self, path: Path | None = None) -> Path:
        target = path or paths.config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return target
