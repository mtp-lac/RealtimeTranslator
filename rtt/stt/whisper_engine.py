"""Faster-Whisper (CTranslate2) STT backend with hallucination filtering."""

import gc
import re
from typing import Iterable

import numpy as np

from rtt.stt.base import BaseSTTEngine, STTResult
from rtt.stt.cuda_dlls import register_nvidia_dll_dirs

HALLUCINATION_PHRASES = {
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "subtitles by",
    "please subscribe",
    "subscribe",
    "mbc 뉴스",
    "ご視聴ありがとうございました",
    "ありがとうございました",
    "お疲れ様でした",
    "you",
}


def _normalize_for_blocklist(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _is_hallucination(text: str) -> bool:
    norm = _normalize_for_blocklist(text)
    if not norm:
        return True
    if norm in HALLUCINATION_PHRASES:
        return True
    return norm.startswith("subtitles by ")


def filter_segments(segments: Iterable, audio_seconds: float) -> str:
    """Reject likely Whisper hallucinations using probability, compression, and phrase guards."""
    clean_texts: list[str] = []
    for seg in segments:
        no_speech_prob = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
        compression_ratio = float(getattr(seg, "compression_ratio", 0.0) or 0.0)
        if no_speech_prob > 0.65:
            continue
        if compression_ratio > 2.4:
            continue
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        if (audio_seconds < 1.0 or no_speech_prob > 0.45) and _is_hallucination(text):
            continue
        clean_texts.append(text)
    return " ".join(clean_texts).strip()


class WhisperEngine(BaseSTTEngine):
    """Primary STT engine backed by faster-whisper (CTranslate2)."""

    def __init__(self):
        self.model = None
        self.model_size = "small"
        self.device_used = "cpu"
        self.compute_type_used = "int8"
        self.fallback_reason: str | None = None

    @property
    def name(self) -> str:
        return f"WhisperEngine({self.model_size}/{self.device_used})"

    def load_model(self, config: dict) -> None:
        register_nvidia_dll_dirs()
        from faster_whisper import WhisperModel

        self.unload()
        self.model_size = str(config.get("model_size", "small"))
        requested_device = str(config.get("device", "cuda"))
        requested_compute = config.get("compute_type") or (
            "float16" if requested_device == "cuda" else "int8"
        )
        self.fallback_reason = None

        if requested_device == "cuda":
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device="cuda",
                    compute_type=requested_compute,
                )
                self.device_used = "cuda"
                self.compute_type_used = str(requested_compute)
                return
            except Exception as exc:
                self.fallback_reason = str(exc)

        self.model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type="int8",
        )
        self.device_used = "cpu"
        self.compute_type_used = "int8"

    def transcribe(self, audio: np.ndarray, language: str | None) -> STTResult | None:
        if self.model is None:
            raise RuntimeError("Whisper model is not loaded.")
        lang = None if not language or language == "auto" else language
        audio_f32 = np.asarray(audio, dtype=np.float32).ravel()
        audio_seconds = float(audio_f32.size) / 16000.0

        segments, info = self.model.transcribe(
            audio_f32,
            language=lang,
            beam_size=3,
            best_of=1,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = filter_segments(segments, audio_seconds)
        if not text:
            return None
        detected_lang = str(getattr(info, "language", None) or lang or "unknown")
        lang_prob = round(float(getattr(info, "language_probability", 1.0) or 1.0), 3)
        return STTResult(text=text, language=detected_lang, language_prob=lang_prob)

    def unload(self) -> None:
        if self.model is not None:
            self.model = None
            gc.collect()
