"""Tests for Whisper segment hallucination and quality guards."""

from types import SimpleNamespace

import numpy as np

from rtt.stt.whisper_engine import WhisperEngine, filter_segments


def test_filter_segments_rejects_no_speech_and_compression():
    segs = [
        SimpleNamespace(text="should skip high no_speech", no_speech_prob=0.70, compression_ratio=1.1),
        SimpleNamespace(text="should skip compression", no_speech_prob=0.10, compression_ratio=2.6),
        SimpleNamespace(text="Hello world.", no_speech_prob=0.05, compression_ratio=1.2),
    ]
    assert filter_segments(segs, audio_seconds=2.5) == "Hello world."


def test_filter_segments_blocklist_short_or_uncertain_audio():
    hallucination = [SimpleNamespace(text="Thank you.", no_speech_prob=0.20, compression_ratio=1.0)]
    assert filter_segments(hallucination, audio_seconds=0.6) == ""
    assert filter_segments(hallucination, audio_seconds=2.0) == "Thank you."

    uncertain = [SimpleNamespace(text="Thanks for watching!", no_speech_prob=0.50, compression_ratio=1.0)]
    assert filter_segments(uncertain, audio_seconds=2.5) == ""

    subtitles = [SimpleNamespace(text="Subtitles by Amara.org", no_speech_prob=0.10, compression_ratio=1.0)]
    assert filter_segments(subtitles, audio_seconds=0.8) == ""


def test_filter_segments_custom_thresholds():
    segs = [SimpleNamespace(text="Borderline speech", no_speech_prob=0.55, compression_ratio=2.1)]
    assert filter_segments(segs, audio_seconds=2.0, no_speech_threshold=0.50) == ""
    assert filter_segments(segs, audio_seconds=2.0, compression_ratio_threshold=2.0) == ""
    assert filter_segments(
        segs, audio_seconds=2.0, no_speech_threshold=0.70, compression_ratio_threshold=2.5
    ) == "Borderline speech"


def test_whisper_engine_runtime_config_forwarded():
    captured_kwargs = {}

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            captured_kwargs.update(kwargs)
            segs = [SimpleNamespace(text="Hello", no_speech_prob=0.1, compression_ratio=1.1)]
            info = SimpleNamespace(language="en", language_probability=0.98)
            return segs, info

    engine = WhisperEngine()
    engine.model = FakeModel()
    engine.apply_runtime_config(
        {
            "task": "translate",
            "beam_size": 5,
            "best_of": 3,
            "temperature": 0.2,
            "vad_filter": True,
            "condition_on_previous_text": True,
            "initial_prompt": "  Glossary: CUDA  ",
            "no_speech_threshold": 0.8,
            "compression_ratio_threshold": 3.0,
        }
    )

    res = engine.transcribe(np.zeros(16000, dtype=np.float32), "ja")
    assert res is not None and res.text == "Hello"
    assert captured_kwargs["task"] == "translate"
    assert captured_kwargs["beam_size"] == 5
    assert captured_kwargs["best_of"] == 3
    assert captured_kwargs["temperature"] == 0.2
    assert captured_kwargs["vad_filter"] is True
    assert captured_kwargs["condition_on_previous_text"] is True
    assert captured_kwargs["initial_prompt"] == "Glossary: CUDA"

    engine.apply_runtime_config({"initial_prompt": "   "})
    engine.transcribe(np.zeros(16000, dtype=np.float32), "auto")
    assert captured_kwargs["initial_prompt"] is None
