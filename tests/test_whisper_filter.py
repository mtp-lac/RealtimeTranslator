"""Tests for Whisper segment hallucination and quality guards."""

from types import SimpleNamespace

from rtt.stt.whisper_engine import filter_segments


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
