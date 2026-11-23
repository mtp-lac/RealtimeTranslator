"""Tests for hysteresis VADSegmenter state machine and pre-roll/overlap behavior."""

import numpy as np

from rtt.vad.segmenter import VADSegmenter


def _tone(amplitude: float = 0.2, marker: float = 0.0) -> np.ndarray:
    arr = np.full(512, amplitude, dtype=np.float32)
    if marker != 0.0:
        arr[0] = marker
    return arr


def test_rms_gate_skips_prob_fn_on_silence():
    calls = 0

    def prob_fn(frame: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return 0.9

    seg = VADSegmenter(prob_fn=prob_fn, rms_gate=0.003)
    out = seg.process(np.zeros(512 * 10, dtype=np.float32))
    assert out == []
    assert calls == 0


def test_min_speech_frames_rejects_short_blip():
    # frames with marker=0.9 are speech, marker=0.1 are non-speech noise above rms_gate
    def prob_fn(frame: np.ndarray) -> float:
        return 0.9 if frame[0] > 0.5 else 0.1

    seg = VADSegmenter(prob_fn=prob_fn, min_speech_frames=4, silence_ms=420)
    frames = [_tone(0.05, marker=0.9)] * 3 + [_tone(0.05, marker=0.1)] * 20
    out = seg.process(np.concatenate(frames))
    assert out == []
    assert not seg.is_speaking


def test_preroll_hysteresis_and_natural_flush():
    resets = 0
    now = [100.0]

    def clock() -> float:
        now[0] += 0.032
        return now[0]

    def prob_fn(frame: np.ndarray) -> float:
        if frame[0] > 0.8:
            return 0.85  # above start_thr 0.50
        if frame[0] > 0.4:
            return 0.40  # between end_thr 0.35 and start_thr 0.50 (hysteresis hold)
        return 0.10

    seg = VADSegmenter(
        prob_fn=prob_fn,
        silence_ms=420,  # 13 frames
        min_speech_frames=4,
        pre_roll_frames=8,
        clock=clock,
        on_reset=lambda: _inc(),
    )

    def _inc():
        nonlocal resets
        resets += 1

    # 8 pre-roll noise frames + 4 speech trigger frames + 3 hysteresis frames + 13 silent frames
    frames = (
        [_tone(0.05, marker=0.1)] * 8
        + [_tone(0.05, marker=0.9)] * 4
        + [_tone(0.05, marker=0.45)] * 3
        + [_tone(0.05, marker=0.1)] * 13
    )
    segments = seg.process(np.concatenate(frames))
    assert len(segments) == 1
    utterance = segments[0]
    assert not utterance.forced
    assert resets == 1
    # 8 frames seeded from pre_roll + 3 hysteresis frames + 13 trailing silence frames = 24 frames
    assert utterance.audio.size == 24 * 512
    # First 4 frames should be the pre-trigger context frames (marker 0.1)
    assert np.isclose(utterance.audio[0], 0.1)
    assert np.isclose(utterance.audio[4 * 512], 0.9)


def test_forced_max_duration_flush_preserves_6_frame_tail():
    resets = 0

    def prob_fn(frame: np.ndarray) -> float:
        return 0.95

    seg = VADSegmenter(
        prob_fn=prob_fn,
        silence_ms=420,
        max_chunk_sec=0.64,  # 20 frames max
        min_speech_frames=4,
        pre_roll_frames=8,
        overlap_tail_frames=6,
        on_reset=lambda: _inc(),
    )

    def _inc():
        nonlocal resets
        resets += 1

    # Tag each frame uniquely at index 1 so we can verify tail continuity
    speech = []
    for idx in range(25):
        f = _tone(0.2)
        f[1] = float(idx)
        speech.append(f)

    out = seg.process(np.concatenate(speech))
    assert len(out) == 1
    assert out[0].forced is True
    assert out[0].audio.size == 20 * 512
    assert resets == 0
    assert seg.is_speaking is True
    # 6-frame overlap tail (frames 14..19) + 5 remaining frames (20..24) = 11 frames pending
    assert len(seg.speech_frames) == 11
    assert seg.speech_frames[0][1] == 14.0

    # Flushing on stop emits the remaining 11 frames including the 6-frame overlap tail
    tail_seg = seg.flush()
    assert tail_seg is not None
    assert tail_seg.audio.size == 11 * 512
    assert tail_seg.audio[1] == 14.0
    assert seg.flush() is None
