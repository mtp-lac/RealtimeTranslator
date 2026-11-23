"""Tests for audio resampling, RMS energy, and the NumPy circular ring buffer."""

import numpy as np

from rtt.audio.resample import TARGET_SR, resample_to_16k, rms
from rtt.audio.ring_buffer import AudioRingBuffer


def test_resample_48k_stereo_to_16k_mono():
    # 0.5 seconds of 48 kHz stereo audio = 24,000 frames * 2 channels = 48,000 interleaved samples
    t = np.linspace(0.0, 0.5, 24000, endpoint=False, dtype=np.float32)
    left = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    stereo = np.column_stack([left, right]).ravel().astype(np.float32)

    out = resample_to_16k(stereo, src_rate=48000, channels=2)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.size == int(0.5 * TARGET_SR)
    assert np.isclose(rms(out), rms(left), atol=0.02)


def test_resample_16k_mono_passthrough_and_empty():
    assert resample_to_16k(np.empty(0, dtype=np.float32), 48000, 2).size == 0
    src = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    out = resample_to_16k(src, src_rate=16000, channels=1)
    assert out.dtype == np.float32
    assert np.allclose(out, src.astype(np.float32))
    assert rms(np.zeros(64, dtype=np.float32)) == 0.0


def test_ring_buffer_write_read_wrap_overflow_and_clear():
    ring = AudioRingBuffer(max_seconds=1.0, sample_rate=10)
    assert ring.capacity == 10
    assert len(ring) == 0

    ring.write(np.arange(6, dtype=np.float32))
    assert len(ring) == 6
    first = ring.read_available()
    assert np.array_equal(first, np.arange(6, dtype=np.float32))
    assert len(ring) == 0
    assert ring.read_available().size == 0

    # Write across wrap boundary without overflow
    ring.write(np.arange(6, 14, dtype=np.float32))  # 8 samples, wraps around 10
    assert len(ring) == 8
    wrapped = ring.read_available()
    assert np.array_equal(wrapped, np.arange(6, 14, dtype=np.float32))

    # Overflow keeps newest `capacity` samples in chronological order
    ring.write(np.arange(0, 5, dtype=np.float32))
    ring.write(np.arange(5, 15, dtype=np.float32))
    assert len(ring) == 10
    overflowed = ring.read_available()
    assert np.array_equal(overflowed, np.arange(5, 15, dtype=np.float32))

    # Single oversized write keeps newest `capacity` samples
    ring.write(np.arange(100, 125, dtype=np.float32))
    assert np.array_equal(ring.read_available(), np.arange(115, 125, dtype=np.float32))

    ring.write(np.array([1.0, 2.0], dtype=np.float32))
    ring.clear()
    assert len(ring) == 0
    assert ring.read_available().size == 0
