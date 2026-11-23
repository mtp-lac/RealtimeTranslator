"""Real-time audio resampling and RMS energy utilities."""

import math
import numpy as np

TARGET_SR = 16000


def resample_to_16k(audio: np.ndarray, src_rate: int, channels: int) -> np.ndarray:
    """Convert interleaved float PCM audio to 16 kHz mono float32."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return np.empty(0, dtype=np.float32)
    if channels > 1:
        usable = (arr.size // channels) * channels
        if usable == 0:
            return np.empty(0, dtype=np.float32)
        arr = arr[:usable].reshape(-1, channels).mean(axis=1)
    arr = arr.astype(np.float32, copy=False)
    if src_rate == TARGET_SR:
        return np.ascontiguousarray(arr, dtype=np.float32)
    from scipy.signal import resample_poly

    gcd = math.gcd(src_rate, TARGET_SR)
    return resample_poly(arr, TARGET_SR // gcd, src_rate // gcd).astype(np.float32)


def rms(frame: np.ndarray) -> float:
    """Compute Root Mean Square energy of an audio buffer."""
    arr = np.asarray(frame)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr, dtype=np.float64))))
