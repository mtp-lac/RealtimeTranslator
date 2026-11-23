"""Thread-safe circular NumPy float32 audio ring buffer."""

import threading
import numpy as np

from rtt.audio.resample import TARGET_SR


class AudioRingBuffer:
    """Fixed-capacity circular buffer holding 16 kHz mono float32 PCM samples."""

    def __init__(self, max_seconds: float = 30.0, sample_rate: int = TARGET_SR):
        self.capacity = max(1, int(max_seconds * sample_rate))
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._write_pos = 0
        self._read_pos = 0
        self._size = 0
        self._lock = threading.Lock()

    def write(self, samples: np.ndarray) -> None:
        arr = np.asarray(samples, dtype=np.float32).ravel()
        n = arr.size
        if n == 0:
            return
        with self._lock:
            if n >= self.capacity:
                self._buf[:] = arr[-self.capacity :]
                self._write_pos = 0
                self._read_pos = 0
                self._size = self.capacity
                return

            first = min(n, self.capacity - self._write_pos)
            self._buf[self._write_pos : self._write_pos + first] = arr[:first]
            rem = n - first
            if rem > 0:
                self._buf[:rem] = arr[first:]
            self._write_pos = (self._write_pos + n) % self.capacity

            overflow = (self._size + n) - self.capacity
            if overflow > 0:
                self._read_pos = (self._read_pos + overflow) % self.capacity
                self._size = self.capacity
            else:
                self._size += n

    def read_available(self) -> np.ndarray:
        with self._lock:
            if self._size == 0:
                return np.empty(0, dtype=np.float32)
            n = self._size
            first = min(n, self.capacity - self._read_pos)
            if first == n:
                out = self._buf[self._read_pos : self._read_pos + n].copy()
            else:
                out = np.concatenate(
                    [self._buf[self._read_pos :], self._buf[: n - first]]
                ).astype(np.float32, copy=False)
            self._read_pos = self._write_pos
            self._size = 0
            return out

    def clear(self) -> None:
        with self._lock:
            self._read_pos = 0
            self._write_pos = 0
            self._size = 0

    def __len__(self) -> int:
        with self._lock:
            return self._size
