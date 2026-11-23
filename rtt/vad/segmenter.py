"""Hysteresis + pre-roll speech segmenter for 16 kHz audio streams."""

from collections import deque
from dataclasses import dataclass
import time
from typing import Callable

import numpy as np


@dataclass
class SpeechSegment:
    """A finalized speech utterance chunk ready for STT transcription."""

    audio: np.ndarray
    start_ts: float
    end_ts: float
    forced: bool = False


class VADSegmenter:
    """Segments continuous 16 kHz audio into utterance chunks using hysteresis thresholds."""

    SAMPLE_RATE = 16000
    FRAME_SAMPLES = 512
    FRAME_SEC = FRAME_SAMPLES / SAMPLE_RATE

    def __init__(
        self,
        prob_fn: Callable[[np.ndarray], float],
        silence_ms: int = 420,
        max_chunk_sec: float = 7.5,
        start_thr: float = 0.50,
        end_thr: float = 0.35,
        min_speech_frames: int = 4,
        pre_roll_frames: int = 8,
        overlap_tail_frames: int = 6,
        rms_gate: float = 0.003,
        clock: Callable[[], float] = time.time,
        on_reset: Callable[[], None] | None = None,
    ):
        self.prob_fn = prob_fn
        self.start_thr = float(start_thr)
        self.end_thr = float(end_thr)
        self.min_speech_frames = max(1, int(min_speech_frames))
        self.pre_roll_frames = max(0, int(pre_roll_frames))
        self.overlap_tail_frames = max(0, int(overlap_tail_frames))
        self.rms_gate = float(rms_gate)
        self.clock = clock
        self.on_reset = on_reset

        self.silence_flush_frames = 1
        self.set_silence_ms(silence_ms)
        self.max_frames = max(1, int((max_chunk_sec * 1000.0) / 32.0))

        self.pre_roll: deque[np.ndarray] = deque(maxlen=self.pre_roll_frames)
        self.speech_frames: list[np.ndarray] = []
        self.is_speaking = False
        self.trigger_count = 0
        self.silence_count = 0
        self.utterance_start_ts = 0.0
        self._remainder = np.empty(0, dtype=np.float32)

    def set_silence_ms(self, silence_ms: int) -> None:
        self.silence_flush_frames = max(1, int(silence_ms / 32))

    def reset(self, *, reset_remainder: bool = True) -> None:
        self.pre_roll.clear()
        self.speech_frames.clear()
        self.is_speaking = False
        self.trigger_count = 0
        self.silence_count = 0
        self.utterance_start_ts = 0.0
        if reset_remainder:
            self._remainder = np.empty(0, dtype=np.float32)
        if self.on_reset is not None:
            self.on_reset()

    def process(self, samples_16k: np.ndarray) -> list[SpeechSegment]:
        arr = np.asarray(samples_16k, dtype=np.float32).ravel()
        if arr.size == 0:
            return []
        buf = arr if self._remainder.size == 0 else np.concatenate([self._remainder, arr])
        num_frames = buf.size // self.FRAME_SAMPLES
        self._remainder = buf[num_frames * self.FRAME_SAMPLES :]

        segments: list[SpeechSegment] = []
        for i in range(num_frames):
            frame = buf[i * self.FRAME_SAMPLES : (i + 1) * self.FRAME_SAMPLES]
            seg = self._step_frame(frame)
            if seg is not None:
                segments.append(seg)
        return segments

    def flush(self) -> SpeechSegment | None:
        """Emit any in-progress speech segment (e.g. when stopping capture)."""
        if not self.is_speaking or not self.speech_frames:
            self.reset()
            return None
        audio = np.concatenate(self.speech_frames).astype(np.float32, copy=False)
        start_ts = self.utterance_start_ts
        end_ts = self.clock()
        self.reset()
        return SpeechSegment(audio=audio, start_ts=start_ts, end_ts=end_ts, forced=False)

    def _step_frame(self, frame: np.ndarray) -> SpeechSegment | None:
        frame_rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        if frame_rms < self.rms_gate:
            prob = 0.0
        else:
            prob = float(self.prob_fn(frame))

        if not self.is_speaking:
            if self.pre_roll_frames > 0:
                self.pre_roll.append(frame)
            if prob >= self.start_thr:
                self.trigger_count += 1
                if self.trigger_count >= self.min_speech_frames:
                    self.is_speaking = True
                    self.utterance_start_ts = self.clock() - (len(self.pre_roll) * self.FRAME_SEC)
                    self.speech_frames = list(self.pre_roll)
                    self.pre_roll.clear()
                    self.silence_count = 0
                    self.trigger_count = 0
            else:
                self.trigger_count = 0
            return None

        self.speech_frames.append(frame)
        if prob >= self.end_thr:
            self.silence_count = 0
        else:
            self.silence_count += 1

        if self.silence_count >= self.silence_flush_frames:
            audio = np.concatenate(self.speech_frames).astype(np.float32, copy=False)
            start_ts = self.utterance_start_ts
            end_ts = self.clock()
            self.reset(reset_remainder=False)
            return SpeechSegment(audio=audio, start_ts=start_ts, end_ts=end_ts, forced=False)

        if len(self.speech_frames) >= self.max_frames:
            audio = np.concatenate(self.speech_frames).astype(np.float32, copy=False)
            start_ts = self.utterance_start_ts
            end_ts = self.clock()
            tail = self.speech_frames[-self.overlap_tail_frames :] if self.overlap_tail_frames > 0 else []
            self.speech_frames = list(tail)
            self.utterance_start_ts = end_ts - (len(tail) * self.FRAME_SEC)
            self.silence_count = 0
            return SpeechSegment(audio=audio, start_ts=start_ts, end_ts=end_ts, forced=True)

        return None
