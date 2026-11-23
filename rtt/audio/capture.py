"""WASAPI audio capture loop with 16 kHz resampling and silence padding."""

import queue
import threading
from typing import Callable

import numpy as np

from rtt.audio.devices import AudioDeviceInfo
from rtt.audio.resample import TARGET_SR, resample_to_16k, rms
from rtt.audio.ring_buffer import AudioRingBuffer


class AudioCapture:
    """Captures audio from a WASAPI mic or loopback endpoint into an AudioRingBuffer."""

    SILENCE_PADDING_SAMPLES = int(TARGET_SR * 0.1)  # 100 ms @ 16 kHz = 1600 samples

    def __init__(
        self,
        device: AudioDeviceInfo,
        ring: AudioRingBuffer,
        on_level: Callable[[float], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ):
        self.device = device
        self.ring = ring
        self.on_level = on_level
        self.on_error = on_error
        self.muted = False

        self._raw_queue: queue.Queue[bytes] = queue.Queue(maxsize=256)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pa = None
        self._stream = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        import pyaudiowpatch as pyaudio

        self._stop_event.clear()
        self._drain_queue()
        self._pa = pyaudio.PyAudio()
        channels = max(1, int(self.device.channels))
        rate = max(8000, int(self.device.sample_rate))
        frames_per_buffer = max(256, int(rate * 0.02))

        def callback(in_data, frame_count, time_info, status_flags):
            if in_data and not self._stop_event.is_set():
                try:
                    self._raw_queue.put_nowait(in_data)
                except queue.Full:
                    try:
                        self._raw_queue.get_nowait()
                        self._raw_queue.put_nowait(in_data)
                    except queue.Empty:
                        pass
            return (None, pyaudio.paContinue)

        try:
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=self.device.index,
                frames_per_buffer=frames_per_buffer,
                stream_callback=callback,
            )
            self._stream.start_stream()
        except Exception as exc:
            self._cleanup_audio()
            raise RuntimeError(f"Failed to open WASAPI audio device '{self.device.name}': {exc}") from exc

        self._thread = threading.Thread(target=self._process_loop, name="AudioCaptureLoop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._cleanup_audio()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._drain_queue()

    def _cleanup_audio(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                if stream.is_active():
                    stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        pa = self._pa
        self._pa = None
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass

    def _drain_queue(self) -> None:
        while True:
            try:
                self._raw_queue.get_nowait()
            except queue.Empty:
                break

    def _process_loop(self) -> None:
        channels = max(1, int(self.device.channels))
        rate = max(8000, int(self.device.sample_rate))
        silence_pad = np.zeros(self.SILENCE_PADDING_SAMPLES, dtype=np.float32)

        while not self._stop_event.is_set():
            try:
                raw = self._raw_queue.get(timeout=0.1)
            except queue.Empty:
                # WASAPI loopback stops delivering packets during complete silence;
                # pad zeros so downstream Silero VAD observes trailing silence.
                self.ring.write(silence_pad)
                if self.on_level is not None:
                    self.on_level(0.0)
                continue

            try:
                samples = np.frombuffer(raw, dtype=np.float32)
                if self.muted:
                    num_out = max(1, int(round((samples.size / channels) * (TARGET_SR / rate))))
                    self.ring.write(np.zeros(num_out, dtype=np.float32))
                    if self.on_level is not None:
                        self.on_level(0.0)
                else:
                    mono_16k = resample_to_16k(samples, rate, channels)
                    if mono_16k.size > 0:
                        self.ring.write(mono_16k)
                        if self.on_level is not None:
                            self.on_level(rms(mono_16k))
            except Exception as exc:
                if self.on_error is not None:
                    self.on_error(str(exc))
