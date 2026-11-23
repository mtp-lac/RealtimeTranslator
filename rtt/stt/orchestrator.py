"""STT Orchestrator managing engine loading, CUDA->CPU retry, and Windows Native fallback."""

import time
from typing import Callable

import numpy as np

from rtt.config import AppConfig
from rtt.stt.base import BaseSTTEngine, STTResult
from rtt.stt.custom_engine import CustomNeuralNetEngine
from rtt.stt.whisper_engine import WhisperEngine
from rtt.stt.windows_native import WindowsNativeSTTEngine


def is_cuda_or_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(tok in msg for tok in ("cuda", "cublas", "cudnn", "out of memory"))


class STTOrchestrator:
    """Coordinates active STT backend selection and automatic runtime fallback."""

    def __init__(
        self,
        config: AppConfig,
        on_status: Callable[[str], None] | None = None,
        whisper_cls: type[BaseSTTEngine] = WhisperEngine,
        native_cls: type[WindowsNativeSTTEngine] = WindowsNativeSTTEngine,
        custom_cls: type[CustomNeuralNetEngine] = CustomNeuralNetEngine,
    ):
        self.config = config
        self.on_status = on_status
        self._whisper_cls = whisper_cls
        self._native_cls = native_cls
        self._custom_cls = custom_cls
        self.active_engine: BaseSTTEngine | None = None

    def _emit(self, message: str) -> None:
        if self.on_status is not None:
            self.on_status(message)

    def _whisper_config(
        self,
        *,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> dict:
        return {
            "model_size": self.config.whisper_model,
            "device": device or self.config.compute_device,
            "compute_type": compute_type or self.config.whisper_compute_type,
            "task": self.config.whisper_task,
            "beam_size": self.config.whisper_beam_size,
            "best_of": self.config.whisper_best_of,
            "temperature": self.config.whisper_temperature,
            "vad_filter": self.config.whisper_vad_filter,
            "condition_on_previous_text": self.config.whisper_condition_on_previous_text,
            "initial_prompt": self.config.whisper_initial_prompt,
            "no_speech_threshold": self.config.whisper_no_speech_threshold,
            "compression_ratio_threshold": self.config.whisper_compression_ratio_threshold,
        }

    def load(self) -> BaseSTTEngine:
        choice = self.config.stt_engine
        if choice == "windows_native":
            engine = self._native_cls()
            engine.load_model({"language": self.config.source_lang})
            self.active_engine = engine
            self._emit(f"Loaded {engine.name}")
            return engine

        if choice == "custom":
            engine = self._custom_cls()
            engine.load_model(
                {
                    "model_path": self.config.custom_model_path,
                    "entrypoint": self.config.custom_entrypoint,
                }
            )
            self.active_engine = engine
            self._emit(f"Loaded {engine.name}")
            return engine

        # Default: Whisper with CUDA -> CPU -> Windows Native fallback
        engine = self._whisper_cls()
        try:
            engine.load_model(self._whisper_config())
            if (
                self.config.compute_device == "cuda"
                and getattr(engine, "device_used", "cpu") != "cuda"
            ):
                reason = getattr(engine, "fallback_reason", None) or "CUDA unavailable"
                self._emit(f"Whisper CUDA failed ({reason}); fell back to CPU int8.")
            else:
                self._emit(f"Loaded {engine.name}")
            self.active_engine = engine
            return engine
        except Exception as exc:
            if self.config.compute_device == "cuda" and is_cuda_or_oom_error(exc):
                self._emit(f"Whisper CUDA load failed ({exc}); retrying on CPU int8...")
                try:
                    cpu_engine = self._whisper_cls()
                    cpu_engine.load_model(self._whisper_config(device="cpu", compute_type="int8"))
                    self.active_engine = cpu_engine
                    self._emit(f"Loaded {cpu_engine.name} (CPU fallback)")
                    return cpu_engine
                except Exception as cpu_exc:
                    exc = cpu_exc

            if self._native_cls.available():
                self._emit(f"Whisper load failed ({exc}); falling back to Windows Native STT.")
                native = self._native_cls()
                native.load_model({"language": self.config.source_lang})
                self.active_engine = native
                return native
            raise

    def transcribe(self, segment_or_audio, language: str | None) -> tuple[STTResult | None, int]:
        if self.active_engine is None:
            raise RuntimeError("No STT engine is loaded.")
        audio = getattr(segment_or_audio, "audio", segment_or_audio)
        t0 = time.perf_counter()

        try:
            res = self.active_engine.transcribe(audio, language)
            return res, int((time.perf_counter() - t0) * 1000)
        except Exception as exc:
            if not is_cuda_or_oom_error(exc):
                raise

            self._emit(f"STT error ({exc}); retrying Whisper on CPU int8...")
            self.active_engine.unload()
            try:
                cpu_engine = self._whisper_cls()
                cpu_engine.load_model(self._whisper_config(device="cpu", compute_type="int8"))
                self.active_engine = cpu_engine
                res = cpu_engine.transcribe(audio, language)
                return res, int((time.perf_counter() - t0) * 1000)
            except Exception as cpu_exc:
                if self._native_cls.available():
                    self._emit(
                        f"CPU Whisper fallback failed ({cpu_exc}); falling back to Windows Native STT."
                    )
                    native = self._native_cls()
                    native.load_model({"language": self.config.source_lang})
                    self.active_engine = native
                    return None, int((time.perf_counter() - t0) * 1000)
                raise

    def unload(self) -> None:
        if self.active_engine is not None:
            self.active_engine.unload()
            self.active_engine = None
