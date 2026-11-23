"""Tests for STTOrchestrator CUDA->CPU retry and Windows Native fallback."""

import numpy as np

from rtt.config import AppConfig
from rtt.stt.base import BaseSTTEngine, STTResult
from rtt.stt.orchestrator import STTOrchestrator


def test_orchestrator_cuda_load_error_retries_cpu_then_native():
    statuses: list[str] = []

    class FakeWhisper(BaseSTTEngine):
        def __init__(self):
            self.device_used = "cpu"

        @property
        def name(self) -> str:
            return f"WhisperEngine(small/{self.device_used})"

        def load_model(self, config: dict) -> None:
            if config["device"] == "cuda":
                raise RuntimeError("CUDA driver version is insufficient")
            self.device_used = "cpu"

        def transcribe(self, audio: np.ndarray, language: str | None) -> STTResult | None:
            return STTResult(text="cpu ok", language="en")

        def unload(self) -> None:
            pass

    class FakeNative(BaseSTTEngine):
        self_capturing = True

        @classmethod
        def available(cls) -> bool:
            return True

        @property
        def name(self) -> str:
            return "WindowsNativeSTT"

        def load_model(self, config: dict) -> None:
            pass

        def transcribe(self, audio: np.ndarray, language: str | None) -> STTResult | None:
            raise NotImplementedError

        def unload(self) -> None:
            pass

    cfg = AppConfig(stt_engine="whisper", compute_device="cuda")
    orch = STTOrchestrator(cfg, on_status=statuses.append, whisper_cls=FakeWhisper, native_cls=FakeNative)
    engine = orch.load()
    assert engine.device_used == "cpu"
    assert any("retrying on CPU" in s for s in statuses)

    res, latency = orch.transcribe(np.zeros(1600, dtype=np.float32), "auto")
    assert res is not None and res.text == "cpu ok"
    assert latency >= 0


def test_orchestrator_runtime_cuda_error_falls_back_to_cpu_and_native():
    statuses: list[str] = []

    class FailingRuntimeWhisper(BaseSTTEngine):
        attempts = 0

        def __init__(self):
            self.device_used = "cuda"

        @property
        def name(self) -> str:
            return f"WhisperEngine(small/{self.device_used})"

        def load_model(self, config: dict) -> None:
            self.device_used = config["device"]

        def transcribe(self, audio: np.ndarray, language: str | None) -> STTResult | None:
            FailingRuntimeWhisper.attempts += 1
            if self.device_used == "cuda":
                raise RuntimeError("cuBLAS error during decode")
            raise RuntimeError("CUDA out of memory on CPU retry too")

        def unload(self) -> None:
            pass

    class FakeNative(BaseSTTEngine):
        self_capturing = True

        @classmethod
        def available(cls) -> bool:
            return True

        @property
        def name(self) -> str:
            return "WindowsNativeSTT"

        def load_model(self, config: dict) -> None:
            pass

        def transcribe(self, audio: np.ndarray, language: str | None) -> STTResult | None:
            raise NotImplementedError

        def unload(self) -> None:
            pass

    cfg = AppConfig(stt_engine="whisper", compute_device="cuda")
    orch = STTOrchestrator(cfg, on_status=statuses.append, whisper_cls=FailingRuntimeWhisper, native_cls=FakeNative)
    orch.load()
    res, _ = orch.transcribe(np.zeros(1600, dtype=np.float32), "auto")
    assert res is None
    assert orch.active_engine.name == "WindowsNativeSTT"
    assert any("Windows Native STT" in s for s in statuses)
