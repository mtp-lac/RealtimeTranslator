"""Modular STT engine layer with Whisper, Windows Native, and custom backends."""

from rtt.stt.base import BaseSTTEngine, STTResult
from rtt.stt.cuda_dlls import register_nvidia_dll_dirs
from rtt.stt.custom_engine import CustomNeuralNetEngine
from rtt.stt.orchestrator import STTOrchestrator
from rtt.stt.whisper_engine import WhisperEngine, filter_segments
from rtt.stt.windows_native import WindowsNativeSTTEngine

__all__ = [
    "BaseSTTEngine",
    "CustomNeuralNetEngine",
    "STTOrchestrator",
    "STTResult",
    "WhisperEngine",
    "WindowsNativeSTTEngine",
    "filter_segments",
    "register_nvidia_dll_dirs",
]
