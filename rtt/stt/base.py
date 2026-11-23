"""Abstract interface and result model for Speech-to-Text backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class STTResult:
    """Normalized transcription output from any STT backend."""

    text: str
    language: str
    language_prob: float = 1.0


class BaseSTTEngine(ABC):
    """Abstract STT interface enabling runtime backend swapping and fallbacks."""

    self_capturing: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine identifier for logs and UI status badges."""

    @abstractmethod
    def load_model(self, config: dict) -> None:
        """Initialize the STT backend with engine-specific parameters."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, language: str | None) -> STTResult | None:
        """Transcribe a 16 kHz mono float32 chunk; return None if no speech found."""

    @abstractmethod
    def unload(self) -> None:
        """Release model weights and device memory."""
