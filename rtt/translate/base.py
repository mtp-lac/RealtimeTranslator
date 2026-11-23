"""Base translation interface and error type."""

from abc import ABC, abstractmethod


class TranslationError(RuntimeError):
    """Raised when a translation provider fails or returns an invalid payload."""


class BaseTranslator(ABC):
    """Abstract async translator interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider identifier for logs and UI status badges."""

    @abstractmethod
    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: list[str] | None = None,
    ) -> str:
        """Translate `text` from `source_lang` into `target_lang`."""

    async def aclose(self) -> None:
        """Release any open HTTP connections."""
        return None
