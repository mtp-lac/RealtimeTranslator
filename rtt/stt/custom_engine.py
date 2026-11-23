"""Extensible custom neural network STT adapter."""

import importlib
from typing import Callable

import numpy as np

from rtt.stt.base import BaseSTTEngine, STTResult


class CustomNeuralNetEngine(BaseSTTEngine):
    """Plugin STT adapter accepting a `package.module:factory` entrypoint."""

    def __init__(self):
        self._predict: Callable[[np.ndarray, str | None], str | STTResult | None] | None = None
        self._label = "CustomNeuralNet"

    @property
    def name(self) -> str:
        return self._label

    def load_model(self, config: dict) -> None:
        entrypoint = str(config.get("entrypoint") or "").strip()
        model_path = str(config.get("model_path") or "").strip()
        if not entrypoint or ":" not in entrypoint:
            raise RuntimeError(
                "Custom STT requires an entrypoint formatted as 'package.module:factory'."
            )
        mod_name, factory_name = entrypoint.split(":", 1)
        module = importlib.import_module(mod_name)
        factory = getattr(module, factory_name, None)
        if not callable(factory):
            raise RuntimeError(f"Custom STT factory '{entrypoint}' is not callable.")
        predictor = factory(model_path)
        if not callable(predictor):
            raise RuntimeError(f"Factory '{entrypoint}' did not return a callable predictor.")
        self._predict = predictor
        self._label = f"Custom({factory_name})"

    def transcribe(self, audio: np.ndarray, language: str | None) -> STTResult | None:
        if self._predict is None:
            raise RuntimeError("Custom STT engine is not loaded.")
        arr = np.asarray(audio, dtype=np.float32).ravel()
        lang = None if not language or language == "auto" else language
        out = self._predict(arr, lang)
        if out is None:
            return None
        if isinstance(out, STTResult):
            return out if out.text.strip() else None
        text = str(out).strip()
        if not text:
            return None
        return STTResult(text=text, language=lang or "unknown", language_prob=1.0)

    def unload(self) -> None:
        self._predict = None
