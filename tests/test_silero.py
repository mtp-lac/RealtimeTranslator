"""Optional integration test for real Silero VAD ONNX weights."""

import numpy as np
import pytest

from rtt import paths
from rtt.vad.silero import SileroVAD


def test_silero_onnx_silence_inference():
    model_path = paths.models_dir() / "silero_vad.onnx"
    if not model_path.is_file():
        pytest.skip("silero_vad.onnx not downloaded in test models dir")

    vad = SileroVAD(model_path=model_path)
    prob = vad(np.zeros(512, dtype=np.float32))
    assert 0.0 <= prob < 0.5
    vad.reset_states()
