"""Smoke test: download Silero ONNX + Whisper tiny, test VAD on silence and Whisper on CPU & CUDA."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rtt.stt.whisper_engine import WhisperEngine
from rtt.vad.segmenter import VADSegmenter
from rtt.vad.silero import SileroVAD


def main() -> int:
    print("1) Loading Silero VAD ONNX...")
    vad = SileroVAD()
    segmenter = VADSegmenter(prob_fn=vad, on_reset=vad.reset_states)
    zeros_2s = np.zeros(32000, dtype=np.float32)
    segments = segmenter.process(zeros_2s)
    print(f"   Silero VAD on 2s silence -> {len(segments)} segments (expected 0)")
    assert len(segments) == 0

    t = np.linspace(0.0, 1.0, 16000, endpoint=False, dtype=np.float32)
    tone_1s = 0.25 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    print("2) Loading Whisper tiny on CPU (int8)...")
    cpu_eng = WhisperEngine()
    cpu_eng.load_model({"model_size": "tiny", "device": "cpu", "compute_type": "int8"})
    cpu_res = cpu_eng.transcribe(tone_1s, language="en")
    print(f"   CPU engine={cpu_eng.name}, synthetic tone result={cpu_res}")
    cpu_eng.unload()

    print("3) Trying Whisper tiny on CUDA (float16)...")
    cuda_eng = WhisperEngine()
    cuda_eng.load_model({"model_size": "tiny", "device": "cuda", "compute_type": "float16"})
    cuda_res = cuda_eng.transcribe(tone_1s, language="en")
    print(f"   Loaded engine={cuda_eng.name} (device_used={cuda_eng.device_used}, fallback={cuda_eng.fallback_reason})")
    print(f"   CUDA/fallback synthetic tone result={cuda_res}")
    cuda_eng.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
