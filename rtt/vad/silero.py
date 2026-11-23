"""Silero VAD v5 inference wrapper backed by onnxruntime (CPU, no PyTorch)."""

from pathlib import Path
import numpy as np

from rtt import paths

SILERO_ONNX_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
)


class SileroVAD:
    """Runs Silero VAD v5 on 512-sample 16 kHz float32 frames using onnxruntime."""

    SAMPLE_RATE = 16000
    FRAME_SAMPLES = 512
    CONTEXT_SAMPLES = 64

    def __init__(self, model_path: Path | str | None = None):
        self.model_path = Path(model_path) if model_path is not None else (paths.models_dir() / "silero_vad.onnx")
        self._ensure_model_downloaded()

        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        inputs = {inp.name: inp for inp in self._session.get_inputs()}
        outputs = self._session.get_outputs()
        required = {"input", "state", "sr"}
        if not required.issubset(inputs.keys()) or len(outputs) < 2:
            raise RuntimeError(
                f"Unexpected Silero ONNX interface: inputs={list(inputs.keys())}, outputs={[o.name for o in outputs]}"
            )

        sr_shape = inputs["sr"].shape
        if sr_shape and len(sr_shape) == 1:
            self._sr = np.array([self.SAMPLE_RATE], dtype=np.int64)
        else:
            self._sr = np.array(self.SAMPLE_RATE, dtype=np.int64)

        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT_SAMPLES), dtype=np.float32)

    def _ensure_model_downloaded(self) -> None:
        if self.model_path.is_file() and self.model_path.stat().st_size > 0:
            return
        import httpx

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.model_path.with_suffix(".onnx.part")
        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            resp = client.get(SILERO_ONNX_URL)
            resp.raise_for_status()
            tmp_path.write_bytes(resp.content)
        tmp_path.replace(self.model_path)

    def reset_states(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT_SAMPLES), dtype=np.float32)

    def __call__(self, frame512: np.ndarray) -> float:
        frame = np.asarray(frame512, dtype=np.float32).reshape(1, self.FRAME_SAMPLES)
        x = np.concatenate([self._context, frame], axis=1)
        out, new_state = self._session.run(
            None,
            {"input": x, "state": self._state, "sr": self._sr},
        )
        self._state = np.asarray(new_state, dtype=np.float32)
        self._context = frame[:, -self.CONTEXT_SAMPLES :]
        return float(np.asarray(out).squeeze())
