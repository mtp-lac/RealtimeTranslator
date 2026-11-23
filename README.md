# RealtimeTranslator

Low-latency Windows real-time audio transcription and translation desktop app built with **PyAudioWPatch** (WASAPI Loopback & Microphone capture), **Silero VAD** (`onnxruntime`, no PyTorch required), **Faster-Whisper** (CTranslate2 CUDA/CPU with Windows Native Speech fallback), **Async Token-Bucket Translation**, **SQLite WAL + streaming logging**, and a **PySide6** control window + translucent floating subtitle overlay.

---

## Setup (Python 3.13 on Windows)

```powershell
cd D:\Code\RealtimeTranslator
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-cuda.txt -r requirements-winnative.txt -r requirements-dev.txt
```

- `requirements.txt` — Core runtime (`numpy`, `scipy`, `PyAudioWPatch`, `onnxruntime`, `faster-whisper`, `httpx`, `PySide6`, `keyring`).
- `requirements-cuda.txt` — NVIDIA CUDA 12 (`cublas` + `cudnn 9.*`) wheels for GPU acceleration.
- `requirements-winnative.txt` — WinRT `Windows.Media.SpeechRecognition` fallback engine.
- `requirements-dev.txt` — `pytest` and `pyinstaller`.

## Running the App & Smoke Checks

```powershell
# Launch the PySide6 GUI + Subtitle Overlay
.\.venv\Scripts\python.exe -m rtt

# Smoke test WASAPI device enumeration and 3s loopback capture
.\.venv\Scripts\python.exe scripts\smoke_devices.py

# Smoke test Silero VAD ONNX + Faster-Whisper (CPU & CUDA)
.\.venv\Scripts\python.exe scripts\smoke_stt.py

# Run test suite
.\.venv\Scripts\python.exe -m pytest
```

---

## Usage & Configuration

### Audio Source (WASAPI Loopback vs Microphone)
- **`[Loopback]` devices** capture desktop audio natively on Windows (games, YouTube, Discord, meetings, anime/streams) without virtual audio cables.
- **`[Mic]` devices** capture microphone input.
- Adjust the **Pause Split** slider (`250–800 ms`) live to tune how quickly Silero VAD finalizes an utterance after trailing silence.

### Translation Providers & API Keys
- **OpenAI / OpenRouter / Local Ollama (`llm`):**
  - **OpenAI:** Base URL `https://api.openai.com/v1`, Model `gpt-4o-mini`
  - **OpenRouter:** Base URL `https://openrouter.ai/api/v1`, Model e.g. `google/gemini-2.0-flash-001`
  - **Ollama (local, no API key needed):** Base URL `http://localhost:11434/v1`, Model e.g. `qwen2.5:7b` (leave API key blank; `Authorization` header is omitted automatically).
- **DeepL (`deepl`):** Supports Free (`:fx` key suffix -> `api-free.deepl.com`) and Pro (`api.deepl.com`).
- **Google Translate (`google`):** Keyless web endpoint fallback (`translate.googleapis.com`).
- **API Key Storage:** Keys are stored securely in **Windows Credential Manager** via `keyring` (`RealtimeTranslator` service) — never written to `config.json`.

### Global Hotkeys & Subtitle Overlay
- **`Ctrl+Shift+T`**: Toggle the floating always-on-top subtitle overlay.
- **`Ctrl+Shift+M`**: Toggle audio capture mute.
- **Click-Through Lock**: Makes the overlay mouse-transparent over games/videos. To unlock click-through mode, uncheck **Click-Through** in the Main Control Window or hide/toggle the overlay with `Ctrl+Shift+T`.

### Logging Modes & Session Exports
Sessions are stored under `%APPDATA%\RealtimeTranslator\logs\`:
- **`sessions.db`** — SQLite 3 database in WAL mode across all non-ephemeral sessions (powers **History Search**).
- **`YYYY-MM-DD_HH-MM-SS\`** — Per-session folder containing `transcript.jsonl`, `bilingual.txt`, `subtitles_bilingual.srt`, and `source_only.txt`.
- **Modes:**
  - **Full Bilingual**: Saves untranslated + translated text to SQLite and all flat files.
  - **Source Only**: Saves only raw STT transcripts (`sessions.db` + `source_only.txt`).
  - **Ephemeral**: Keeps entries in RAM only for the live session (writes nothing to disk; disables history search).

---

## Packaging (CPU vs CUDA Builds)

Separate **CPU** and **CUDA** Windows bundles are supported via `RealtimeTranslator.spec` and `build.ps1`:

```powershell
# Lean CPU-only distribution
.\build.ps1

# Full CUDA distribution (bundles nvidia-cublas-cu12 / nvidia-cudnn-cu12)
.\build.ps1 -Cuda
```

Output is created in `dist\RealtimeTranslator\`.

---

## Troubleshooting

- **CUDA DLL errors (`cublas64_12.dll` / `cudnn_ops64_9.dll` not found):** `rtt/stt/cuda_dlls.py` automatically registers `site-packages/nvidia/*/bin` before importing CTranslate2. If GPU inference still fails or runs out of VRAM, `STTOrchestrator` automatically retries Whisper on CPU (`int8`) and can fall back to Windows Native STT.
- **Silent desktop audio:** Windows WASAPI loopback pauses packet delivery during pure silence; `AudioCapture` automatically pads `100 ms` silence frames on queue timeouts so Silero VAD detects sentence endings cleanly.
- **Windows Native STT + Loopback warning:** `Windows.Media.SpeechRecognition` binds to the Windows default recording microphone rather than WASAPI loopback streams. The GUI displays a warning whenever `Windows Native STT` is combined with a `[Loopback]` endpoint.
