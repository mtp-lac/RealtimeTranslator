"""Isolated Qt worker threads for audio capture, Silero+STT, and async translation."""

import asyncio
import threading
import time

from PySide6.QtCore import QObject, QThread, Signal

from rtt import secrets
from rtt.audio.capture import AudioCapture
from rtt.audio.devices import AudioDeviceInfo
from rtt.audio.ring_buffer import AudioRingBuffer
from rtt.config import AppConfig
from rtt.logs.transcript_logger import DualTranscriptLogger
from rtt.stt.base import STTResult
from rtt.stt.orchestrator import STTOrchestrator
from rtt.translate.base import BaseTranslator
from rtt.translate.deepl import DeepLTranslator
from rtt.translate.google import GoogleTranslator
from rtt.translate.llm import LLMTranslator
from rtt.translate.pipeline import TranslationPipeline, UtteranceTask, split_text_chunks
from rtt.translate.rate_limit import AsyncTokenBucket
from rtt.vad.segmenter import SpeechSegment, VADSegmenter
from rtt.vad.silero import SileroVAD


class AudioCaptureThread(QObject):
    """Manages AudioCapture and emits live RMS levels to the Qt UI thread."""

    level = Signal(float)
    error = Signal(str)

    def __init__(self, device: AudioDeviceInfo, ring: AudioRingBuffer, parent: QObject | None = None):
        super().__init__(parent)
        self._capture = AudioCapture(
            device=device,
            ring=ring,
            on_level=lambda v: self.level.emit(float(v)),
            on_error=lambda msg: self.error.emit(str(msg)),
        )

    def start(self) -> None:
        self._capture.start()

    def stop(self) -> None:
        self._capture.stop()

    def set_muted(self, muted: bool) -> None:
        self._capture.muted = bool(muted)


class STTWorker(QThread):
    """Runs Silero VAD + STTOrchestrator on a dedicated QThread."""

    utterance_transcribed = Signal(object)
    status = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        config: AppConfig,
        ring: AudioRingBuffer,
        logger: DualTranscriptLogger,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.config = config
        self.ring = ring
        self.logger = logger
        self._stop_event = threading.Event()
        self._seq_id = 0
        self._seq_lock = threading.Lock()
        self._silence_ms = int(config.silence_ms)
        self._max_sentence_chars = max(20, int(config.max_sentence_chars))
        self._segmenter: VADSegmenter | None = None
        self._orchestrator: STTOrchestrator | None = None

    def set_silence_ms(self, silence_ms: int) -> None:
        self._silence_ms = int(silence_ms)
        if self._segmenter is not None:
            self._segmenter.set_silence_ms(self._silence_ms)

    def set_max_sentence_chars(self, max_chars: int) -> None:
        self._max_sentence_chars = max(20, int(max_chars))

    def update_whisper_config(self, config: AppConfig) -> None:
        orch = self._orchestrator
        if orch is None:
            return
        orch.config = config
        engine = orch.active_engine
        updater = getattr(engine, "apply_runtime_config", None)
        if callable(updater):
            updater(orch._whisper_config())

    def stop(self) -> None:
        self._stop_event.set()

    def _next_seq_id(self) -> int:
        with self._seq_lock:
            self._seq_id += 1
            return self._seq_id

    def _emit_utterance(
        self,
        res: STTResult,
        start_ts: float,
        end_ts: float,
        engine_name: str,
        latency_ms: int,
    ) -> None:
        chunks = split_text_chunks(res.text, self._max_sentence_chars)
        for idx, chunk in enumerate(chunks):
            seq_id = self._next_seq_id()
            task = UtteranceTask(
                seq_id=seq_id,
                start_ts=start_ts,
                end_ts=end_ts,
                source_lang=res.language,
                source_text=chunk,
            )
            self.logger.log_source(
                seq_id=seq_id,
                start_ts=start_ts,
                end_ts=end_ts,
                stt_engine=engine_name,
                source_lang=res.language,
                source_text=chunk,
                stt_latency_ms=latency_ms if idx == 0 else 0,
            )
            self.utterance_transcribed.emit(task)

    def _handle_segment(self, seg: SpeechSegment, orchestrator: STTOrchestrator) -> None:
        res, latency_ms = orchestrator.transcribe(seg, self.config.source_lang)
        if res is None or not res.text.strip():
            return
        engine_name = orchestrator.active_engine.name if orchestrator.active_engine else "STT"
        self._emit_utterance(res, seg.start_ts, seg.end_ts, engine_name, latency_ms)

    def run(self) -> None:
        self._stop_event.clear()
        orchestrator = STTOrchestrator(self.config, on_status=lambda m: self.status.emit(m))
        self._orchestrator = orchestrator
        try:
            self.status.emit("Loading STT model...")
            engine = orchestrator.load()

            if getattr(engine, "self_capturing", False):
                def _on_native(res: STTResult) -> None:
                    now = time.time()
                    self._emit_utterance(res, now, now, engine.name, 0)

                engine.start(_on_native)
                self.status.emit(f"Listening ({engine.name})")
                while not self._stop_event.wait(0.05):
                    pass
                return

            self.status.emit("Loading Silero VAD...")
            vad = SileroVAD()
            segmenter = VADSegmenter(
                prob_fn=vad,
                silence_ms=self._silence_ms,
                on_reset=vad.reset_states,
            )
            self._segmenter = segmenter
            self.status.emit(f"Listening ({engine.name})")

            while not self._stop_event.is_set():
                samples = self.ring.read_available()
                if samples.size > 0:
                    segmenter.set_silence_ms(self._silence_ms)
                    for seg in segmenter.process(samples):
                        if self._stop_event.is_set():
                            break
                        self._handle_segment(seg, orchestrator)
                else:
                    self._stop_event.wait(0.03)

            tail = segmenter.flush()
            if tail is not None:
                self._handle_segment(tail, orchestrator)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self._segmenter = None
            self._orchestrator = None
            try:
                orchestrator.unload()
            except Exception:
                pass
            self.status.emit("STT stopped")


def build_translator(provider: str, config: AppConfig) -> BaseTranslator | None:
    """Instantiate a translation backend for the given provider identifier."""
    kind = (provider or "none").lower()
    if kind == "none":
        return None
    if kind == "llm":
        return LLMTranslator(
            base_url=config.llm_base_url,
            api_key=secrets.get_api_key("llm"),
            model=config.llm_model,
        )
    if kind == "google":
        return GoogleTranslator()
    if kind == "deepl":
        return DeepLTranslator(api_key=secrets.get_api_key("deepl"))
    raise ValueError(f"Unknown translator provider: {provider}")


class TranslationWorker(QThread):
    """Runs the asyncio TranslationPipeline and token bucket on its own QThread."""

    utterance_translated = Signal(int, object, str, object, str, int, str)
    status = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        config: AppConfig,
        logger: DualTranscriptLogger,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.config = config
        self.logger = logger
        self._loop: asyncio.AbstractEventLoop | None = None
        self._bucket: AsyncTokenBucket | None = None
        self._pipeline: TranslationPipeline | None = None
        self._run_task: asyncio.Task | None = None
        self._primary: BaseTranslator | None = None
        self._fallback: BaseTranslator | None = None
        self._ready = threading.Event()
        self._stop_requested = False

    def submit(self, task: UtteranceTask) -> None:
        self._ready.wait(timeout=2.0)
        if self._pipeline is not None and self._loop is not None and not self._loop.is_closed():
            self._pipeline.submit_threadsafe(task, self._loop)

    def update_rate(self, req_per_min: int) -> None:
        rate_per_sec = max(1, int(req_per_min)) / 60.0
        if self._bucket is not None and self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._bucket.set_rate, rate_per_sec)

    def update_max_sentence_chars(self, max_chars: int) -> None:
        if self._pipeline is not None and self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._pipeline.set_coalesce_max_chars, max_chars)

    def stop(self) -> None:
        self._stop_requested = True
        loop = self._loop
        if self._pipeline is not None:
            self._pipeline.stop()
        if loop is not None and loop.is_running():
            def _cancel():
                if self._run_task is not None and not self._run_task.done():
                    self._run_task.cancel()

            loop.call_soon_threadsafe(_cancel)

    def _on_result(
        self,
        primary_seq_id: int,
        merged_ids: list[int],
        combined_src: str,
        translated: str | None,
        provider_name: str,
        latency_ms: int,
        status: str,
    ) -> None:
        self.logger.log_translation(
            primary_seq_id=primary_seq_id,
            merged_seq_ids=merged_ids,
            combined_source_text=combined_src,
            translated_text=translated,
            provider=provider_name,
            translation_latency_ms=latency_ms,
            status=status,
        )
        self.utterance_translated.emit(
            primary_seq_id,
            merged_ids,
            combined_src,
            translated,
            provider_name,
            latency_ms,
            status,
        )

    def run(self) -> None:
        self._stop_requested = False
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)

        async def _main():
            self._primary = build_translator(self.config.primary_translator, self.config)
            fb_name = self.config.fallback_translator
            if fb_name == self.config.primary_translator:
                self._fallback = None
            else:
                self._fallback = build_translator(fb_name, self.config)

            rate_per_sec = max(1, int(self.config.max_requests_per_min)) / 60.0
            self._bucket = AsyncTokenBucket(rate_per_sec=rate_per_sec, burst=2)
            self._pipeline = TranslationPipeline(
                primary=self._primary,
                fallback=self._fallback,
                bucket=self._bucket,
                target_lang=self.config.target_lang,
                on_result=self._on_result,
                coalesce_max_chars=max(20, int(self.config.max_sentence_chars)),
            )
            self._run_task = asyncio.create_task(self._pipeline.run())
            self._ready.set()
            if self._stop_requested:
                self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass
            finally:
                for tr in (self._primary, self._fallback):
                    if tr is not None:
                        try:
                            await tr.aclose()
                        except Exception:
                            pass

        try:
            loop.run_until_complete(_main())
        except Exception as exc:
            self._ready.set()
            self.error.emit(str(exc))
        finally:
            self._ready.set()
            loop.close()
            self._loop = None
            self._pipeline = None
            self._bucket = None
            self.status.emit("Translation stopped")
