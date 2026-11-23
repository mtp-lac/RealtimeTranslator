"""Windows Native Speech Recognition fallback engine via winrt."""

import asyncio
import concurrent.futures
import threading
from typing import Callable

import numpy as np

from rtt.stt.base import BaseSTTEngine, STTResult


class WindowsNativeSTTEngine(BaseSTTEngine):
    """Self-capturing fallback STT backed by Windows.Media.SpeechRecognition."""

    self_capturing = True

    def __init__(self):
        self._language: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._recognizer = None
        self._token = None
        self._running = False

    @property
    def name(self) -> str:
        return "WindowsNativeSTT"

    @classmethod
    def available(cls) -> bool:
        try:
            import winrt.windows.globalization  # noqa: F401
            import winrt.windows.media.speechrecognition  # noqa: F401
            return True
        except Exception:
            return False

    def load_model(self, config: dict) -> None:
        if not self.available():
            raise RuntimeError(
                "Windows Native Speech Recognition (winrt) is not available in this environment."
            )
        lang = config.get("language")
        self._language = None if not lang or lang == "auto" else str(lang)

    def start(self, on_text: Callable[[STTResult], None]) -> None:
        if self._running:
            return
        if not self.available():
            raise RuntimeError("Windows Native Speech Recognition is unavailable.")

        ready_future: concurrent.futures.Future = concurrent.futures.Future()
        self._loop = asyncio.new_event_loop()

        async def _begin():
            from winrt.windows.globalization import Language
            from winrt.windows.media.speechrecognition import SpeechRecognizer

            if self._language:
                recognizer = SpeechRecognizer(Language(self._language))
            else:
                recognizer = SpeechRecognizer()

            await recognizer.compile_constraints_async()
            session = recognizer.continuous_recognition_session

            def _on_result(sender, args):
                text = (getattr(args.result, "text", "") or "").strip()
                if text:
                    lang = self._language or str(recognizer.current_language.language_tag)
                    on_text(STTResult(text=text, language=lang, language_prob=1.0))

            self._token = session.add_result_generated(_on_result)
            await session.start_async()
            self._recognizer = recognizer
            self._running = True

        def _thread_main():
            assert self._loop is not None
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(_begin())
                ready_future.set_result(True)
                self._loop.run_forever()
            except Exception as exc:
                if not ready_future.done():
                    ready_future.set_exception(exc)
            finally:
                self._loop.close()

        self._thread = threading.Thread(target=_thread_main, name="WindowsNativeSTTLoop", daemon=True)
        self._thread.start()
        try:
            ready_future.result(timeout=15.0)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        loop = self._loop
        recognizer = self._recognizer
        if loop is not None and loop.is_running():
            async def _shutdown():
                try:
                    if recognizer is not None:
                        session = recognizer.continuous_recognition_session
                        if self._token is not None:
                            try:
                                session.remove_result_generated(self._token)
                            except Exception:
                                pass
                        try:
                            await session.stop_async()
                        except Exception:
                            pass
                        try:
                            recognizer.close()
                        except Exception:
                            pass
                finally:
                    loop.stop()

            fut = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
            try:
                fut.result(timeout=5.0)
            except Exception:
                loop.call_soon_threadsafe(loop.stop)

        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._thread = None
        self._loop = None
        self._recognizer = None
        self._token = None
        self._running = False

    def transcribe(self, audio: np.ndarray, language: str | None) -> STTResult | None:
        raise NotImplementedError(
            "WindowsNativeSTTEngine captures directly from the default microphone; use start(on_text)."
        )

    def unload(self) -> None:
        self.stop()
