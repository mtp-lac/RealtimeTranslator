"""Bounded, coalescing, non-blocking async translation pipeline."""

import asyncio
import concurrent.futures
from dataclasses import dataclass
import time
from typing import Callable

from rtt.translate.base import BaseTranslator
from rtt.translate.rate_limit import AsyncTokenBucket


@dataclass
class UtteranceTask:
    """Single STT utterance submitted to the async translation queue."""

    seq_id: int
    start_ts: float
    end_ts: float
    source_lang: str
    source_text: str


OnResultCallback = Callable[
    [int, list[int], str, str | None, str, int, str],
    None,
]


class TranslationPipeline:
    """Consumes UtteranceTasks with token-bucket pacing, coalescing, context, and fallback."""

    def __init__(
        self,
        primary: BaseTranslator | None,
        fallback: BaseTranslator | None,
        bucket: AsyncTokenBucket,
        target_lang: str,
        on_result: OnResultCallback,
        timeout: float = 4.0,
        max_queue: int = 32,
        coalesce_max_chars: int = 140,
    ):
        self.primary = primary
        self.fallback = fallback
        self.bucket = bucket
        self.target_lang = target_lang
        self.on_result = on_result
        self.timeout = float(timeout)
        self.coalesce_max_chars = int(coalesce_max_chars)
        self.queue: asyncio.Queue[UtteranceTask] = asyncio.Queue(maxsize=max_queue)
        self._stopped = False

    def submit_threadsafe(
        self,
        task: UtteranceTask,
        loop: asyncio.AbstractEventLoop,
    ) -> concurrent.futures.Future | None:
        if self._stopped or loop.is_closed():
            return None
        return asyncio.run_coroutine_threadsafe(self.queue.put(task), loop)

    def stop(self) -> None:
        self._stopped = True

    async def run(self) -> None:
        recent_context: list[str] = []
        while not self._stopped:
            item = await self.queue.get()
            merged_items = [item]
            try:
                while (
                    not self.queue.empty()
                    and sum(len(u.source_text) for u in merged_items) < self.coalesce_max_chars
                ):
                    try:
                        nxt = self.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    merged_items.append(nxt)

                combined_src = " ".join(
                    u.source_text.strip() for u in merged_items if u.source_text.strip()
                ).strip()
                merged_ids = [u.seq_id for u in merged_items]
                if not combined_src:
                    continue

                if self.primary is None and self.fallback is None:
                    self.on_result(
                        item.seq_id,
                        merged_ids,
                        combined_src,
                        None,
                        "None",
                        0,
                        "NOT_TRANSLATED",
                    )
                    continue

                t0 = time.monotonic()
                translated: str | None = None
                provider_name = "None"
                status = "FAILED"

                if self.primary is not None:
                    await self.bucket.acquire()
                    provider_name = self.primary.name
                    try:
                        translated = await asyncio.wait_for(
                            self.primary.translate(
                                combined_src,
                                item.source_lang,
                                self.target_lang,
                                context=recent_context[-4:],
                            ),
                            timeout=self.timeout,
                        )
                        status = "TRANSLATED"
                    except Exception:
                        if self.fallback is not None:
                            provider_name = self.fallback.name
                            try:
                                translated = await asyncio.wait_for(
                                    self.fallback.translate(
                                        combined_src,
                                        item.source_lang,
                                        self.target_lang,
                                        context=None,
                                    ),
                                    timeout=self.timeout,
                                )
                                status = "FALLBACK_TRANSLATED"
                            except Exception:
                                translated = None
                                status = "FAILED"
                        else:
                            translated = None
                            status = "FAILED"
                elif self.fallback is not None:
                    provider_name = self.fallback.name
                    try:
                        translated = await asyncio.wait_for(
                            self.fallback.translate(
                                combined_src,
                                item.source_lang,
                                self.target_lang,
                                context=None,
                            ),
                            timeout=self.timeout,
                        )
                        status = "FALLBACK_TRANSLATED"
                    except Exception:
                        translated = None
                        status = "FAILED"

                latency_ms = int((time.monotonic() - t0) * 1000)
                recent_context.append(combined_src)
                if len(recent_context) > 8:
                    recent_context = recent_context[-4:]

                self.on_result(
                    item.seq_id,
                    merged_ids,
                    combined_src,
                    translated,
                    provider_name,
                    latency_ms,
                    status,
                )
            finally:
                for _ in merged_items:
                    self.queue.task_done()
