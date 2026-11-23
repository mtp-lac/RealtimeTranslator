"""Bounded, coalescing, non-blocking async translation pipeline."""

import asyncio
import concurrent.futures
from dataclasses import dataclass
import time
from typing import Callable

from rtt.translate.base import BaseTranslator
from rtt.translate.rate_limit import AsyncTokenBucket


_SENTENCE_BREAKS_SPACE = (". ", "! ", "? ")
_SENTENCE_BREAKS_CHAR = ("。", "！", "？", "\n")
_CLAUSE_BREAKS_SPACE = (", ", "; ", ": ")
_CLAUSE_BREAKS_CHAR = ("、", "，", "；", "：")
_CJK_END_PUNCT = frozenset("。！？、，")


def split_text_chunks(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most `max_chars` characters on natural boundaries."""
    remaining = (text or "").strip()
    if not remaining:
        return []
    limit = int(max_chars)
    if limit <= 0 or len(remaining) <= limit:
        return [remaining]

    chunks: list[str] = []
    while len(remaining) > limit:
        split_idx = _find_split_index(remaining, limit)
        chunk = remaining[:split_idx].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_idx:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _best_punct_split(text: str, limit: int, space_tokens: tuple[str, ...], char_tokens: tuple[str, ...]) -> int:
    best = 0
    extended = text[: min(len(text), limit + 1)]
    for tok in space_tokens:
        idx = extended.rfind(tok)
        if idx > 0 and idx + 1 <= limit:
            best = max(best, idx + 1)
    window = text[:limit]
    for tok in char_tokens:
        idx = window.rfind(tok)
        if idx > 0:
            best = max(best, idx + 1)
    return best


def _find_split_index(text: str, limit: int) -> int:
    idx = _best_punct_split(text, limit, _SENTENCE_BREAKS_SPACE, _SENTENCE_BREAKS_CHAR)
    if idx > 0:
        return idx

    idx = _best_punct_split(text, limit, _CLAUSE_BREAKS_SPACE, _CLAUSE_BREAKS_CHAR)
    if idx > 0:
        return idx

    if limit < len(text) and text[limit] == " ":
        return limit
    space_idx = text[:limit].rfind(" ")
    if space_idx > 0:
        return space_idx

    return limit


def _join_translated_chunks(parts: list[str]) -> str:
    out = ""
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        if not out:
            out = piece
        elif out[-1] in _CJK_END_PUNCT:
            out += piece
        else:
            out += " " + piece
    return out


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

    def set_coalesce_max_chars(self, max_chars: int) -> None:
        self.coalesce_max_chars = max(20, int(max_chars))

    def stop(self) -> None:
        self._stopped = True

    async def _translate_with(
        self,
        translator: BaseTranslator,
        text: str,
        source_lang: str,
        context: list[str] | None,
    ) -> str:
        chunks = split_text_chunks(text, self.coalesce_max_chars)
        if not chunks:
            return ""
        if len(chunks) == 1:
            return await asyncio.wait_for(
                translator.translate(
                    chunks[0],
                    source_lang,
                    self.target_lang,
                    context=context,
                ),
                timeout=self.timeout,
            )

        translated_parts: list[str] = []
        for chunk in chunks:
            part = await asyncio.wait_for(
                translator.translate(
                    chunk,
                    source_lang,
                    self.target_lang,
                    context=context,
                ),
                timeout=self.timeout,
            )
            translated_parts.append(part)
        return _join_translated_chunks(translated_parts)

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
                        translated = await self._translate_with(
                            self.primary,
                            combined_src,
                            item.source_lang,
                            recent_context[-4:],
                        )
                        status = "TRANSLATED"
                    except Exception:
                        if self.fallback is not None:
                            provider_name = self.fallback.name
                            try:
                                translated = await self._translate_with(
                                    self.fallback,
                                    combined_src,
                                    item.source_lang,
                                    None,
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
                        translated = await self._translate_with(
                            self.fallback,
                            combined_src,
                            item.source_lang,
                            None,
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
