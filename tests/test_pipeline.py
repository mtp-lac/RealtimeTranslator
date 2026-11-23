"""Tests for TranslationPipeline coalescing, timeout fallback, context window, and task_done balance."""

import asyncio

from rtt.translate.base import BaseTranslator, TranslationError
from rtt.translate.pipeline import TranslationPipeline, UtteranceTask, split_text_chunks
from rtt.translate.rate_limit import AsyncTokenBucket


class RecordingTranslator(BaseTranslator):
    def __init__(self, name_str="Primary", delay=0.0, fail=False, prefix="TR:"):
        self._name = name_str
        self.delay = delay
        self.fail = fail
        self.prefix = prefix
        self.calls: list[tuple[str, list[str] | None]] = []

    @property
    def name(self) -> str:
        return self._name

    async def translate(self, text: str, source_lang: str, target_lang: str, context: list[str] | None = None) -> str:
        self.calls.append((text, list(context) if context is not None else None))
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise TranslationError("boom")
        return f"{self.prefix}{text}"


def test_pipeline_coalescing_context_and_task_done():
    async def run():
        primary = RecordingTranslator("LLM")
        fallback = RecordingTranslator("Google", prefix="FB:")
        results = []

        pipe = TranslationPipeline(
            primary=primary,
            fallback=fallback,
            bucket=AsyncTokenBucket(rate_per_sec=100.0, burst=10),
            target_lang="en",
            on_result=lambda *args: results.append(args),
            coalesce_max_chars=140,
        )
        # Pre-fill queue with 3 short items so they coalesce into one batch
        for seq in (1, 2, 3):
            await pipe.queue.put(UtteranceTask(seq, 0.0, 1.0, "ja", f"line {seq}"))

        runner = asyncio.create_task(pipe.run())
        await asyncio.wait_for(pipe.queue.join(), timeout=2.0)

        # Send 5 sequential items one-by-one so context accumulates and caps at 4
        for seq in range(4, 9):
            await pipe.queue.put(UtteranceTask(seq, float(seq), float(seq + 1), "ja", f"turn {seq}"))
            await asyncio.wait_for(pipe.queue.join(), timeout=2.0)

        pipe.stop()
        runner.cancel()
        try:
            await runner
        except asyncio.CancelledError:
            pass

        # First batch coalesced 1,2,3
        assert results[0][0] == 1
        assert results[0][1] == [1, 2, 3]
        assert results[0][2] == "line 1 line 2 line 3"
        assert results[0][3] == "TR:line 1 line 2 line 3"
        assert results[0][6] == "TRANSLATED"

        # Last call received rolling context capped at 4 items
        last_ctx = primary.calls[-1][1]
        assert last_ctx is not None
        assert len(last_ctx) == 4
        assert last_ctx == ["turn 4", "turn 5", "turn 6", "turn 7"]

    asyncio.run(run())


def test_pipeline_timeout_fallback_and_both_fail_and_none():
    async def run():
        slow = RecordingTranslator("SlowLLM", delay=0.3)
        fb = RecordingTranslator("Google", prefix="FB:")
        results = []
        pipe = TranslationPipeline(
            primary=slow,
            fallback=fb,
            bucket=AsyncTokenBucket(rate_per_sec=100.0, burst=5),
            target_lang="en",
            on_result=lambda *args: results.append(args),
            timeout=0.05,
        )
        await pipe.queue.put(UtteranceTask(1, 0.0, 1.0, "ja", "hello"))
        runner = asyncio.create_task(pipe.run())
        await asyncio.wait_for(pipe.queue.join(), timeout=2.0)

        # Both fail -> FAILED
        pipe.primary = RecordingTranslator("FailLLM", fail=True)
        pipe.fallback = RecordingTranslator("FailFB", fail=True)
        await pipe.queue.put(UtteranceTask(2, 1.0, 2.0, "ja", "world"))
        await asyncio.wait_for(pipe.queue.join(), timeout=2.0)

        # Both None -> NOT_TRANSLATED
        pipe.primary = None
        pipe.fallback = None
        await pipe.queue.put(UtteranceTask(3, 2.0, 3.0, "ja", "raw"))
        await asyncio.wait_for(pipe.queue.join(), timeout=2.0)

        pipe.stop()
        runner.cancel()
        try:
            await runner
        except asyncio.CancelledError:
            pass

        assert results[0][3] == "FB:hello"
        assert results[0][6] == "FALLBACK_TRANSLATED"
        assert results[1][3] is None
        assert results[1][6] == "FAILED"
        assert results[2][3] is None
        assert results[2][6] == "NOT_TRANSLATED"

    asyncio.run(run())


def test_split_text_chunks_boundaries():
    assert split_text_chunks("", 50) == []
    assert split_text_chunks("short line", 50) == ["short line"]

    # Sentence punctuation boundary
    en_text = "First sentence. Second sentence is longer."
    assert split_text_chunks(en_text, 20) == [
        "First sentence.",
        "Second sentence is",
        "longer.",
    ]

    # Japanese/Chinese sentence punctuation
    ja_text = "おはようございます。今日はいい天気ですね。明日も晴れます。"
    ja_chunks = split_text_chunks(ja_text, 15)
    assert ja_chunks == [
        "おはようございます。",
        "今日はいい天気ですね。",
        "明日も晴れます。",
    ]

    # Hard split fallback when there are no spaces or punctuation
    assert split_text_chunks("abcdefghij", 4) == ["abcd", "efgh", "ij"]


def test_pipeline_splits_oversized_utterance_and_updates_limit():
    async def run():
        primary = RecordingTranslator("LLM")
        results = []
        pipe = TranslationPipeline(
            primary=primary,
            fallback=None,
            bucket=AsyncTokenBucket(rate_per_sec=100.0, burst=10),
            target_lang="en",
            on_result=lambda *args: results.append(args),
            coalesce_max_chars=25,
        )
        long_text = "Hello world. How are you doing today?"
        await pipe.queue.put(UtteranceTask(1, 0.0, 1.0, "en", long_text))

        runner = asyncio.create_task(pipe.run())
        await asyncio.wait_for(pipe.queue.join(), timeout=2.0)

        # Dynamically update limit and confirm shorter chunking
        pipe.set_coalesce_max_chars(20)
        await pipe.queue.put(UtteranceTask(2, 1.0, 2.0, "en", "Alpha beta gamma delta"))
        await asyncio.wait_for(pipe.queue.join(), timeout=2.0)

        pipe.stop()
        runner.cancel()
        try:
            await runner
        except asyncio.CancelledError:
            pass

        assert len(primary.calls) == 4
        assert all(len(call_text) <= 25 for call_text, _ in primary.calls[:2])
        assert all(len(call_text) <= 20 for call_text, _ in primary.calls[2:])
        assert results[0][3] == "TR:Hello world. TR:How are you doing today?"
        assert results[1][3] == "TR:Alpha beta gamma TR:delta"

    asyncio.run(run())
