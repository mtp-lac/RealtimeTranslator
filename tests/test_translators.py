"""Tests for LLM, DeepL, and Google translators via httpx.MockTransport."""

import asyncio
import json
import httpx
import pytest

from urllib.parse import parse_qs

from rtt.translate.base import TranslationError
from rtt.translate.deepl import DeepLTranslator
from rtt.translate.google import (
    GoogleTranslator,
    parse_batchexecute_response,
    parse_dict_chrome_response,
    parse_google_response,
)
from rtt.translate.llm import LLMTranslator


def test_llm_translator_payload_auth_and_errors():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), dict(request.headers), json.loads(request.content)))
        if len(seen) == 3:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Translated line"}}]},
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            # With API key + rolling context
            tr = LLMTranslator("https://api.example.com/v1", "sk-secret", "gpt-4o-mini", client=client)
            out = await tr.translate("Bonjour", "fr", "en", context=["c1", "c2", "c3", "c4", "c5"])
            assert out == "Translated line"

            # Without API key (e.g., local Ollama): Authorization header must be omitted
            ollama = LLMTranslator("http://localhost:11434/v1/", "", "qwen2.5", client=client)
            await ollama.translate("Salut", "auto", "en")

            # HTTP 429 raises TranslationError
            with pytest.raises(TranslationError):
                await tr.translate("Fail", "auto", "en")

    asyncio.run(run())

    url1, hdr1, body1 = seen[0]
    assert url1 == "https://api.example.com/v1/chat/completions"
    assert hdr1["authorization"] == "Bearer sk-secret"
    sys_msg = body1["messages"][0]["content"]
    assert "c1" not in sys_msg
    assert "c2" in sys_msg and "c5" in sys_msg
    assert body1["messages"][1]["content"] == "Bonjour"

    url2, hdr2, _ = seen[1]
    assert url2 == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in hdr2


def test_deepl_host_selection_and_target_mapping():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), dict(request.headers), json.loads(request.content)))
        return httpx.Response(200, json={"translations": [{"text": "Hello"}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            free = DeepLTranslator("abc123:fx", client=client)
            assert await free.translate("こんにちは", "ja", "en") == "Hello"
            pro = DeepLTranslator("abc123-pro", client=client)
            assert await pro.translate("こんにちは", "auto", "vi") == "Hello"

    asyncio.run(run())
    assert seen[0][0] == "https://api-free.deepl.com/v2/translate"
    assert seen[0][1]["authorization"] == "DeepL-Auth-Key abc123:fx"
    assert seen[0][2]["target_lang"] == "EN-US"
    assert seen[0][2]["source_lang"] == "JA"

    assert seen[1][0] == "https://api.deepl.com/v2/translate"
    assert "source_lang" not in seen[1][2]
    assert seen[1][2]["target_lang"] == "VI"


def _make_batchexecute_body(segments: list[list]) -> str:
    inner = [None, [[[None, None, None, None, None, segments]]]]
    outer = [["wrb.fr", "MkEWBc", json.dumps(inner)]]
    return ")]}'\n\n" + json.dumps(outer)


def test_google_translator_and_parser():
    assert parse_google_response([[["Hello ", "こんにちは"], ["world", "世界"]]]) == "Hello world"
    with pytest.raises(TranslationError):
        parse_google_response([])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/translate_a/single"
        assert request.url.params["client"] == "at"
        assert request.url.params["sl"] == "auto"
        assert request.url.params["tl"] == "en"
        form = parse_qs(request.content.decode("utf-8"))
        assert form["q"] == ["おはようございます"]
        return httpx.Response(200, json=[[["Good morning", "おはようございます", None, None, 10]]])

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            g = GoogleTranslator(client=client)
            assert await g.translate("おはようございます", "auto", "en") == "Good morning"

    asyncio.run(run())


def test_google_batchexecute_and_dict_chrome_parsers():
    en_raw = _make_batchexecute_body([
        ["Good morning.", None, None, None, None, None, "おはようございます。", 1],
        ["How are you?", None, True, None, None, None, "お元気ですか。", 1],
    ])
    assert parse_batchexecute_response(en_raw) == "Good morning. How are you?"

    zh_raw = _make_batchexecute_body([
        ["你好世界。", None, None, None, None, None, "Hello world.", 1],
        ["你好吗？", None, None, None, None, None, "How are you?", 1],
    ])
    assert parse_batchexecute_response(zh_raw) == "你好世界。你好吗？"

    for bad in ("", "not json", ")]}'\n[[1,2,3]]", ")]}'\n" + json.dumps([["wrb.fr", "MkEWBc", "{}"]])):
        with pytest.raises(TranslationError):
            parse_batchexecute_response(bad)

    assert parse_dict_chrome_response(["Hello world"]) == "Hello world"
    assert parse_dict_chrome_response([["Hello ", "ja"], ["world", "ja"]]) == "Hello world"
    for bad_dict in ([], {}, [["", "ja"]]):
        with pytest.raises(TranslationError):
            parse_dict_chrome_response(bad_dict)


def test_google_fallback_cooldown_and_cache():
    calls: list[str] = []

    def handler_batch_fallback(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/translate_a/single":
            return httpx.Response(429, text="rate limited")
        if request.url.path == "/_/TranslateWebserverUi/data/batchexecute":
            body = _make_batchexecute_body([["Hello", None, None]])
            return httpx.Response(200, text=body)
        raise AssertionError(f"Unexpected path: {request.url.path}")

    async def run_batch():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler_batch_fallback)) as client:
            g = GoogleTranslator(client=client, cooldown_seconds=60.0)
            assert await g.translate("こんにちは", "ja", "en") == "Hello"
            # Cached hit should make zero extra requests
            assert await g.translate("こんにちは", "ja", "en") == "Hello"
            # New text should skip cooled-down single_at and go straight to batchexecute
            assert await g.translate("こんばんは", "ja", "en") == "Hello"

    asyncio.run(run_batch())
    assert calls == [
        "/translate_a/single",
        "/_/TranslateWebserverUi/data/batchexecute",
        "/_/TranslateWebserverUi/data/batchexecute",
    ]

    calls2: list[str] = []

    def handler_dict_fallback(request: httpx.Request) -> httpx.Response:
        calls2.append(request.url.path)
        if request.url.path in ("/translate_a/single", "/_/TranslateWebserverUi/data/batchexecute"):
            return httpx.Response(429, text="rate limited")
        if request.url.path == "/translate_a/t":
            return httpx.Response(200, json=[["Good evening", "ja"]])
        raise AssertionError(f"Unexpected path: {request.url.path}")

    async def run_dict():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler_dict_fallback)) as client:
            g = GoogleTranslator(client=client, cooldown_seconds=60.0)
            assert await g.translate("こんばんは", "auto", "en") == "Good evening"
            assert await g.translate("おはよう", "auto", "en") == "Good evening"

    asyncio.run(run_dict())
    assert calls2 == [
        "/translate_a/single",
        "/_/TranslateWebserverUi/data/batchexecute",
        "/translate_a/t",
        "/translate_a/t",
    ]

    def handler_all_fail(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    async def run_fail():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler_all_fail)) as client:
            g = GoogleTranslator(client=client)
            with pytest.raises(TranslationError):
                await g.translate("テスト", "ja", "en")

    asyncio.run(run_fail())
