"""Tests for LLM, DeepL, and Google translators via httpx.MockTransport."""

import asyncio
import json
import httpx
import pytest

from rtt.translate.base import TranslationError
from rtt.translate.deepl import DeepLTranslator
from rtt.translate.google import GoogleTranslator, parse_google_response
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


def test_google_translator_and_parser():
    assert parse_google_response([[["Hello ", "こんにちは"], ["world", "世界"]]]) == "Hello world"
    with pytest.raises(TranslationError):
        parse_google_response([])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["sl"] == "auto"
        assert request.url.params["tl"] == "en"
        return httpx.Response(200, json=[[ ["Good morning", "おはようございます", None, None, 10] ]])

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            g = GoogleTranslator(client=client)
            assert await g.translate("おはようございます", "auto", "en") == "Good morning"

    asyncio.run(run())
