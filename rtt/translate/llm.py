"""OpenAI-compatible LLM translator (works with OpenAI, OpenRouter, Ollama, vLLM)."""

import httpx

from rtt.translate.base import BaseTranslator, TranslationError


class LLMTranslator(BaseTranslator):
    """Translates utterances using an OpenAI-compatible `/chat/completions` endpoint."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4o-mini",
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = (api_key or "").strip()
        self.model = model or "gpt-4o-mini"
        self._owns_client = client is None
        self._client = client

    @property
    def name(self) -> str:
        return f"LLM({self.model})"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: list[str] | None = None,
    ) -> str:
        src_hint = (
            f"from {source_lang} "
            if source_lang and source_lang.lower() != "auto"
            else ""
        )
        system_lines = [
            f"You are a professional real-time subtitle translator. Translate ONLY the user's new utterance {src_hint}into {target_lang}.",
            "Output ONLY the natural translation with no quotes, explanations, or commentary.",
        ]
        if context:
            recent = "\n".join(f"- {line}" for line in context[-4:] if line.strip())
            if recent:
                system_lines.append(
                    "Recent prior source sentences (read-only context for pronouns and terminology; do NOT translate or repeat them):\n"
                    + recent
                )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "\n\n".join(system_lines)},
                {"role": "user", "content": text},
            ],
        }

        try:
            resp = await self._get_client().post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise TranslationError(f"LLM HTTP request failed: {exc}") from exc

        if resp.status_code >= 400:
            raise TranslationError(f"LLM HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise TranslationError(f"Malformed LLM response: {exc}") from exc

        translated = (content or "").strip()
        if not translated:
            raise TranslationError("LLM returned empty translation.")
        return translated

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
