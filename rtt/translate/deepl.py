"""DeepL API translator supporting Free (`:fx`) and Pro endpoints."""

import httpx

from rtt.translate.base import BaseTranslator, TranslationError


def _normalize_target_lang(target_lang: str) -> str:
    code = (target_lang or "en").strip().upper()
    if code == "EN":
        return "EN-US"
    if code == "PT":
        return "PT-PT"
    return code


class DeepLTranslator(BaseTranslator):
    """Translates utterances via DeepL's `/v2/translate` REST API."""

    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None):
        self.api_key = (api_key or "").strip()
        self._owns_client = client is None
        self._client = client

    @property
    def name(self) -> str:
        return "DeepL"

    @property
    def endpoint(self) -> str:
        host = "api-free.deepl.com" if self.api_key.endswith(":fx") else "api.deepl.com"
        return f"https://{host}/v2/translate"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=8.0)
        return self._client

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: list[str] | None = None,
    ) -> str:
        if not self.api_key:
            raise TranslationError("DeepL API key is missing.")

        payload: dict = {
            "text": [text],
            "target_lang": _normalize_target_lang(target_lang),
        }
        if source_lang and source_lang.lower() != "auto":
            payload["source_lang"] = source_lang.split("-")[0].upper()

        headers = {
            "Authorization": f"DeepL-Auth-Key {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = await self._get_client().post(self.endpoint, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise TranslationError(f"DeepL HTTP request failed: {exc}") from exc

        if resp.status_code >= 400:
            raise TranslationError(f"DeepL HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            data = resp.json()
            translated = str(data["translations"][0]["text"]).strip()
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise TranslationError(f"Malformed DeepL response: {exc}") from exc

        if not translated:
            raise TranslationError("DeepL returned empty translation.")
        return translated

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
