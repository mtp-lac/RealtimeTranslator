"""Free Google Translate web endpoint fallback translator."""

import httpx

from rtt.translate.base import BaseTranslator, TranslationError


def parse_google_response(data) -> str:
    """Extract concatenated translated text from `translate_a/single` JSON."""
    if not isinstance(data, list) or not data or not isinstance(data[0], list):
        raise TranslationError("Unexpected Google Translate response structure.")
    parts: list[str] = []
    for seg in data[0]:
        if isinstance(seg, list) and seg and isinstance(seg[0], str):
            parts.append(seg[0])
    result = "".join(parts).strip()
    if not result:
        raise TranslationError("Google Translate returned empty text.")
    return result


class GoogleTranslator(BaseTranslator):
    """Non-key fallback translator backed by Google Translate's `gtx` endpoint."""

    ENDPOINT = "https://translate.googleapis.com/translate_a/single"

    def __init__(self, client: httpx.AsyncClient | None = None):
        self._owns_client = client is None
        self._client = client

    @property
    def name(self) -> str:
        return "GoogleTranslate"

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
        sl = source_lang if source_lang and source_lang.lower() != "auto" else "auto"
        params = {
            "client": "gtx",
            "sl": sl,
            "tl": target_lang,
            "dt": "t",
            "q": text,
        }
        try:
            resp = await self._get_client().get(self.ENDPOINT, params=params)
        except httpx.HTTPError as exc:
            raise TranslationError(f"Google Translate request failed: {exc}") from exc

        if resp.status_code >= 400:
            raise TranslationError(f"Google Translate HTTP {resp.status_code}")

        try:
            payload = resp.json()
        except ValueError as exc:
            raise TranslationError(f"Invalid Google Translate JSON: {exc}") from exc

        return parse_google_response(payload)

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
