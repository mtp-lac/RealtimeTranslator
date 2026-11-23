"""Free Google Translate web endpoint translator with multi-endpoint fallback."""

from collections import OrderedDict
import json
import time

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


def parse_batchexecute_response(raw_text: str) -> str:
    """Extract translated text from Google Translate `batchexecute` (`MkEWBc`) response."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise TranslationError("Empty Google batchexecute response.")

    inner_json: str | None = None
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("[["):
            continue
        try:
            outer = json.loads(stripped)
        except ValueError:
            continue
        if not isinstance(outer, list):
            continue
        for item in outer:
            if (
                isinstance(item, list)
                and len(item) >= 3
                and item[0] == "wrb.fr"
                and item[1] == "MkEWBc"
                and isinstance(item[2], str)
            ):
                inner_json = item[2]
                break
        if inner_json is not None:
            break

    if inner_json is None:
        raise TranslationError("Unexpected Google batchexecute response structure.")

    try:
        inner = json.loads(inner_json)
        segments = inner[1][0][0][5]
    except (ValueError, TypeError, IndexError, KeyError) as exc:
        raise TranslationError("Malformed Google batchexecute payload.") from exc

    if not isinstance(segments, list):
        raise TranslationError("Google batchexecute payload missing segments.")

    parts: list[str] = []
    for idx, seg in enumerate(segments):
        if isinstance(seg, list) and seg and isinstance(seg[0], str):
            if idx > 0 and len(seg) > 2 and seg[2] is True:
                parts.append(" ")
            parts.append(seg[0])

    result = "".join(parts).strip()
    if not result:
        raise TranslationError("Google batchexecute returned empty text.")
    return result


def parse_dict_chrome_response(data) -> str:
    """Extract translated text from `clients5.google.com/translate_a/t` JSON."""
    if not isinstance(data, list) or not data:
        raise TranslationError("Unexpected Google dict-chrome-ex response structure.")

    parts: list[str] = []
    for item in data:
        if isinstance(item, str):
            if item:
                parts.append(item)
        elif isinstance(item, list) and item and isinstance(item[0], str):
            if item[0]:
                parts.append(item[0])

    result = "".join(parts).strip()
    if not result:
        raise TranslationError("Google dict-chrome-ex returned empty text.")
    return result


class _EndpointHTTPStatusError(TranslationError):
    """Internal error carrying HTTP status code for cooldown decisions."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class GoogleTranslator(BaseTranslator):
    """Non-key translator using multiple Google Translate web endpoints with fallback."""

    ENDPOINT = "https://translate.google.com/translate_a/single"
    BATCHEXECUTE_ENDPOINT = "https://translate.google.com/_/TranslateWebserverUi/data/batchexecute"
    DICT_CHROME_ENDPOINT = "https://clients5.google.com/translate_a/t"
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        cooldown_seconds: float = 60.0,
        cache_size: int = 256,
    ):
        self._owns_client = client is None
        self._client = client
        self._cooldown_seconds = float(cooldown_seconds)
        self._cache_size = max(0, int(cache_size))
        self._cooldown_until: dict[str, float] = {}
        self._cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()

    @property
    def name(self) -> str:
        return "GoogleTranslate"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(3.5, connect=2.0),
                headers=self.DEFAULT_HEADERS,
                follow_redirects=True,
            )
        return self._client

    @staticmethod
    def _raise_for_status(resp: httpx.Response, label: str) -> None:
        if resp.status_code >= 400:
            raise _EndpointHTTPStatusError(
                resp.status_code,
                f"{label} HTTP {resp.status_code}",
            )

    async def _translate_single_at(
        self, client: httpx.AsyncClient, text: str, sl: str, tl: str
    ) -> str:
        resp = await client.post(
            self.ENDPOINT,
            params={
                "client": "at",
                "dt": "t",
                "sl": sl,
                "tl": tl,
                "ie": "UTF-8",
                "oe": "UTF-8",
            },
            data={"q": text},
        )
        self._raise_for_status(resp, "Google Translate single_at")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise TranslationError(f"Invalid Google Translate JSON: {exc}") from exc
        return parse_google_response(payload)

    async def _translate_batchexecute(
        self, client: httpx.AsyncClient, text: str, sl: str, tl: str
    ) -> str:
        rpc_payload = json.dumps([[text, sl, tl, True], [None]], separators=(",", ":"))
        f_req = json.dumps(
            [[["MkEWBc", rpc_payload, None, "generic"]]],
            separators=(",", ":"),
        )
        resp = await client.post(
            self.BATCHEXECUTE_ENDPOINT,
            params={
                "rpcids": "MkEWBc",
                "hl": "en",
                "soc-app": "1",
                "soc-platform": "1",
                "soc-device": "1",
            },
            data={"f.req": f_req},
            headers={
                "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                "Origin": "https://translate.google.com",
                "Referer": "https://translate.google.com/",
            },
        )
        self._raise_for_status(resp, "Google Translate batchexecute")
        return parse_batchexecute_response(resp.text)

    async def _translate_dict_chrome(
        self, client: httpx.AsyncClient, text: str, sl: str, tl: str
    ) -> str:
        resp = await client.get(
            self.DICT_CHROME_ENDPOINT,
            params={
                "client": "dict-chrome-ex",
                "sl": sl,
                "tl": tl,
                "q": text,
            },
        )
        self._raise_for_status(resp, "Google Translate dict_chrome_ex")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise TranslationError(f"Invalid Google dict-chrome-ex JSON: {exc}") from exc
        return parse_dict_chrome_response(payload)

    def _ordered_endpoints(self, now: float):
        endpoints = [
            ("single_at", self._translate_single_at),
            ("batchexecute", self._translate_batchexecute),
            ("dict_chrome_ex", self._translate_dict_chrome),
        ]
        ready = [(n, fn) for n, fn in endpoints if self._cooldown_until.get(n, 0.0) <= now]
        cooling = sorted(
            ((n, fn) for n, fn in endpoints if self._cooldown_until.get(n, 0.0) > now),
            key=lambda pair: self._cooldown_until[pair[0]],
        )
        return ready + cooling

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: list[str] | None = None,
    ) -> str:
        clean_text = text.strip()
        if not clean_text:
            raise TranslationError("Empty text.")

        sl = source_lang.strip() if source_lang and source_lang.strip().lower() != "auto" else "auto"
        tl = target_lang.strip()
        cache_key = (sl, tl, clean_text)

        if self._cache_size > 0 and cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        client = self._get_client()
        now = time.monotonic()
        last_error: Exception | None = None

        for ep_name, fn in self._ordered_endpoints(now):
            try:
                translated = await fn(client, clean_text, sl, tl)
            except _EndpointHTTPStatusError as exc:
                last_error = exc
                if exc.status_code in (429, 403):
                    self._cooldown_until[ep_name] = time.monotonic() + self._cooldown_seconds
                continue
            except (httpx.HTTPError, ValueError, TranslationError) as exc:
                last_error = exc
                continue

            self._cooldown_until.pop(ep_name, None)
            if self._cache_size > 0:
                self._cache[cache_key] = translated
                self._cache.move_to_end(cache_key)
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            return translated

        if isinstance(last_error, TranslationError):
            raise last_error
        raise TranslationError(f"Google Translate request failed: {last_error}")

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
