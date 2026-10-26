"""Async HTTP client for Metacritic: browser-like headers, polite delays, retries."""
from __future__ import annotations

import asyncio
import random

import httpx

BASE_URL = "https://www.metacritic.com"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Phrases in <title> of an anti-bot interstitial instead of the real page.
_BLOCK_MARKERS = ("just a moment", "access denied", "attention required", "captcha", "are you a robot")


def _title(html: str) -> str:
    start = html.lower().find("<title")
    if start == -1:
        return ""
    end = html.lower().find("</title>", start)
    return html[start : end if end != -1 else start + 300].lower()


class BlockedError(RuntimeError):
    """Metacritic answered with a block / challenge page."""


class MetacriticClient:
    def __init__(
        self,
        proxy: str | None = None,
        delay: tuple[float, float] = (1.0, 2.5),
        timeout: float = 20.0,
        retries: int = 3,
    ) -> None:
        self._delay = delay
        self._retries = retries
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            follow_redirects=True,
            proxy=proxy,
        )

    async def __aenter__(self) -> "MetacriticClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self._client.aclose()

    async def get(self, path: str, params: dict | None = None) -> str:
        """GET a page and return HTML. Retries 429/5xx/network errors with backoff."""
        last_exc: Exception | None = None
        for attempt in range(1, self._retries + 1):
            await asyncio.sleep(random.uniform(*self._delay))
            try:
                resp = await self._client.get(path, params=params)
            except httpx.TransportError as exc:
                last_exc = exc
            else:
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_exc = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp
                    )
                elif resp.status_code == 403:
                    raise BlockedError(f"403 for {resp.url}")
                else:
                    resp.raise_for_status()
                    title = _title(resp.text)
                    if any(m in title for m in _BLOCK_MARKERS):
                        raise BlockedError(f"challenge page for {resp.url}: {title!r}")
                    return resp.text
            await asyncio.sleep(2**attempt)  # 2s, 4s, 8s
        raise RuntimeError(f"GET {path} failed after {self._retries} attempts: {last_exc}")

    # --- page helpers -------------------------------------------------------
    async def home(self) -> str:
        return await self.get("/game/")

    async def browse_new(self, page: int = 1) -> str:
        params = {"page": page} if page > 1 else None
        return await self.get("/browse/game/all/all/all-time/new/", params=params)

    async def game(self, slug: str, platform: str | None = None) -> str:
        return await self.get(f"/game/{slug}/", params={"platform": platform} if platform else None)

    async def critic_reviews(self, slug: str, platform: str | None = None) -> str:
        return await self.get(
            f"/game/{slug}/critic-reviews/", params={"platform": platform} if platform else None
        )

    async def user_reviews(self, slug: str, platform: str | None = None) -> str:
        return await self.get(
            f"/game/{slug}/user-reviews/", params={"platform": platform} if platform else None
        )
