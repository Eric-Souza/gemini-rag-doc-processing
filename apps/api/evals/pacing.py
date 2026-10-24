"""Request pacing for the Gemini free tier, shared by the eval runners."""

from __future__ import annotations

import asyncio
import re
import time


def rate_limit_message(exc: BaseException) -> str | None:
    """The error text if `exc` (or anything it wraps) is a 429 rate-limit error, else None."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current)
        if (
            type(current).__name__ in {"RateLimitError", "GoogleRateLimitError"}
            or "RESOURCE_EXHAUSTED" in text
            or re.search(r"\b429\b", text)
        ):
            return text
        current = current.__cause__ or current.__context__
    return None


class Pacer:
    """Spaces out chat requests to stay under the Gemini free tier's requests-per-minute limit,
    and waits out a 429 using the retry delay the API suggests."""

    def __init__(self, rpm: float, retries: int = 3) -> None:
        self.interval = 60 / rpm if rpm > 0 else 0.0
        self.retries = retries
        self.requests = 0
        self.rate_limited = 0
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        """Block until the next request is allowed. Call once per API request."""
        async with self._lock:
            delay = self._next - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next = time.monotonic() + self.interval
            self.requests += 1

    async def call(self, fn, *args, paced: bool = True, **kwargs):
        """Run `fn`, retrying on per-minute 429s. Pass paced=False when `fn` already paces
        each of its requests (for example through an HTTP client hook)."""
        for attempt in range(self.retries + 1):
            if paced:
                await self.wait()
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                message = rate_limit_message(exc)
                if message is None:
                    raise
                if "PerDay" in message:
                    raise SystemExit(
                        "Gemini free-tier daily request quota is used up for this model. "
                        "Retry tomorrow or set GEMINI_EVAL_MODEL to a model with quota left."
                    ) from exc
                if attempt == self.retries:
                    raise
                self.rate_limited += 1
                match = re.search(r"retry in ([\d.]+)s", message)
                delay = float(match.group(1)) + 1 if match else 60.0
                print(f"     rate limited, retrying in {delay:.0f}s")
                await asyncio.sleep(delay)
