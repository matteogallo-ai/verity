"""Request guards for the HTTP API: API-key authentication, per-key rate limit, per-key
spending budget, and upload type checks.

Everything here is deterministic and in process: one API process serves one tenant
(see SECURITY.md), so the limiter does not need shared state across processes.
"""

from __future__ import annotations

import hmac
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import structlog
from fastapi import HTTPException, Request

from verity.config import Settings

_ANONYMOUS = "anonymous"


class RateLimiter:
    """Sliding-window limit of ``limit`` hits per ``window_s`` seconds, per principal.

    Keyed by configured keys, the number of buckets is bounded by configuration. Keyed
    by client address, it is not, so ``max_principals`` caps it: expired buckets go
    first, then the oldest. A client that controls more addresses than the cap can
    reset its own count; the cap bounds memory, it does not stop a distributed guesser.
    """

    def __init__(
        self,
        *,
        limit: int,
        window_s: float,
        clock: Callable[[], float] = time.monotonic,
        max_principals: int | None = None,
    ) -> None:
        self._limit = limit
        self._window_s = window_s
        self._clock = clock
        self._max_principals = max_principals
        self._hits: dict[str, deque[float]] = {}

    def __len__(self) -> int:
        return len(self._hits)

    def _live(self, hits: deque[float], now: float) -> deque[float]:
        while hits and hits[0] <= now - self._window_s:
            hits.popleft()
        return hits

    def _evict(self, now: float) -> None:
        for principal in [p for p, hits in self._hits.items() if not self._live(hits, now)]:
            del self._hits[principal]
        while self._max_principals is not None and len(self._hits) >= self._max_principals:
            del self._hits[next(iter(self._hits))]

    def blocked(self, principal: str) -> float | None:
        """Like :meth:`hit` without recording: ``None`` while a slot is free."""
        now = self._clock()
        hits = self._live(self._hits.get(principal, deque()), now)
        return hits[0] + self._window_s - now if len(hits) >= self._limit else None

    def hit(self, principal: str) -> float | None:
        """Record one hit. ``None`` when allowed, else the seconds until a slot frees."""
        now = self._clock()
        cap = self._max_principals
        if principal not in self._hits and cap is not None and len(self._hits) >= cap:
            self._evict(now)
        hits = self._live(self._hits.setdefault(principal, deque()), now)
        if len(hits) >= self._limit:
            return hits[0] + self._window_s - now
        hits.append(now)
        return None


class SpendBudget:
    """At most ``budget_usd`` of LLM spend per ``window_s`` seconds, per principal.

    Spend is recorded after each answer from the cost it reports, so a request is
    admitted while the window's spend is under budget: the answers in flight when
    it is crossed (at most the per-key rate limit) overshoot it.
    """

    def __init__(
        self,
        *,
        budget_usd: float,
        window_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if budget_usd <= 0 or window_s <= 0:
            raise ValueError("budget_usd and window_s must be positive")
        self._budget_usd = budget_usd
        self._window_s = window_s
        self._clock = clock
        self._spend: dict[str, deque[tuple[float, float]]] = {}

    def _current(self, principal: str) -> deque[tuple[float, float]]:
        now = self._clock()
        spend = self._spend.setdefault(principal, deque())
        while spend and spend[0][0] <= now - self._window_s:
            spend.popleft()
        return spend

    def record(self, principal: str, cost_usd: float) -> None:
        if cost_usd > 0:
            self._current(principal).append((self._clock(), cost_usd))

    def exhausted(self, principal: str) -> float | None:
        """``None`` while under budget, else the seconds until enough spend expires."""
        spend = self._current(principal)
        total = sum(cost for _, cost in spend)
        if total < self._budget_usd:
            return None
        for at, cost in spend:
            total -= cost
            if total < self._budget_usd:
                return at + self._window_s - self._clock()
        raise AssertionError("unreachable: the loop empties the window")


def retry_after_header(seconds: float) -> dict[str, str]:
    """``Retry-After`` in whole seconds, rounded up and at least 1, as HTTP requires."""
    return {"Retry-After": str(max(1, math.ceil(seconds)))}


def _principal_for(supplied: str | None, settings: Settings) -> str | None:
    """The configured key ``supplied`` matches, by index; ``None`` if none does.

    Every key is compared, with ``hmac.compare_digest``, so the response time does not
    reveal which key or how much of it matched.
    """
    if not supplied:
        return None
    matched: str | None = None
    for index, key in enumerate(settings.api_keys):
        if hmac.compare_digest(supplied.encode(), key.get_secret_value().encode()):
            matched = f"key-{index}"
    return matched


async def require_api_key(request: Request) -> str:
    """FastAPI dependency: authenticate ``X-API-Key`` and apply the rate limit.

    Returns the principal (``key-<n>`` or ``anonymous``), bound on the log context.
    The key itself is never logged. Async on purpose: FastAPI runs a sync dependency
    in a worker thread, whose context-variable binding the handler would not see.

    Failed attempts are limited per client address, checked before the key so a
    blocked client cannot tell a right guess from a wrong one. Behind a reverse proxy
    every request comes from the proxy's address unless uvicorn trusts its forwarded
    headers (``--proxy-headers --forwarded-allow-ips``), and one guesser would then
    block every client.
    """
    settings: Settings = request.app.state.settings
    if settings.api_auth_disabled:
        principal = _ANONYMOUS
    else:
        client = request.client.host if request.client else "unknown"
        failures: RateLimiter = request.app.state.auth_failures
        wait = failures.blocked(client)
        if wait is not None:
            raise HTTPException(
                status_code=429,
                detail="Too many failed key attempts.",
                headers=retry_after_header(wait),
            )
        found = _principal_for(request.headers.get("x-api-key"), settings)
        if found is None:
            failures.hit(client)
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")
        principal = found
    limiter: RateLimiter = request.app.state.rate_limiter
    retry_after = limiter.hit(principal)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded.",
            headers=retry_after_header(retry_after),
        )
    structlog.contextvars.bind_contextvars(principal=principal)
    return principal


# --- uploads ---------------------------------------------------------------------------

# Sent by browsers and curl when the OS has no mapping for the extension (common for
# .md); accepted only because the content is then sniffed below.
_GENERIC_MIME = "application/octet-stream"
_TEXT_MIMES = frozenset({"text/plain", "text/markdown", "text/x-markdown"})
_ALLOWED: dict[str, frozenset[str]] = {
    ".md": _TEXT_MIMES,
    ".markdown": _TEXT_MIMES,
    ".txt": frozenset({"text/plain"}),
    ".pdf": frozenset({"application/pdf"}),
    ".docx": frozenset({"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}),
}
_MAGIC = {".pdf": b"%PDF-", ".docx": b"PK\x03\x04"}
ALLOWED_EXTENSIONS: tuple[str, ...] = tuple(_ALLOWED)


@dataclass(frozen=True)
class UploadRejectedError(Exception):
    message: str


def check_upload(filename: str, content_type: str | None, content: bytes) -> str:
    """The parser extension for an upload, or :class:`UploadRejectedError`.

    Three checks, all required: the extension is allow-listed, the declared MIME type
    matches it (or is the generic octet-stream), and the bytes look like that type
    (magic number for PDF/DOCX, UTF-8 without NUL for text).
    """
    lowered = filename.lower()
    ext = next((e for e in ALLOWED_EXTENSIONS if lowered.endswith(e)), None)
    if ext is None:
        raise UploadRejectedError(f"Unsupported file type; allowed: {', '.join(_ALLOWED)}.")
    mime = (content_type or _GENERIC_MIME).split(";", 1)[0].strip().lower()
    if mime != _GENERIC_MIME and mime not in _ALLOWED[ext]:
        raise UploadRejectedError(f"Content type {mime!r} does not match {ext}.")
    magic = _MAGIC.get(ext)
    if magic is not None:
        if not content.startswith(magic):
            raise UploadRejectedError(f"Content is not a {ext} file.")
    elif b"\x00" in content or not _is_utf8(content):
        raise UploadRejectedError("Text uploads must be UTF-8.")
    return ext


def _is_utf8(content: bytes) -> bool:
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


__all__ = [
    "ALLOWED_EXTENSIONS",
    "RateLimiter",
    "SpendBudget",
    "UploadRejectedError",
    "check_upload",
    "require_api_key",
    "retry_after_header",
]
