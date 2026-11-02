"""Request guards for the HTTP API: API-key authentication, per-key rate limit, and
upload type checks.

Everything here is deterministic and in process: one API process serves one tenant
(see SECURITY.md), so the limiter does not need shared state across processes.
"""

from __future__ import annotations

import hmac
import math
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass

import structlog
from fastapi import HTTPException, Request

from verity.config import Settings

_ANONYMOUS = "anonymous"


class RateLimiter:
    """Sliding-window limit of ``limit`` hits per ``window_s`` seconds, per principal.

    Principals are configured keys (plus ``anonymous`` when auth is disabled) or, for
    failed authentications, client addresses. Addresses are chosen by the client, so
    at most ``max_principals`` buckets are kept; the least recently hit is dropped.
    """

    def __init__(
        self,
        *,
        limit: int,
        window_s: float,
        max_principals: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window_s = window_s
        self._max_principals = max_principals
        self._clock = clock
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()

    @property
    def n_principals(self) -> int:
        return len(self._hits)

    def _window(self, principal: str, now: float) -> deque[float]:
        hits = self._hits.get(principal)
        if hits is None:
            return deque()
        while hits and hits[0] <= now - self._window_s:
            hits.popleft()
        return hits

    def retry_after(self, principal: str) -> float | None:
        """Without recording a hit: ``None`` when allowed, else the seconds to wait."""
        now = self._clock()
        hits = self._window(principal, now)
        if len(hits) >= self._limit:
            return hits[0] + self._window_s - now
        return None

    def hit(self, principal: str) -> float | None:
        """Record one hit. ``None`` when allowed, else the seconds until a slot frees."""
        retry = self.retry_after(principal)
        if retry is not None:
            return retry
        hits = self._hits.setdefault(principal, deque())
        hits.append(self._clock())
        self._hits.move_to_end(principal)
        while len(self._hits) > self._max_principals:
            self._hits.popitem(last=False)
        return None


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
    """
    settings: Settings = request.app.state.settings
    if settings.api_auth_disabled:
        principal = _ANONYMOUS
    else:
        # Checked before the key: once a client has spent its failure budget, a right
        # guess must not be distinguishable from a wrong one.
        failures: RateLimiter = request.app.state.auth_failure_limiter
        client = request.client.host if request.client else "unknown"
        _raise_if_limited(failures.retry_after(client))
        found = _principal_for(request.headers.get("x-api-key"), settings)
        if found is None:
            failures.hit(client)
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")
        principal = found
    limiter: RateLimiter = request.app.state.rate_limiter
    _raise_if_limited(limiter.hit(principal))
    structlog.contextvars.bind_contextvars(principal=principal)
    return principal


def _raise_if_limited(retry_after: float | None) -> None:
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded.",
            headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
        )


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
    "UploadRejectedError",
    "check_upload",
    "require_api_key",
]
