"""Readiness: can this process serve ``/ask`` right now?

Liveness (``/health``) only says the process answers. Readiness requires the startup
preload (corpus and models) to be done, and reaches the two dependencies an answer
needs: the store (Postgres, when it is the configured store) and the LLM providers. The providers are probed the way
:class:`~verity.llm.routing.MultiProviderRoutingClient` uses them: in order, moving to
the next one only when a provider is unavailable, and stopping on any other failure
(a rejected key, an unknown model), because routing would stop there too.

The response names each check's state only. The cause (host, driver message) goes to
the log: ``/ready`` needs no key.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

import structlog

from verity.api.state import AppRuntime
from verity.llm.base import ProviderUnavailableError
from verity.llm.clients import AnthropicClient, OpenAIClient
from verity.retrieval.pgvector_store import PgVectorStore

_log = structlog.get_logger("verity.api.readiness")

# A dependency slower than this to answer one trivial request is not ready to serve.
CHECK_TIMEOUT_S = 2.0
# /ready is public: a result is reused for this long so that a caller looping on it
# cannot turn each request into a new Postgres connection and a provider request.
CACHE_S = 5.0

OK, NOT_USED, UNREACHABLE, TIMEOUT = "ok", "not used", "unreachable", "timeout"


async def _probe(name: str, call: Callable[[], Awaitable[None]]) -> tuple[str, bool]:
    """``(state, transient)``: ``transient`` is true when routing would try the next
    provider after this failure."""
    try:
        await asyncio.wait_for(call(), CHECK_TIMEOUT_S)
    except TimeoutError:
        _log.warning("readiness_timeout", check=name, timeout_s=CHECK_TIMEOUT_S)
        return TIMEOUT, True
    # Readiness reports a dependency's state and never raises: any failure, typed or
    # not (a driver error mid-query, a bad key), makes the dependency not ready. The
    # cause is logged here; the response carries only the state.
    except Exception as exc:
        _log.warning("readiness_failed", check=name, error_type=type(exc).__name__, error=str(exc))
        return UNREACHABLE, isinstance(exc, ProviderUnavailableError)
    return OK, False


async def _store_state(runtime: AppRuntime) -> str:
    if not isinstance(runtime.store, PgVectorStore):
        return NOT_USED
    state, _ = await _probe("store", runtime.store.ping)
    return state


async def _llm_state(runtime: AppRuntime) -> str:
    pingable = [p for p in runtime.providers if isinstance(p, (AnthropicClient, OpenAIClient))]
    if not pingable:
        return NOT_USED
    states: list[str] = []
    for provider in pingable:
        state, transient = await _probe(f"llm:{provider.provider.value}", provider.ping)
        if state == OK or not transient:
            return state
        states.append(state)
    return TIMEOUT if all(s == TIMEOUT for s in states) else UNREACHABLE


class Readiness:
    """One per app: probes the dependencies at most once per :data:`CACHE_S`, and
    concurrent callers share the probe in flight."""

    def __init__(self, runtime: AppRuntime) -> None:
        self._runtime = runtime
        self._lock = asyncio.Lock()
        self._last: tuple[float, bool, dict[str, str]] | None = None

    async def check(self) -> tuple[bool, dict[str, str]]:
        """``(ready, {check: state})``; ready when the preload is done and no dependency
        is unreachable or timed out."""
        probed_ready, probed = await self._probed()
        # The preload state is in memory, so it is read on every call rather than
        # cached: the process turns ready as soon as the preload ends.
        preload = self._runtime.preload_state
        return probed_ready and preload == OK, {**probed, "preload": preload}

    async def _probed(self) -> tuple[bool, dict[str, str]]:
        async with self._lock:
            now = time.monotonic()
            if self._last is not None and now - self._last[0] < CACHE_S:
                return self._last[1], self._last[2]
            # A new preload attempt re-ingests the whole corpus: starting it here, with
            # the probes, bounds attempts to one per CACHE_S however often /ready is hit.
            if self._runtime.preload_state == "failed":
                self._runtime.start_preload()
            store, llm = await asyncio.gather(
                _store_state(self._runtime), _llm_state(self._runtime)
            )
            checks = {"store": store, "llm": llm}
            ready = all(s in (OK, NOT_USED) for s in checks.values())
            self._last = (time.monotonic(), ready, checks)
            return ready, checks
