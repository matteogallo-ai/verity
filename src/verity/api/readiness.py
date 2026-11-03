"""Readiness: can this process serve ``/ask`` right now?

Liveness (``/health``) only says the process answers. Readiness reaches the two
dependencies an answer needs: the store (Postgres, when it is the configured store)
and the LLM providers. One reachable provider is enough, because the routing client
falls back to the next one. Each check is cut at :data:`CHECK_TIMEOUT_S`, so a probe
never waits on a hung dependency longer than its own timeout would.

The response names each check's state only. The cause (host, driver message) goes to
the log: ``/ready`` needs no key.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import structlog

from verity.api.state import AppRuntime
from verity.llm.clients import AnthropicClient, OpenAIClient
from verity.retrieval.pgvector_store import PgVectorStore

_log = structlog.get_logger("verity.api.readiness")

# Below the default probe timeouts of orchestrators (Kubernetes: 1 s period 10 s, so a
# probe must answer well within it); a dependency slower than this is not ready.
CHECK_TIMEOUT_S = 2.0

OK, NOT_USED, UNREACHABLE, TIMEOUT = "ok", "not used", "unreachable", "timeout"


async def _probe(name: str, call: Callable[[], Awaitable[None]]) -> str:
    try:
        await asyncio.wait_for(call(), CHECK_TIMEOUT_S)
    except TimeoutError:
        _log.warning("readiness_timeout", check=name, timeout_s=CHECK_TIMEOUT_S)
        return TIMEOUT
    # Readiness reports a dependency's state and never raises: any failure, typed or
    # not (a driver error mid-query, a bad key), makes the dependency not ready. The
    # cause is logged here; the response carries only the state.
    except Exception as exc:
        _log.warning("readiness_failed", check=name, error_type=type(exc).__name__, error=str(exc))
        return UNREACHABLE
    return OK


async def _store_state(runtime: AppRuntime) -> str:
    if not isinstance(runtime.store, PgVectorStore):
        return NOT_USED
    return await _probe("store", runtime.store.ping)


async def _llm_state(runtime: AppRuntime) -> str:
    pingable = [p for p in runtime.providers if isinstance(p, (AnthropicClient, OpenAIClient))]
    if not pingable:
        return NOT_USED
    states = await asyncio.gather(*(_probe(f"llm:{p.provider.value}", p.ping) for p in pingable))
    if OK in states:
        return OK
    return TIMEOUT if all(s == TIMEOUT for s in states) else UNREACHABLE


async def readiness(runtime: AppRuntime) -> tuple[bool, dict[str, str]]:
    """``(ready, {check: state})``; ready when no check is unreachable or timed out."""
    store, llm = await asyncio.gather(_store_state(runtime), _llm_state(runtime))
    checks = {"store": store, "llm": llm}
    return all(s in (OK, NOT_USED) for s in checks.values()), checks
