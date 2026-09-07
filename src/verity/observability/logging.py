"""Structured logging plumbing — one contextvar for the trace id.

`structlog.contextvars.merge_contextvars` reads any values bound via
`structlog.contextvars.bind_contextvars` and injects them on every subsequent
log call within the same task. That is exactly the shape we want for a request
trace_id: bind once at the top of ``RagAgent.answer``, every log line in every
stage (retriever, synthesizer, scorer, …) carries it automatically without any
call-site plumbing.

The API server + CLI call :func:`configure_logging` once at process start. Tests
that don't touch logging don't need to call anything — structlog degrades to the
default text renderer and the ``trace_id`` field, if present, prints alongside
the event.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import structlog

from verity.types import TraceId

_CONFIGURED = False

# Task-local trace id — used by ``HybridRetriever`` (and anywhere else that
# wants to open a span under the current request) to look up the current
# request's trace id without threading it through every function signature.
# The public Retriever/Chunker/… protocols stay untouched.
current_trace_id: ContextVar[TraceId | None] = ContextVar("verity_trace_id", default=None)


def configure_logging(level: str = "INFO", *, json_output: bool | None = None) -> None:
    """Configure structlog to render every event with the current contextvars.

    ``json_output`` defaults to ``True`` when a level >= INFO is requested (prod
    default), ``False`` at DEBUG (human-readable dev output). Callers may
    override either way. Idempotent — safe to call multiple times.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    if json_output is None:
        json_output = numeric_level >= logging.INFO

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    processors.append(
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def bind_trace(trace_id: TraceId) -> None:
    """Bind ``trace_id`` on the current task's contextvars so every subsequent
    log record inside this request carries it, and set the trace-scoped
    :data:`current_trace_id` so nested components (retriever, judge, …) can
    open spans parented under the same id without argument threading."""
    structlog.contextvars.bind_contextvars(trace_id=str(trace_id))
    current_trace_id.set(trace_id)


def unbind_trace() -> None:
    """Clear the request-scoped trace binding — call at the end of a request."""
    structlog.contextvars.unbind_contextvars("trace_id")
    current_trace_id.set(None)


@contextmanager
def request_trace(trace_id: TraceId):  # type: ignore[no-untyped-def]
    """Bind trace_id for the duration of the with-block, then clear it.

    Preferred over manual bind/unbind pairs in application code so structural
    guarantees (unbind on exception) come for free.
    """
    token = current_trace_id.set(trace_id)
    structlog.contextvars.bind_contextvars(trace_id=str(trace_id))
    try:
        yield
    finally:
        structlog.contextvars.unbind_contextvars("trace_id")
        current_trace_id.reset(token)


__all__ = ["bind_trace", "configure_logging", "request_trace", "unbind_trace"]
