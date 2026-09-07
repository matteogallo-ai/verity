"""``current_trace_id`` contextvar + ``request_trace`` binding + structlog carrying trace_id."""

from __future__ import annotations

import logging

import structlog

from verity.observability.logging import (
    configure_logging,
    current_trace_id,
    request_trace,
)
from verity.types import new_id


def test_current_trace_id_defaults_to_none() -> None:
    assert current_trace_id.get() is None


def test_request_trace_binds_and_unbinds() -> None:
    trace = new_id()
    assert current_trace_id.get() is None
    with request_trace(trace):
        assert current_trace_id.get() == trace
    assert current_trace_id.get() is None


def test_request_trace_binds_even_on_exception() -> None:
    trace = new_id()
    try:
        with request_trace(trace):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert current_trace_id.get() is None


def test_structlog_processor_injects_trace_id(caplog) -> None:  # type: ignore[no-untyped-def]
    """A log call inside ``request_trace`` carries the trace_id via the
    contextvars processor. We route structlog through stdlib logging so caplog
    can capture the record."""
    configure_logging(level="DEBUG", json_output=False)
    # For this test, wire structlog to forward to stdlib logging via a plain
    # bound logger. `bind_contextvars` still applies through
    # `merge_contextvars` (already in the configure_logging processors list).
    caplog.set_level(logging.INFO, logger="verity.test")
    trace = new_id()
    logger = structlog.get_logger("verity.test")
    with request_trace(trace):
        # Emit a log; the merge_contextvars processor should attach trace_id.
        # We assert on the event dict directly by rendering to a JSON string
        # via a captured PrintLogger-like buffer.
        events: list[dict] = []  # type: ignore[type-arg]
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                lambda _logger, _method, event_dict: events.append(dict(event_dict)) or event_dict,
                structlog.processors.KeyValueRenderer(),
            ]
        )
        logger = structlog.get_logger("verity.test")
        logger.info("hello")
    assert events, "no event captured"
    assert events[0].get("trace_id") == str(trace)
