"""Shared fixtures for the Verity test suite.

Tests that are neither ``slow`` nor ``integration`` (the fast lane) run with
network sockets disabled: a hidden download or API call fails the test instead of
passing on a machine that happens to be online. Unix sockets stay allowed for the
asyncio event loop.
"""

from __future__ import annotations

import pytest
import pytest_socket

from verity.config import Settings
from verity.types import (
    Chunk,
    Confidence,
    Document,
    RetrievalHit,
    RetrieverKind,
    SourceType,
)

# A developer's .env may hold a real provider key; tests must never read it, or a
# test meant to run keyless would reach a paid API (the slow and integration lanes
# leave the network open). Settings in tests come from the environment only.
Settings.model_config["env_file"] = None


@pytest.fixture
def document() -> Document:
    return Document(
        source_type=SourceType.MARKDOWN,
        uri="file:///corpus/msa.md",
        title="Master Services Agreement",
        text="1. Term. Either party may terminate on 30 days written notice.",
    )


@pytest.fixture
def chunk(document: Document) -> Chunk:
    return Chunk(
        document_id=document.id,
        text="Either party may terminate on 30 days written notice.",
        ordinal=0,
        char_start=10,
        char_end=62,
        section="1. Term",
    )


@pytest.fixture
def hit(chunk: Chunk) -> RetrievalHit:
    return RetrievalHit(chunk=chunk, score=0.82, kind=RetrieverKind.RERANKED, rank=0)


@pytest.fixture
def confident() -> Confidence:
    return Confidence(score=0.9, refused=False, rationale="Directly supported by the term clause.")


@pytest.fixture
def refused() -> Confidence:
    return Confidence(score=0.2, refused=True, rationale="Question is out of corpus scope.")


_NETWORKED_MARKERS = frozenset({"slow", "integration"})


@pytest.fixture(autouse=True)
def _block_network_in_fast_lane(request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
    if _NETWORKED_MARKERS.isdisjoint(m.name for m in request.node.iter_markers()):
        pytest_socket.disable_socket(allow_unix_socket=True)
        yield
        pytest_socket.enable_socket()
    else:
        yield
