"""Shared fixtures for the Verity test suite."""

from __future__ import annotations

import pytest

from verity.types import (
    Chunk,
    Confidence,
    Document,
    RetrievalHit,
    RetrieverKind,
    SourceType,
)


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
