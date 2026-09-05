"""Ingestion contracts: parse → chunk → embed.

Three orthogonal responsibilities, three protocols. A ``Chunker`` never knows which
``Parser`` produced the document; an ``Embedder`` never knows how the text was chunked.
This is what lets us swap docling for a naive parser in tests, or a local embedder for
a hosted one, without touching the rest of the pipeline.

Implementations land in S1.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from verity.types import Chunk, Document, EmbeddedChunk, SourceType


@runtime_checkable
class Parser(Protocol):
    """Turns raw bytes (or a URL) into a normalised :class:`Document`.

    Structure-aware: implementations must preserve headings/sections in the text and
    in per-region metadata so the chunker can chunk along semantic boundaries.
    """

    def supports(self, source_type: SourceType) -> bool: ...

    async def parse(self, uri: str, raw: bytes | None = None) -> Document:
        """Parse ``raw`` (or fetch ``uri`` when ``raw`` is None) into a Document."""
        ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a document into retrievable chunks along semantic/structural boundaries.

    Must set correct ``char_start``/``char_end`` offsets into ``document.text`` — these
    are load-bearing for citation highlighting downstream.
    """

    def chunk(self, document: Document) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    """Produces dense vectors for a batch of texts. Batched by contract for throughput."""

    @property
    def model(self) -> str: ...

    @property
    def dim(self) -> int: ...

    async def embed(self, texts: list[str]) -> list[tuple[float, ...]]: ...

    async def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]: ...
