"""End-to-end ingestion: discover sources → parse → chunk → embed → upsert.

The pipeline is intentionally small — it wires the stage protocols together and does
no heavy lifting itself. That is the whole point of the protocols: this file has no
knowledge of docling, sentence-transformers, or Postgres. Swap any of them for a fake
and the pipeline runs the same way.

Errors are handled per-source, not per-batch: one bad PDF must not sink an ingestion
run over a directory of 500 documents. Failures are logged with structured context
and the pipeline continues. The exceptions are :class:`StoreSchemaError` and
:class:`StoreUnavailableError`: the store cannot accept any document, so they
propagate at the first one.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import structlog

from verity.ingestion.base import Chunker, Embedder, Parser
from verity.ingestion.ids import content_sha256
from verity.retrieval.base import IndexedDocument, StoreSchemaError, StoreUnavailableError
from verity.types import Chunk, Document, EmbeddedChunk

log = structlog.get_logger(__name__)

_SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf", ".docx"}


class VectorSink(Protocol):
    """Narrow write-side of the store the pipeline actually depends on."""

    async def indexed_document(self, document_id: str) -> IndexedDocument | None: ...

    async def replace_document(self, document: Document, chunks: list[EmbeddedChunk]) -> None: ...


@dataclass
class IngestionFailure:
    uri: str
    error: str


@dataclass
class IngestionResult:
    """Summary of a single ingestion run — surfaced by the CLI and by tests."""

    documents: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    failures: list[IngestionFailure] = field(default_factory=list)
    # Documents whose text, chunks and embedding model match the index: not embedded.
    unchanged: int = 0
    elapsed_s: float = 0.0

    @property
    def n_documents(self) -> int:
        return len(self.documents)

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)


def discover_sources(path_or_url: str) -> list[str]:
    """Expand a CLI argument into a concrete list of URIs to parse.

    - A URL (``http(s)://…``) → one entry, that URL.
    - A regular file → one entry, its ``file://`` URI (kept as absolute path if local).
    - A directory → recursively finds every supported extension.
    """
    parsed = urlparse(path_or_url)
    if parsed.scheme in {"http", "https"}:
        return [path_or_url]

    path = Path(path_or_url).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"No such file or directory: {path_or_url}")
    if path.is_file():
        return [str(path)]
    return sorted(
        str(p) for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    )


class IngestionPipeline:
    """Parses, chunks, embeds and (optionally) upserts a batch of sources."""

    def __init__(
        self,
        parser: Parser,
        chunker: Chunker,
        embedder: Embedder,
        sink: VectorSink | None = None,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._sink = sink

    async def ingest(self, uris: Iterable[str]) -> IngestionResult:
        result = IngestionResult()
        started = time.perf_counter()

        for uri in uris:
            # Each stage below catches everything on purpose: one unreadable source is
            # logged and recorded as a failure, and the rest of the batch continues
            # (module docstring). Store errors that affect every document re-raise.
            try:
                doc = await self._parser.parse(uri)
            except Exception as exc:
                log.error("parse_failed", uri=uri, error=str(exc))
                result.failures.append(IngestionFailure(uri=uri, error=f"parse: {exc}"))
                continue

            try:
                chunks = self._chunker.chunk(doc)
            except Exception as exc:
                log.error("chunk_failed", uri=uri, document_id=str(doc.id), error=str(exc))
                result.failures.append(IngestionFailure(uri=uri, error=f"chunk: {exc}"))
                continue

            try:
                indexed = self._sink is not None and await self._is_indexed(doc, chunks)
            except (StoreSchemaError, StoreUnavailableError):
                raise
            except Exception as exc:
                log.error("index_lookup_failed", uri=uri, document_id=str(doc.id), error=str(exc))
                result.failures.append(IngestionFailure(uri=uri, error=f"index lookup: {exc}"))
                continue
            if indexed:
                result.unchanged += 1
                result.documents.append(doc)
                result.chunks.extend(chunks)
                log.info("unchanged", uri=uri, document_id=str(doc.id))
                continue

            try:
                embedded = await self._embedder.embed_chunks(chunks)
            except Exception as exc:
                log.error("embed_failed", uri=uri, document_id=str(doc.id), error=str(exc))
                result.failures.append(IngestionFailure(uri=uri, error=f"embed: {exc}"))
                continue

            if self._sink is not None:
                try:
                    # Replacing, not upserting: an edit that moves chunk boundaries
                    # changes chunk ids, and the old chunks must not stay retrievable.
                    await self._sink.replace_document(doc, embedded)
                except (StoreSchemaError, StoreUnavailableError):
                    raise
                except Exception as exc:
                    log.error(
                        "upsert_failed",
                        uri=uri,
                        document_id=str(doc.id),
                        error=str(exc),
                    )
                    result.failures.append(IngestionFailure(uri=uri, error=f"upsert: {exc}"))
                    continue

            result.documents.append(doc)
            result.chunks.extend(chunks)
            log.info(
                "ingested",
                uri=uri,
                document_id=str(doc.id),
                n_chunks=len(chunks),
            )

        result.elapsed_s = time.perf_counter() - started
        return result

    async def _is_indexed(self, doc: Document, chunks: Sequence[Chunk]) -> bool:
        """Whether the index already holds exactly this text, chunked this way and
        embedded by this model. Chunk ids carry their spans, so a chunker change is
        caught too."""
        assert self._sink is not None
        held = await self._sink.indexed_document(str(doc.id))
        return (
            held is not None
            and held.content_sha256 == content_sha256(doc.text)
            and held.chunk_ids == frozenset(str(c.id) for c in chunks)
            and held.embedding_models == frozenset({self._embedder.model})
        )


__all__ = [
    "IngestionFailure",
    "IngestionPipeline",
    "IngestionResult",
    "VectorSink",
    "discover_sources",
]
