"""End-to-end ingestion: discover sources → parse → chunk → embed → upsert.

The pipeline is intentionally small — it wires the stage protocols together and does
no heavy lifting itself. That is the whole point of the protocols: this file has no
knowledge of docling, sentence-transformers, or Postgres. Swap any of them for a fake
and the pipeline runs the same way.

Errors are handled per-source, not per-batch: one bad PDF must not sink an ingestion
run over a directory of 500 documents. Failures are logged with structured context
and the pipeline continues.
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
from verity.types import Chunk, Document

log = structlog.get_logger(__name__)

_SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf", ".docx"}


class VectorSink(Protocol):
    """Narrow write-side of the VectorStore the pipeline actually depends on."""

    async def upsert_documents(self, documents: Sequence[Document]) -> None: ...

    async def upsert(self, chunks: list) -> None: ...  # type: ignore[type-arg]


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
                embedded = await self._embedder.embed_chunks(chunks)
            except Exception as exc:
                log.error("embed_failed", uri=uri, document_id=str(doc.id), error=str(exc))
                result.failures.append(IngestionFailure(uri=uri, error=f"embed: {exc}"))
                continue

            if self._sink is not None:
                try:
                    await self._sink.upsert_documents([doc])
                    await self._sink.upsert(embedded)
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


__all__ = [
    "IngestionFailure",
    "IngestionPipeline",
    "IngestionResult",
    "VectorSink",
    "discover_sources",
]
