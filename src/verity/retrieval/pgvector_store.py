"""Postgres + pgvector implementation of the :class:`VectorStore` protocol.

Only the primitives live here: upsert, dense_search, sparse_search, get_chunks.
Fusion, rerank, and the hybrid retriever land in S2 — this module deliberately does
none of that.

Concurrency model: one shared :class:`psycopg.AsyncConnection` per store instance is
fine at the throughput of an ingestion CLI + an evaluation loop; if that stops being
true, swap in :class:`psycopg_pool.AsyncConnectionPool` behind the same interface
without touching callers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from verity.config import get_settings
from verity.types import Chunk, EmbeddedChunk, RetrievalHit, RetrieverKind


def _vector_literal(embedding: Sequence[float]) -> str:
    """pgvector accepts a text literal of the form ``'[0.1,0.2,...]'``."""
    return "[" + ",".join(f"{x:.7f}" for x in embedding) + "]"


class PgVectorStore:
    """Async pgvector-backed store. Instantiate with the DB URL from settings."""

    def __init__(self, database_url: str | None = None) -> None:
        self._url = database_url or get_settings().database_url

    async def _connect(self) -> psycopg.AsyncConnection[dict[str, Any]]:
        conn = await psycopg.AsyncConnection.connect(self._url)
        return cast(psycopg.AsyncConnection[dict[str, Any]], conn)

    async def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        """Insert or update chunks + parent documents in a single transaction.

        The document row is upserted first (so the FK on chunks resolves) using the
        parent metadata carried on each chunk's parent; since a Chunk doesn't hold
        the Document reference at runtime, callers hand us EmbeddedChunk plus, in
        the pipeline, a ``upsert_documents`` step. Here we assume ``documents`` are
        already present — see :meth:`upsert_documents`.
        """
        if not chunks:
            return
        conn = await self._connect()
        try:
            async with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
                for embedded in chunks:
                    c = embedded.chunk
                    await cur.execute(
                        """
                        INSERT INTO chunks (
                            id, document_id, ordinal, char_start, char_end,
                            section, page, text, metadata, embedding, embedding_model
                        )
                        VALUES (
                            %(id)s, %(document_id)s, %(ordinal)s, %(char_start)s, %(char_end)s,
                            %(section)s, %(page)s, %(text)s, %(metadata)s,
                            %(embedding)s::vector, %(embedding_model)s
                        )
                        ON CONFLICT (id) DO UPDATE SET
                            document_id = EXCLUDED.document_id,
                            ordinal = EXCLUDED.ordinal,
                            char_start = EXCLUDED.char_start,
                            char_end = EXCLUDED.char_end,
                            section = EXCLUDED.section,
                            page = EXCLUDED.page,
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding,
                            embedding_model = EXCLUDED.embedding_model
                        """,
                        {
                            "id": str(c.id),
                            "document_id": str(c.document_id),
                            "ordinal": c.ordinal,
                            "char_start": c.char_start,
                            "char_end": c.char_end,
                            "section": c.section,
                            "page": c.page,
                            "text": c.text,
                            "metadata": json.dumps(c.metadata),
                            "embedding": _vector_literal(embedded.embedding),
                            "embedding_model": embedded.model,
                        },
                    )
        finally:
            await conn.close()

    async def upsert_documents(self, documents: Sequence[Any]) -> None:
        """Upsert ``Document`` rows. Kept off the ``VectorStore`` protocol on purpose
        so the retrieval-side API stays chunk-centric; the pipeline calls this
        before upserting chunks."""
        if not documents:
            return
        conn = await self._connect()
        try:
            async with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
                for doc in documents:
                    await cur.execute(
                        """
                        INSERT INTO documents (
                            id, source_type, uri, title, text, metadata, ingested_at
                        )
                        VALUES (
                            %(id)s, %(source_type)s, %(uri)s, %(title)s,
                            %(text)s, %(metadata)s, %(ingested_at)s
                        )
                        ON CONFLICT (id) DO UPDATE SET
                            source_type = EXCLUDED.source_type,
                            uri = EXCLUDED.uri,
                            title = EXCLUDED.title,
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata,
                            ingested_at = EXCLUDED.ingested_at
                        """,
                        {
                            "id": str(doc.id),
                            "source_type": doc.source_type.value,
                            "uri": doc.uri,
                            "title": doc.title,
                            "text": doc.text,
                            "metadata": json.dumps(doc.metadata),
                            "ingested_at": doc.ingested_at,
                        },
                    )
        finally:
            await conn.close()

    async def dense_search(self, query_embedding: tuple[float, ...], k: int) -> list[RetrievalHit]:
        """Cosine similarity via pgvector's ``<=>`` operator (distance, ascending)."""
        if k <= 0:
            return []
        conn = await self._connect()
        try:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT id, document_id, ordinal, char_start, char_end,
                           section, page, text, metadata,
                           (embedding <=> %(qv)s::vector) AS distance
                      FROM chunks
                     WHERE embedding IS NOT NULL
                     ORDER BY embedding <=> %(qv)s::vector
                     LIMIT %(k)s
                    """,
                    {"qv": _vector_literal(query_embedding), "k": k},
                )
                rows = await cur.fetchall()
                return [
                    RetrievalHit(
                        chunk=_row_to_chunk(row),
                        score=1.0 - float(row["distance"]),
                        kind=RetrieverKind.DENSE,
                        rank=i,
                    )
                    for i, row in enumerate(rows)
                ]
        finally:
            await conn.close()

    async def sparse_search(self, query: str, k: int) -> list[RetrievalHit]:
        """FTS via ``websearch_to_tsquery`` + ``ts_rank`` over the generated tsvector."""
        if k <= 0 or not query.strip():
            return []
        conn = await self._connect()
        try:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT id, document_id, ordinal, char_start, char_end,
                           section, page, text, metadata,
                           ts_rank(text_tsv, websearch_to_tsquery('english', %(q)s)) AS score
                      FROM chunks
                     WHERE text_tsv @@ websearch_to_tsquery('english', %(q)s)
                     ORDER BY score DESC
                     LIMIT %(k)s
                    """,
                    {"q": query, "k": k},
                )
                rows = await cur.fetchall()
                return [
                    RetrievalHit(
                        chunk=_row_to_chunk(row),
                        score=float(row["score"]),
                        kind=RetrieverKind.SPARSE,
                        rank=i,
                    )
                    for i, row in enumerate(rows)
                ]
        finally:
            await conn.close()

    async def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """Hydrate ``Chunk``s in the requested order (missing ids are skipped)."""
        if not chunk_ids:
            return []
        conn = await self._connect()
        try:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT id, document_id, ordinal, char_start, char_end,
                           section, page, text, metadata
                      FROM chunks
                     WHERE id = ANY(%(ids)s::uuid[])
                    """,
                    {"ids": chunk_ids},
                )
                rows = await cur.fetchall()
                by_id = {str(row["id"]): _row_to_chunk(row) for row in rows}
                return [by_id[cid] for cid in chunk_ids if cid in by_id]
        finally:
            await conn.close()

    async def count_chunks(self) -> int:
        """Debug/observability helper: how many chunks are currently stored."""
        conn = await self._connect()
        try:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("SELECT COUNT(*) AS n FROM chunks")
                row = await cur.fetchone()
                return int(row["n"]) if row else 0
        finally:
            await conn.close()


def _row_to_chunk(row: dict[str, Any]) -> Chunk:
    metadata_raw = row.get("metadata") or {}
    metadata: dict[str, str]
    if isinstance(metadata_raw, str):
        metadata = json.loads(metadata_raw)
    else:
        metadata = {k: str(v) for k, v in metadata_raw.items()}
    return Chunk(
        id=UUID(str(row["id"])),
        document_id=UUID(str(row["document_id"])),
        text=row["text"],
        ordinal=int(row["ordinal"]),
        char_start=int(row["char_start"]),
        char_end=int(row["char_end"]),
        section=row.get("section"),
        page=row.get("page"),
        metadata=metadata,
    )


__all__ = ["PgVectorStore"]
