"""Postgres + pgvector implementation of the :class:`VectorStore` protocol.

Only the primitives live here: upsert, dense_search, sparse_search, get_chunks.
Fusion and rerank live in :mod:`verity.retrieval.hybrid`.

Connection model: every operation opens its own :class:`psycopg.AsyncConnection` and
closes it on return. That is enough for the ingestion CLI and the retrieval
scorecard; under concurrent load, a :class:`psycopg_pool.AsyncConnectionPool` behind
``_connect`` would avoid one connection setup per call without touching callers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast
from uuid import UUID

import psycopg
from psycopg.rows import dict_row, tuple_row

from verity.config import get_settings
from verity.db.redact import redact_conninfo
from verity.retrieval.base import DocumentSummary, StoreSchemaError, StoreUnavailableError
from verity.types import Chunk, EmbeddedChunk, RetrievalHit, RetrieverKind


def _vector_literal(embedding: Sequence[float]) -> str:
    """pgvector accepts a text literal of the form ``'[0.1,0.2,...]'``."""
    return "[" + ",".join(f"{x:.7f}" for x in embedding) + "]"


async def _check_embedding_dim(
    conn: psycopg.AsyncConnection[dict[str, Any]], expected: int
) -> None:
    """Compare ``expected`` with the declared dimension of ``chunks.embedding``.

    For a pgvector ``VECTOR(n)`` column, ``pg_attribute.atttypmod`` holds ``n``.
    The query runs in its own transaction block: a bare SELECT would open an
    implicit transaction, turning the caller's ``conn.transaction()`` into a
    savepoint that is rolled back when the connection closes.
    """
    async with conn.transaction(), conn.cursor(row_factory=tuple_row) as cur:
        await cur.execute(
            """
            SELECT a.atttypmod
              FROM pg_attribute a
             WHERE a.attrelid = to_regclass('chunks')
               AND a.attname = 'embedding'
               AND NOT a.attisdropped
            """
        )
        row = await cur.fetchone()
    if row is None:
        raise SchemaNotMigratedError(
            "table chunks (or its embedding column) not found; run `verity db upgrade`"
        )
    declared = int(row[0])
    if declared != expected:
        raise EmbeddingDimensionMismatchError(
            f"configured embedding_dim={expected} but chunks.embedding is VECTOR({declared}). "
            "Change VERITY_EMBEDDING_DIM / the embedding model, or add a migration that "
            "alters the column and re-embeds the corpus."
        )


class EmbeddingDimensionMismatchError(StoreSchemaError):
    """The configured embedding dimension does not fit the ``chunks.embedding`` column."""


class SchemaNotMigratedError(StoreSchemaError):
    """The ``chunks`` table does not exist yet."""


class PgVectorStore:
    """Async pgvector-backed store. Instantiate with the DB URL from settings.

    The ``chunks.embedding`` column has a fixed dimension in the migration
    (``VECTOR(384)``), while ``embedding_dim`` is configurable. The first operation
    on a store reads the column's declared dimension and raises
    :class:`EmbeddingDimensionMismatchError` if they differ, so a changed embedding
    model fails on the first connection with an actionable message instead of a
    driver error on insert. The check is retried on each connection until it
    passes; ``IngestionPipeline`` lets it propagate, so ``verity ingest`` stops at
    the first document.
    """

    def __init__(
        self, database_url: str | None = None, *, embedding_dim: int | None = None
    ) -> None:
        settings = get_settings()
        self._url = database_url or settings.database_url.get_secret_value()
        self._embedding_dim = embedding_dim if embedding_dim is not None else settings.embedding_dim
        self._schema_checked = False

    async def _connect(self) -> psycopg.AsyncConnection[dict[str, Any]]:
        try:
            conn = cast(
                psycopg.AsyncConnection[dict[str, Any]],
                await psycopg.AsyncConnection.connect(self._url),
            )
        # ProgrammingError is a connection string libpq cannot parse; its message
        # quotes the offending fragment, which can be the password. The driver error
        # is not chained (``from None``) so no traceback renders its raw message.
        except psycopg.Error as exc:
            reason = redact_conninfo(str(exc).strip(), self._url)
            raise StoreUnavailableError(
                f"cannot connect to Postgres: {type(exc).__name__}: {reason}"
            ) from None
        if not self._schema_checked:
            try:
                await _check_embedding_dim(conn, self._embedding_dim)
            # BaseException so cancellation also closes the connection before
            # propagating; nothing is swallowed.
            except BaseException:
                await conn.close()
                raise
            self._schema_checked = True
        return conn

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

    async def list_documents(self) -> list[DocumentSummary]:
        conn = await self._connect()
        try:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT d.id, d.uri, d.title, COUNT(c.id) AS n_chunks
                      FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
                     GROUP BY d.id
                     ORDER BY d.uri
                    """
                )
                rows = await cur.fetchall()
        finally:
            await conn.close()
        return [
            DocumentSummary(
                document_id=str(row["id"]),
                uri=row["uri"],
                title=row["title"],
                n_chunks=int(row["n_chunks"]),
            )
            for row in rows
        ]

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
        """Full-text search: any of the query's English lexemes, ranked by ``ts_rank``.

        ``websearch_to_tsquery`` / ``plainto_tsquery`` AND every term, so a natural
        question ("What is the notice period for ...?") matched no chunk at all for 14
        of the 19 eval questions. OR-ing the lexemes, as BM25 does, lets ``ts_rank``
        order partial matches. The cast re-parses the normalised lexemes as-is.
        """
        if k <= 0 or not query.strip():
            return []
        conn = await self._connect()
        try:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    WITH q AS (
                        SELECT replace(plainto_tsquery('english', %(q)s)::text, '&', '|')
                               ::tsquery AS tsq
                    )
                    SELECT id, document_id, ordinal, char_start, char_end,
                           section, page, text, metadata,
                           ts_rank(text_tsv, q.tsq) AS score
                      FROM chunks, q
                     WHERE text_tsv @@ q.tsq
                     ORDER BY score DESC, id
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


__all__ = [
    "EmbeddingDimensionMismatchError",
    "PgVectorStore",
    "SchemaNotMigratedError",
]
