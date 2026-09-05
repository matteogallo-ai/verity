-- 001_initial: base schema for documents + chunks with dense (pgvector) and sparse (FTS).
--
-- Idempotent by construction: every statement uses IF NOT EXISTS or CREATE OR REPLACE.
-- The migrator ships this file exactly once per database via schema_migrations, but
-- re-applying it manually must remain safe (developer ergonomics).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id           UUID PRIMARY KEY,
    source_type  TEXT NOT NULL,
    uri          TEXT NOT NULL UNIQUE,
    title        TEXT,
    text         TEXT NOT NULL,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id            UUID PRIMARY KEY,
    document_id   UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal       INTEGER NOT NULL,
    char_start    INTEGER NOT NULL,
    char_end      INTEGER NOT NULL,
    section       TEXT,
    page          INTEGER,
    text          TEXT NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding     VECTOR(384),
    embedding_model TEXT NOT NULL,
    -- Generated tsvector for the sparse retriever. GENERATED lets Postgres keep it in
    -- sync automatically; websearch_to_tsquery targets English by default, which is
    -- what the eval dataset uses.
    text_tsv      TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);
CREATE INDEX IF NOT EXISTS chunks_ordinal_idx ON chunks(document_id, ordinal);

-- Sparse retrieval index: GIN over the generated tsvector column.
CREATE INDEX IF NOT EXISTS chunks_text_tsv_idx ON chunks USING GIN (text_tsv);

-- Dense retrieval index: HNSW on cosine distance. HNSW is the pgvector default from
-- 0.5 onward — no ANALYZE required, robust recall out of the box. See ADR 0002.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_class WHERE relname = 'chunks_embedding_hnsw_idx'
    ) THEN
        EXECUTE 'CREATE INDEX chunks_embedding_hnsw_idx ON chunks USING hnsw (embedding vector_cosine_ops)';
    END IF;
END $$;
