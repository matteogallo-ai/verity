# Migrations

Numbered SQL files applied in lexical order by `verity db upgrade`
(`src/verity/db/migrator.py`). Each applied file is recorded in `schema_migrations`
with its sha256. Applied files are never edited; a schema change is a new file.

## Embedding dimension

`001_initial.sql` declares `chunks.embedding VECTOR(384)`, the output size of the
default embedding model `BAAI/bge-small-en-v1.5`. The dimension is fixed in SQL
because pgvector needs it to build the HNSW index.

`VERITY_EMBEDDING_DIM` stays configurable, so the two can disagree. `PgVectorStore`
compares them on its first database operation and raises
`EmbeddingDimensionMismatchError` when they differ
(`tests/integration/test_pgvector_store.py::test_store_refuses_an_embedding_dim_the_schema_cannot_hold`).

To change the embedding model to one with a different dimension: add a migration
that drops the HNSW index, alters the column to `VECTOR(<n>)`, and recreates the
index; then re-ingest the corpus, because stored vectors cannot be converted.
