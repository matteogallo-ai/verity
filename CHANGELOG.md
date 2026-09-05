# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- **Portable IDs (S1.1).** `document_id_for_uri` now normalises its input through a
  new `identity_for` helper: local file paths are hashed as their POSIX path
  *relative to the repo root* (e.g. `datasets/corpus/msa.md`), not as absolute
  machine-specific paths. `Document.uri` still stores the absolute path for I/O and
  provenance. Consequence: `datasets/eval/questions.jsonl` chunk ids are now stable
  across machines and CI runs (regenerated via `scripts/build_eval_labels.py`).

### Added
- `tests/integration/test_eval_labels_integrity.py`: asserts every
  `relevant_chunk_id` resolves to a chunk containing the expected anchor phrase,
  and that unanswerable questions keep an empty label set. Would have caught the
  non-portable id bug on the first CI run.
- Chunker emits a structured `chunk_over_budget` WARNING when an emitted chunk
  exceeds `max_words` (defensive guard against silent embedding truncation).

## [0.2.0] — 2026-09-05

### Added
- **S1 — ingestion & indexing.**
  - Structure-aware `Parser` implementations: native TXT + Markdown (heading hierarchy
    preserved), docling for PDF/DOCX (headings, sections, pages captured), httpx +
    trafilatura for web pages. `SourceTypeParser` dispatches by URI/extension.
  - `StructureAwareChunker` packs paragraphs into ~512-token chunks along section
    boundaries with light overlap. Load-bearing invariant:
    `document.text[chunk.char_start:chunk.char_end] == chunk.text` (asserted on the
    example corpus).
  - Deterministic identifiers: `Document.id = uuid5(NAMESPACE_URL, uri)` and
    `Chunk.id = uuid5(NAMESPACE_URL, f"{document.id}:{ordinal}:{cs}:{ce}")`. Re-ingesting
    the same source produces the same ids — the vector store upsert is idempotent and
    eval labels stay stable.
  - `LocalEmbedder` around sentence-transformers (default `BAAI/bge-small-en-v1.5`,
    384-dim). Batched, wrapped in `asyncio.to_thread`, dim-checked against
    `config.embedding_dim`, backend factory is injectable so unit tests avoid model
    downloads. **The model weights download on the first run** — plan ~130 MB plus
    torch (~2 GB via `uv sync`).
  - `PgVectorStore` (pgvector): idempotent `upsert(chunks)` and `upsert_documents`,
    cosine `dense_search`, `websearch_to_tsquery` + `ts_rank` `sparse_search`,
    `get_chunks(ids)`. HNSW index on `embedding vector_cosine_ops`, GIN on the
    generated `tsvector`. Migrations live in `migrations/`, applied by a small runner
    (advisory-locked, per-file transactions) via `verity db upgrade`.
  - `verity ingest <path|dir|url>` wired end-to-end (parse → chunk → embed → upsert),
    with a Rich summary and continue-on-error per source (structured logs via
    structlog).
- **Example corpus + eval labels.** `datasets/corpus/msa.md` (30-day termination
  clause, $1M liability cap tied to Acme Corporation as data-processing entity,
  no-competitor assignment clause) and `datasets/corpus/financial_summary_fy2024.md`
  (revenue + YoY figure). `scripts/build_eval_labels.py` reruns the ingestion pipeline
  in dry-run mode and writes `relevant_chunk_ids` for q-001/002/003/005; q-004/006
  remain out-of-scope by design (see ADR 0003).
- **CI integration lane.** `.github/workflows/ci.yml` adds an `integration` job that
  starts a `pgvector/pgvector:pg16` service, runs `verity db upgrade`, and executes
  `pytest -m integration` — so the round-trip test runs on every push without a local
  Postgres. Unit lane (`-m "not integration"`) stays DB-free and free.

### Notes
- No fabricated eval numbers — the harness lands in S4. The `pending S4` placeholders
  in the README remain in place.
- Retriever/Fusion/Reranker are still S2. This release only implements the
  `VectorStore` primitives.

## [0.1.0] — S0 scaffold

### Added
- **S0 — scaffold.** Typed domain model (`verity.types`), stage protocols
  (ingestion, retrieval, llm, agent, eval, observability), typed settings,
  CLI surface (`verity`, `verity eval run/compare`), Docker Compose (app + pgvector),
  CI pipeline (lint · typecheck · test · eval gate), docs skeleton with ADRs,
  evaluation methodology, and a dataset specification including unanswerable questions.

_No implementation logic yet — S0 defines the contracts every later stage fills._

## Roadmap
- S1 ingestion & indexing · S2 hybrid retrieval + RRF + rerank · S3 agentic pipeline
  (decomposition, citations, confidence, refusal) · S4 eval harness + headline scorecard
  · S5 observability · S6 Next.js UI (v1.1) · S7 README + demo + `v1.0.0` tag.
