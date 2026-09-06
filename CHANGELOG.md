# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] — 2026-09-06

### Added
- **S3 — agentic RAG with citations + calibrated refusal.**
  - **LLM runtime** (`verity.llm`): `AnthropicClient`, `OpenAIClient` (vendor
    errors translated to `AuthenticationError` / `ProviderUnavailableError`),
    `StubLLMClient` (scripted, offline, deterministic). Thin `.prompt` adapter
    (per ADR 0004) parses the shipped PromptLang sources and renders them into
    `Message` lists with strict variable substitution.
  - **`MultiProviderRoutingClient`**: honours `settings.provider_order`.
    `ProviderUnavailableError` triggers ordered fallback; **`AuthenticationError`
    short-circuits** (no silent fallback across a bad key — audit-trail intent,
    covered by a dedicated test).
  - **Agent stack** (`verity.agent`):
    - `LLMQuestionDecomposer` — parses JSON list, degrades to `[question]` on
      malformed output, cap at `settings.max_agent_steps`.
    - `CitedSynthesizer` — draft answer + validated `Citation`s. Every citation's
      `quote` is verified to be a verbatim substring of the referenced chunk and
      the offsets are re-anchored to `document.text` (`chunk.char_start + pos`).
      Hallucinated citations are dropped.
    - `LLMConfidenceScorer` — calibrated confidence + refusal against
      `settings.confidence_threshold`; malformed scorer output forces refusal
      (fail-open on principle).
    - `RagAgent` orchestrator: decompose → per-sub-q retrieve → aggregate hits →
      synthesize → score → refuse-or-ship. Refusal is a first-class output
      (empty citations, honest "I don't know" text, rationale, hits_used
      preserved for traceability).
  - **`prompts/synthesize.prompt`** — new PromptLang source for cited-answer
    synthesis (JSON schema `{answer, citations:[{quote, chunk_index}]}`).
  - **CLI `verity ask`** — question → cited answer + confidence + rationale +
    usage table. `--stub` flag runs the scripted responder end-to-end without
    keys (same script the integration test asserts against).
- **README**: one line under "Measured results" pointing to
  `datasets/eval/runs/` so reviewers can find the actual retrieval scorecards
  without waiting for S4.

### Notes
- No faithfulness / refusal numbers are published in this release — that
  scorecard is S4. The retrieval scorecard from 0.3.0 remains the only
  measured chiffre in the repo.

## [0.3.0] — 2026-09-06

### Added
- **S2 — hybrid retrieval + first measured scorecard.**
  - `ReciprocalRankFusion` (Fusion protocol): score = Σ 1/(rrf_k + rank_i),
    dedup by `chunk.id`, deterministic tie-break on chunk id for
    bit-reproducible fusion output.
  - `CrossEncoderReranker` (Reranker protocol): local
    `cross-encoder/ms-marco-MiniLM-L-6-v2`, batched, async via
    `asyncio.to_thread`, backend factory injectable (unit lane never loads torch).
  - `HybridRetriever` (Retriever protocol): embed query → dense ∥ sparse
    (`asyncio.gather`) → RRF → cross-encoder rerank. Structured `stage`
    logs per phase (`dense`/`sparse`/`fused`/`reranked` + top score + count).
    Robustness: a stage error or empty sparse degrades to `[]` without crashing.
  - `InMemoryVectorStore` — second implementation of the `VectorStore` protocol,
    backed by numpy cosine + `rank_bm25`. Makes the unit lane exercise the full
    hybrid pipeline offline and lets `verity eval retrieval --in-memory`
    produce a scorecard on any laptop with no Postgres.
  - Retrieval metrics: `precision_at_k`, `recall_at_k`, `ndcg_at_k` with binary
    gain + macro-averaging (`BinaryRetrievalMetric` implements the
    `RetrievalMetric` protocol from `eval.base`). Undefined-on-empty-gold cases
    raise explicitly rather than silently returning NaN.
  - CLI:
    - `verity retrieve "…"` — one-shot retrieval, Rich table of top hits.
    - `verity eval retrieval` — real measured retrieval scorecard tagged to
      the current git SHA, persisted as JSON under `datasets/eval/runs/`.
      Supports `--floor-precision / --floor-recall / --floor-ndcg` for CI
      anti-regression gates.
  - CI integration lane adds `verity eval retrieval --no-in-memory
    --floor-ndcg 0.5 --floor-recall 0.75` as a hard gate.

### Measured retrieval scorecard (in-memory backend, git SHA 82ebaa5, k=8, 4 answerable questions)
- precision@8: **0.156** (capped by k > gold size; 1-2 gold chunks / 8 slots)
- recall@8: **1.000**
- nDCG@8: **0.877**
- per-example nDCG: q-001 0.631 · q-002 0.877 · q-003 1.000 · q-005 1.000

The pgvector run will produce equivalent numbers (same embedder, same
reranker, same fusion) — will be re-verified in the first CI run on 0.3.0.

## [0.2.0] — 2026-09-05

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
