# Architecture

Verity is a staged pipeline where every stage is a typed protocol (`verity/*/base.py`)
speaking the domain model in `verity/types.py`. Nothing crosses a stage boundary except
those frozen types, which is what makes each stage independently testable and each
metric independently attributable.

## Stages

### 1. Ingestion (`verity.ingestion`)
`Parser → Chunker → Embedder`.
Structure-aware parsing (docling) preserves headings/sections so chunking follows semantic
boundaries rather than a fixed window. Chunks carry `char_start`/`char_end` offsets into
the parent document — the substrate for exact-passage citations. Embedding is a separate
protocol so chunking has no dependency on the embedding model (and CI can embed locally).

### 2. Retrieval (`verity.retrieval`)
`VectorStore` → `HybridRetriever` runs dense and sparse search → `Fusion` (RRF, 0-based
ranks) → `Reranker` (local cross-encoder). RRF avoids the score-scale mismatch between
dense and sparse; the reranker orders the top-k the agent reasons over. Each stage is a
traced span (latency, hit count).

Two `VectorStore` implementations exist:
- `InMemoryVectorStore`: cosine dense search + BM25 (`rank-bm25`). Used by `verity ask`,
  `verity eval run` and the API (`verity serve`) by default; the corpus is ingested at
  start-up.
- `PgVectorStore`: pgvector HNSW dense search + Postgres full-text search (`ts_rank`).
  Written by `verity ingest`, read by `verity retrieve --no-in-memory` and
  `verity eval retrieval --no-in-memory`, and exercised by the integration test lane and the CI retrieval
  gate. BM25 and `ts_rank` are different sparse scorers; their rankings have not been
  compared.

### 3. Agentic RAG (`verity.agent`)
`QuestionDecomposer → (multi-step retrieve) → synthesize with citations → ConfidenceScorer`.
Simple questions pass straight through (single sub-question); complex ones decompose,
retrieve per sub-question, and combine evidence. Synthesis emits inline citations mapped to
exact passages. The confidence scorer owns the **refusal** decision. The prompts live in
`prompts/*.prompt`, read by a local parser (`verity.llm.prompt`); ADR 0004 describes a
PromptLang integration that the code does not implement.

### 4. LLM runtime (`verity.llm`)
`LLMClient` per provider behind a `RoutingClient` with ordered fallback. Fallback on
transient/availability errors; **short-circuit on authentication errors** (a bad key is a
config bug, not a fallback trigger) — same decision as PromptLang's RoutingClient.

### 5. Observability (`verity.observability`)
Every stage opens a `StageSpan` bound to the query's `trace_id`: latency, token/cost usage,
hit counts and scores. Spans go to an in-process OpenTelemetry `InMemorySpanExporter` that
backs `/metrics` and the dashboard. OTLP export is attempted only when
`VERITY_OTEL_ENDPOINT` is set AND `opentelemetry-exporter-otlp-proto-http` is installed;
that package is not a declared dependency, and when it is missing the endpoint is ignored
without a warning. Logs are structured (structlog) and carry `trace_id` on every line.

### 6. Evaluation (`verity.eval`)
Runs the whole agent over the labeled dataset and emits a `Scorecard` saved as JSON with
the git SHA. In CI the agent and the judge are scripted stubs (a pipeline smoke test); the
headline scorecard comes from one live run. `FileRunStore` keeps runs as JSON files for
`verity eval compare`. Definitions: `docs/evaluation-methodology.md`.

## The seams that matter

| Concern | Isolated behind | Why |
|---|---|---|
| Vector DB choice | `VectorStore` | swap pgvector→Qdrant without touching the pipeline (ADR 0002) |
| Prompt files | `verity.llm.prompt` | callers depend on `PromptTemplate.render` only |
| Provider choice | `LLMClient` / `RoutingClient` | add/remove providers without agent changes |
| Refusal policy | `ConfidenceScorer` | one threshold in one place, measured by the refusal metrics |

## Data flow (one query)

```
question
  └─ tracer.stage("decompose") ─ QuestionDecomposer → [sub-questions]
       └─ per sub-question:
            tracer.stage("retrieve") ─ Retriever → dense+sparse → RRF → rerank → hits
       └─ tracer.stage("synthesize") ─ LLM → draft answer + citations
       └─ tracer.stage("confidence") ─ ConfidenceScorer → Confidence
            ├─ score ≥ threshold and not refused → Answer(text, citations, hits_used, usage)
            └─ score <  threshold, or scorer refuses → Answer(refused=True, "I don't know", rationale)
```

Every stage contributes to `usage` (tokens, cost, latency) and to the trace, so the same
run yields the user's answer, the observability data, and the eval inputs.
