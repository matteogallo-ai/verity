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
`VectorStore` (pgvector: dense + FTS/BM25) → `Retriever` runs both → `Fusion` (RRF)
→ `Reranker` (local cross-encoder). Hybrid retrieval fuses dense recall with sparse
precision; RRF avoids score-scale mismatch between the two; the reranker tightens the
top-k that the agent actually reasons over. Every intermediate list is traced so retrieval
regressions localise to a stage.

### 3. Agentic RAG (`verity.agent`)
`QuestionDecomposer → (multi-step retrieve) → synthesize with citations → ConfidenceScorer`.
Simple questions pass straight through (single sub-question); complex ones decompose,
retrieve per sub-question, and combine evidence. Synthesis emits inline citations mapped to
exact passages. The confidence scorer owns the **refusal** decision. Decomposition, the
judge, and confidence prompts are authored in PromptLang (ADR 0004).

### 4. LLM runtime (`verity.llm`)
`LLMClient` per provider behind a `RoutingClient` with ordered fallback. Fallback on
transient/availability errors; **short-circuit on authentication errors** (a bad key is a
config bug, not a fallback trigger) — same decision as PromptLang's RoutingClient.

### 5. Observability (`verity.observability`)
Every stage opens a `StageSpan` bound to the query's `trace_id`: latency, token/cost usage,
hit counts and scores. Exported via OpenTelemetry; the same data backs `/metrics` and the
dashboard. Logs are structured (structlog) and carry `trace_id` on every line.

### 6. Evaluation (`verity.eval`)
Runs the whole agent over the labeled dataset and emits a `Scorecard` persisted with the
git SHA. Deterministic in CI (local embeddings + stubbed judge); real judge out-of-band for
the headline scorecard. `RunStore` powers regression tracking.

## The seams that matter

| Concern | Isolated behind | Why |
|---|---|---|
| Vector DB choice | `VectorStore` | swap pgvector→Qdrant without touching the pipeline (ADR 0002) |
| Prompt authoring | `verity.llm` adapter | swap PromptLang→templating if it ever blocks (ADR 0004) |
| Provider choice | `LLMClient` / `RoutingClient` | add/remove providers without agent changes |
| Refusal policy | `ConfidenceScorer` | tune calibration in one place, measured in eval |

## Data flow (one query)

```
question
  └─ tracer.stage("decompose") ─ QuestionDecomposer → [sub-questions]
       └─ per sub-question:
            tracer.stage("retrieve") ─ Retriever → dense+sparse → RRF → rerank → hits
       └─ tracer.stage("synthesize") ─ LLM → draft answer + citations
       └─ tracer.stage("confidence") ─ ConfidenceScorer → Confidence
            ├─ score ≥ threshold → Answer(text, citations, hits_used, usage)
            └─ score <  threshold → Answer(refused=True, honest "I don't know", rationale)
```

Every stage contributes to `usage` (tokens, cost, latency) and to the trace, so the same
run yields the user's answer, the observability data, and the eval inputs.
