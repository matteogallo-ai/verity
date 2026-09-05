<div align="center">

# Verity

**An evaluated, observable RAG agent for enterprise documents.**
Retrieves, reasons over multiple steps, cites exact passages, and *refuses to answer
when the evidence is weak* — instead of hallucinating. Every answer is measured.

[![ci](https://github.com/matteogallo-ai/verity/actions/workflows/ci.yml/badge.svg)](https://github.com/matteogallo-ai/verity/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)
![python](https://img.shields.io/badge/python-3.12-blue.svg)

</div>

> **Status — S1 ingestion & indexing.** Parse (docling PDF/DOCX, native TXT/MD,
> httpx+trafilatura Web) · structure-aware chunker with load-bearing offset invariant ·
> deterministic ids · local embedder (sentence-transformers, offline) · pgvector store
> (dense + FTS) · migrations + `verity ingest`. Hybrid retriever/RRF/rerank land at S2.
> The scorecard below is still a **placeholder** — a real, git-SHA-tagged eval run
> lands at S4. No numbers here are fabricated in the meantime.

---

## The problem

Enterprise RAG demos answer confidently and are wrong just often enough to be dangerous.
The failure that matters is not "retrieval is imperfect" — it is **a fluent answer with
no grounding and no signal that it is unreliable**. For documents that drive real
decisions (contracts, filings, policies), a system that *knows when it doesn't know* is
worth more than one that always answers.

Verity treats that as an engineering problem, not a prompt trick: measure grounding,
measure refusal calibration, make both visible, and gate them in CI.

## Demo

<!-- 90-second demo GIF lands at S7. Script: docs/demo-script.md -->
_Demo GIF — S7._

## Measured results

<!-- Replaced at S4 by `verity eval run` output, tagged to a git SHA. -->

| Metric | Value | Notes |
|---|---|---|
| Retrieval precision@8 | _pending S4_ | on labeled dataset |
| Retrieval recall@8 | _pending S4_ | |
| Answer faithfulness | _pending S4_ | claim-level, LLM-judge |
| Citation accuracy | _pending S4_ | |
| Hallucination rate | _pending S4_ | ↓ better |
| Refusal precision / recall | _pending S4_ | the headline metric |
| Latency p50 / p99 | _pending S4_ | end-to-end |
| Cost / query | _pending S4_ | USD |

Methodology, including how refusal is scored and why the dataset contains unanswerable
questions, is in [`docs/evaluation-methodology.md`](docs/evaluation-methodology.md).

## Architecture

```
            ┌──────────────┐
 docs ─────▶│  Ingestion   │ parse (structure-aware) → semantic chunk → embed
            └──────┬───────┘
                   ▼
            ┌──────────────┐   dense (pgvector) ┐
 query ────▶│  Retrieval   │   sparse (BM25/FTS)├─▶ RRF fusion ─▶ cross-encoder rerank
            └──────┬───────┘                    ┘
                   ▼
            ┌──────────────┐  decompose → multi-step retrieve → synthesize with citations
            │ Agentic RAG  │  → confidence scoring → REFUSE if weak
            └──────┬───────┘  (orchestration prompts authored in PromptLang)
                   ▼
            ┌──────────────┐
            │ Answer +     │  inline citations · confidence · chunks used · cost · latency
            │ observability│  OTel traces per stage · /metrics · structured logs
            └──────────────┘
                   ▲
            ┌──────────────┐
            │ Eval harness │  retrieval P/R/nDCG · faithfulness · refusal calibration
            └──────────────┘  · scorecard tagged to git SHA · regression tracking
```

Full diagram and rationale: [`docs/architecture.md`](docs/architecture.md).
Key decisions are recorded as ADRs in [`docs/adr/`](docs/adr/).

## Stack

Python 3.12 · FastAPI · pgvector (single stateful service: vectors + FTS + eval history)
· local embeddings + cross-encoder reranker (offline, no keys) · multi-provider LLM
runtime (Anthropic / OpenAI / local) with authentication-aware fallback · OpenTelemetry
+ structlog · Next.js UI (v1.1). Tooling: uv · ruff · mypy strict · pytest · GitHub Actions.

## Quickstart

```bash
git clone https://github.com/matteogallo-ai/verity && cd verity
cp .env.example .env                       # keys optional — retrieval + eval run offline
docker compose up -d db                    # pgvector
uv sync --dev                              # ~2 GB (torch + docling) on first sync
uv run verity db upgrade                   # apply migrations
uv run verity ingest datasets/corpus       # parse → chunk → embed → upsert
uv run verity eval run --ci                # deterministic eval (local embeddings, stubbed judge)
```

The first ingestion run downloads the default embedding model
(`BAAI/bge-small-en-v1.5`, ~130 MB) into the Hugging Face cache — subsequent runs are
fully offline.

Full local run (app + db) once S5/S6 land: `docker compose up`.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest -m "not integration"
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under [MIT](LICENSE).
