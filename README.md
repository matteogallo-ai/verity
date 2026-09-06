<div align="center">

# Verity

**An evaluated, observable RAG agent for enterprise documents.**
Retrieves, reasons over multiple steps, cites exact passages, and *refuses to answer
when the evidence is weak* — instead of hallucinating. Every answer is measured.

[![ci](https://github.com/matteogallo-ai/verity/actions/workflows/ci.yml/badge.svg)](https://github.com/matteogallo-ai/verity/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)
![python](https://img.shields.io/badge/python-3.12-blue.svg)

</div>

> **Status — S4 eval harness.** Full pipeline live: ingestion (S1) · hybrid
> retrieval + RRF + rerank (S2) · agentic RAG with citations + calibrated refusal
> (S3) · eval harness with retrieval + refusal + answer-quality tracks (S4).
> Retrieval numbers below are real and reproducible. Refusal chiffres are
> mechanically deterministic but were measured on the scripted stub agent used
> in CI — the real-agent refusal calibration lands with the `--judge live` run
> (pending). Faithfulness / citation accuracy / hallucination rate stay _pending
> real judge_ until that same live run.

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

Numbers below come from `verity eval run`, tagged to a git SHA, on the 16-question
labeled dataset. **Every row carries the provenance it was measured under** — a
number is only as trustworthy as the components that produced it.

- **Retrieval** — real, deterministic. Local embeddings + local reranker, no LLM
  in the loop. Reproducible bit-for-bit.
- **Refusal calibration** — the *maths* is deterministic (a confusion matrix
  built from the `answerable` labels and the observed refuses), but the observed
  refuses come from whichever agent produced the answers. The scorecard below was
  measured on `agent=stub-agent` (the scripted stub used for offline CI), so it
  reflects the **stub's** refusal calibration — not Verity's real behaviour under
  a live LLM. A `--judge live` run against a real agent lands the true refusal
  chiffres alongside faithfulness.
- **Answer quality** (faithfulness / citation accuracy / hallucination rate) —
  the shipped scorecard uses `judge_model=stub-judge-v1`, a mechanical judge, so
  these three stay marked _pending real judge_.

| Metric | Value | Provenance |
|---|---|---|
| Retrieval precision@8 | **0.148** | REAL · on 11 answerable questions (gold cap 1-2 chunks / 8 slots) |
| Retrieval recall@8 | **1.000** | REAL · every gold chunk retrieved |
| Retrieval nDCG@8 | **0.950** | REAL · macro-averaged ranking quality |
| Refusal precision | 1.000 | REAL calc · **agent=stub-agent** (not real Verity) |
| Refusal recall | 1.000 | REAL calc · **agent=stub-agent** · 5/5 out-of-scope refused incl. 3 near-miss |
| Answer faithfulness | _pending real judge_ | (stub judge scorecard shows 1.000, not published) |
| Citation accuracy | _pending real judge_ | |
| Hallucination rate | _pending real judge_ | ↓ better |
| Latency p50 / p95 / p99 | 84 / 1477 / 4784 ms | REAL · in-memory backend, stub LLM |
| Cost / query | $0.00 | stub LLM; real cost measured on live-judge run |

Retrieval is a real, standalone chiffre. Refusal — including the 5/5 near-miss
result — measures how a *deterministic scripted agent* behaves; the real-agent
refusal calibration lands with the `--judge live` run (pending). Reproduce with
`uv run verity eval run --judge stub`. Persisted scorecards under
[`datasets/eval/runs/`](datasets/eval/runs/) carry the full provenance triple
(`embedding_model`, `agent_model`, `judge_model`).

Methodology — how refusal is scored and why the dataset contains unanswerable +
near-miss questions — is in
[`docs/evaluation-methodology.md`](docs/evaluation-methodology.md).

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
