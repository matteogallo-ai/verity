<div align="center">

# Verity

**A RAG agent over enterprise documents that refuses when its evidence is weak, and
publishes how often it gets that right.**
Cited answers with verbatim quotes. Threshold-gated refusal on out-of-scope questions.
Every published number traceable to the run artefact that produced it.

[![ci](https://github.com/matteogallo-ai/verity/actions/workflows/ci.yml/badge.svg)](https://github.com/matteogallo-ai/verity/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)
![python](https://img.shields.io/badge/python-3.12-blue.svg)

</div>

## The problem

Enterprise RAG demos answer confidently and are wrong just often enough to be
dangerous: a fluent, cited-looking answer with no real grounding is the failure that
matters for contracts, filings and policies.

Verity drafts an answer, asks a scorer model how well the evidence supports it, and
refuses below a threshold. It publishes faithfulness, citation accuracy,
hallucination rate and refusal precision / recall from one live run whose
per-question audit trail is committed next to the code.

## What it looks like

**Cited answer**: click a citation and the source pane highlights the exact clause.
The confidence readout shows the scorer's self-reported score, the rerank scores and
the evidence hits. A provenance badge names the model that answered.

![Cited answer with source panel + confidence readout](docs/assets/screenshot-answer.png)

**Refusal**: for questions the corpus does not cover, the agent returns *"I don't
know based on the provided documents"* with the scorer's rationale.

![Refusal on an out-of-scope question](docs/assets/screenshot-refusal.png)

*Both screenshots are captures of `make demo`, which runs the scripted stub agent
offline: the badge reads `stub-agent` and the scores are the stub's canned values.*

## Features

Each feature maps to its implementation and tests in [`docs/CLAIMS.md`](docs/CLAIMS.md);
CI job `claims` runs those tests.

- Verbatim citations: every quote is a substring of its source document; others are dropped (C01).
- Threshold-gated refusal with a rationale (C02); unparseable scorer output is refused (C03).
- Hybrid retrieval: dense + sparse, reciprocal rank fusion, cross-encoder rerank (C04).
- Two vector stores: in-memory with cosine and BM25 (C05); Postgres + pgvector with full-text search (C06), the default for the API, the demo and the eval (C22).
- Markdown and plain-text ingestion with structure-aware chunking and machine-independent ids (C07); PDF and DOCX through docling (C08).
- Multi-provider LLM routing with fallback, stopping on authentication errors (C09).
- Per-request provenance: each answer records the model that actually served it (C10).
- Evaluation harness: retrieval, refusal and LLM-judged answer-quality metrics, one audit record per question (C11).
- Live run artefacts protected from being overwritten by stub runs (C12).
- README results generated from the scorecard and checked in CI (C13); CLI labels match scorecard keys (C14).
- SDK call signatures (C15) and LLM output parsers (C16) tested against the shapes real models produce.
- Per-stage tracing behind `/metrics`, `/metrics/prom` and `/dashboard` (C17); optional OTLP export, bounded in-memory spans (C29).
- HTTP API with `/status`, `/ask` and content-addressed `/ingest` (C18), and a Next.js UI with citation highlighting and a provenance badge (C19).
- Provider API keys held as secrets, absent from any printed settings (C20).
- `docker compose up`: non-root image, Postgres, migrations, corpus ingestion and the API in order, smoke-tested in CI (C26).
- API hardening: API keys on `/ask` and `/ingest`, per-key rate limit, failed-key throttling, bounded inputs, file-type allow-list, generic errors with a request ID (C23).
- Prompt-injection defence: delimited, untrusted-labelled evidence, and a deterministic support check (citation coverage, cited relevance, injection markers) the model's score cannot override; adversarial eval questions (C24).
- Provider and store outages reported as errors (HTTP 503), never as refusals (C25).

## Measured results

<!-- RESULTS:START -->
**Measured on**: `agent=claude-sonnet-4-6 (56/56)` · `judge=claude-opus-5-5 (34/56 answered — 22 refusals bypass the judge)` · `embedding=BAAI/bge-small-en-v1.5` · commit `969e1d5` · 2026-11-02 · dataset `test` · n=56 (36 answerable + 20 unanswerable) · **0 errored** (kept out of every metric) · total cost **$1.2768**

| Metric | Value | Notes |
|---|---:|---|
| **Refusal recall** | **0.850** | unanswerable questions refused / unanswerable (20) · threshold 0 |
| **Refusal precision** | **0.773** | refusals that were unanswerable / refusals (22) |
| Faithfulness, per question | 0.990 | mean over 34 judged answers · judge unvalidated |
| Faithfulness, per claim | 0.991 | pooled over 109 claims · judge unvalidated |
| Citation accuracy, per question | 0.968 | mean over 34 judged answers · judge unvalidated |
| Citation accuracy, per claim | 0.963 | pooled over 109 claims · judge unvalidated |
| Hallucination rate | 0.000 | ↓ better · answers with an unsupported claim, over 31 answered answerable questions |
| Unanswerable answered | 3/20 | ↓ better · an answer shipped where the corpus has none |
| Retrieval nDCG@8 | 0.929 | deterministic (no LLM) |
| Retrieval recall@8 | 0.986 | deterministic |
| Retrieval precision@8 | 0.146 | k>gold ceiling; interpret with recall |
| Latency p50 / p95 / p99 | 5445.7 ms / 8949.3 ms / 10929.3 ms | warm; cold start `8460.1 ms` measured separately |
| Cost / query | $0.0132 | total $1.2768 = agent $0.7408 + judge $0.5360 |

Refusals by cause: `irrelevant_citation` 4 (4 answerable) · `low_citation_coverage` 7 (1 answerable) · `model_refused` 2 (0 answerable) · `no_citation` 6 (0 answerable) · `suspected_injection` 3 (0 answerable).
No refusal has the cause `below_threshold`: the confidence threshold decided none of them.

Generated from [`scorecards/live-v1.1.0.json`](scorecards/live-v1.1.0.json), itself computed from the run artefact `datasets/eval/runs/live/eval_969e1d5_test.json` (`scripts/generate_headline_results.py --check` recomputes both).

<!-- RESULTS:END -->

The v1.0.0 artefact
[`eval_d0ca1d5_questions.json`](datasets/eval/runs/live/eval_d0ca1d5_questions.json)
is immutable and still stores citation accuracy **0.7917**, under the earlier
definition that averaged its 4 refusals in at 1.0. Under the single definition used
above (refusals excluded) it is **0.7222**, per question over 12 judged answers; the
recomputation is in
[`eval_d0ca1d5_questions.recomputed.json`](datasets/eval/runs/live/eval_d0ca1d5_questions.recomputed.json)
(`scripts/recompute_live_aggregates.py`).

Every value in the table is copied from the committed scorecard JSON by
[`scripts/generate_headline_results.py`](scripts/generate_headline_results.py);
`--check` fails CI if the README drifts. Each answer-quality metric carries its
`scope` and `n_denominator`. The full audit trail (per-question answer, citations,
per-claim judge verdict) is
[`datasets/eval/runs/live/eval_d0ca1d5_questions.json`](datasets/eval/runs/live/eval_d0ca1d5_questions.json).
Retrieval numbers come from the in-memory store (BM25 sparse arm), which that run
used; `verity eval run` now defaults to pgvector (`--store memory` reproduces it). Definitions: [`docs/evaluation-methodology.md`](docs/evaluation-methodology.md).

## Open issues: what the numbers do not say

- **n = 16 is a method check, not a benchmark.** 16 questions over 5 documents; one
  question moves refusal recall by 0.2. There is no dev/test split: the 0.55
  threshold was set by hand and is reported on the same examples.
- **The judge is unvalidated.** It is the same model as the agent
  (`claude-sonnet-4-6`) and its agreement with human labels has not been measured.
- **Citation accuracy is a per-question mean.** Aggregated per claim instead, the
  same artefact gives 21 of 34 claims with `citation_ok` (0.618), against 0.722 per
  question. All 13 failing claims are `supported = true`: the judge found the
  attached citation too coarse (a paragraph-level quote for several facts), not the
  claim unsupported. An earlier v1.0.0 headline published 0.792, which averaged over
  all 16 questions with refusals counted as 1.0; v1.0.1 restricted it to the 12
  answered questions.
- **The confidence score is not calibrated.** It is the scorer model's
  self-report; refusal is "threshold-gated", and calibration has not been measured.
- **q-015 is a labelling ambiguity.** *"Which specific cloud provider hosts the
  production infrastructure?"* is labelled out-of-scope (the corpus names no
  provider). The agent answered that the evidence does not name one, with one
  citation (the artefact does not record its confidence score). The label only
  checks `refused`, so this counts as an out-of-scope question answered, and
  refusal recall is reported as 0.800.
- **Latency includes cold start.** The run used `--no-warmup`, so p95/p99 include
  the first question's model loading; percentiles come from 16 calls in one run.

## Roadmap

Not implemented, or implemented without a test. Each item is a `roadmap` row in
[`docs/CLAIMS.md`](docs/CLAIMS.md).

- Web-page ingestion: the trafilatura parser exists but no test exercises it (C30).
- Calibrated refusal: measure calibration (ECE, reliability diagram) before claiming it (C21).
- A judge validated against human labels, from a different model family (C27).
- ≥ 80 questions with a frozen dev/test split; micro and macro aggregates (C28).

## Architecture

```mermaid
flowchart LR
    subgraph Ingest["Ingestion"]
        docs[/"documents<br/>MD · TXT · PDF · DOCX (web untested)"/] --> parser[parser<br/>structure-aware]
        parser --> chunker[chunker]
        chunker --> embedder[BAAI/bge-small-en-v1.5]
    end

    embedder --> pg[(pgvector, default<br/>HNSW + FTS)]
    embedder --> mem[(in-memory store<br/>cosine + BM25)]

    subgraph Retrieve["Hybrid retrieval"]
        query[/"query"/] --> dense[dense search]
        query --> sparse[sparse search]
        dense --> rrf[RRF fusion]
        sparse --> rrf
        rrf --> rerank[cross-encoder<br/>ms-marco-MiniLM]
    end

    mem -.->|VERITY_STORE=memory| dense
    pg --> dense
    pg --> sparse

    subgraph Agent["Agent"]
        rerank --> decompose[decompose]
        decompose --> synthesize[synthesize<br/>+ verbatim citations]
        synthesize --> score[confidence scorer]
        score -->|below threshold or refused| refuse[REFUSE<br/>with rationale]
        score -->|otherwise| ship[Answer + citations]
    end

    subgraph Eval["Evaluation"]
        ship --> judge[LLM judge<br/>supported · citation_ok]
        refuse --> refusal[refusal metrics<br/>TP/FP/FN/TN]
        judge --> scorecard["scorecards/live-v1.0.0.json<br/>+ per-question audit"]
        refusal --> scorecard
    end

    scorecard --> readme["README results table<br/>(generated)"]
```

Stage-by-stage description: [`docs/architecture.md`](docs/architecture.md).

## Quickstart

```bash
git clone https://github.com/matteogallo-ai/verity && cd verity
uv sync --dev                              # Python 3.12; CPU-only torch on Linux
VERITY_STORE=memory uv run verity eval run --ci  # pipeline smoke test on stubs, no database
make demo                                  # pgvector + API :8000 + Next.js :3000, stub agent
```

The first run downloads the embedding and reranker models (~200 MB) into the
Hugging Face cache. `make demo` needs Docker (for pgvector), Node 22 and corepack.

pgvector is the default store (`VERITY_STORE`, or `--store` on the CLI); every command
that reads the corpus first upserts it, idempotently:

```bash
docker compose up -d db                    # pgvector on :5432
uv run verity db upgrade                   # apply migrations
uv run verity eval retrieval               # ingests datasets/corpus, then scores retrieval
```

`VERITY_STORE=memory` runs everything in process memory, without a database.

Everything in containers (no provider key means stub mode). The image embeds every
local model, so it runs offline, PDF and DOCX uploads included: 4.77 GB uncompressed
on linux/arm64, of which 1.89 GB is the Python environment (torch CPU, docling),
765 MB the docling layout, table and OCR models, 227 MB the embedder and reranker and
204 MB the X11/GL libraries opencv needs.

```bash
VERITY_API_KEYS='["a-long-random-key"]' docker compose up   # or VERITY_API_AUTH_DISABLED=true
curl -H 'X-API-Key: a-long-random-key' -H 'content-type: application/json' \
  -d '{"question": "What is the notice period?"}' http://127.0.0.1:8000/ask
```

The published numbers are recomputed from the committed run artefact, with no model
call:

```bash
uv run python scripts/generate_headline_results.py --check   # scorecard and README ≡ artefact
```

A new live evaluation calls paid models (agent and judge): estimate its cost first.
The threshold is chosen on the dev split and the test split is run once at it
([methodology](docs/evaluation-methodology.md#refusal-deterministic-over-the-agents-decisions)):

```bash
export VERITY_ANTHROPIC_API_KEY=sk-ant-...
uv run verity eval run --judge live --no-agent-stub \
  --dataset datasets/eval/dev.jsonl --budget-usd 2.0 --report-only-floors
uv run python scripts/refusal_analysis.py --dev datasets/eval/runs/live/eval_<sha>_dev.json \
  --out-json scorecards/refusal-analysis.json --out-png docs/figures/reliability.png
VERITY_CONFIDENCE_THRESHOLD=<chosen> uv run verity eval run --judge live --no-agent-stub \
  --dataset datasets/eval/test.jsonl --budget-usd 2.8 --report-only-floors
uv run python scripts/generate_headline_results.py \
  --artefact datasets/eval/runs/live/eval_<sha>_test.json --scorecard scorecards/<name>.json
```

The v1.0.0 run (`scorecards/live-v1.0.0.json`) used 16 questions that predate the
dev/test split; its artefact stays in `datasets/eval/runs/live/`.

## How this was built

Verity was implemented with Claude Code as the implementation engine. The process:
a written spec with acceptance criteria, an agent implementing on a branch, a review
of the diff by a second agent without the builder's context, a human audit of the
claims and critical paths, then CI. The coding agent writes most first drafts; the
author is accountable for what merges. Standards:
[`docs/ENGINEERING_STANDARDS.md`](docs/ENGINEERING_STANDARDS.md).

Human audits caught what the agents and CI did not:

1. **Machine-specific ids.** Document ids were derived from absolute file paths, so
   the gold chunk ids in the eval dataset only matched on the machine that built
   them. Ids now hash the repo-relative path, and
   `tests/integration/test_eval_labels_integrity.py` checks every gold id resolves.
2. **Scorecard denominators.** The first v1.0.0 headline averaged citation accuracy
   over all 16 questions, counting the 4 refusals as 1.0: 0.792. Restricted to the
   12 answered questions it is 0.722 (v1.0.1). Every answer-quality metric now
   carries its scope and denominator.
3. **An external audit (2026-11-02)** found public claims ahead of the code:
   "calibrated" refusal that was never measured, a methodology defining per-claim
   aggregation while the code averaged per question, a Docker image that cannot
   serve live answers, an API with no input limits or prompt-injection defence,
   pgvector described as the default store, docstrings contradicting the code, and
   sprint-era narrative in shipped files. This README, `docs/CLAIMS.md` and the
   `claims` / `stale-narrative` CI jobs are the first part of the response; the
   Roadmap lists what remains.

## Stack

Python 3.12 · FastAPI · Postgres + pgvector (or an in-memory store) · local
embeddings and cross-encoder reranker (offline, no keys) · Anthropic and OpenAI
clients behind an ordered fallback router · OpenTelemetry spans (in-process) +
structlog · Next.js 15 UI. Tooling: uv · ruff · mypy strict · vulture · deptry ·
pytest · GitHub Actions. Eval history is kept as versioned JSON files.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run vulture && uv run deptry src
HF_HUB_OFFLINE=1 uv run pytest -m "not integration and not slow"  # fast lane, no network
uv run pytest -m "slow and not integration"        # loads the local models
VERITY_DATABASE_URL=postgresql://verity:verity@localhost:5432/verity \
  uv run pytest -m integration                     # needs pgvector
uv run python scripts/check_claims.py              # docs/CLAIMS.md ≡ tests ≡ README
uv run python scripts/check_stale_narrative.py
uv run python scripts/generate_headline_results.py --check   # README ≡ scorecard
```

[`CLAUDE.md`](CLAUDE.md) is the working guide for coding agents. See
[CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). Licensed under
[MIT](LICENSE).
