<div align="center">

# Verity

**A RAG agent over enterprise documents that refuses when its evidence is weak, and
publishes how often it gets that right.**
Cited answers with verbatim quotes. Threshold-gated refusal on out-of-scope questions.
Every published number traceable to the run artefact that produced it.

[![ci](https://github.com/matteogallo-ai/verity/actions/workflows/ci.yml/badge.svg)](https://github.com/matteogallo-ai/verity/actions/workflows/ci.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/matteogallo-ai/verity/badge)](https://scorecard.dev/viewer/?uri=github.com/matteogallo-ai/verity)
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
- API hardening: API keys on `/status`, `/ask` and `/ingest`, per-key rate limit, a limit on failed key attempts, bounded inputs, file-type allow-list, generic errors with a request ID (C23).
- Bounded spend: per-key spending budget on `/ask`, a per-answer cap on `verity ask` checked before each LLM call, bounded `max_tokens` and timeouts (C31).
- Liveness and readiness split: `/health` answers while the process lives; `/ready` answers `503` until the corpus and the local models are loaded at startup, and unless Postgres answers and the LLM providers can serve as routing uses them, without spending tokens (C32).
- Retrieved text reaches the synthesizer datamarked (spotlighting): a mark between every pair of words lets its prompt label all of it as document content. Against 12 naturally worded poisoned documents, 3 live runs per arm, no case was steered with or without the mark (0/12 each, Wilson 95 % [0, 0.243]), though the attacks made answerable questions refuse (4/24 answers with the mark, 7/24 without): this set does not show an effect of the mark (C33).
- Prompt-injection defence: delimited, untrusted-labelled evidence, and a deterministic support check (citation coverage, cited relevance, injection markers) the model's score cannot override; adversarial eval questions (C24).
- Provider and store outages reported as errors (HTTP 503), never as refusals (C25).

## Measured results

<!-- RESULTS:START -->
**Measured on**: `agent=claude-sonnet-4-6 (16/16)` · `judge=claude-sonnet-4-6 (12/16 answered — 4 refusals bypass the judge)` · `embedding=BAAI/bge-small-en-v1.5` · commit `d0ca1d5` · 2026-09-07 · n=16 (11 answerable + 5 out-of-scope) · total cost **$0.2713**

| Metric | Value | Notes |
|---|---:|---|
| **Refusal recall** | **0.800** | headline · fraction of out-of-scope questions correctly refused |
| **Refusal precision** | **1.000** | of the refusals, fraction that were genuinely unanswerable |
| Faithfulness (answered, n=12) | 1.000 | per-question mean of claim-level judge verdicts over 12 answered questions · judge unvalidated |
| Citation accuracy (answered, n=12) | 0.722 | per-question mean over 12 answered questions (refusals bypass the judge) · judge unvalidated |
| Hallucination rate (answerable_and_answered, n=11) | 0.000 | ↓ better · at-risk set = answerable questions the agent answered |
| Out-of-scope answered | 1/5 | ↓ better · questions we shipped an answer for that were unanswerable |
| Retrieval nDCG@8 | 0.953 | deterministic (no LLM) |
| Retrieval recall@8 | 1.000 | deterministic |
| Retrieval precision@8 | 0.148 | k>gold ceiling; interpret with recall |
| Latency p50 / p95 / p99 | 9374.7 ms / 32526.5 ms / 36653.1 ms | no warmup: includes the first question's model loading |
| Cost / query | $0.0122 | total $0.2713 = agent $0.1950 + judge $0.0763 |

Numbers are copied verbatim from [`scorecards/live-v1.0.0.json`](scorecards/live-v1.0.0.json) (committed artefact of the single live run of record — never averaged, never re-run to nudge a number).

<!-- RESULTS:END -->

Every value in the table is copied from the committed scorecard JSON by
[`scripts/generate_headline_results.py`](scripts/generate_headline_results.py);
`--check` fails CI if the README drifts. Each answer-quality metric carries its
`scope` and `n_denominator`. The full audit trail (per-question answer, citations,
per-claim judge verdict) is
[`datasets/eval/runs/live/eval_d0ca1d5_questions.json`](datasets/eval/runs/live/eval_d0ca1d5_questions.json).
Retrieval numbers come from the in-memory store (BM25 sparse arm), which that run
used; `verity eval run` now defaults to pgvector (`--store memory` reproduces it). Definitions: [`docs/evaluation-methodology.md`](docs/evaluation-methodology.md).
Each proportion with its k/n and Wilson 95 % interval, the approved SLOs and whether the run meets them, the intended
use and the known limits: [`docs/SYSTEM_CARD.md`](docs/SYSTEM_CARD.md), generated from
the same scorecard.

## Open issues: what the numbers do not say

- **n = 16 is a method check, not a benchmark.** 16 questions over 5 documents; one
  question moves refusal recall by 0.2. There is no dev/test split: the 0.55
  threshold was set by hand and is reported on the same examples.
- **The numbers predate the current answering pipeline.** The run of record
  (`d0ca1d5`, 2026-09-07) ran before the deterministic support check and the
  delimited, untrusted-labelled prompts. The support check's two thresholds were
  chosen on these same 16 questions, so they cannot measure its false-refusal cost,
  and no live run of the current code is published here.
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

Stage-by-stage description, and what I'd do differently:
[`docs/architecture.md`](docs/architecture.md#what-id-do-differently).

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
that reads the corpus first syncs it: unchanged documents are skipped, edited ones have
their chunks replaced, and documents whose file was removed leave the index:

```bash
docker compose up -d db                    # pgvector on :5432
uv run verity db upgrade                   # apply migrations
uv run verity eval retrieval               # ingests datasets/corpus, then scores retrieval
```

`VERITY_STORE=memory` runs everything in process memory, without a database.

Everything in containers (the image embeds the local models; no provider key means stub
mode):

```bash
VERITY_API_KEYS='["a-long-random-key"]' docker compose up   # or VERITY_API_AUTH_DISABLED=true
curl -H 'X-API-Key: a-long-random-key' -H 'content-type: application/json' \
  -d '{"question": "What is the notice period?"}' http://127.0.0.1:8000/ask
```

Reproducing the v1.0.0 headline calls Anthropic models (the original run cost
$0.27):

```bash
export VERITY_ANTHROPIC_API_KEY=sk-ant-...
uv run verity eval run \
  --judge live --no-agent-stub --no-warmup \
  --dataset datasets/eval/questions.jsonl \
  --scorecard-json scorecards/live-v1.0.0.json \
  --budget-usd 8.0 --report-only-floors \
  --floor-refusal-recall 0.6 --floor-ndcg 0.5 --floor-recall 0.75
```

It writes `datasets/eval/runs/live/eval_<sha>_questions.json` and the scorecard;
`uv run python scripts/generate_headline_results.py` regenerates the README block.

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
[CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), the
[threat model](docs/THREAT_MODEL.md) and the [runbook](docs/RUNBOOK.md). Licensed under
[MIT](LICENSE).
