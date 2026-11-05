<div align="center">

# Verity

**A RAG agent over enterprise documents that refuses when its evidence is weak, and
publishes how often it gets that right.**
Cited answers with verbatim quotes. Refusal when the citations do not support the answer.
Every published number traceable to the run artefact that produced it.

[![ci](https://github.com/matteogallo-ai/verity/actions/workflows/ci.yml/badge.svg)](https://github.com/matteogallo-ai/verity/actions/workflows/ci.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/matteogallo-ai/verity/badge)](https://scorecard.dev/viewer/?uri=github.com/matteogallo-ai/verity)
[![license: MIT](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)
![python](https://img.shields.io/badge/python-3.12-blue.svg)

</div>

## What it is

Enterprise RAG demos answer confidently and are wrong just often enough to be
dangerous. Verity drafts an answer, checks deterministically that its citations cover
and match it, asks a scorer model how well the evidence supports it, and refuses when
a check fails, the scorer refuses, or the score is below a threshold.

## Key result

<!-- KEY-RESULT:START -->
**Refusal recall 0.850**: 17/20 unanswerable questions refused, Wilson 95 % CI [0.640, 0.948], at refusal precision 0.773 (17/22, CI [0.566, 0.899]).
One live run on the held-out `test` split, n = 56, agent `claude-sonnet-4-6`, commit `969e1d5`. Every metric, the judge's validation and the limits: [`docs/results.md`](docs/results.md).
<!-- KEY-RESULT:END -->

## What it looks like

![Cited answer with source panel + confidence readout](docs/assets/screenshot-answer.png)

Click a citation and the source pane highlights the exact clause; out-of-corpus
questions get *"I don't know based on the provided documents"*
([refusal screenshot](docs/assets/screenshot-refusal.png)). Both are captures of
`make demo`, which runs the scripted stub agent offline.

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
The threshold is chosen on the dev split and the test split is run once at it, with
no change to the system in between
([methodology](docs/evaluation-methodology.md#refusal-deterministic-over-the-agents-decisions)):

```bash
export VERITY_ANTHROPIC_API_KEY=sk-ant-...
uv run verity eval run --judge live --no-agent-stub \
  --dataset datasets/eval/dev.jsonl --budget-usd 2.0 --report-only-floors
uv run python scripts/refusal_analysis.py --dev datasets/eval/runs/live/eval_<sha>_dev.json \
  --out-json scorecards/refusal-analysis.json --out-png docs/figures/reliability.png
VERITY_CONFIDENCE_THRESHOLD=<chosen> uv run verity eval run --judge live --no-agent-stub \
  --dataset datasets/eval/test.jsonl --budget-usd 1.6 --report-only-floors
uv run python scripts/refusal_analysis.py --dev datasets/eval/runs/live/eval_<sha>_dev.json \
  --test datasets/eval/runs/live/eval_<sha>_test.json \
  --out-json scorecards/refusal-analysis.json --out-png docs/figures/reliability.png
uv run python scripts/generate_headline_results.py \
  --artefact datasets/eval/runs/live/eval_<sha>_test.json --scorecard scorecards/<name>.json
```

Run artefacts in `datasets/eval/runs/live/` are pinned in its `SHA256SUMS` and never
edited; v1.1.0 used the dev run `eval_6ec544d_dev.json` and the test run
`eval_969e1d5_test.json`. The v1.0.0 run (`scorecards/live-v1.0.0.json`) used 16
questions that predate the dev/test split.

## Features

Each feature maps to its implementation and tests in [`docs/CLAIMS.md`](docs/CLAIMS.md);
CI job `claims` runs those tests.

- Verbatim citations: every quote is a substring of its source document; others are dropped (C01).
- Refusal with a rationale when a citation check fails, the scorer refuses or its score is below a threshold (C02); unparseable scorer output is refused (C03).
- Hybrid retrieval: dense + sparse, reciprocal rank fusion, cross-encoder rerank (C04).
- Two vector stores: in-memory with cosine and BM25 (C05); Postgres + pgvector with full-text search (C06), the default for the API, the demo and the eval (C22).
- Markdown and plain-text ingestion with structure-aware chunking and machine-independent ids (C07); PDF and DOCX through docling (C08).
- Multi-provider LLM routing with fallback, stopping on authentication errors (C09).
- Per-request provenance: each answer records the model that actually served it (C10).
- Evaluation harness: retrieval, refusal and LLM-judged answer-quality metrics, one audit record per question (C11).
- Live run artefacts protected from being overwritten by stub runs (C12).
- 95 questions over 21 documents (≥ 25 unanswerable, ≥ 5 adversarial) with a frozen, seeded dev/test split; per-question and per-claim aggregates (C28).
- README results generated from the scorecard and checked in CI (C13); CLI labels match scorecard keys (C14).
- SDK call signatures (C15) and LLM output parsers (C16) tested against the shapes real models produce.
- Per-stage tracing behind `/metrics`, `/metrics/prom` and `/dashboard` (C17); optional OTLP export, bounded in-memory spans (C29).
- HTTP API with `/status`, `/ask` and content-addressed `/ingest` (C18), and a Next.js UI with citation highlighting and a provenance badge (C19).
- Provider API keys held as secrets, absent from any printed settings (C20).
- `docker compose up`: non-root image, Postgres, migrations, corpus ingestion and the API in order, smoke-tested in CI (C26).
- API hardening: API keys on `/status`, `/ask` and `/ingest`, per-key rate limit, a limit on failed key attempts, bounded inputs, file-type allow-list, generic errors with a request ID (C23).
- Bounded spend: per-key spending budget on `/ask`, a per-answer cap on `verity ask` checked before each LLM call, bounded `max_tokens` and timeouts (C31).
- Liveness and readiness split: `/health` answers while the process lives; `/ready` answers `503` until the corpus and the local models are loaded at startup, and unless Postgres answers and the LLM providers can serve as routing uses them, without spending tokens (C32).
- Retrieved text reaches the synthesizer datamarked (spotlighting): a mark between every pair of words lets its prompt label all of it as document content. Against 12 naturally worded poisoned documents, 3 live runs per arm on the earlier 6-document corpus, no case was steered with or without the mark (0/12 each, Wilson 95 % [0, 0.243]), though the attacks made answerable questions refuse (4/24 answers with the mark, 7/24 without): this set does not show an effect of the mark (C33).
- Prompt-injection defence: delimited, untrusted-labelled evidence, and a deterministic support check (citation coverage, cited relevance, injection markers) the model's score cannot override; adversarial eval questions (C24).
- Provider and store outages reported as errors (HTTP 503), never as refusals (C25).
- Judge validated against blind human labels on 60 claims, real and perturbed to be false: κ, agreement, specificity and sensitivity with n and intervals, generated from a pinned artefact (C27).

## Roadmap

Not implemented, or implemented without a test. Each item is a `roadmap` row in
[`docs/CLAIMS.md`](docs/CLAIMS.md).

- Web-page ingestion: the trafilatura parser exists but no test exercises it (C30).
- Calibrated refusal: the measured test ECE (0.116) does not support it (C21).
- A judge from a different model family, and a second human annotator (C34).

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
