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

## The problem

Enterprise RAG demos answer confidently and are wrong just often enough to be
dangerous: a fluent, cited-looking answer with no real grounding is the failure that
matters for contracts, filings and policies.

Verity drafts an answer, checks deterministically that its citations cover and
match it, asks a scorer model how well the evidence supports it, and refuses when a
check fails, the scorer refuses, or the score is below a threshold. It publishes faithfulness, citation accuracy,
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
- 95 questions over 21 documents (≥ 25 unanswerable, ≥ 5 adversarial) with a frozen, seeded dev/test split; per-question and per-claim aggregates (C28).
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

## Measured results

<!-- RESULTS:START -->
**Measured on**: `agent=claude-sonnet-4-6 (56/56)` · `judge=claude-opus-5-5 (34/56 answered — 22 refusals bypass the judge)` · `embedding=BAAI/bge-small-en-v1.5` · commit `969e1d5` · 2026-11-02 · dataset `test` · n=56 (36 answerable + 20 unanswerable) · **0 errored** (kept out of every metric) · total cost **$1.2768**

| Metric | Value | Notes |
|---|---:|---|
| **Refusal recall** | **0.850** | unanswerable questions refused / unanswerable: 17/20 · 95 % CI [0.640, 0.948] · threshold 0 |
| **Refusal precision** | **0.773** | refusals that were unanswerable / refusals: 17/22 · 95 % CI [0.566, 0.899] |
| Faithfulness, per question | 0.990 | mean over 34 judged answers · judge κ 0.71 vs blind human (n=60) |
| Faithfulness, per claim | 0.991 | pooled over claims: 108/109 · 95 % CI [0.950, 0.998] · judge κ 0.71 vs blind human (n=60) |
| Citation accuracy, per question | 0.968 | mean over 34 judged answers · judge κ 0.71 vs blind human (n=60) |
| Citation accuracy, per claim | 0.963 | pooled over claims: 105/109 · 95 % CI [0.909, 0.986] · judge κ 0.71 vs blind human (n=60) |
| Hallucination rate | 0.000 | ↓ better · answers with an unsupported claim / answered answerable questions: 0/31 · 95 % CI [0.000, 0.110] |
| Unanswerable answered | 3/20 | ↓ better · an answer shipped where the corpus has none: 3/20 · 95 % CI [0.052, 0.360] |
| Retrieval nDCG@8 | 0.929 | scored without an LLM on hits retrieved for the LLM sub-questions |
| Retrieval recall@8 | 0.986 | as nDCG |
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
refusal cause, confidence score, per-claim judge verdict) is
[`datasets/eval/runs/live/eval_969e1d5_test.json`](datasets/eval/runs/live/eval_969e1d5_test.json).
The run used the in-memory store (BM25 sparse arm); `verity eval run` defaults to
pgvector (`--store memory` reproduces it). Definitions:
[`docs/evaluation-methodology.md`](docs/evaluation-methodology.md).
Each proportion with its k/n and Wilson 95 % interval, the approved SLOs and whether
this run meets them, the intended use and the known limits:
[`docs/SYSTEM_CARD.md`](docs/SYSTEM_CARD.md), generated from the same scorecard.

## Judge validation

The answer-quality rows above are judged by a model. Its agreement with blind human
labels is the result of record; an adjudication of the disagreements, published
separately, does not replace it.

<!-- JUDGE:START -->
Judge `claude-opus-5-5` with judge prompt v0.1.0 (`datasets/eval/judge_validation/judge_faithfulness.v0.1.0.prompt`, sha256 `2c5f8eca49ba`) against **blind human labels**, the reference: 60 claims labelled without origin or judge verdict, 40 real claims (16 from the dev run, 24 from the test run) and 20 claims perturbed to be false (one number, date, negation or entity changed). The κ below belongs to this prompt version: a changed judge prompt is a new judge and needs its own κ.

| Measure | Value | n · notes |
|---|---|---|
| Cohen's κ, all claims | 0.71 | 60 claims · observed agreement 0.883 (53/60 · 95 % CI [0.778, 0.942]) · 46 supported / 14 unsupported by the human |
| Cohen's κ, real claims only | 0.48 | 40 claims · observed agreement 0.950 (38/40 · 95 % CI [0.835, 0.986]) · **fragile**: only 1 of 40 claims is in the smaller human class, so one or two disagreements move κ widely |
| Judge specificity on real claims (human: supported) | 0.949 | 37/39 · 95 % CI [0.831, 0.986] |
| Sensitivity: perturbed claims the judge finds unsupported (**upper bound**) | 0.800 | 16/20 · 95 % CI [0.584, 0.919] · by kind: date 5/5 · entity 3/5 · negation 5/5 · number 3/5 |
| … among perturbed claims the human finds unsupported | 0.923 | 12/13 · 95 % CI [0.667, 0.986] |
| Perturbed claims the human finds unsupported | 0.650 | 13/20 · 95 % CI [0.433, 0.819] |

Disagreements: the judge rejects 6 claims the human accepts and accepts 1 the human rejects. Notes from the artefact:

- kappa has no Wilson interval: it is not a proportion; with 60 items it is uncertain by itself.
- a perturbed claim counts as detected if any sub-claim the judge made of it is unsupported, even outside the changed span: sensitivity is an upper bound (per-claim detail in perturbed_verdicts.json).
- robustness to injected content is not measured: judge prompt v0.1.0 puts the retrieved text in front of the judge without delimiters or an untrusted-content instruction ([THREAT_MODEL](docs/THREAT_MODEL.md#llm01-prompt-injection)), and only 4 of the 60 claims had the corpus note written to carry an injection in the judge's context.

Generated from [`datasets/eval/judge_validation/agreement.json`](datasets/eval/judge_validation/agreement.json) (`scripts/render_judge_validation.py --check`); the labels and the key are pinned in `datasets/eval/judge_validation/SHA256SUMS`.
<!-- JUDGE:END -->

### Adjudicated agreement (not the result of record)

<!-- JUDGE-ADJUDICATED:START -->
10 claims re-read with both verdicts visible, by the annotator of the blind labels: every disagreement, and every perturbed claim judged true blind. 7 labels changed, **6 of them to the judge's verdict**. The judge's verdict was in view, so a change towards it may be persuasion as much as correction: these figures bound agreement from above and validate nothing. Of the 7 perturbed claims judged true blind, 2 stay true after the change and 5 were annotator misses.

| Measure | Blind (of record) | Adjudicated |
|---|---|---|
| Cohen's κ, all claims | 0.71 | 0.93 |
| Cohen's κ, real claims only (fragile: 3 unsupported) | 0.48 | 1.00 |
| Observed agreement, all claims | 53/60 · 95 % CI [0.778, 0.942] | 58/60 · 95 % CI [0.886, 0.991] |
| Perturbed claims the human finds unsupported | 13/20 · 95 % CI [0.433, 0.819] | 18/20 · 95 % CI [0.699, 0.972] |
| … of which the judge finds unsupported | 12/13 · 95 % CI [0.667, 0.986] | 16/18 · 95 % CI [0.672, 0.969] |

Generated from [`datasets/eval/judge_validation/agreement_adjudicated.json`](datasets/eval/judge_validation/agreement_adjudicated.json); each decision and its reason is in `adjudication.csv`.
<!-- JUDGE-ADJUDICATED:END -->

### Judge prompt v0.2.0 (not the judge of the runs of record)

<!-- JUDGE-V2:START -->
Judge `claude-opus-5-5` with judge prompt v0.2.0 (sha256 `aa66e62370a2`), the prompt in `prompts/`: its inputs are delimited and its evidence datamarked, and it names the citation of each claim. Rejudged against the same blind labels, for $1.30, on **52 of 60 claims only: the budget stopped the rejudge**, so the figures below are incomplete. Every claim, real or perturbed, is judged as a one-claim answer; v0.1.0 used the live whole-answer verdict for the real claims, so a change on real claims mixes the prompt and the protocol. The runs of record were judged by v0.1.0: their answer-quality figures carry the v0.1.0 κ above.

| Measure | v0.1.0 (runs of record) | v0.2.0 |
|---|---|---|
| Cohen's κ, all claims judged | 0.71 (60) | 0.60 (52) |
| Cohen's κ, the same 52 claims | 0.64 | 0.60 |
| Observed agreement, the same 52 claims | 45/52 · 95 % CI [0.747, 0.933] | 44/52 · 95 % CI [0.725, 0.920] |
| Cohen's κ, real claims only (fragile: 1 unsupported) | 0.48 | 0.37 |
| Judge specificity on real claims | 37/39 · 95 % CI [0.831, 0.986] | 32/35 · 95 % CI [0.776, 0.970] |
| Sensitivity on perturbed claims (upper bound) | 16/20 · 95 % CI [0.584, 0.919] | 12/16 · 95 % CI [0.505, 0.898] |

The judge named a valid citation it found supporting for 40 of the 57 claims it returned. The blind labels judge support only, so this count is not validated against a human.

Generated from [`datasets/eval/judge_validation/judge-v0.2.0/agreement.json`](datasets/eval/judge_validation/judge-v0.2.0/agreement.json); per-claim verdicts in `verdicts.json` beside it.
<!-- JUDGE-V2:END -->

## Open issues: what the numbers do not say

- **The runs predate the current prompts.** Both v1.1.0 runs used
  `synthesize.prompt` v0.2.0 and judge prompt v0.1.0: the datamarked synthesis (C33)
  and the delimited judge came after them, and no live run of the current pipeline is
  published here.
- **The threshold decides nothing on this data.** Every refusal above has a
  deterministic cause (no citation, low citation coverage, irrelevant citation,
  suspected injection) or is the scorer's own refusal; none is `below_threshold`.
  The threshold was chosen on the dev split by a rule fixed before the run (highest
  refusal recall at precision ≥ 0.9, else highest F1): no threshold reached 0.9,
  refusal precision and recall are identical for every threshold from 0.00 to 0.95,
  and the rule gave 0.00. Sweep and figure:
  [`scorecards/refusal-analysis.json`](scorecards/refusal-analysis.json),
  [`docs/figures/reliability.png`](docs/figures/reliability.png).
- **Refusal precision is a known limit** (row above, with its interval). Every
  refusal of an answerable question comes from the citation checks
  (`irrelevant_citation`, `low_citation_coverage`: counts under "Refusals by cause");
  dev shows the same cause (`scorecards/refusal-analysis.json`). Tuning those checks
  is done on dev only; a new test run belongs to a new version.
- **The confidence score is not calibrated.** On test, 35 of the 36 scored answers
  score 0.9 or more and 31 of them are answerable: ECE 0.116 over n = 36, with almost
  every answer in one bin. Isotonic and Platt maps fitted on dev hold one
  unanswerable example and do not fix it (ECE 0.097 and 0.091). The score is the
  scorer model's self-report.
- **The judge is validated on 60 claims by one annotator, and is a Claude model.**
  It is `claude-opus-5-5`, a different model from the agent (`claude-sonnet-4-6`),
  not a different family. κ against the owner's blind labels and the sensitivity on
  perturbed claims are under "Judge validation"; κ on real claims alone rests on one
  human-unsupported claim and is fragile. One annotator gives no inter-annotator
  agreement.
- **n = 56 is small.** The 95 % Wilson intervals in the table show how little it
  pins down: no hallucination in the sample does not rule out a rate up to the
  interval's upper bound.
  Per-claim intervals assume independent claims; claims of one answer are not, so
  those intervals are too narrow. The corpus is 21 fictional documents in three
  domains, with no scanned PDFs or tables.
- **Unanswerable questions were answered** (row "Unanswerable answered"): q-074,
  q-085 and q-089, with scores 0.72, 0.92 and 0.90.

## Roadmap

Not implemented, or implemented without a test. Each item is a `roadmap` row in
[`docs/CLAIMS.md`](docs/CLAIMS.md).

- Web-page ingestion: the trafilatura parser exists but no test exercises it (C30).
- Calibrated refusal: the measured test ECE (0.116) does not support it (C21).
- A judge from a different model family, and a second human annotator (C34).

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
        judge --> scorecard["scorecards/live-v1.1.0.json<br/>+ per-question audit"]
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
