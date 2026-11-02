# System card — Verity

**Owner:** Matteo Gallo · **Version:** 1.0.x · **Licence:** MIT (code, prompts, corpus,
eval set) · Security posture: [`SECURITY.md`](../SECURITY.md) · Threats:
[`THREAT_MODEL.md`](THREAT_MODEL.md) · Operations: [`RUNBOOK.md`](RUNBOOK.md)

Every number on this page is generated from the committed scorecard by
`scripts/generate_headline_results.py`; CI runs it with `--check`. Every measured figure
is in the generated block; the prose around it quotes none.

## What it is

A retrieval-augmented question-answering agent over one team's documents. A question
is decomposed, answered from hybrid-retrieved passages with verbatim quotes, checked
deterministically (citation coverage, cited relevance, injection markers), then scored
by a model; below the threshold, or when any check fails, it answers "I don't know
based on the provided documents" with the reason. Surfaces: the `verity` CLI, an HTTP
API (`/ask`, `/ingest`, `/status`) and a Next.js UI. Each shipped capability is a
`shipped` row of [`CLAIMS.md`](CLAIMS.md) with the tests that prove it.

## Intended use

- Answering factual questions about a bounded set of English business documents
  (contracts, policies, filings) where a reader checks the cited clause before acting.
- One tenant per deployment: one API process, one corpus, a list of API keys that all
  see the whole corpus ([ADR 0006](adr/0006-api-security-baseline.md)).
- A reference implementation for measuring refusal: the eval harness and its committed
  artefacts are part of the deliverable.

## Out of scope

- **Several tenants, or users with different rights to documents.** There is no
  per-document access control and no tenant column in the schema
  ([THREAT_MODEL.md, LLM08](THREAT_MODEL.md#llm08-vector-and-embedding-weaknesses)).
- **Decisions without a human reading the citation.** The confidence score is the
  scorer model's self-report; its calibration is not measured (claim C21, roadmap).
- **Non-English documents.** The embedding model is English-only and Postgres full-text
  search uses the `english` configuration.
- **Web-page ingestion.** The parser exists without a test (claim C30, roadmap).
- **Corpora beyond 200 documents through the API**, and deleting a document: no
  endpoint deletes one.
- **Untrusted uploaders.** Any key holder can add a document every other client then
  retrieves.

## Data

| Artefact | Content | Licence / terms |
|---|---|---|
| `datasets/corpus/` | 6 Markdown documents: an MSA, a DPA, an SLA, an employee handbook, an FY2024 financial summary (all naming a company "Acme Corporation"), and `vendor_onboarding_note.md`, an adversarial document with injected instructions | MIT, with the repository; their origin is not documented in the repository |
| `datasets/eval/questions.jsonl` | 19 labelled questions: 11 answerable, 8 out-of-scope, 3 of those adversarial; gold chunk ids for every answerable one | MIT, with the repository |
| Run of record | `datasets/eval/runs/live/eval_d0ca1d5_questions.json` and the scorecard below; q-001 to q-016 over the first 5 documents (it predates the adversarial document) | MIT, with the repository |
| Embedding model | `BAAI/bge-small-en-v1.5`, 384 dimensions, run locally | MIT (its model card) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, run locally | stated on its model card; not vendored here |
| Generation | `claude-sonnet-4-6` through the Anthropic API, `gpt-4.1-mini` as fallback | the providers' API terms; question and retrieved passages are sent to them |

The API does not detect or filter personal data in uploaded documents.

## Measured performance

One live run, never averaged or re-run; read every value with its n and its interval.
The answer-quality rows come from an LLM judge (named in the source line) that is not
validated against human labels (claim C27, roadmap). Faithfulness and citation accuracy are means of
per-question fractions, not proportions, so the scorecard holds no k/n for them.
Definitions: [`evaluation-methodology.md`](evaluation-methodology.md).

<!-- SYSTEM-CARD:START -->
Source: [`scorecards/live-v1.0.0.json`](../scorecards/live-v1.0.0.json) · commit `d0ca1d5` · 2026-09-07 · agent `claude-sonnet-4-6 (16/16)` · judge `claude-sonnet-4-6 (12/16 answered — 4 refusals bypass the judge)` · n=16 (11 answerable + 5 out-of-scope).

| Metric | Value | k/n | 95 % CI (Wilson) |
|---|---:|---:|---:|
| Refusal recall | 0.800 | 4/5 | [0.376, 0.964] |
| Refusal precision | 1.000 | 4/4 | [0.510, 1.000] |
| Out-of-scope answered ↓ | 0.200 | 1/5 | [0.036, 0.624] |
| Hallucination rate ↓ (judge) | 0.000 | 0/11 | [0.000, 0.259] |
| Faithfulness, per-question mean (judge) | 1.000 | n=12 | no interval |
| Citation accuracy, per-question mean (judge) | 0.722 | n=12 | no interval |
| Retrieval nDCG@8, per-question mean | 0.953 | n=11 | no interval |
| Retrieval Recall@8, per-question mean | 1.000 | n=11 | no interval |

| SLO | Target | Measured on the run of record | Status |
|---|---|---|---|
| p95 latency of one answer | ≤ 15,000 ms (proposed, pending owner approval) | 32526.5 ms (n=16, no warmup: includes model loading) | over target |
| Error rate (requests ending in `answer_failed`) | ≤ 1% (proposed, pending owner approval) | not recorded by this scorecard | not measured |
| Cost per answer (agent calls; the judge is eval-only) | ≤ $0.02 (proposed, pending owner approval) | $0.0122 (mean, n=16) | within target |
<!-- SYSTEM-CARD:END -->

## Service level objectives

The targets in the second table above are proposals from the implementer, **pending
owner approval**; nothing enforces or alerts on them. How each is measured on a
running server, with the signals the code emits (responses: [`RUNBOOK.md`](RUNBOOK.md)):

| SLO | Measured from | Caveat |
|---|---|---|
| p95 latency | `latency_ms` of the `answer_done` and `answer_failed` log lines (wall clock of one `agent.answer`); offline, the scorecard's `latency_ms.p95` | `/metrics` and `/metrics/prom` expose per-stage percentiles only, not the end-to-end one; the run of record includes model loading |
| Error rate | count of `answer_failed` lines ÷ (`answer_done` + `answer_failed`) over the window; the same requests are `/ask` responses with status 503 | `verity_requests_total` counts `agent.confidence` spans, which a failed request may never open, so it cannot be the denominator |
| Cost per answer | `cost_usd` of `answer_done`; `verity_cost_usd_total` ÷ `verity_requests_total` for the session | priced from the `model_prices` table in `verity.config`, not from provider invoices; the judge's cost is excluded (eval-only) |

## Known limits

- **Sample size.** The n in the table above; no dev/test split, and the confidence
  threshold and the support-check thresholds were chosen on the same examples.
- **Unvalidated judge.** Its agreement with human labels is not measured.
- **Uncalibrated confidence.** Refusal is threshold-gated on a self-reported score.
- **Citation granularity.** A paragraph-level quote can cover several claims; the judge
  marks such citations not ok while the claim is supported.
- **Limits of the run of record itself** (latency with or without model loading, an
  error count the scorecard may not hold) are stated in the generated rows.
- **Corpus poisoning.** A document that states a false fact relevant to the question is
  answered from; the defences target instructions, not falsehoods.
