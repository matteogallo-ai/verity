# System card — Verity

**Owner:** Matteo Gallo · **Version:** 1.1.x · **Licence:** MIT (code, prompts, corpus,
eval set) · Security posture: [`SECURITY.md`](../SECURITY.md) · Threats:
[`THREAT_MODEL.md`](THREAT_MODEL.md) · Operations: [`RUNBOOK.md`](RUNBOOK.md)

Every measured number on this page is generated from the committed scorecard by
`scripts/generate_headline_results.py`; CI runs it with `--check`. Every measured figure
is in the generated block; the prose around it quotes none.

## What it is

A retrieval-augmented question-answering agent over one team's documents. A question
is decomposed, answered from hybrid-retrieved passages with verbatim quotes, checked
deterministically (citation coverage, injection markers; cited relevance only when
`VERITY_IRRELEVANT_CITATION_CHECK` is set), then scored by a model; when a check fails
or the scorer refuses, it answers "I don't know based on the provided documents" with
the reason. The score threshold defaults to 0 and refuses nothing. Surfaces: the `verity` CLI, an HTTP
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
| `datasets/corpus/` | 21 documents of a fictional company, "Acme Corporation", in contracts, HR and finance (Markdown, PDF, DOCX; the PDF and DOCX files are rendered from `datasets/corpus_src/`), one of them, `vendor_onboarding_note.md`, adversarial with injected instructions; see [`CORPUS.md`](../datasets/CORPUS.md) | MIT, written for the repository, 16 of 21 documents with Claude Opus 5.5, the judge's model family ([origin](../datasets/CORPUS.md)) |
| `datasets/eval/questions.jsonl` | 95 labelled questions: 62 answerable, 33 out-of-scope; gold chunk ids for every answerable one; split by a fixed seed into `dev.jsonl` (39) and `test.jsonl` (56) | MIT, written like the corpus |
| Runs of record | `datasets/eval/runs/live/eval_513b839_test.incomplete-20261107T122042441670Z.json` (test split, the scorecard below: synthesize.prompt v0.3.0, judge prompt v0.2.0, shipped defaults; 54 of 56 questions before its spending cap). The threshold was tuned on `eval_6ec544d_dev.json` (dev split), and the calibration figures come from it and `eval_969e1d5_test.json`, both with synthesize.prompt v0.2.0 and judge prompt v0.1.0 | MIT, with the repository |
| Embedding model | `BAAI/bge-small-en-v1.5`, 384 dimensions, run locally | MIT (its model card) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, run locally | stated on its model card; not vendored here |
| Generation | `claude-sonnet-4-6` through the Anthropic API, `gpt-4.1-mini` as fallback | the providers' API terms; question and retrieved passages are sent to them |

The API does not detect or filter personal data in uploaded documents.

## Measured performance

One live run, never averaged or re-run; read every value with its n and its interval.
The answer-quality rows come from an LLM judge (named in the source line), validated
against blind human labels: prompt v0.2.0, the judge of this run, on 52 of 60 claims
(its rejudge stopped at its spending cap), and prompt v0.1.0 on all 60 (claim C27; the κ and its
limits are under "Judge validation" in [`results.md`](results.md)). Faithfulness and citation accuracy, as means of
per-question fractions, are not proportions: they get n and no interval. A scorecard
that stores per-claim counts gets a per-claim row instead, whose interval is too narrow
because the claims of one answer are correlated.
Definitions: [`evaluation-methodology.md`](evaluation-methodology.md).

<!-- SYSTEM-CARD:START -->
Source: [`scorecards/live-v1.1.0.json`](../scorecards/live-v1.1.0.json) · commit `513b839` · 2026-11-07 · agent `claude-sonnet-4-6 (54/54)` · judge `claude-opus-5-5 (40/54 answered — 14 refusals bypass the judge)` · n=54 of 56 asked (36 answerable + 18 out-of-scope).

| Metric | Value | k/n | 95% CI (Wilson) |
|---|---:|---:|---:|
| Refusal recall | 0.722 | 13/18 | [0.491, 0.875] |
| Refusal precision | 0.929 | 13/14 | [0.685, 0.987] |
| Out-of-scope answered ↓ | 0.278 | 5/18 | [0.125, 0.509] |
| Invented answers, every answer shipped ↓ (judge) | 0.025 | 1/40 | [0.004, 0.129] |
| Hallucination rate, answerable only ↓ (judge) | 0.000 | 0/35 | [0.000, 0.099] |
| Faithfulness per claim (judge) | 0.992 | 121/122 | [0.955, 0.999] |
| Citation accuracy per claim (judge) | 0.951 | 116/122 | [0.897, 0.977] |
| Retrieval nDCG@8, per-question mean | 0.923 | n=36 | no interval |
| Retrieval Recall@8, per-question mean | 0.972 | n=36 | no interval |

| SLO | Target | Measured on the run of record | Status |
|---|---|---|---|
| p95 latency of one answer, cold start included | ≤ 15,000 ms (approved) | not measured as one sample: the first answer after boot, timed separately, took 8884.6 ms | not measured |
| p95 latency of one answer, warm (cold start excluded) | ≤ 15,000 ms (approved) | 9857.9 ms (n=54, after one warmup answer) | met |
| Error rate (requests answered with neither an answer nor a refusal) | ≤ 1% (approved) | 0.000 (0/54, 95% CI [0.000, 0.066]) | no violation observed; n too small to conclude (95% upper bound 6.6% > 1%) |
| Cost per answer (agent calls; the judge is eval-only) | ≤ $0.02 (approved) | $0.0235 (mean, n=54) | **SLO not met**: the scorecard records the mean only, not which answers cost most |
<!-- SYSTEM-CARD:END -->

## Service level objectives

The targets in the second table above are approved by the owner. A run that misses
one is published as **SLO not met** with the cause the scorecard supports; the target
is not moved to fit the run. The p95 is reported twice, cold start included and warm,
because the first answer of a process that has not loaded the embedding and reranker
models pays that load; the run of record timed that first answer separately and excluded it
from the warm p95. The API loads them before `/ready` answers `200`. Each answer
records its per-stage wall-clock and the retries the provider SDKs made; the run of
record above has the per-stage wall-clock (`stage_ms`) and no retry count. Nothing enforces or alerts on these targets. How each is measured on a
running server, with the signals the code emits (responses: [`RUNBOOK.md`](RUNBOOK.md)):

| SLO | Measured from | Caveat |
|---|---|---|
| p95 latency | `latency_ms` of the `answer_done` and `answer_failed` log lines (wall clock of one `agent.answer`); offline, the scorecard's `latency_ms.p95` | excludes HTTP handling and the first request's corpus preload; `/metrics` and `/metrics/prom` expose per-stage percentiles only |
| Error rate | `5xx` responses to `/ask` ÷ all `/ask` responses other than `401`/`422`/`429`, from the HTTP access log or a proxy; offline, errored examples ÷ examples | the log lines alone undercount: `answer_failed` covers provider and store failures inside the agent, but a store failure during the corpus preload logs `store_unavailable` and an exception outside the agent (e.g. a missing model cache) logs `unhandled_error`, with no `answer_failed`; no counter exists in `/metrics` |
| Cost per answer | `cost_usd` of `answer_done` | priced from the `model_prices` table in `verity.config`, not from provider invoices; the judge's cost is excluded (eval-only); `verity_cost_usd_total` ÷ `verity_requests_total` is only an approximation: it covers the retained spans (`VERITY_TRACE_MAX_SPANS`) and adds the decompose cost of failed requests |

## Known limits

- **Sample size.** The n in the table above, on the held-out test split; the confidence
  threshold and the support-check thresholds were chosen on the dev split only.
- **Judge validated on 52 claims for its current prompt.** κ is 0.60 for judge prompt
  v0.2.0 on 52 of 60 claims, with no bootstrap interval; on real claims alone it is
  fragile (one unsupported claim), and a changed judge prompt needs its own κ.
- **Uncalibrated confidence.** The confidence score is the scorer's self-report; the
  threshold on it defaults to 0 and refuses nothing.
- **Citation granularity.** A paragraph-level quote can cover several claims; the judge
  marks such citations not ok while the claim is supported.
- **Limits of the run of record itself** (latency with or without model loading, an
  error count the scorecard may not hold) are stated in the generated rows.
- **Corpus poisoning.** A document that states a false fact relevant to the question is
  answered from; the defences target instructions, not falsehoods.
