# System card — Verity

**Owner:** Matteo Gallo · **Version:** 1.0.x · **Licence:** MIT (code, prompts, corpus,
eval set) · Security posture: [`SECURITY.md`](../SECURITY.md)

Every number on this page is generated from the committed scorecard by
`scripts/generate_headline_results.py`; CI runs it with `--check`. Every measured figure
is in the generated block; the prose around it quotes none.

## Measured performance

One live run, never averaged or re-run; read every value with its n and its interval. The answer-quality rows come
from an LLM judge that is the same model as the agent and is not validated against
human labels (claim C27, roadmap). Faithfulness and citation accuracy are means of
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
running server, with the signals the code emits:

| SLO | Measured from | Caveat |
|---|---|---|
| p95 latency | `latency_ms` of the `answer_done` and `answer_failed` log lines (wall clock of one `agent.answer`); offline, the scorecard's `latency_ms.p95` | `/metrics` and `/metrics/prom` expose per-stage percentiles only, not the end-to-end one; the run of record includes model loading |
| Error rate | count of `answer_failed` lines ÷ (`answer_done` + `answer_failed`) over the window; the same requests are `/ask` responses with status 503 | `verity_requests_total` counts `agent.confidence` spans, which a failed request may never open, so it cannot be the denominator |
| Cost per answer | `cost_usd` of `answer_done`; `verity_cost_usd_total` ÷ `verity_requests_total` for the session | priced from the `model_prices` table in `verity.config`, not from provider invoices; the judge's cost is excluded (eval-only) |
