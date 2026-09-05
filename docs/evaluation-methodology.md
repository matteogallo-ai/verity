# Evaluation methodology

This page is the reason Verity exists. Anyone can wire up RAG; the discipline is in
measuring it honestly, gating it in CI, and tracking it over time. This document states
exactly what is measured, how, and what would make a number untrustworthy.

## Principles

1. **No fabricated numbers.** Every figure in the README comes from `verity eval run`
   tied to a git SHA. Placeholders are labeled as such until the harness lands (S4).
2. **CI eval is free and deterministic.** Retrieval and refusal metrics run with local
   embeddings and a stubbed judge — no API keys, reproducible bit-for-bit, gate-able on
   every PR. The published headline scorecard is a separate, real run (real judge model).
3. **Refusal is measured, not assumed.** The dataset contains questions with no answer in
   the corpus; the system is scored on getting those *right by refusing*.

## The dataset

Labeled examples (`datasets/eval/`), each with:

| Field | Meaning |
|---|---|
| `question` | the query |
| `answerable` | whether the corpus actually contains the answer |
| `reference_answer` | gold answer, or `null` when `answerable = false` |
| `relevant_chunk_ids` | gold set of chunks that should be retrieved |
| `tags` | slices for per-category analysis (e.g. `multi-hop`, `numeric`, `out-of-scope`) |

The dataset deliberately mixes: single-hop answerable, **multi-hop** (needs decomposition),
**numeric/table** questions, and **unanswerable / out-of-scope** questions. A system that
scores well on answerable questions but can't refuse the out-of-scope ones fails Verity's
bar regardless of its faithfulness number.

## Metrics

### Retrieval (deterministic)
- **precision@k / recall@k** against `relevant_chunk_ids`, measured on the final
  fused+reranked set.
- **nDCG@k** to credit ranking quality, not just presence.
- Reported per stage in traces (dense-only, sparse-only, fused, reranked) so a retrieval
  regression can be localised.

### Answer quality (LLM-as-judge)
- **Faithfulness** — the drafted answer is decomposed into atomic claims; each claim is
  checked for support in the *retrieved context*. Faithfulness = supported claims / total
  claims. This measures grounding, independent of whether the answer matches the reference.
- **Citation accuracy** — each inline citation is verified to actually support the claim it
  is attached to (a citation that points to an irrelevant passage is a failure even if the
  claim happens to be true).
- **Hallucination rate** — fraction of answerable answers containing ≥1 unsupported claim.

### Refusal calibration (headline)
Framed as a classification problem over `answerable`:
- **Refusal recall** — of the unanswerable questions, the fraction correctly refused.
  Low recall = the system hallucinates on out-of-scope questions.
- **Refusal precision** — of the questions the system refused, the fraction that were
  genuinely unanswerable. Low precision = the system is uselessly timid on answerable ones.
- The `confidence_threshold` is the operating point on this precision/recall trade-off; the
  methodology reports the point used and, where feasible, the curve.

### Operational
- **Latency** p50 / p95 / p99, end-to-end and per stage (from traces).
- **Cost/query** in USD, aggregated from token usage across all LLM calls in a query.

## Judge reliability

The faithfulness/citation judge is an LLM and is itself a source of error. Mitigations:
- The judge prompt is authored in PromptLang, versioned, and its own agreement with human
  labels is spot-checked on a held-out slice; the agreement figure is reported next to the
  scorecard so the judge's trust level is explicit.
- Claim-level judging (not whole-answer) reduces variance and makes disagreements
  inspectable.
- In CI the judge is a deterministic stub, so CI never depends on judge variance; only the
  out-of-band headline run uses the real judge.

## Regression tracking

Each run is persisted as an `EvalRun` (scorecard + git SHA + embedding/judge model ids).
`verity eval compare` surfaces deltas across recent runs so a quality regression shows up
as a diff, the same way a production AI team catches model regressions before shipping.

## What would make a number untrustworthy

Stated plainly, because reviewers will ask:
- a scorecard without a git SHA;
- a headline run using the stubbed judge (that's CI mode, not a real result);
- faithfulness reported without the judge's human-agreement figure;
- a refusal recall near 1.0 with refusal precision unreported (trivially achieved by
  refusing often);
- eval examples that leak into any tuning of the retriever or prompts.
