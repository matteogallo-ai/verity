# ADR 0003 — Refusal is a first-class, measured behaviour

**Status:** accepted · 2026-09

## Context
Most RAG systems always answer. The dangerous failure mode is a fluent, ungrounded answer
presented with no reliability signal. Verity's thesis is that *calibrated refusal* is
more valuable than marginal answer coverage for high-stakes documents.

## Decision
Treat refusal as a first-class output, not an error path:
- The agent produces a calibrated `Confidence`; below `confidence_threshold` it returns an
  honest "I don't know" with a rationale instead of an answer.
- The eval dataset **intentionally contains unanswerable questions** (answer not in corpus).
- The harness measures **refusal precision and recall** as headline metrics, alongside
  faithfulness and hallucination rate.

## Consequences
- Coverage is not the optimisation target; grounded-coverage-with-honest-refusal is.
  Contributions that raise answer rate by answering when they shouldn't are regressions
  (see CONTRIBUTING).
- The dataset schema carries an `answerable` flag and supports `reference_answer = None`.
- Confidence must be *calibrated*, not a raw model self-report; methodology is in
  `docs/evaluation-methodology.md`.
