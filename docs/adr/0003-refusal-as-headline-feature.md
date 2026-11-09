# ADR 0003 — Refusal is a first-class, measured behaviour

**Status:** partly superseded · 2026-11-02 (accepted · 2026-09)

> Superseded: the mechanism, refusal below a calibrated `confidence_threshold`, is
> replaced by deterministic checks (no citation, low citation coverage, suspected
> injection), because the confidence score is not calibrated and the threshold now
> defaults to 0, which refuses nothing
> ([results, open issues](../results.md#open-issues-what-the-numbers-do-not-say)).
> Still in force: refusal is a first-class output, and refusal precision and recall are
> headline metrics.

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
