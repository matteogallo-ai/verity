# ADR 0007 — Eval reproducibility rests on recorded artefacts, not on sampling

**Status:** accepted · 2026-11

## Context
The agent and the judge used to send `temperature=0.0` (Anthropic through `extra_body`,
because `anthropic` 1.x had removed the typed argument) on the assumption that it made
live runs repeatable. Two facts make that untenable:

- Anthropic API version `2026-10-21` rejects `temperature`, `top_p` and `top_k` with a
  `400`, even at their default values and even when sent through `extra_body`. Claude
  4.7+ models already accept only the defaults. `anthropic` 2.x is the first SDK line
  that sends this API version.
- Even where it was accepted, `temperature=0` never guaranteed identical outputs across
  calls; a rerun of a live eval was always a new sample.

## Decision
- No LLM call sends a sampling parameter. The `LLMClient` / `RoutingClient` protocol
  has no `temperature` argument; every provider, OpenAI included, uses its default, so
  there is one sampling policy to reason about.
- What is reproduced is the **run artefact**, not the model output. A live run is
  stored once as an `EvalRun` (scorecard, git SHA, agent / judge / embedding model ids,
  and one `PerExampleAudit` per example: answer, refusal rationale, citations, hits
  used, per-claim judge verdicts, token usage). Published numbers are computed from that
  artefact; re-reading it costs nothing and gives the same numbers every time.
- Runs of record under `datasets/eval/runs/live/` are never overwritten. A new live run
  is a new artefact, compared with the previous one by `verity eval compare`.

## Consequences
- Two live runs on the same commit can differ. That difference is model variance, and
  it is reported as such, not hidden behind a sampling setting.
- `tests/unit/test_llm_sdk_schema.py` binds the request kwargs to the installed SDK and
  fails if any sampling key reaches the request, directly or through `extra_body`.
- A major `anthropic` bump changes the API version; the pin `>=2.0,<3.0` makes that an
  explicit decision.
