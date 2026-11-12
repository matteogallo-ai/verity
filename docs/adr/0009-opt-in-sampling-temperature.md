# ADR 0009 — An opt-in sampling temperature, sent on API version 2023-06-01

**Status:** proposed · 2026-11. The owner accepts it by merging the branch that adds
`VERITY_LLM_TEMPERATURE`. It amends [ADR 0007](0007-reproducibility-from-artefacts.md);
with the setting unset, ADR 0007 holds unchanged.

## Context

ADR 0007 sends no sampling parameter, so every call samples at the provider default
(temperature 1 on Anthropic). On a 20-question prospecting bench, two runs on the same
commit, corpus and chunks scored 25/40 and 29/40: four questions changed grade with
nothing but sampling between them. For answers about a collective labour agreement, the
same question getting a different answer is a product defect, not only eval noise.

The pinned `anthropic` 2.x SDK sends API version `2026-10-21`, which rejects
`temperature`, `top_p` and `top_k` with a 400 on every model. API version `2023-06-01`
still accepts a temperature on models before Claude 4.7, Claude Sonnet 4.6 included.

## Decision

- `Settings.llm_temperature` (`VERITY_LLM_TEMPERATURE`, 0 to 1) is unset by default.
- Unset, every request is the ADR 0007 request, on the SDK's API version.
- Set, the Anthropic client sends that temperature through `extra_body` on API version
  `2023-06-01`, for that request only (`extra_headers`); the OpenAI fallback sends the
  same temperature, so the router keeps one sampling policy.
- Runs record the value (`run_settings.llm_temperature` in the prospecting bench).

## Consequences

- The setting works only with models before Claude 4.7; with a later model, or once
  `2023-06-01` is retired, every call returns a 400 and the router reports the
  provider down. Unsetting the setting restores the default request.
- Temperature 0 lowers the variance of an answer; it does not make it identical
  across calls. Runs on the same commit are still separate samples, and ADR 0007's
  rule that the artefact, not a rerun, is the reproducible record still holds.
- `tests/unit/test_llm_temperature.py` checks, through the SDK's HTTP layer on a mock
  transport, the version header and body sent with and without the setting.
