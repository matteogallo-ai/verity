# Prompts (authored in PromptLang)

Verity's orchestration prompts are authored here as PromptLang sources and compiled/
consumed through `verity.llm` (ADR 0004). They are versioned and testable units, not
inline f-strings. Three prompts drive the agent:

- `decompose.prompt` — split a complex question into ordered sub-questions.
- `judge_faithfulness.prompt` — LLM-as-judge for claim-level grounding + citation checks.
- `confidence.prompt` — calibrated confidence + refusal rationale.

The `.prompt` files below are S0 specifications (typed inputs + intent). They are wired to
the compiler in S3/S4. Kept provider-agnostic; the runtime binds the model.
