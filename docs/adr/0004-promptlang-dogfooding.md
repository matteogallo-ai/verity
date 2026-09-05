# ADR 0004 — Orchestration prompts authored in PromptLang

**Status:** accepted · 2026-09

## Context
Verity's agent depends on several non-trivial prompts: question decomposition, the
faithfulness LLM-judge, and confidence/refusal scoring. These are production artefacts —
they need versioning, testing, and typed inputs, not f-strings scattered in code.
PromptLang (github.com/matteogallo-ai/promptlang) exists precisely to treat prompts as
typed, versioned, testable code.

## Decision
Author Verity's orchestration prompts in **PromptLang** (`prompts/*.prompt`), compiled and
consumed through the LLM runtime behind a thin adapter (`verity.llm`). Verity dogfoods
PromptLang rather than embedding raw strings.

## Rationale
- Prompts become versioned, testable units with typed inputs — the same discipline the
  rest of the codebase holds.
- It makes the portfolio narrative coherent: PromptLang is a *tool proven in production*
  by two systems that use it (Praxis, Verity), not a standalone artefact.

## Consequences
- A build dependency on PromptLang at the prompt-compilation boundary. The adapter is
  deliberately thin so PromptLang can be swapped for direct templating if it ever becomes
  a blocker — the agent depends on `verity.llm` interfaces, not on PromptLang directly.
- Prompt changes go through PromptLang's own test/lint flow before landing here.
