# ADR 0005 — Prompt files parsed locally, without PromptLang

**Status:** accepted · 2026-11 · supersedes [ADR 0004](0004-promptlang-dogfooding.md)

## Context
ADR 0004 decided that Verity's prompts would be authored in PromptLang and compiled
through it. That integration was never built. The prompts live in `prompts/*.prompt`
and `verity.llm.prompt` reads them with three regular expressions: `@directive` lines,
`system: |` / `user: |` block scalars, and `{{ var }}` substitution. PromptLang is not a
dependency. The README and architecture notes had to carry a disclaimer that the ADR
described something the code does not do.

## Decision
Keep the local parser and record it as the design. Prompts stay in versioned
`prompts/*.prompt` files, outside Python code, with this contract:

- A variable the template references but the caller does not supply raises `KeyError`;
  no prompt is ever rendered with a silent blank (`tests/unit/test_prompt.py`).
- `@version` is bumped with every behavioural change to a prompt. Changing any
  non-comment line changes model behaviour, so it is reviewed like code.
- Untrusted inputs (evidence, question, draft answer) enter a prompt only inside the
  delimiters rendered by `verity.agent.evidence`, after neutralisation, and every prompt's
  system section says they are data (`tests/unit/test_prompt_injection.py`).

## Rationale
- The format needs four features; a 140-line module with tests covers them. A compiler
  dependency would add a release coupling to another repository for no feature Verity uses.
- Callers depend only on `PromptTemplate.render`, so adopting a richer engine later is a
  change to one module.

Rejected: building the PromptLang integration now. Nothing in the agent needs
conditionals, partials or typed inputs beyond what `render` checks, and the
dependency would be added to satisfy a narrative, not a requirement.

## Consequences
- No conditionals, loops, partials or few-shot templating in prompts.
- `@model` is recorded but not used; the model comes from `Settings`.
- The syntax is "borrowed from PromptLang" only in the sense that the files look alike;
  they are not guaranteed to compile with PromptLang.
