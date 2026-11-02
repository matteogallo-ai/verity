# Prompts

The agent's and the judge's prompts, one file each, loaded at runtime by
`verity.llm.prompt.load_prompt`:

- `decompose.prompt`: split a question into ordered sub-questions.
- `synthesize.prompt`: draft an answer with verbatim citations from the evidence.
- `confidence.prompt`: confidence score and refusal decision for a draft answer.
- `judge_faithfulness.prompt`: LLM judge for claim-level grounding and citation checks
  (eval only).

## Format

The syntax is borrowed from PromptLang but parsed by Verity's own loader (regular
expressions in `verity/llm/prompt.py`); the PromptLang compiler is not used.

- Lines starting with `#` before the first block are comments and are not sent to
  the model; inside a block, `#` is ordinary text.
- `@version` and `@model` directives are recorded, not used at runtime.
- `system: |` and `user: |` blocks hold the message text; `{{ name }}` is replaced
  by the variable passed to `render()`. A missing variable raises `KeyError`.

Changing any non-comment line changes what the model sees. Such a change needs a
live eval run before its effect on the published numbers is known.
