"""Thin adapter that consumes ``.prompt`` files (PromptLang sources).

ADR 0004 records the decision to author orchestration prompts in PromptLang and to
front them behind a *thin adapter* so the runtime can swap for direct templating if
PromptLang ever becomes a blocker. This module is that adapter.

The subset of PromptLang syntax we consume:

- ``@version`` and ``@model`` directive lines (recorded, not used at runtime).
- ``system: |`` and ``user: |`` block-scalar sections.
- ``{{ var }}`` substitutions inside the rendered text (whitespace around the
  variable name is tolerated).

Full PromptLang is a separate project; if S3+ needs conditionals, few-shot templating
or partials, we swap this module for the real compiler behind the same interface
(:class:`PromptTemplate.render`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from verity.types import Message, Role

_DIRECTIVE_RE = re.compile(r"^@(\w+)\s+(.+?)\s*$")
_BLOCK_HEADER_RE = re.compile(r"^(system|user|assistant):\s*\|\s*$")
_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")

_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


@dataclass(frozen=True)
class PromptTemplate:
    """A parsed ``.prompt`` file — three ordered message templates + directives."""

    name: str
    version: str
    model: str
    system: str = ""
    user: str = ""
    assistant: str = ""
    directives: dict[str, str] = field(default_factory=dict)

    def render(self, **variables: object) -> list[Message]:
        """Substitute ``{{ var }}`` occurrences and emit a ``list[Message]``.

        Any variable referenced in the template but not supplied raises ``KeyError``
        — silent-blank rendering would produce prompts that look right but omit the
        user's actual question, which is exactly the class of bug PromptLang exists
        to prevent.
        """
        messages: list[Message] = []
        for role, body in (
            (Role.SYSTEM, self.system),
            (Role.USER, self.user),
            (Role.ASSISTANT, self.assistant),
        ):
            if not body.strip():
                continue
            rendered = _substitute(body, variables)
            messages.append(Message(role=role, content=rendered))
        return messages


def _substitute(template: str, variables: dict[str, object]) -> str:
    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variables:
            raise KeyError(f"prompt variable {key!r} not supplied")
        return str(variables[key])

    return _VAR_RE.sub(_replace, template)


def parse_prompt(source: str, *, name: str) -> PromptTemplate:
    """Parse a ``.prompt`` source string into a :class:`PromptTemplate`."""
    directives: dict[str, str] = {}
    blocks: dict[str, list[str]] = {"system": [], "user": [], "assistant": []}
    current_block: str | None = None
    block_indent: int | None = None

    for raw_line in source.splitlines():
        stripped = raw_line.strip()
        if current_block is None:
            if not stripped or stripped.startswith("#"):
                continue
            directive_match = _DIRECTIVE_RE.match(stripped)
            if directive_match:
                directives[directive_match.group(1)] = directive_match.group(2)
                continue
            block_match = _BLOCK_HEADER_RE.match(stripped)
            if block_match:
                current_block = block_match.group(1)
                block_indent = None
                continue
        else:
            if not raw_line.strip() and block_indent is None:
                # Preserve blank lines *inside* the block once we've started capturing.
                blocks[current_block].append("")
                continue
            block_match = _BLOCK_HEADER_RE.match(stripped) if stripped else None
            leading = len(raw_line) - len(raw_line.lstrip())
            if block_match and (block_indent is None or leading <= (block_indent or 0) - 1):
                # A new block starts — commit and switch.
                current_block = block_match.group(1)
                block_indent = None
                continue
            if block_indent is None and raw_line.strip():
                block_indent = leading
            content = raw_line[block_indent:] if block_indent is not None else raw_line.lstrip()
            blocks[current_block].append(content)

    def _join(section: list[str]) -> str:
        return "\n".join(section).rstrip() + ("\n" if section and section[-1] == "" else "")

    return PromptTemplate(
        name=name,
        version=directives.get("version", "0.0.0"),
        model=directives.get("model", "any"),
        system=_join(blocks["system"]).rstrip(),
        user=_join(blocks["user"]).rstrip(),
        assistant=_join(blocks["assistant"]).rstrip(),
        directives=directives,
    )


@lru_cache(maxsize=32)
def load_prompt(name: str, *, prompts_dir: Path | None = None) -> PromptTemplate:
    """Load ``prompts/<name>.prompt`` (or ``prompts_dir/<name>.prompt``) and cache it."""
    base = prompts_dir or _PROMPTS_DIR
    path = base / f"{name}.prompt"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt {name!r} not found at {path}")
    return parse_prompt(path.read_text(encoding="utf-8"), name=name)


__all__ = ["PromptTemplate", "load_prompt", "parse_prompt"]
