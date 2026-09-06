"""Prompt loader: block parsing, variable substitution, missing-var strictness."""

from __future__ import annotations

from pathlib import Path

import pytest

from verity.llm.prompt import PromptTemplate, load_prompt, parse_prompt
from verity.types import Role


def test_parse_shipped_decompose_prompt() -> None:
    p = load_prompt("decompose")
    assert isinstance(p, PromptTemplate)
    assert p.version == "0.1.0"
    assert "decompose enterprise-document questions" in p.system.lower()
    assert "{{ question }}" in p.user


def test_render_substitutes_variables() -> None:
    src = (
        "@version 1.0.0\n@model any\nsystem: |\n  You are helpful.\nuser: |\n  Hello {{ name }}!\n"
    )
    tpl = parse_prompt(src, name="hello")
    messages = tpl.render(name="Matteo")
    assert [m.role for m in messages] == [Role.SYSTEM, Role.USER]
    assert messages[1].content.strip() == "Hello Matteo!"


def test_render_missing_variable_raises() -> None:
    src = "@version 1 @model any\n" + "user: |\n  {{ x }}\n"
    tpl = parse_prompt(src, name="x")
    with pytest.raises(KeyError, match="x"):
        tpl.render()


def test_load_prompt_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("nonexistent-prompt", prompts_dir=tmp_path)


def test_parse_handles_only_directives() -> None:
    tpl = parse_prompt("@version 0.1.0\n@model any\n", name="empty")
    assert tpl.system == ""
    assert tpl.user == ""
