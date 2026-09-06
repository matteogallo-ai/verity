"""CLI surface is stable and every command runs (scaffold bodies exit 0)."""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from verity import __version__
from verity.cli import MISSING_KEY_MESSAGE, app
from verity.config import get_settings

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _clean(output: str) -> str:
    """Strip ANSI escapes and collapse whitespace so wrapping / colouring can't
    break substring assertions on the terminal output."""
    without_ansi = _ANSI_RE.sub("", output)
    return re.sub(r"\s+", " ", without_ansi).strip()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_eval_run_ci_exits_clean() -> None:
    result = runner.invoke(app, ["eval", "run", "--ci"])
    assert result.exit_code == 0


def test_eval_compare_exits_clean() -> None:
    result = runner.invoke(app, ["eval", "compare"])
    assert result.exit_code == 0


def test_ingest_exits_clean(tmp_path: object) -> None:
    result = runner.invoke(app, ["ingest", str(tmp_path)])
    assert result.exit_code == 0


def test_missing_key_message_contains_actionable_hints() -> None:
    """Directly assert on the guard message constant — the source of truth.

    Reading the constant instead of parsing terminal output makes the assertion
    completely independent of Rich's panel wrapping, ANSI colouring, and terminal
    width. This is the primary regression guard.
    """
    assert "VERITY_ANTHROPIC_API_KEY" in MISSING_KEY_MESSAGE
    assert "VERITY_OPENAI_API_KEY" in MISSING_KEY_MESSAGE
    assert "--stub" in MISSING_KEY_MESSAGE


def test_ask_without_key_and_without_stub_errors_actionably(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end CLI check: no key → non-zero exit and the guard message reaches
    the terminal without a Python traceback.

    Robustness measures:
    - ``COLUMNS=200`` prevents Rick's BadParameter panel from wrapping tokens like
      ``VERITY_ANTHROPIC_API_KEY`` across lines;
    - ``TERM=dumb`` neutralises ANSI colouring;
    - :func:`_clean` strips any residual ANSI and collapses whitespace so the
      substring checks are tokenisation-independent.
    """
    monkeypatch.delenv("VERITY_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VERITY_OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()
    try:
        result = runner.invoke(
            app,
            ["ask", "does the contract permit termination?"],
            env={"COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1"},
        )
    finally:
        get_settings.cache_clear()

    # Behaviour first: the guard triggered and produced a non-zero exit.
    assert result.exit_code != 0

    combined = _clean((result.output or "") + (str(result.exception) if result.exception else ""))
    assert "VERITY_ANTHROPIC_API_KEY" in combined
    assert "--stub" in combined
    # Explicit "no traceback leaked" check — BadParameter must render as a Typer
    # message, not a raw Python stack.
    assert "Traceback" not in combined
