"""CLI surface is stable and every command runs (scaffold bodies exit 0)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from verity import __version__
from verity.cli import app
from verity.config import get_settings

runner = CliRunner()


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


def test_ask_without_key_and_without_stub_errors_actionably(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must show the user the exact env vars to set — no stacktrace, no
    silent fallback to the stub. The user asked for a real answer; a stub answer
    would be dishonest."""
    monkeypatch.delenv("VERITY_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VERITY_OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()
    try:
        result = runner.invoke(app, ["ask", "does the contract permit termination?"])
    finally:
        get_settings.cache_clear()
    assert result.exit_code != 0
    combined = (result.output or "") + (str(result.exception) if result.exception else "")
    assert "VERITY_ANTHROPIC_API_KEY" in combined
    assert "--stub" in combined
    # Explicit "no traceback leaked" check — BadParameter renders as a Typer message.
    assert "Traceback" not in combined
