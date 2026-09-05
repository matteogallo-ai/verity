"""CLI surface is stable and every command runs (scaffold bodies exit 0)."""

from __future__ import annotations

from typer.testing import CliRunner

from verity import __version__
from verity.cli import app

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
