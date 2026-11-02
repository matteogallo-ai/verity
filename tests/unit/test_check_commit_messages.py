"""scripts/check_commit_messages.py — which subjects pass, and the git range."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_commit_messages.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_commit_messages", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ccm = _load()


@pytest.mark.parametrize(
    "subject",
    [
        "feat: add x",
        "fix(config): hold keys as SecretStr",
        "docs(src): describe what the code does",
        "refactor!: drop the v1 API",
        "ci: gate on vulture",
        "build(deps): bump torch",
    ],
)
def test_accepts_conventional_subjects(subject: str) -> None:
    assert ccm.is_conventional(subject)


@pytest.mark.parametrize(
    "subject",
    [
        "S6: per-request provenance stamping",
        "v1.0.1: honest scorecard denominators",
        "Feat: capitalised type",
        "feat:missing space",
        "feat: ",
        "feature: unknown type",
        "fix(Config): uppercase scope",
    ],
)
def test_rejects_other_subjects(subject: str) -> None:
    assert not ccm.is_conventional(subject)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_main_checks_only_the_given_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "S0: legacy subject")
    base = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "fix: good")
    monkeypatch.chdir(tmp_path)

    assert ccm.main([f"{base}..HEAD"]) == 0

    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "bad subject")
    assert ccm.main([f"{base}..HEAD"]) == 1
    assert "not a Conventional Commit: bad subject" in capsys.readouterr().out
