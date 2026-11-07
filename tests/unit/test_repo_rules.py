"""The repo-specific rules of CLAUDE.md that are not checked by another job.

"make demo, CI and the tests never call a paid model": the tests are covered by the
network guard (tests/unit/test_network_guard.py); these cover the demo and CI.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from verity.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]
KEYS = ("VERITY_ANTHROPIC_API_KEY", "VERITY_OPENAI_API_KEY")


def _serve_command(target: str) -> str:
    """The ``verity serve`` line of a target, as make expands it (``-n`` runs nothing)."""
    out = subprocess.run(
        ["make", "-n", "--no-print-directory", target],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines = [line for line in out.splitlines() if "verity serve" in line]
    assert len(lines) == 1, (target, out)
    return lines[0]


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
@pytest.mark.parametrize("target", ["demo", "demo-api"])
def test_the_demo_targets_blank_the_provider_keys(target: str) -> None:
    """A key in .env would otherwise switch the demo API to the live agent."""
    command = _serve_command(target)
    for key in KEYS:
        assert f"{key}= " in command, (target, key)


def test_a_blank_environment_key_overrides_a_key_in_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("VERITY_ANTHROPIC_API_KEY=sk-not-a-real-key\n", encoding="utf-8")
    for key in KEYS:
        monkeypatch.setenv(key, "")
    assert not Settings(_env_file=env_file).has_llm_api_key  # type: ignore[call-arg]


def test_no_workflow_reads_a_secret_other_than_the_github_token() -> None:
    for workflow in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        used = set(re.findall(r"secrets\.(\w+)", workflow.read_text(encoding="utf-8")))
        assert used <= {"GITHUB_TOKEN"}, (workflow.name, used)


def test_every_marked_block_has_a_generator() -> None:
    """A block between START/END markers is generated: a hand-written one would carry
    figures no --check recomputes."""
    docs = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").rglob("*.md"))]
    names = {
        name
        for doc in docs
        for name in re.findall(r"<!-- ([A-Z0-9-]+):START -->", doc.read_text(encoding="utf-8"))
    }
    scripts = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted((REPO_ROOT / "scripts").glob("*.py"))
    )
    assert names
    assert sorted(n for n in names if f"{n}:START" not in scripts) == []
