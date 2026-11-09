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
    env, _, _ = command.partition("uv run verity serve")
    for key in KEYS:
        assert f" {key}= " in f" {env}", (target, key)


def test_a_blank_environment_key_overrides_a_key_in_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("VERITY_ANTHROPIC_API_KEY=sk-not-a-real-key\n", encoding="utf-8")
    for key in KEYS:
        monkeypatch.setenv(key, "")
    assert not Settings(_env_file=env_file).has_llm_api_key  # type: ignore[call-arg]


def test_no_workflow_reads_a_secret_other_than_the_github_token() -> None:
    """Any use of the secrets context (dot, index or toJSON) other than GITHUB_TOKEN."""
    for workflow in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        text = workflow.read_text(encoding="utf-8").replace("secrets.GITHUB_TOKEN", "")
        assert not re.search(r"\bsecrets\b", text), workflow.name


def test_every_marked_block_has_a_generator() -> None:
    """A block between START/END markers is generated: a hand-written one would carry
    figures no --check recomputes."""
    docs = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").rglob("*.md"))]
    names = {
        name
        for doc in docs
        for name in re.findall(r"<!-- ([A-Z0-9-]+):START -->", doc.read_text(encoding="utf-8"))
    }
    scripts = {
        p: p.read_text(encoding="utf-8") for p in sorted((REPO_ROOT / "scripts").glob("*.py"))
    }
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    tests = [p.read_text(encoding="utf-8") for p in sorted((REPO_ROOT / "tests").rglob("*.py"))]

    def checked_in_ci(script: Path) -> bool:
        if re.search(rf"scripts/{re.escape(script.name)}[^\n]*--check", ci):
            return True
        return any(script.name in t and '"--check"' in t for t in tests)

    assert names
    missing = []
    for name in sorted(names):
        generators = [p for p, text in scripts.items() if f'"<!-- {name}:START -->"' in text]
        if not generators or not any(checked_in_ci(p) for p in generators):
            missing.append(name)
    assert missing == []


def test_the_version_is_written_once_in_pyproject() -> None:
    """``verity.__version__`` is read from the installed metadata, so a release bump
    edits pyproject.toml only and the CLI, /health and the dashboard follow."""
    import tomllib

    import verity

    declared = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert verity.__version__ == declared["project"]["version"]
    source = (REPO_ROOT / "src" / "verity" / "__init__.py").read_text(encoding="utf-8")
    assert not re.search(r"""__version__\s*=\s*["']""", source)


def test_ci_runs_the_quickstart_eval_command() -> None:
    """The README's smoke test is the command the evaluation job runs, floors added, so
    a quickstart that stops working fails CI."""
    import yaml

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = re.search(
        r"^VERITY_STORE=memory (uv run verity eval run [^#\n]*?)\s*#", readme, re.M
    )
    assert quickstart is not None
    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())
    steps = [
        s for s in workflow["jobs"]["evaluation"]["steps"] if "verity eval run" in s.get("run", "")
    ]
    assert len(steps) == 1
    run = " ".join(steps[0]["run"].replace("\\\n", " ").split())
    assert steps[0]["env"]["VERITY_STORE"] == "memory"
    assert run.startswith(quickstart.group(1) + " ")
    assert "--floor-ndcg" in run and "--floor-recall" in run
