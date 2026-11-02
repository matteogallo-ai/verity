"""scripts/check_stale_narrative.py — what it flags and what it lets through."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_stale_narrative.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_stale_narrative", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


csn = _load()


def _kinds(text: str) -> list[str]:
    return [f.kind for f in csn.scan_text("x.py", text)]


@pytest.mark.parametrize(
    "line",
    [
        "Implementations land in S1.",
        "the S4.2 compare step",
        "Eval harness (S4-S7)",
        "sprints S1\u2013S4",
        "if S3+ needs conditionals",
        "the S1 invariant that citations point at real chunk text",
        "pre-S6.1 callers that build Answer",
        "Floors are the S3-established gates",
    ],
)
def test_flags_sprint_labels(line: str) -> None:
    assert "sprint" in _kinds(line)


@pytest.mark.parametrize(
    "line",
    [
        "Fusion lands in S2",
        "the scorecard lands at S7",
        "Next.js UI scheduled for S6",
        "streaming is coming in v1.2",
        "planned for v2",
        "Implementation coming",
        "Implementations coming soon",
    ],
)
def test_flags_future_promises(line: str) -> None:
    assert "future" in _kinds(line)


@pytest.mark.parametrize("line", ["# TODO tidy this", "FIXME: race", "XXX hack"])
def test_flags_todo_without_issue(line: str) -> None:
    assert _kinds(line) == ["todo"]


@pytest.mark.parametrize(
    "line",
    [
        "# TODO(#42) remove after the migration",
        "uploads go to an S3-compatible bucket",
        "Amazon S3",
        "AWS S3 bucket",
        "files/S3/path",
        "SHA d05355b and S10 are not sprints",
        "version 1.0.1 of the package",
        "Python 3.12",
    ],
)
def test_lets_legitimate_text_through(line: str) -> None:
    assert _kinds(line) == []


def test_scan_respects_scope_and_exclusions(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1  # lands in S3\n")
    (tmp_path / "src" / "image.png").write_bytes(b"\x89PNG S3 lands in S3")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "ENGINEERING_STANDARDS.md").write_text("Forbidden: lands in S<n>, S4.2\n")
    (tmp_path / "CHANGELOG.md").write_text("## S4.2 shipped\n")
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    (tmp_path / "tests" / "unit" / "t.py").write_text("# S4 regression\n")
    (tmp_path / "tests" / "unit" / "test_check_stale_narrative.py").write_text("# S4 fixture\n")
    (tmp_path / "Makefile").write_text("# demo path (S6)\n")
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text("# S3-established floors\n")
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "page.tsx").write_text("// S7 banner\n")
    (tmp_path / "README.md").write_text("ok\nEval harness (S4-S7)\n")

    findings = csn.scan(tmp_path)

    assert [str(f) for f in findings] == [
        ".github/workflows/ci.yml:1: sprint: # S3-established floors",
        "Makefile:1: sprint: # demo path (S6)",
        "README.md:2: sprint: Eval harness (S4-S7)",
        "src/a.py:1: sprint: x = 1  # lands in S3",
        "src/a.py:1: future: x = 1  # lands in S3",
        "tests/unit/t.py:1: sprint: # S4 regression",
        "web/page.tsx:1: sprint: // S7 banner",
    ]


def test_main_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    assert csn.main(["--root", str(tmp_path)]) == 0
    (tmp_path / "src" / "b.py").write_text("# FIXME\n")
    assert csn.main(["--root", str(tmp_path)]) == 1
    assert "src/b.py:1: todo: # FIXME" in capsys.readouterr().out
