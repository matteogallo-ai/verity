"""scripts/check_claims.py — table parsing, README consistency, test outcomes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_claims.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_claims", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cc = _load()

TABLE = """
| ID | Claim | Implementation | Tests | Status |
|---|---|---|---|---|
| C01 | works | `a.py` | `tests/test_x.py::test_pass` | shipped |
| C02 | later | — | — | roadmap |
"""

README = """# X

## Features

- It works (C01).

## Roadmap

- Later (C02).

## Other
"""


def test_parse_claims_reads_ids_status_and_tests() -> None:
    claims, problems = cc.parse_claims(TABLE)
    assert problems == []
    assert [(c.claim_id, c.status, c.tests) for c in claims] == [
        ("C01", "shipped", ["tests/test_x.py::test_pass"]),
        ("C02", "roadmap", []),
    ]


def test_parse_claims_flags_bad_rows() -> None:
    table = TABLE + "| C01 | dup | x | — | shipped |\n| C03 | odd | x | — | done |\n"
    _claims, problems = cc.parse_claims(table)
    assert any("duplicate claim id C01" in p for p in problems)
    assert any("shipped claim C01 lists no test" in p for p in problems)
    assert any("C03 has status 'done'" in p for p in problems)


def test_readme_consistent_with_claims() -> None:
    claims, _ = cc.parse_claims(TABLE)
    assert cc.check_readme(README, claims) == []


def test_readme_roadmap_item_listed_as_feature_fails() -> None:
    claims, _ = cc.parse_claims(TABLE)
    readme = README.replace("- It works (C01).", "- It works (C01).\n- Shiny (C02).")
    problems = cc.check_readme(readme, claims)
    assert any("Features: C02 is roadmap, not shipped" in p for p in problems)


def test_readme_feature_without_claim_id_fails() -> None:
    claims, _ = cc.parse_claims(TABLE)
    readme = README.replace("- It works (C01).", "- It works (C01).\n- Unbacked promise.")
    problems = cc.check_readme(readme, claims)
    assert any("bullet without a claim id: - Unbacked promise." in p for p in problems)


@pytest.mark.parametrize("bullet", ["* Unbacked promise.", "+ Unbacked promise."])
def test_readme_star_or_plus_bullet_without_claim_id_fails(bullet: str) -> None:
    claims, _ = cc.parse_claims(TABLE)
    readme = README.replace("- It works (C01).", f"- It works (C01).\n{bullet}")
    problems = cc.check_readme(readme, claims)
    assert any(f"bullet without a claim id: {bullet}" in p for p in problems)


def test_three_digit_claim_ids_are_matched_in_the_readme() -> None:
    claims, _ = cc.parse_claims(TABLE + "| C100 | more | x | `tests/test_x.py` | shipped |\n")
    readme = README.replace("- It works (C01).", "- It works (C01).\n- More (C100).")
    assert cc.check_readme(readme, claims) == []


def test_readme_missing_shipped_claim_fails() -> None:
    claims, _ = cc.parse_claims(TABLE)
    problems = cc.check_readme(README.replace("- It works (C01).", ""), claims)
    assert any("shipped claim C01 not listed" in p for p in problems)


def test_junit_key_handles_classes_and_params() -> None:
    assert cc._junit_key("tests/unit/test_a.py::TestX::test_y[1-2]") == (
        "tests.unit.test_a.TestX",
        "test_y",
    )
    assert cc._junit_key("tests/unit/test_a.py") == ("tests.unit.test_a", "")


def test_check_tests_reports_failed_skipped_and_missing(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "import pytest\n"
        "def test_pass():\n    assert True\n"
        "def test_fail():\n    assert False\n"
        "@pytest.mark.skip(reason='x')\ndef test_skip():\n    pass\n"
    )
    claims = [
        cc.Claim("C01", "shipped", ["tests/test_x.py::test_pass"]),
        cc.Claim("C02", "shipped", ["tests/test_x.py::test_fail"]),
        cc.Claim("C03", "shipped", ["tests/test_x.py::test_skip"]),
        cc.Claim("C04", "shipped", ["tests/test_x.py"]),
    ]
    problems = cc.check_tests(claims, tmp_path, run=True)
    lines = [p for p in problems if not p.startswith("pytest output")]
    assert lines == [
        "C02 (shipped): tests/test_x.py::test_fail failed",
        "C03 (shipped): tests/test_x.py::test_skip skipped",
        "C04 (shipped): tests/test_x.py failed",
    ]


def test_check_tests_reports_missing_files(tmp_path: Path) -> None:
    claims = [cc.Claim("C01", "shipped", ["tests/nope.py::test_a", "web/tests/A.test.tsx"])]
    problems = cc.check_tests(claims, tmp_path, run=True)
    assert problems == [
        "C01: test file not found: tests/nope.py::test_a",
        "C01: test file not found: web/tests/A.test.tsx",
    ]


def test_repository_claims_are_structurally_consistent() -> None:
    root = SCRIPT.parents[1]
    assert cc.main(["--root", str(root), "--no-run"]) == 0


@pytest.mark.parametrize(
    "nodeid",
    ["tests/t.py::test_renamed", "tests/t.py::TestBox::test_gone", "tests/t.py::Missing::test_a"],
)
def test_a_listed_test_that_is_not_defined_is_reported_without_running(
    tmp_path: Path, nodeid: str
) -> None:
    """A renamed test makes pytest abort the whole claims run in CI, where Postgres is;
    the structure check catches it on any machine."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "t.py").write_text(
        "def test_kept():\n    pass\n"
        "class TestBox:\n    def test_kept(self):\n        pass\n"
        "async def test_async():\n    pass\n"
    )
    kept = ["tests/t.py::test_kept", "tests/t.py::TestBox::test_kept", "tests/t.py::test_async"]
    claims = [cc.Claim("C01", "shipped", [*kept, "tests/t.py", nodeid])]
    assert cc.check_tests(claims, tmp_path, run=False) == [
        f"C01: test not defined in its file: {nodeid}"
    ]
