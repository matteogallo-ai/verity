"""The naive-baseline comparison applies the pre-registered rule, nothing else."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    path = REPO_ROOT / "scripts" / "compare_naive_baseline.py"
    spec = importlib.util.spec_from_file_location("compare_naive_baseline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _m(k: int, n: int, lo: float, hi: float) -> dict[str, object]:
    return {"k": k, "n": n, "value": k / n, "ci95_wilson": [lo, hi]}


def test_the_abstention_patterns_are_the_pre_registered_ones() -> None:
    prereg = (REPO_ROOT / "docs/preregistration/naive-rag-baseline.md").read_text()
    # The block is indented under its list item in the Markdown.
    block = re.search(r"```\n(.*?)\n\s*```", prereg, re.DOTALL)
    assert block is not None
    patterns = tuple(line.strip() for line in block.group(1).splitlines())
    from verity.eval.abstention import ABSTENTION_PATTERNS

    assert patterns == ABSTENTION_PATTERNS


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The documents do not mention a CEO's home address.", True),
        ("The provided evidence doesn't specify the retention period.", True),
        ("There is no information about parking in the documents.", True),
        ("I don't know based on the provided documents.", True),
        ("The notice period is 90 days.", False),
        ("Revenue did not decline; it rose 12 %.", False),
    ],
)
def test_self_abstention(text: str, expected: bool) -> None:
    assert _load().is_self_abstention(text) is expected


def test_overlapping_intervals_read_not_distinguishable() -> None:
    read = _load().read
    assert read(_m(1, 13, 0.01, 0.33), _m(3, 13, 0.08, 0.50)) == "not distinguishable"
    # Touching bounds overlap.
    assert read(_m(1, 13, 0.01, 0.33), _m(9, 13, 0.33, 0.87)) == "not distinguishable"


def test_disjoint_intervals_read_distinguishable_with_verity_direction() -> None:
    read = _load().read
    assert read(_m(1, 13, 0.01, 0.33), _m(13, 13, 0.77, 1.0)) == "distinguishable: Verity lower"
    assert read(_m(13, 13, 0.77, 1.0), _m(1, 13, 0.01, 0.33)) == "distinguishable: Verity higher"


def test_an_empty_denominator_is_not_read() -> None:
    empty = {"k": 0, "n": 0, "value": None, "ci95_wilson": None}
    assert _load().read(empty, _m(1, 13, 0.01, 0.33)) == "not computable (n = 0)"


def test_missing_markers_fail_loudly() -> None:
    with pytest.raises(SystemExit, match="markers"):
        _load().replace_block("no markers\n", "block")


def test_the_committed_comparison_matches_its_artefacts() -> None:
    assert _load().main(["--check"]) == 0


def test_the_comparison_runs_on_the_questions_both_arms_completed() -> None:
    import json

    data = json.loads((REPO_ROOT / "scorecards/naive-baseline-comparison.json").read_text())
    for arm in data["arms"].values():
        assert arm["n_examples"] == len(data["compared_ids"])
        assert arm["metrics"]["answerable_answered"]["n"] + arm["metrics"]["unanswerable_answered"][
            "n"
        ] == len(data["compared_ids"])


def _committed_copy(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    import shutil

    scorecard, readme = tmp_path / "cmp.json", tmp_path / "README.md"
    shutil.copy(REPO_ROOT / "scorecards/naive-baseline-comparison.json", scorecard)
    shutil.copy(REPO_ROOT / "README.md", readme)
    return scorecard, readme, ["--scorecard", str(scorecard), "--readme", str(readme)]


@pytest.mark.parametrize("target", ["scorecard", "readme"])
def test_check_fails_on_drift_and_writes_nothing(tmp_path: Path, target: str) -> None:
    mod = _load()
    scorecard, readme, args = _committed_copy(tmp_path)
    # The block links its scorecard by path: render once so only the drift below differs.
    assert mod.main(args) == 0
    assert mod.main(["--check", *args]) == 0
    path = scorecard if target == "scorecard" else readme
    text = path.read_text()
    stale = text.replace("not distinguishable", "distinguishable: Verity lower", 1)
    assert stale != text
    path.write_text(stale)
    assert mod.main(["--check", *args]) == 1
    assert path.read_text() == stale


def test_arms_are_compared_on_the_questions_both_completed(tmp_path: Path) -> None:
    """Cut the naive arm to its first 10 audits: only those 10 are compared."""
    import json

    data = json.loads((REPO_ROOT / "scorecards/naive-baseline-comparison.json").read_text())
    naive = json.loads((REPO_ROOT / data["arms"]["naive"]["source_artefact"]).read_text())
    naive["per_audit"] = naive["per_audit"][:10]
    cut = tmp_path / "naive.json"
    cut.write_text(json.dumps(naive))
    built = _load().build(REPO_ROOT / data["arms"]["verity"]["source_artefact"], cut)
    first_ten = [a["example_id"] for a in naive["per_audit"]]
    assert built["compared_ids"] == first_ten
    assert built["arms"]["verity"]["n_examples"] == 10
    assert set(built["completed_by_one_arm_only"]) == set(data["compared_ids"]) - set(first_ten)


def test_paired_rows_carry_an_exact_mcnemar_test() -> None:
    """The arms answer the same questions: answerable answered is 16/21 vs 21/21, five
    discordant questions all one way, McNemar exact p = 2 * 0.5**5 = 0.0625."""
    import json

    data = json.loads((REPO_ROOT / "scorecards/naive-baseline-comparison.json").read_text())
    test = data["paired_tests"]["answerable_answered"]
    assert (test["verity_only"], test["naive_only"], test["n_paired"]) == (0, 5, 21)
    assert test["p_mcnemar_exact"] == pytest.approx(0.0625)
    assert data["readings"]["answerable_answered"] == "not distinguishable (McNemar p = 0.062)"
    # Per-claim rows are not paired: the two arms make different claims.
    assert data["paired_tests"]["faithfulness_per_claim"] is None


def test_agreeing_distinguishable_readings_are_not_reported_as_differing() -> None:
    """Both rules say Verity answers fewer unanswerable questions: they agree."""
    mod = _load()
    test = {"verity_only": 0, "naive_only": 9, "n_paired": 9, "p_mcnemar_exact": 0.0039}
    paired = mod.verdict_paired(test, "lower")
    overlap = mod.verdict_overlap(_m(0, 9, 0.0, 0.3), _m(9, 9, 0.7, 1.0), "lower")
    assert paired == overlap == "Verity better"


def test_a_row_too_small_to_reach_the_threshold_says_so() -> None:
    """With 2 paired questions the smallest exact McNemar p is 0.5: no result is possible."""
    reading = _load().read_paired(
        {"verity_only": 0, "naive_only": 2, "n_paired": 2, "p_mcnemar_exact": 0.5}, "lower"
    )
    assert reading == "not testable: 2 paired questions, smallest possible p = 0.500"
