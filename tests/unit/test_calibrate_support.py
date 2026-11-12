"""tools/prospection/calibrate_support.py: replay a support rule on recorded bench runs.

No model is called: the replay reads the answers and the refused drafts the bench
recorded, with their verified quotes, and applies ``verity.agent.support`` to them.
"""

from __future__ import annotations

from typing import Any

from tools.prospection.calibrate_support import Outcome, replay_run, tally

QUESTION = "Quel est le salaire minimum d'un jardinier qui vient d'obtenir son CFC ?"
QUOTE = "B2) salaire minimum dès l'obtention du CFC 26.05 4'775.85"
SECTION = "Article 8 Salaires"
RIGHT = (
    "Selon l'art. 8, le salaire minimum d'un jardinier qualifié qui vient d'obtenir son "
    "CFC est de 26.05 CHF à l'heure, soit 4'775.85 CHF par mois."
)


def _question(qid: int, *, expect_refusal: bool = False) -> dict[str, Any]:
    return {
        "id": qid,
        "question": QUESTION,
        "article": "Art. 8",
        "expected": "26.05",
        "articles": ["8"],
        "facts": [] if expect_refusal else [["26.05"]],
        "expect_refusal": expect_refusal,
    }


def _answered(qid: int, answer: str, code: str) -> dict[str, Any]:
    return {
        "question": _question(qid),
        "answer": answer,
        "refused": False,
        "citations": [{"quote": QUOTE, "section": SECTION}],
        "decision": {"reason": None, "draft_citations": None},
        "graded": {"code": code},
    }


def _refused_draft(qid: int, draft: str, *, verified: bool = True) -> dict[str, Any]:
    return {
        "question": _question(qid),
        "answer": "Refus.",
        "refused": True,
        "citations": [],
        "decision": {
            "reason": "low_citation_coverage",
            "draft": draft,
            "draft_citations": [
                {"quote": QUOTE, "chunk_index": 0, "section": SECTION, "verified": verified}
            ],
        },
        "graded": {"code": "NT"},
    }


def _run(*results: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "live",
        "run_settings": {"language": "fr", "min_citation_coverage": 0.5},
        "results": list(results),
    }


def test_a_correct_draft_refused_for_coverage_now_ships_and_grades_right() -> None:
    (row,) = replay_run(_run(_refused_draft(1, RIGHT)), rule="figures")
    assert row.outcome is Outcome.NT_TO_RIGHT
    assert row.new_reason is None


def test_a_draft_whose_figure_is_only_in_an_unverified_quote_stays_refused() -> None:
    (row,) = replay_run(_run(_refused_draft(1, RIGHT, verified=False)), rule="figures")
    assert row.outcome is Outcome.STILL_REFUSED


def test_a_wrong_figure_in_a_refused_draft_would_ship_as_wrong_only_if_quoted() -> None:
    wrong = RIGHT.replace("26.05", "4'775.85 ou 26.50")
    (row,) = replay_run(_run(_refused_draft(1, wrong)), rule="figures")
    assert row.outcome is Outcome.STILL_REFUSED
    assert row.new_reason == "unquoted_figure"


def test_an_answer_that_was_right_and_now_fails_is_counted_as_lost() -> None:
    extra = RIGHT + " La base annuelle est de 2'200 heures."
    rows = replay_run(_run(_answered(1, RIGHT, "R+S"), _answered(2, extra, "R+S")), rule="figures")
    assert [r.outcome for r in rows] == [Outcome.UNCHANGED, Outcome.RIGHT_TO_NT]


def test_a_wrong_answer_that_now_fails_is_counted_as_caught() -> None:
    wrong = RIGHT.replace("26.05", "27.95")
    (row,) = replay_run(_run(_answered(1, wrong, "F")), rule="figures")
    assert row.outcome is Outcome.F_CAUGHT


def test_the_tally_counts_each_outcome() -> None:
    rows = replay_run(
        _run(_refused_draft(1, RIGHT), _answered(2, RIGHT.replace("26.05", "27.95"), "F")),
        rule="figures",
    )
    counts = tally(rows)
    assert counts[Outcome.NT_TO_RIGHT] == 1
    assert counts[Outcome.F_CAUGHT] == 1
    assert counts[Outcome.NEW_F] == 0


def test_a_figure_from_the_cited_passage_counts_once_the_passages_are_given() -> None:
    dated = RIGHT + " Ces salaires valent depuis l'avenant du 13 décembre 2022."
    run = _run(_answered(1, dated, "R+S"))
    (without,) = replay_run(run, rule="figures")
    assert without.outcome is Outcome.RIGHT_TO_NT
    passage = f"Avenant du 13 décembre 2022\n{QUOTE}"
    (with_passage,) = replay_run(run, rule="figures", passages=[passage])
    assert with_passage.outcome is Outcome.UNCHANGED
