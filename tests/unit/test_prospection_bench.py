"""The prospection bench grades answers with the test grid, deterministically."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.prospection.bench import (
    CitedQuote,
    Question,
    Result,
    Run,
    _fingerprint,
    grade,
    load_questions,
    main,
    normalise,
    render_table,
)

_SALARY = Question(
    id=1,
    question="Quel est le salaire minimum d'un jardinier CFC ?",
    article="Art. 8.1, cat. B2",
    expected="CHF 26.05 de l'heure",
    articles=("8",),
    facts=(("26.05",), ("4775.85",)),
)
_NIGHT = Question(
    id=20,
    question="Quel supplément pour le travail de nuit ?",
    article="Aucun",
    expected="Pas dans le texte",
    expect_refusal=True,
)


def _grade(question: Question, answer: str, **kwargs: object) -> tuple[str, int, bool]:
    options: dict[str, object] = {"refused": False, "error": None, "citations": ()}
    options.update(kwargs)
    g = grade(question, answer, **options)  # type: ignore[arg-type]
    return g.code, g.points, g.needs_review


@pytest.mark.parametrize("written", ["CHF 4'775.85", "4 775,85", "4\u2019775.85", "4775.85"])
def test_swiss_amounts_normalise_to_one_form(written: str) -> None:
    assert "4775.85" in normalise(written)


def test_accents_and_case_are_ignored() -> None:
    assert normalise("Lundi du Jeûne, ASCENSION") == "lundi du jeune, ascension"


def test_a_figure_with_its_article_scores_two() -> None:
    answer = "Selon l'art. 8.1, CHF 26.05 de l'heure, soit CHF 4'775.85 par mois."
    assert _grade(_SALARY, answer) == ("R+S", 2, False)


def test_the_section_of_a_cited_chunk_counts_as_the_source() -> None:
    cited = (CitedQuote(quote="B2) 26.05", section="CHAPITRE III > Article 8 \u2013 Salaires"),)
    assert _grade(_SALARY, "26.05 / heure, 4'775.85 / mois", citations=cited)[0] == "R+S"


def test_a_quote_that_starts_with_the_article_number_counts_as_the_source() -> None:
    cited = (CitedQuote(quote="8.1 Le barème des salaires", section=None),)
    assert _grade(_SALARY, "26.05 / heure, 4'775.85 / mois", citations=cited)[0] == "R+S"


def test_a_section_headed_art_counts_as_the_source() -> None:
    cited = (CitedQuote(quote="z", section="Art. 8 Salaires"),)
    assert _grade(_SALARY, "26.05 / heure, 4'775.85 / mois", citations=cited)[0] == "R+S"


def test_another_article_is_not_the_source() -> None:
    cited = (CitedQuote(quote="12.1 repas", section="Article 12 \u2013 Indemnité"),)
    assert _grade(_SALARY, "Art. 12 : 26.05 / heure, 4'775.85 / mois", citations=cited) == (
        "R",
        1,
        False,
    )


def test_a_missing_fact_is_wrong_and_flagged_for_review() -> None:
    assert _grade(_SALARY, "Art. 8.1 : CHF 26.05 de l'heure.") == ("F", -1, True)


def test_a_figure_inside_a_longer_number_does_not_count() -> None:
    assert _grade(_SALARY, "CHF 126.05 et 4775.850")[0] == "F"


def test_refusing_an_answerable_question_is_not_found() -> None:
    assert _grade(_SALARY, "I don't know", refused=True) == ("NT", 0, False)


def test_refusing_where_the_text_is_silent_scores_two() -> None:
    assert _grade(_NIGHT, "I don't know", refused=True) == ("R+S", 2, False)


def test_answering_where_the_text_is_silent_is_wrong() -> None:
    assert _grade(_NIGHT, "Un supplément de 25 %.") == ("F", -1, True)


def test_an_error_is_not_found_and_flagged() -> None:
    assert _grade(_SALARY, "", error="budget_exceeded: cap") == ("NT", 0, True)


def _write(path: Path, *questions: dict[str, object]) -> Path:
    path.write_text("\n".join(json.dumps(q) for q in questions) + "\n")
    return path


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    q = _SALARY.model_dump(mode="json")
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(_write(tmp_path / "q.jsonl", q, q))


def test_an_answerable_question_without_facts_is_rejected(tmp_path: Path) -> None:
    q = {**_SALARY.model_dump(mode="json"), "facts": []}
    with pytest.raises(ValueError, match="needs facts"):
        load_questions(_write(tmp_path / "q.jsonl", q))


@pytest.mark.parametrize("facts", [[[]], [[""]]])
def test_an_empty_fact_is_rejected(tmp_path: Path, facts: list[list[str]]) -> None:
    q = {**_SALARY.model_dump(mode="json"), "facts": facts}
    with pytest.raises(ValueError, match="empty"):
        load_questions(_write(tmp_path / "q.jsonl", q))


def test_the_fingerprint_names_every_ingested_file(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "annexe.md").write_text("# A")
    (tmp_path / "cct.md").write_text("# B")
    (tmp_path / "notes.xyz").write_text("not ingested")
    assert sorted(_fingerprint(tmp_path)) == ["cct.md", "sub/annexe.md"]


def _result(question: Question, code: str, points: int, error: str | None = None) -> Result:
    return Result.model_validate(
        {
            "question": question.model_dump(),
            "answer": "",
            "refused": False,
            "error": error,
            "citations": [],
            "cost_usd": 0.0,
            "latency_ms": 0.0,
            "graded": {"code": code, "points": points, "needs_review": False, "reason": "x"},
        }
    )


def _run(mode: str, *results: Result) -> Run:
    return Run.model_validate(
        {
            "mode": mode,
            "model": None,
            "git_sha": "abc",
            "corpus": {},
            "n_chunks": 0,
            "budget_usd": 3.0,
            "results": [r.model_dump() for r in results],
        }
    )


def test_the_table_totals_and_scales_to_100() -> None:
    table = render_table(_run("live", _result(_SALARY, "R+S", 2), _result(_NIGHT, "R", 1)))
    assert "R+S : 1 R : 1 NT : 0 F : 0 · score 3 / 4 · 75 / 100" in table
    assert "STUB" not in table


def test_a_negative_score_floors_at_zero_on_100() -> None:
    table = render_table(_run("live", _result(_SALARY, "F", -1)))
    assert "score -1 / 2 · 0 / 100" in table


def test_a_run_cut_short_says_so() -> None:
    cut = _result(_NIGHT, "NT", 0, error="budget_exceeded")
    table = render_table(_run("live", _result(_SALARY, "R", 1), cut))
    assert "**Run incomplet** : pas de réponse pour les questions [20]" in table


def test_a_complete_run_has_no_incomplete_line() -> None:
    assert "incomplet" not in render_table(_run("live", _result(_SALARY, "R", 1)))


def test_a_stub_run_says_its_score_measures_nothing() -> None:
    assert render_table(_run("stub")).startswith("**STUB")


@pytest.mark.slow
def test_a_stub_run_writes_both_files(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "cct.md").write_text("# Article 8 \u2013 Salaires\n\n8.1 Jardinier CFC : 26.05.\n")
    questions = _write(tmp_path / "q.jsonl", _SALARY.model_dump(mode="json"))
    assert main(["--stub", "--corpus", str(corpus), "--questions", str(questions)]) == 0
    run = json.loads((tmp_path / "resultats.json").read_text())
    assert run["mode"] == "stub"
    assert [r["question"]["id"] for r in run["results"]] == [1]
    assert (tmp_path / "tableau.md").read_text().startswith("**STUB")
