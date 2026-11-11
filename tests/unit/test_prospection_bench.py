"""The prospection bench grades answers with the test grid, deterministically."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from tools.prospection.bench import (
    CitedQuote,
    Question,
    Result,
    Run,
    _fingerprint,
    decision_of,
    grade,
    load_questions,
    main,
    normalise,
    render_table,
)

from verity.agent.synthesizer import SynthesisResult
from verity.types import (
    Answer,
    Chunk,
    Citation,
    Confidence,
    RefusalReason,
    RetrievalHit,
    RetrieverKind,
    UsageStats,
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


_DOC = uuid4()
_CHUNK = Chunk(
    document_id=_DOC,
    text="B2) salaire minimum dès l'obtention du CFC 26.05 4'775.85",
    ordinal=0,
    char_start=0,
    char_end=57,
    section="CHAPITRE III > Article 8 - Salaires",
)
_HIT = RetrievalHit(chunk=_CHUNK, score=3.2, kind=RetrieverKind.RERANKED, rank=0)


_RUN: dict[str, Any] = {"language": "en", "max_tokens": 512}


def _refused(reason: RefusalReason, rationale: str) -> Answer:
    return Answer(
        query="q",
        text="refus",
        confidence=Confidence(score=0.0, refused=True, rationale=rationale, reason=reason),
        hits_used=(_HIT,),
    )


def _draft(text: str, quote: str, output_tokens: int = 40) -> SynthesisResult:
    citation = Citation(
        chunk_id=_CHUNK.id, document_id=_DOC, quote=quote, char_start=0, char_end=len(quote)
    )
    return SynthesisResult(
        text=text,
        citations=(citation,),
        usage=UsageStats(output_tokens=output_tokens),
        provider=None,
        model=None,
    )


def test_a_refusal_records_its_reason_coverage_draft_and_passages() -> None:
    draft = _draft("Le salaire minimum est de 26.05 de l'heure", "CFC 26.05")
    decision = decision_of(
        _refused(RefusalReason.LOW_CITATION_COVERAGE, "Only 20% ..."), draft, **_RUN
    )
    assert decision.reason == "low_citation_coverage"
    assert decision.rationale == "Only 20% ..."
    assert decision.draft == "Le salaire minimum est de 26.05 de l'heure"
    assert decision.citation_coverage is not None and 0 < decision.citation_coverage < 0.5
    assert decision.cited_rerank_score == 3.2
    assert [(h.rank, h.kind, h.section) for h in decision.hits] == [
        (0, "reranked", "CHAPITRE III > Article 8 - Salaires")
    ]


def test_without_a_draft_there_is_no_coverage() -> None:
    decision = decision_of(_refused(RefusalReason.NO_CITATION, "none"), None, **_RUN)
    assert decision.citation_coverage is None and decision.draft is None
    assert decision.draft_output_tokens is None and decision.draft_truncated is None


def test_a_draft_that_used_its_whole_budget_is_marked_truncated() -> None:
    """An unparseable draft cut off by max_tokens reads as an empty, uncited draft."""
    cut = decision_of(_refused(RefusalReason.NO_CITATION, "none"), _draft("", "x", 512), **_RUN)
    assert (cut.draft_output_tokens, cut.draft_truncated) == (512, True)
    whole = decision_of(_refused(RefusalReason.NO_CITATION, "none"), _draft("", "x", 300), **_RUN)
    assert whole.draft_truncated is False


def test_the_recorded_coverage_is_computed_in_the_run_language() -> None:
    draft = _draft("Le montant est calculé sur l'AVS", "le montant se calcule sur l'ÁVS")
    refusal = _refused(RefusalReason.LOW_CITATION_COVERAGE, "low")
    french = decision_of(refusal, draft, language="fr", max_tokens=512)
    english = decision_of(refusal, draft, language="en", max_tokens=512)
    assert french.citation_coverage == 1.0
    assert english.citation_coverage is not None and english.citation_coverage < 1.0


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
            "decision": None,
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
            "run_settings": {},
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


def test_a_blank_refusal_text_is_rejected(tmp_path: Path) -> None:
    questions = _write(tmp_path / "q.jsonl", _SALARY.model_dump(mode="json"))
    with pytest.raises(SystemExit):
        main(
            [
                "--stub",
                "--corpus",
                str(tmp_path),
                "--questions",
                str(questions),
                "--refusal-text",
                " ",
            ]
        )


@pytest.mark.parametrize("cap", ["0", "-5"])
def test_a_draft_cap_that_is_not_positive_is_rejected(
    tmp_path: Path, cap: str, capsys: pytest.CaptureFixture[str]
) -> None:
    questions = _write(tmp_path / "q.jsonl", _SALARY.model_dump(mode="json"))
    with pytest.raises(SystemExit):
        main(
            [
                "--stub",
                "--corpus",
                str(tmp_path),
                "--questions",
                str(questions),
                "--draft-max-tokens",
                cap,
            ]
        )
    assert "--draft-max-tokens must be positive" in capsys.readouterr().err


@pytest.mark.slow
def test_a_stub_run_writes_both_files(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "cct.md").write_text("# Article 8 \u2013 Salaires\n\n8.1 Jardinier CFC : 26.05.\n")
    questions = _write(tmp_path / "q.jsonl", _SALARY.model_dump(mode="json"))
    referral = "Contactez le secrétariat de l'association."
    args = ["--stub", "--corpus", str(corpus), "--questions", str(questions)]
    assert main([*args, "--refusal-text", referral]) == 0
    run = json.loads((tmp_path / "resultats.json").read_text())
    assert run["mode"] == "stub"
    assert run["run_settings"]["min_citation_coverage"] == 0.5
    assert run["run_settings"]["synthesis_max_tokens"] == 512
    assert run["run_settings"]["language"] == "en"
    assert run["run_settings"]["name_articles"] is False
    assert run["run_settings"]["refusal_text"] == referral
    # The scripted stub knows none of this corpus's themes, so it refuses.
    assert run["results"][0]["refused"]
    assert run["results"][0]["answer"] == referral
    decision = run["results"][0]["decision"]
    assert decision["hits"] and decision["hits"][0]["excerpt"]
    assert [r["question"]["id"] for r in run["results"]] == [1]
    assert (tmp_path / "tableau.md").read_text().startswith("**STUB")
    # A French draft that cites several articles outgrows the repo's 512 tokens; the
    # bench raises the cap for its own run without touching the default.
    assert main([*args, "--draft-max-tokens", "1024"]) == 0
    run = json.loads((tmp_path / "resultats.json").read_text())
    assert run["run_settings"]["synthesis_max_tokens"] == 1024
