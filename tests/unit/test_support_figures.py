"""VERITY_SUPPORT_CHECK=figures: every figure of the answer must be quoted.

On the French collective agreements, the word-coverage threshold refused correct
answers that restate the question ("un jardinier qui vient d'obtenir son CFC") around a
quoted wage: 9 of the 10 answerable questions refused on the prospection bench's
latest runs were right. The ``figures`` rule moves the weight to what a wrong answer
gets wrong: each amount, duration, date and percentage must be in a verified quote
(or in the question, or in the cited article's heading), and the words the answer
takes from the question or the heading count as covered. ``coverage`` stays the
default, unchanged.

The drafts and quotes below are from the bench's runs on public agreements.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterator
from typing import ClassVar
from uuid import uuid4

import pytest
from pydantic import ValidationError

from verity.agent import support
from verity.agent.figures import figures
from verity.agent.support import check_support
from verity.agent.synthesizer import CitedSynthesizer
from verity.config import Settings, get_settings
from verity.types import (
    Chunk,
    Citation,
    Completion,
    Message,
    Provider,
    RefusalReason,
    RetrievalHit,
    RetrieverKind,
    UsageStats,
)


def _hit(text: str, section: str | None) -> RetrievalHit:
    chunk = Chunk(
        document_id=uuid4(),
        text=text,
        ordinal=0,
        char_start=0,
        char_end=len(text),
        section=section,
    )
    return RetrievalHit(chunk=chunk, score=5.0, kind=RetrieverKind.RERANKED, rank=0)


def _cited(
    text: str, section: str | None = None, passage: str | None = None
) -> tuple[Citation, list[RetrievalHit]]:
    """A citation quoting ``text`` from a passage (by default, the quote itself)."""
    hit = _hit(passage or text, section)
    citation = Citation(
        chunk_id=hit.chunk.id,
        document_id=hit.chunk.document_id,
        quote=text,
        char_start=0,
        char_end=len(text),
    )
    return citation, [hit]


def _check(
    answer: str,
    quote: str,
    section: str | None,
    question: str,
    rule: str,
    *,
    passage: str | None = None,
    uncited: str | None = None,
) -> object:
    citation, hits = _cited(quote, section, passage)
    if uncited is not None:
        hits.append(_hit(uncited, None))
    return check_support(
        answer,
        (citation,),
        hits,
        min_coverage=0.5,
        min_rerank_score=-1e9,
        language="fr",
        rule=rule,  # type: ignore[arg-type]
        question=question,
    )


GARDENER_QUESTION = "Quel est le salaire minimum d'un jardinier qui vient d'obtenir son CFC ?"
GARDENER_DRAFT = (
    "Selon l'art. 8 de la convention, le salaire minimum d'un jardinier qualifié qui vient "
    "d'obtenir son CFC (catégorie B2) est de 26.05 CHF à l'heure, soit 4'775.85 CHF par mois."
)
GARDENER_QUOTE = "B2) \N{EN DASH} salaire minimum dès 1'obtention du CFC 26.05 4'775.85"
GARDENER_SECTION = "Article 8 \N{EN DASH} Salaires"


# --- figures -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "written", ["CHF 4'775.85", "4 775,85", "4775.85", "4\u2019775.85", "4\u202f775,85"]
)
def test_a_swiss_amount_reads_the_same_however_it_is_written(written: str) -> None:
    assert figures(written, language="fr") == {"4775.85"}


def test_trailing_zeros_and_dashes_do_not_change_an_amount() -> None:
    assert figures("CHF 31.00", language="fr") == figures("31.\N{EN DASH}", language="fr") == {"31"}


def test_list_numbers_and_article_references_are_not_figures() -> None:
    answer = "Selon l'art. 15 ch. 5, al. 2 :\n1. **Paiement** à 125%\n2. Report de 28 heures"
    assert figures(answer, language="fr", skip_references=True) == {"125", "28"}


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Après 5 ans, 99 jours de vacances.", {"5", "99"}),
        ("25 % et 99 %", {"25", "99"}),
        ("1, 2 et 9 mois", {"1", "2", "9"}),
        ("Entre 8 et 99 semaines.", {"8", "99"}),
        ("Selon l'art. 12, 5 semaines de vacances.", {"5"}),
        ("Selon les art. 3, 4 et 5, le délai est de 2 mois.", {"4", "5", "2"}),
    ],
)
def test_only_the_number_right_after_a_reference_label_is_skipped(
    answer: str, expected: set[str]
) -> None:
    assert figures(answer, language="fr", skip_references=True) == expected


@pytest.mark.parametrize(("written", "value"), [("$1,000", "1000"), ("4,775.85", "4775.85")])
def test_an_english_comma_groups_thousands(written: str, value: str) -> None:
    assert figures(written, language="en") == {value}


def test_a_compound_number_word_is_not_read_as_its_parts() -> None:
    """ "vingt-quatre" is not 20 and 4; reading it so would let a wrong 4 through."""
    assert figures("après vingt-quatre mois dans un bâtiment neuf", language="fr") == set()


def test_a_number_written_in_words_in_the_quote_counts() -> None:
    quote = "Le passage interviendra après trois ans d'expérience."
    assert "3" in figures(quote, language="fr")


def test_a_date_and_an_ordinal_keep_their_digits() -> None:
    assert figures("dès le 1.1.2026, 13e salaire", language="fr") == {"1.1.2026", "13"}


# --- the figures rule ---------------------------------------------------------------


def test_a_correct_answer_that_restates_the_question_passes() -> None:
    """The bench's paysagistes Q1: refused at 45 % coverage, right to the centime."""
    coverage = _check(
        GARDENER_DRAFT, GARDENER_QUOTE, GARDENER_SECTION, GARDENER_QUESTION, "coverage"
    )
    figures_ = _check(
        GARDENER_DRAFT, GARDENER_QUOTE, GARDENER_SECTION, GARDENER_QUESTION, "figures"
    )
    assert coverage.reason is RefusalReason.LOW_CITATION_COVERAGE  # type: ignore[attr-defined]
    assert figures_.reason is None  # type: ignore[attr-defined]


def test_a_wrong_amount_is_refused_even_when_the_words_match() -> None:
    wrong = GARDENER_DRAFT.replace("26.05", "27.95")
    check = _check(wrong, GARDENER_QUOTE, GARDENER_SECTION, GARDENER_QUESTION, "figures")
    assert check.reason is RefusalReason.UNQUOTED_FIGURE  # type: ignore[attr-defined]
    assert "27.95" in check.rationale  # type: ignore[attr-defined]


def test_a_figure_the_member_gave_in_the_question_may_be_repeated() -> None:
    question = "Combien de jours de vacances par an pour un travailleur de 58 ans ?"
    draft = "Un travailleur de 58 ans a droit à 30 jours ouvrables de vacances (art. 50)."
    quote = "Dès 56 ans révolus, le travailleur a droit à 30 jours ouvrables de vacances."
    check = _check(draft, quote, "Art. 50 Vacances", question, "figures")
    assert check.reason is None  # type: ignore[attr-defined]


def test_a_figure_of_the_cited_article_may_frame_the_quoted_one() -> None:
    """Métal-Vaud Q1, the demo's wage question: the quote is the table row, and the
    amendment's date is in the heading of the same cited passage."""
    question = (
        "Quel est le salaire horaire minimum d'un travailleur titulaire d'un CFC, dès la "
        "fin de son apprentissage ?"
    )
    draft = (
        "Selon l'avenant n°2 du 13 décembre 2022 (art. 38, classe e), le salaire horaire "
        "minimum d'un travailleur titulaire d'un CFC dès la fin de son apprentissage est "
        "de CHF 25.60."
    )
    quote = "e) Travailleur avec CFC dès fin de l'apprentissage — Salaire horaire (CHF) : 25.60"
    passage = f"Art. 38 Salaires (Avenant N°2 du 13 décembre 2022)\n{quote}"
    check = _check(draft, quote, "Art. 38 Salaires", question, "figures", passage=passage)
    assert check.reason is None  # type: ignore[attr-defined]


def test_a_figure_only_in_a_passage_the_answer_does_not_cite_is_refused() -> None:
    """The draft saw the 2'200 hours in another article, and cited only the wage row."""
    question = "Combien payer un aide-jardinier sans CFC qui a 2 ans d'expérience ?"
    draft = (
        "Selon l'art. 8 (catégorie C2), le salaire minimum est de 23.70 CHF à l'heure, "
        "calculé sur la base de 2'200 heures annuelles."
    )
    quote = "aide-jardinier sans CFC ... expérience de 2 ans dans le métier - salaire minimum 23.70"
    uncited = "Article 3 - Durée du travail : 2200 heures par an"
    check = _check(draft, quote, "Article 8 - Salaires", question, "figures", uncited=uncited)
    assert check.reason is RefusalReason.UNQUOTED_FIGURE  # type: ignore[attr-defined]
    assert check.missing_figures == ("2200",)  # type: ignore[attr-defined]


def test_a_wrong_figure_after_a_comma_is_refused() -> None:
    quote = "Après 5 ans de service, le travailleur a droit à 25 jours de vacances."
    for draft in (
        "Après 5 ans de service, 99 jours de vacances.",
        "Après 5 ans de service, le travailleur a droit à 25 et 99 jours de vacances.",
    ):
        check = _check(draft, quote, None, "Combien de jours de vacances ?", "figures")
        assert check.reason is RefusalReason.UNQUOTED_FIGURE  # type: ignore[attr-defined]
        assert check.missing_figures == ("99",)  # type: ignore[attr-defined]


def test_words_found_nowhere_still_refuse() -> None:
    """The rule weighs ordinary words less; it does not ignore them."""
    draft = "Le congé maternité dure seize semaines payées par l'employeur."
    check = _check(draft, "Le congé dure deux jours.", None, "Quel congé ?", "figures")
    assert check.reason is RefusalReason.LOW_CITATION_COVERAGE  # type: ignore[attr-defined]


def test_the_question_counts_only_under_the_figures_rule() -> None:
    check = _check(GARDENER_DRAFT, GARDENER_QUOTE, GARDENER_SECTION, GARDENER_QUESTION, "coverage")
    assert check.coverage < 0.5  # type: ignore[attr-defined]


# --- the setting ----------------------------------------------------------------------


@pytest.fixture
def fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_coverage_stays_the_default_and_an_unknown_rule_is_rejected() -> None:
    assert Settings().support_check == "coverage"
    with pytest.raises(ValidationError):
        Settings(support_check="semantic")


def test_the_agent_applies_the_configured_rule_with_the_question(
    monkeypatch: pytest.MonkeyPatch, fresh_settings: None
) -> None:
    from tests.unit.test_agent import _seed_retriever
    from verity.agent import agent as agent_module
    from verity.agent.scripted_stub import build_stub_router

    seen: list[tuple[str, str]] = []
    real = support.check_support

    def spy(*args: object, **kwargs: object) -> object:
        seen.append((str(kwargs["rule"]), str(kwargs["question"])))
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(agent_module, "check_support", spy)
    monkeypatch.setenv("VERITY_SUPPORT_CHECK", "figures")
    question = "What is the notice period for terminating the master services agreement?"

    async def ask() -> None:
        agent = agent_module.create_default_agent(
            retriever=await _seed_retriever(), client=build_stub_router()
        )
        await agent.answer(question)

    asyncio.run(ask())
    assert seen == [("figures", question)]


# --- the draft prompt -----------------------------------------------------------------


class _Client:
    messages: ClassVar[list[Message]] = []

    async def complete(self, messages: list[Message], *, max_tokens: int = 256) -> Completion:
        _Client.messages = messages
        text = json.dumps({"answer": "a", "citations": []})
        return Completion(text=text, provider=Provider.ANTHROPIC, model="m", usage=UsageStats())


def _draft_prompt(**kwargs: object) -> str:
    asyncio.run(CitedSynthesizer(_Client(), **kwargs).synthesize("q?", []))  # type: ignore[arg-type]
    return "\x00".join(f"{m.role}:{m.content}" for m in _Client.messages)


def test_under_the_figures_rule_the_draft_is_asked_to_quote_every_figure() -> None:
    assert "every amount, duration, date and percentage" in _draft_prompt(quote_figures=True)


# sha256 of the draft messages for "q?" and no evidence, taken before the figure note
# existed: the published runs and the bench runs were drafted with exactly these.
@pytest.mark.parametrize(
    ("options", "digest"),
    [
        ({}, "6cb87523de889c1bd996cc013e5ba0adffc7a395f39b89ec5793c349e09c5afd"),
        (
            {"name_articles": True, "complete_answers": True},
            "9acf510e4695d5c6511ad5093595cea611c1bff65bb8e76ed4f144148c66dc4d",
        ),
    ],
)
def test_without_it_the_draft_prompt_is_unchanged(
    options: dict[str, object], digest: str, fresh_settings: None
) -> None:
    assert hashlib.sha256(_draft_prompt(**options).encode()).hexdigest() == digest
