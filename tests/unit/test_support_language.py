"""The citation coverage check reads French when the corpus is French.

The English word rule (``[a-z0-9]+``, English stopwords) cuts French words at every
accent ("réalisé" gives "alis") and counts "dans", "est", "pour" as content, so a
French answer that its quotes support scores far below the threshold. ``fr``
tokenises on Unicode letters, compares accent-folded words, and drops French
stopwords. English stays the default and is unchanged.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from verity.agent import support
from verity.agent.support import citation_coverage, content_words
from verity.config import Settings
from verity.types import Citation


def _cite(quote: str) -> tuple[Citation, ...]:
    return (
        Citation(
            chunk_id=uuid4(), document_id=uuid4(), quote=quote, char_start=0, char_end=len(quote)
        ),
    )


def test_english_is_the_default_and_keeps_its_word_rule() -> None:
    # Pinned output of the English rule on accented text, accent cuts included.
    assert content_words("réalisé dans l'année") == {"alis", "dans", "ann"}
    assert Settings().language == "en"


def test_french_words_are_whole_and_folded() -> None:
    words = content_words("Le 13e salaire est réalisé dans l'année", language="fr")
    assert words == {"13e", "salai", "reali", "annee"}


def test_french_accents_match_on_both_sides() -> None:
    answer = "Le montant est calculé sur le revenu AVS."
    quote = "le montant se calcule sur le revenu ÁVS"
    assert citation_coverage(answer, _cite(quote), language="fr") == 1.0
    assert citation_coverage(answer, _cite(quote), language="en") < 1.0


def test_french_numbers_are_compared_whole() -> None:
    assert citation_coverage("Le taux est de 8,33 %.", _cite("taux 8,34 %"), language="fr") < 1.0


def test_a_french_answer_that_says_more_than_its_quote_is_still_low() -> None:
    """Folding must not make an unsupported answer pass: the check keeps its teeth."""
    answer = "Le congé maternité dure seize semaines payées par l'employeur."
    assert citation_coverage(answer, _cite("Le congé dure deux jours."), language="fr") < 0.5


@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=200))
def test_on_ascii_text_the_rules_differ_only_by_their_stopwords(text: str) -> None:
    english = content_words(text)
    french = content_words(text, language="fr")
    assert french - english <= {w[:5] for w in support._STOPWORDS}
    assert english - french <= {w[:5] for w in support._STOPWORDS_FR}


def test_an_unknown_language_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(language="de")


def test_the_agent_checks_coverage_in_the_configured_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verity.agent import agent as agent_module

    seen: list[str] = []
    real = support.check_support

    def spy(*args: object, **kwargs: object) -> object:
        seen.append(str(kwargs["language"]))
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(agent_module, "check_support", spy)
    monkeypatch.setenv("VERITY_LANGUAGE", "fr")
    from verity.config import get_settings

    get_settings.cache_clear()
    import asyncio

    from tests.unit.test_agent import _seed_retriever
    from verity.agent.scripted_stub import build_stub_router

    async def ask() -> None:
        agent = agent_module.create_default_agent(
            retriever=await _seed_retriever(), client=build_stub_router()
        )
        await agent.answer(
            "What is the notice period for terminating the master services agreement?"
        )

    asyncio.run(ask())
    assert seen == ["fr"]


@pytest.mark.parametrize(
    ("one", "other"),
    [("plus de 30 jours", "moins de 30 jours"), ("sans frais", "avec frais")],
)
def test_french_comparatives_carry_meaning(one: str, other: str) -> None:
    """English keeps more, less and without; French drops only the negation, whose
    polarity the scorer judges."""
    assert content_words(one, language="fr") != content_words(other, language="fr")


def test_french_ligatures_fold_to_their_letters() -> None:
    assert content_words("cœur ŒUVRE", language="fr") == content_words(
        "coeur oeuvre", language="fr"
    )
