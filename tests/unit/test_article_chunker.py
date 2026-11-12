"""VERITY_CHUNKING=articles: one passage per article of a legal text.

The structure chunker packs paragraphs under the heading docling reports, which on a
collective labour agreement is often not the article: on the landscapers' CCT, art.
9.1 (Sunday work, 50 %) ended a chunk labelled "D) Jardiniers-grimpeurs" that opened
on the wage scale. The article chunker cuts at each article, labels the passage with
the article's real number and title, splits a long article by paragraph under the same
label, and keeps a table with the article or annex that holds it; in an annex, a short
title line (a canton) labels the tables under it.

The fixtures are verbatim excerpts of the three agreements as the bench parses them
(docling, tables flattened): the landscapers' CCT of the canton of Vaud, the national
hotel-restaurant CCNT and the CCT of the French-speaking finishing trades (CCT-SOR).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from verity.ingestion.article_chunker import ArticleChunker
from verity.ingestion.chunker import StructureAwareChunker
from verity.types import Chunk, Document, SourceType

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "articles"


def _doc(text: str) -> Document:
    return Document(source_type=SourceType.PDF, uri="file:///cct.pdf", text=text)


def _fixture(name: str) -> Document:
    return _doc((FIXTURES / name).read_text())


def _chunks(document: Document, **kwargs: int) -> list[Chunk]:
    chunks = ArticleChunker(**kwargs).chunk(document)
    for chunk in chunks:
        assert document.text[chunk.char_start : chunk.char_end] == chunk.text
    # Every word of the document is in exactly one passage.
    covered = " ".join(c.text for c in chunks)
    assert covered.split() == document.text.split()
    return chunks


def _holding(chunks: list[Chunk], text: str) -> Chunk:
    (chunk,) = [c for c in chunks if text in c.text]
    return chunk


def test_an_article_is_its_own_passage_with_its_real_number_and_title() -> None:
    chunks = _chunks(_fixture("paysagistes_vd_art8-10.txt"))
    sunday = _holding(chunks, "supplément de 50%.")
    assert sunday.section == "Article 9 \N{EN DASH} Travail du dimanche ou jours fériés"
    assert sunday.metadata == {
        "article": "9",
        "article_title": "Travail du dimanche ou jours fériés",
    }
    assert sunday.text.startswith("Article 9 \N{EN DASH} Travail du dimanche")
    assert "Jardiniers-grimpeurs" not in sunday.text
    assert [c.metadata["article"] for c in chunks] == ["8", "9", "10"]


def test_a_repeated_table_caption_and_a_reference_stay_in_their_article() -> None:
    """The flattened wage scale repeats "Art. 10 Salaires minimums" over each table."""
    document = _fixture("ccnt_art10-12.txt")
    assert document.text.count("Art. 10 Salaires minimums") > 1
    chunks = _chunks(document)
    assert {c.metadata["article"] for c in chunks} == {"10", "11", "12"}
    scale = _holding(chunks, "3'713")
    assert scale.section == "Art. 10 Salaires minimums"


def test_a_long_article_is_split_by_paragraph_under_the_same_label() -> None:
    chunks = _chunks(_fixture("ccnt_art10-12.txt"), max_words=120)
    article_10 = [c for c in chunks if c.metadata["article"] == "10"]
    assert len(article_10) > 1
    assert {c.section for c in article_10} == {"Art. 10 Salaires minimums"}
    assert all(len(c.text.split()) <= 120 or "\n\n" not in c.text for c in article_10)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_a_cantonal_table_is_labelled_with_its_canton_not_the_table_before_it() -> None:
    """On the CCT-SOR, the Geneva upholsterers' scale (class A: 4'958 / 27.90) sat under
    the heading of the all-cantons scale (class A: 5'509 / 31.00) and was read as it."""
    chunks = _chunks(_fixture("sor_annexe_ii_2026.txt"))
    every_canton = _holding(chunks, "5'509 ; dès 3ème année après CFC : 31.00")
    geneva = _holding(chunks, "4'958 ; dès 3ème année après CFC : 27.90")
    assert every_canton is not geneva
    assert every_canton.section == "ANNEXE II > Tous les cantons > Métiers du second œuvre"
    assert geneva.section == "ANNEXE II > Genève"
    assert "Tous les cantons" not in geneva.text
    assert geneva.metadata == {"annex": "ANNEXE II"}


def test_an_annex_that_lists_articles_has_no_article_of_its_own() -> None:
    """Annex IV names the articles apprentices are exempt from, under group titles; a
    heading with no provision below it is not an article of the annex."""
    chunks = _chunks(_fixture("sor_annexe_iv.txt"))
    assert all(c.metadata == {"annex": "Annexe IV"} for c in chunks)
    assert all((c.section or "").startswith("Annexe IV") for c in chunks)


def test_the_articles_of_an_annex_carry_the_annex() -> None:
    chunks = _chunks(_fixture("sor_annexe_vi.txt"))
    assert [c.section for c in chunks] == [
        "Annexe VI > Art. 1 Principe",
        "Annexe VI > Art. 2 Utilisation",
    ]
    assert chunks[0].text.startswith("Annexe VI\n\nCaution\n\nArt. 1 Principe")


def test_an_article_one_far_into_an_annex_starts_a_new_text() -> None:
    """The early-retirement agreement (CCRA) follows annex IX of the CCT-SOR; its art. 1
    is not an article of annex IX."""
    chunks = _chunks(_fixture("sor_annexe_ix_ccra.txt"))
    article_1 = [c for c in chunks if c.metadata.get("article") == "1"]
    assert article_1
    assert {c.section for c in article_1} == {
        "Art. 1 Du point de vue du genre de travaux et du territoire"
    }
    assert all("annex" not in c.metadata for c in article_1)


@pytest.mark.parametrize(
    "line",
    [
        "Art. 10 al. 4 lit. b) de la présente CCT s'applique.",
        "art. 12.1 CCT",
        "Articles 3 à 5",
    ],
)
def test_a_reference_at_the_start_of_a_line_is_not_an_article(line: str) -> None:
    text = f"Art. 9 Vacances\n\nLe travailleur a droit à cinq semaines.\n\n{line}\n\nFin du texte."
    (chunk,) = _chunks(_doc(text))
    assert chunk.metadata["article"] == "9"


def test_a_lower_number_inside_the_text_does_not_cut_it() -> None:
    text = (
        "Art. 9 Vacances\n\nLe travailleur a droit à cinq semaines.\n\n"
        "Art. 4 Engagement\n\nVoir les règles de l'engagement ci-dessus."
    )
    (chunk,) = _chunks(_doc(text))
    assert chunk.metadata["article"] == "9"


@pytest.mark.parametrize(
    ("heading", "number", "title"),
    [
        ("Article premier Champ d'application", "1", "Champ d'application"),
        ("Art. 5bis Indemnités", "5bis", "Indemnités"),
        ("§ 3 Durée du travail", "3", "Durée du travail"),
        ("Art. 7", "7", ""),
    ],
)
def test_heading_forms(heading: str, number: str, title: str) -> None:
    (chunk,) = _chunks(_doc(f"{heading}\n\nLe texte de cet article tient en une phrase."))
    assert chunk.metadata == {"article": number, "article_title": title}
    assert chunk.section == heading


def test_a_text_without_articles_is_chunked_as_before() -> None:
    text = "# Policy\n\nEmployees accrue leave monthly.\n\n## Notice\n\nGive thirty days."
    document = _doc(text)
    assert ArticleChunker().chunk(document) == StructureAwareChunker().chunk(document)


def test_the_setting_selects_the_chunker(monkeypatch: pytest.MonkeyPatch) -> None:
    from verity.config import get_settings
    from verity.runtime import create_chunker

    get_settings.cache_clear()
    assert isinstance(create_chunker(), StructureAwareChunker)
    monkeypatch.setenv("VERITY_CHUNKING", "articles")
    get_settings.cache_clear()
    assert isinstance(create_chunker(), ArticleChunker)


def test_the_bench_source_check_reads_the_real_article_from_the_passage() -> None:
    """Q11 of the landscapers' bench: a citation from the art. 9 passage sources art. 9."""
    from tools.prospection.bench import CitedQuote, Question, grade

    sunday = _holding(_chunks(_fixture("paysagistes_vd_art8-10.txt")), "supplément de 50%.")
    question = Question(
        id=11,
        question="Quel supplément pour le dimanche ou un jour férié travaillé ?",
        article="Art. 9.1",
        expected="50 %",
        articles=("9",),
        facts=(("50",),),
    )
    cited = (CitedQuote(quote="un supplément de 50%", section=sunday.section),)
    graded = grade(question, "Un supplément de 50 %.", refused=False, error=None, citations=cited)
    assert graded.code == "R+S"
