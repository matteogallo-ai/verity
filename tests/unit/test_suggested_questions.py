"""Suggested questions come from the corpus folder (suggested_questions.json).

A demo on one agreement shows questions about that agreement; a folder without the
file shows none, rather than the shipped corpus's English examples.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.api_app import api_app
from verity.api.state import SUGGESTED_QUESTIONS_FILE, read_suggested_questions
from verity.config import Settings
from verity.ingestion.pipeline import discover_sources


def _corpus(tmp_path: Path, questions: object | None, *, raw: str | None = None) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "cct.md").write_text("# Art. 1 Vacances\n\nCinq semaines par an.\n")
    if raw is not None:
        (corpus / SUGGESTED_QUESTIONS_FILE).write_text(raw, encoding="utf-8")
    elif questions is not None:
        (corpus / SUGGESTED_QUESTIONS_FILE).write_text(json.dumps(questions), encoding="utf-8")
    return corpus


def test_the_questions_are_read_from_the_corpus_folder(tmp_path: Path) -> None:
    questions = ["Combien de jours de vacances ?", "Quel est le délai de congé ?"]
    assert read_suggested_questions(_corpus(tmp_path, questions)) == questions


def test_a_folder_without_the_file_suggests_nothing(tmp_path: Path) -> None:
    assert read_suggested_questions(_corpus(tmp_path, None)) == []


@pytest.mark.parametrize(
    ("questions", "raw"),
    [
        (None, "not json"),
        ({"q": "Combien ?"}, None),
        (["Combien ?", 3], None),
        (["Combien ?", "  "], None),
        (["x" * 301], None),
    ],
    ids=["malformed", "not-a-list", "not-a-string", "blank", "over-long"],
)
def test_an_invalid_file_suggests_nothing(
    tmp_path: Path, questions: object | None, raw: str | None
) -> None:
    assert read_suggested_questions(_corpus(tmp_path, questions, raw=raw)) == []


def test_at_most_eight_questions_are_suggested(tmp_path: Path) -> None:
    questions = [f"Question {i} ?" for i in range(12)]
    assert read_suggested_questions(_corpus(tmp_path, questions)) == questions[:8]


def test_the_file_is_not_ingested_as_a_document(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, ["Combien ?"])
    assert [Path(uri).name for uri in discover_sources(str(corpus))] == ["cct.md"]


def test_status_carries_the_questions_of_the_served_corpus(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, ["Combien de jours de vacances ?"])
    settings = Settings(api_auth_disabled=True)
    with api_app(tmp_path, settings, corpus_dir=corpus) as client:
        assert client.get("/status").json()["suggested_questions"] == [
            "Combien de jours de vacances ?"
        ]


def test_the_shipped_corpus_keeps_its_demo_questions(tmp_path: Path) -> None:
    """make demo shows the questions the UI carried before they moved to the corpus."""
    with api_app(tmp_path, Settings(api_auth_disabled=True)) as client:
        questions = client.get("/status").json()["suggested_questions"]
    assert questions == [
        "What is the notice period for terminating the master services agreement?",
        "Which entity is liable for data-processing breaches, and up to what cap?",
        "Where is personal data stored, and what mechanism covers EU-to-US transfers?",
        "What is the response time for a Priority 1 support request?",
        "What is the CEO's home address?",
    ]
