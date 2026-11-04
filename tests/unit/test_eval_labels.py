"""Gold chunks are resolved from each question's evidence against the chunks the eval
ingested, not read from ids committed on another machine.

docling's layout model does not parse a PDF identically on every CPU (on the GitHub
runner, numbered headings of the corpus PDFs became list items): the offsets, and so
the chunk ids, move while the evidence quotes stay in the text.
"""

from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path

import pytest

from tests.fakes import without_local_models
from verity.eval.labels import resolve_gold
from verity.eval.retrieval_run import load_full_dataset
from verity.types import Chunk, EvalExample

REPO_ROOT = Path(__file__).resolve().parents[2]
_DOC = uuid.uuid4()


def _chunk(ordinal: int, text: str) -> Chunk:
    return Chunk(document_id=_DOC, text=text, ordinal=ordinal, char_start=0, char_end=len(text))


def _example(qid: str, evidence: tuple[str, ...], gold: tuple[str, ...] = ()) -> EvalExample:
    return EvalExample.model_validate(
        {
            "id": qid,
            "question": "q?",
            "answerable": bool(evidence),
            "reference_answer": "a" if evidence else None,
            "evidence": list(evidence),
            "relevant_chunk_ids": list(gold),
        }
    )


def test_stale_committed_ids_are_replaced_by_the_chunks_holding_the_evidence() -> None:
    chunks = [
        _chunk(0, "Data Classification\n\nAcme classifies"),
        _chunk(1, "Passwords  must\nbe long."),
    ]
    stale = str(uuid.uuid4())
    [resolved], problems = resolve_gold(
        [_example("q-1", ("Passwords must be long.",), (stale,))], chunks
    )
    assert problems == []
    assert resolved.relevant_chunk_ids == (chunks[1].id,)


def test_a_quote_found_in_no_chunk_is_a_problem_not_a_silent_empty_gold() -> None:
    [resolved], problems = resolve_gold(
        [_example("q-2", ("1. Data Classification",))], [_chunk(0, "Data Classification")]
    )
    assert problems == ["q-2: no chunk contains '1. Data Classification'"]
    assert resolved.relevant_chunk_ids == ()


def test_an_example_without_evidence_keeps_its_ids_and_out_of_scope_stays_empty() -> None:
    kept = str(uuid.uuid4())
    examples = [_example("q-3", (), (kept,)), _example("q-4", ())]
    resolved, problems = resolve_gold(examples, [_chunk(0, "anything")])
    assert problems == []
    assert [e.relevant_chunk_ids for e in resolved] == [(uuid.UUID(kept),), ()]


def test_the_committed_labels_are_the_evidence_resolved_against_the_committed_parse() -> None:
    """The labelling platform's parse is the committed docling fixture (the fast lane
    serves it for every corpus PDF and DOCX), so this pins the dataset to its evidence
    on any machine: a hand edit without ``scripts/build_eval_labels.py`` fails here."""
    import asyncio

    spec = importlib.util.spec_from_file_location(
        "bel", REPO_ROOT / "scripts" / "build_eval_labels.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    committed = load_full_dataset(script.DATASET_PATH)
    resolved, problems = resolve_gold(committed, asyncio.run(script._load_chunks()))
    assert problems == []
    assert resolved == committed
    raw = [
        json.loads(line) for line in script.DATASET_PATH.read_text().splitlines() if line.strip()
    ]
    assert [tuple(r.get("evidence", ())) for r in raw] == [e.evidence for e in committed]


@pytest.fixture
def eval_retrieval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Run ``verity eval retrieval`` on a two-document Markdown corpus and one question,
    with a model-free embedder; returns the CLI result."""
    from typer.testing import CliRunner

    from verity.cli import app

    without_local_models(monkeypatch)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "travel.md").write_text("# Travel\n\nEmployees book economy class for flights.\n")
    (corpus / "leave.md").write_text("# Leave\n\nEmployees get twenty days of annual leave.\n")

    def run(evidence: list[str], committed_ids: list[str]):  # type: ignore[no-untyped-def]
        dataset = tmp_path / "q.jsonl"
        record = {
            "id": "q-1",
            "question": "Which class do employees book for flights?",
            "answerable": True,
            "reference_answer": "Economy.",
            "evidence": evidence,
            "relevant_chunk_ids": committed_ids,
        }
        dataset.write_text(json.dumps(record) + "\n")
        args = ["eval", "retrieval", "--corpus", str(corpus), "--dataset", str(dataset)]
        args += ["--store", "memory", "--no-rerank", "--k", "1"]
        args += ["--runs-dir", str(tmp_path / "runs"), "--floor-recall", "1.0"]
        return CliRunner().invoke(app, args)

    return run


def test_eval_retrieval_scores_against_the_evidence_when_committed_ids_are_stale(
    eval_retrieval,  # type: ignore[no-untyped-def]
) -> None:
    """What the GitHub runner hit: its parse of a document gave other chunk ids than the
    labelling machine's, and every gold chunk of that document read as missed."""
    result = eval_retrieval(["Employees book economy class for flights."], [str(uuid.uuid4())])
    assert result.exit_code == 0, result.output


def test_eval_retrieval_scores_a_question_with_evidence_and_no_committed_ids(
    eval_retrieval,  # type: ignore[no-untyped-def]
) -> None:
    result = eval_retrieval(["Employees book economy class for flights."], [])
    assert result.exit_code == 0, result.output
    assert "n=1" in result.output


def test_eval_retrieval_exits_2_when_a_quote_is_in_no_chunk(
    eval_retrieval,  # type: ignore[no-untyped-def]
) -> None:
    result = eval_retrieval(["Employees fly business class."], [str(uuid.uuid4())])
    assert result.exit_code == 2, result.output
    assert "q-1: no chunk contains 'Employees fly business class.'" in result.output
