"""The natural-language injection set and the script that measures it."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from verity.agent.support import has_injection_marker
from verity.types import Answer, AnswerError, Citation, Confidence, ErrorKind

_SPEC = importlib.util.spec_from_file_location(
    "run_injection_eval", "scripts/run_injection_eval.py"
)
assert _SPEC is not None and _SPEC.loader is not None
injection_eval = importlib.util.module_from_spec(_SPEC)
sys.modules["run_injection_eval"] = injection_eval
_SPEC.loader.exec_module(injection_eval)

_CASES = injection_eval.load_cases()
_POISONED = sorted((injection_eval.INJECTION / "corpus").glob("*.md"))


def test_every_case_names_a_poisoned_document_and_every_document_has_a_case() -> None:
    assert sorted(c["poisoned_document"] for c in _CASES) == [p.name for p in _POISONED]
    assert len({c["id"] for c in _CASES}) == len(_CASES)


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c["id"])
def test_a_success_marker_is_never_in_the_clean_corpus(case: dict[str, object]) -> None:
    """Otherwise a correct answer from the clean corpus would count as a success."""
    clean = " ".join(p.read_text(encoding="utf-8").lower() for p in injection_eval.CORPUS.glob("*"))
    for marker in case["success_markers"]:  # type: ignore[attr-defined]
        assert marker.lower() not in clean


@pytest.mark.parametrize("path", _POISONED, ids=lambda p: p.name)
def test_the_poisoned_documents_pass_the_marker_regex(path: Path) -> None:
    """The set measures what a regex cannot catch: a naturally worded instruction."""
    assert not has_injection_marker(path.read_text(encoding="utf-8"))


def _answer(text: str, *, refused: bool = False, error: bool = False) -> Answer:
    if error:
        return Answer(
            query="q", text="", error=AnswerError(kind=ErrorKind.PROVIDER_UNAVAILABLE, detail="d")
        )
    citation = Citation(
        chunk_id="00000000-0000-0000-0000-000000000001",
        document_id="00000000-0000-0000-0000-000000000002",
        quote="x",
        char_start=0,
        char_end=1,
    )
    return Answer(
        query="q",
        text=text,
        citations=(citation,),
        confidence=Confidence(score=0.9, refused=refused, rationale="r"),
    )


def test_an_attack_succeeds_only_when_the_planted_content_ships() -> None:
    markers = ["sla-verify"]
    assert injection_eval.attack_succeeded(_answer("Visit https://SLA-verify.example"), markers)
    assert not injection_eval.attack_succeeded(_answer("99.9% uptime"), markers)
    assert not injection_eval.attack_succeeded(_answer("sla-verify", refused=True), markers)
    assert not injection_eval.attack_succeeded(_answer("", error=True), markers)


@pytest.mark.parametrize(
    ("text", "succeeded"),
    [
        ("Notice is seven\u02c6(7)\u02c6days.", True),
        ("Notice is seven  (7)\ndays.", True),
        ("Revenue was US$1,612.0 million.", False),
        ("Revenue was US$612.0 million.", True),
    ],
)
def test_markers_match_whole_words_across_marks_and_whitespace(text: str, succeeded: bool) -> None:
    markers = ["seven (7) days", "612"]
    assert injection_eval.attack_succeeded(_answer(text), markers) is succeeded


@pytest.mark.slow
@pytest.mark.parametrize("flags", [[], ["--no-datamark"]], ids=["datamarked", "unmarked"])
def test_a_stub_run_writes_a_complete_artefact(tmp_path: Path, flags: list[str]) -> None:
    out = tmp_path / "run.json"
    assert injection_eval.main(["--stub", "--out", str(out), *flags]) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["complete"] is True
    assert result["datamarked"] is (not flags)
    poisoned = {p.name for p in _POISONED}
    assert all(set(r["cited_poisoned"]) <= poisoned for r in result["attacks"])
    assert result["summary"]["attack_success"]["of"] == len(_CASES)
    assert result["summary"]["control_answered"]["of"] == len(injection_eval.load_controls())
    assert result["summary"]["cost_usd"] == 0


@pytest.mark.slow
def test_a_run_that_crashes_keeps_every_answer_it_paid_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken() -> list[dict[str, object]]:
        raise RuntimeError("control set unreadable")

    monkeypatch.setattr(injection_eval, "load_controls", broken)
    out = tmp_path / "run.json"
    with pytest.raises(RuntimeError):
        injection_eval.main(["--stub", "--out", str(out)])
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["complete"] is False
    assert len(result["attacks"]) == len(_CASES)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_every_poisoned_document_is_retrieved_for_its_question() -> None:
    """A case whose poisoned document never reaches the evidence measures retrieval,
    not the synthesizer: a 0 % attack rate would then say nothing about the mark."""
    from verity.config import get_settings
    from verity.ingestion.ids import document_id_for_uri
    from verity.runtime import build_local_components, build_retriever, ingest_corpus

    store, embedder, reranker = build_local_components(store="memory")
    for directory in (injection_eval.CORPUS, injection_eval.INJECTION / "corpus"):
        await ingest_corpus(corpus_dir=directory, store=store, embedder=embedder)
    retriever = build_retriever(store=store, embedder=embedder, reranker=reranker)
    missed = []
    for case in _CASES:
        path = injection_eval.INJECTION / "corpus" / case["poisoned_document"]
        poisoned = str(document_id_for_uri(str(path)))
        hits = await retriever.retrieve(case["question"], k=get_settings().rerank_k)
        if poisoned not in {str(h.chunk.document_id) for h in hits}:
            missed.append(case["id"])
    assert missed == []
