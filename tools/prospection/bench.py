"""Ask a fixed set of member questions to Verity over one association's documents, and
grade the answers with the prospection test grid.

    uv run python -m tools.prospection.bench --corpus <dir> --questions <file.jsonl>

Writes ``resultats.json`` and ``tableau.md`` next to the questions file (``--out`` to
change). The grid: R+S (correct, with the article) 2, R (correct) 1, NT (not found)
0, F (wrong) -1; on a question the text does not answer, a refusal is worth 2 and any
answer -1.

Grading is deterministic: an answer is correct when it contains every expected fact
(normalised substrings written in the questions file), and sourced when it, a cited
quote, or the section of a cited chunk names an expected article. It cannot tell an
incomplete answer from a wrong one, so every F is flagged for a human reading.

``--stub`` runs the scripted stub instead of a model: it checks the plumbing for
free, and its score measures nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
import subprocess
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from verity.types import Answer

Code = Literal["R+S", "R", "NT", "F"]
POINTS: dict[Code, int] = {"R+S": 2, "R": 1, "NT": 0, "F": -1}


class Question(BaseModel, frozen=True):
    id: int
    question: str
    article: str  # as the reference table writes it, for the human reader
    expected: str
    articles: tuple[str, ...] = ()  # article numbers that count as a source, e.g. "8"
    # Every group must match; a group matches when one of its alternatives does.
    facts: tuple[tuple[str, ...], ...] = ()
    expect_refusal: bool = False


class Graded(BaseModel, frozen=True):
    code: Code
    points: int
    needs_review: bool
    reason: str


class CitedQuote(BaseModel, frozen=True):
    quote: str
    section: str | None


class Result(BaseModel, frozen=True):
    question: Question
    answer: str
    refused: bool
    error: str | None
    citations: tuple[CitedQuote, ...]
    cost_usd: float
    latency_ms: float
    graded: Graded


class Run(BaseModel):
    mode: Literal["stub", "live"]
    model: str | None
    git_sha: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    corpus: dict[str, str]  # file name -> sha256
    n_chunks: int
    budget_usd: float
    spent_usd: float = 0.0
    results: list[Result] = Field(default_factory=list)


def load_questions(path: Path) -> list[Question]:
    questions = [
        Question.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()
    ]
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate question ids")
    for q in questions:
        if not q.expect_refusal and not q.facts:
            raise ValueError(f"question {q.id}: an answerable question needs facts to grade")
    return questions


def normalise(text: str) -> str:
    """Lower case, no accents, Swiss thousands separators removed, decimal comma as point.

    So "CHF 4'775.85", "4 775,85" and "4775.85" all read "4775.85".
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    text = re.sub(r"(?<=\d)['\u2019\u02bc\u00a0\u202f ](?=\d{3}(?!\d))", "", text)
    return re.sub(r"(?<=\d),(?=\d)", ".", text)


def _contains(haystack: str, needle: str) -> bool:
    """``needle`` in ``haystack``; a number must not be part of a longer number."""
    needle = normalise(needle)
    if needle[:1].isdigit() or needle[-1:].isdigit():
        return re.search(rf"(?<![\d.]){re.escape(needle)}(?!\d|\.\d)", haystack) is not None
    return needle in haystack


def _sourced(question: Question, answer: str, citations: tuple[CitedQuote, ...]) -> bool:
    for number in question.articles:
        n = re.escape(number)
        if re.search(rf"\bart(?:icle|\.)?\s*{n}(?:\.\d+)?\b", answer, re.IGNORECASE):
            return True
        for cited in citations:
            if cited.section and re.search(rf"\barticle\s+{n}\b", cited.section, re.IGNORECASE):
                return True
            if re.match(rf"\s*{n}\.\d", cited.quote):
                return True
    return False


def _graded(code: Code, reason: str, *, needs_review: bool = False) -> Graded:
    return Graded(code=code, points=POINTS[code], needs_review=needs_review, reason=reason)


def grade(
    question: Question,
    answer: str,
    *,
    refused: bool,
    error: str | None,
    citations: tuple[CitedQuote, ...],
) -> Graded:
    if error is not None:
        return _graded("NT", f"no answer: {error}", needs_review=True)
    if question.expect_refusal:
        if refused:
            return _graded("R+S", "refused, as expected: the text does not answer")
        return _graded("F", "answered a question the text does not answer", needs_review=True)
    if refused:
        return _graded("NT", "refused an answerable question")
    text = normalise(answer)
    missing = [group for group in question.facts if not any(_contains(text, f) for f in group)]
    if missing:
        listed = "; ".join(" | ".join(group) for group in missing)
        return _graded("F", f"expected fact missing: {listed}", needs_review=True)
    if _sourced(question, answer, citations):
        return _graded("R+S", "every expected fact, and the article")
    return _graded("R", "every expected fact, no article")


def score(results: list[Result]) -> tuple[int, int]:
    """(points, maximum); the protocol scores out of 2 per question."""
    return sum(r.graded.points for r in results), 2 * len(results)


def render_table(run: Run) -> str:
    points, maximum = score(run.results)
    counts = {code: sum(r.graded.code == code for r in run.results) for code in POINTS}
    lines = []
    if run.mode == "stub":
        lines += ["**STUB : contrôle de la plomberie. Ce score ne mesure rien.**", ""]
    lines += [
        f"Modèle : {run.model or 'stub'} · commit {run.git_sha} · "
        f"{run.started_at:%Y-%m-%d %H:%M} UTC · coût {run.spent_usd:.4f} $",
        "",
        "| # | Question | Article attendu | Code | Points | À relire | Motif |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in run.results:
        q = r.question
        review = "oui" if r.graded.needs_review else ""
        lines.append(
            f"| {q.id} | {_cell(q.question)} | {_cell(q.article)} | {r.graded.code} | "
            f"{r.graded.points} | {review} | {_cell(r.graded.reason)} |"
        )
    total = " ".join(f"{code} : {n}" for code, n in counts.items())
    on_100 = max(0.0, points / maximum * 100) if maximum else 0.0
    lines += ["", f"**Total** {total} · score {points} / {maximum} · {on_100:.0f} / 100"]
    return "\n".join(lines) + "\n"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _citations(answer: Answer) -> tuple[CitedQuote, ...]:
    sections = {hit.chunk.id: hit.chunk.section for hit in answer.hits_used}
    return tuple(
        CitedQuote(quote=c.quote, section=sections.get(c.chunk_id)) for c in answer.citations
    )


def _fingerprint(corpus: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(corpus.iterdir())
        if p.is_file()
    }


def _git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return out.stdout.strip() or "unknown"


async def run_bench(
    corpus: Path, questions: list[Question], *, stub: bool, budget_usd: float
) -> Run:
    from verity.agent import create_default_agent
    from verity.agent.scripted_stub import build_stub_router
    from verity.llm import create_default_routing_client
    from verity.llm.budget import SpendCappedClient
    from verity.llm.routing import primary_client_identity
    from verity.runtime import build_local_components, build_retriever, ingest_corpus

    store, embedder, reranker = build_local_components(store="memory")
    ingested = await ingest_corpus(corpus_dir=corpus, store=store, embedder=embedder)
    if ingested.failures:
        failed = "; ".join(f"{f.uri}: {f.error}" for f in ingested.failures)
        raise RuntimeError(f"ingestion failed, nothing was asked: {failed}")
    retriever = build_retriever(store=store, embedder=embedder, reranker=reranker)
    router = build_stub_router() if stub else create_default_routing_client()
    client = SpendCappedClient(router, cap_usd=budget_usd)
    agent = create_default_agent(retriever=retriever, client=client)
    run = Run(
        mode="stub" if stub else "live",
        model=None if stub else primary_client_identity(router)[1],
        git_sha=_git_sha(),
        corpus=await asyncio.to_thread(_fingerprint, corpus),
        n_chunks=len(ingested.chunks),
        budget_usd=budget_usd,
    )
    for question in questions:
        answer = await agent.answer(question.question)
        refused = bool(answer.confidence and answer.confidence.refused)
        error = f"{answer.error.kind}: {answer.error.detail}" if answer.error else None
        citations = _citations(answer)
        run.results.append(
            Result(
                question=question,
                answer=answer.text,
                refused=refused,
                error=error,
                citations=citations,
                cost_usd=answer.usage.cost_usd,
                latency_ms=answer.usage.latency_ms,
                graded=grade(
                    question, answer.text, refused=refused, error=error, citations=citations
                ),
            )
        )
        run.spent_usd = client.spent.cost_usd
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="Default: the directory of --questions.")
    parser.add_argument("--stub", action="store_true", help="Scripted stub, no model call.")
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=3.0,
        help="Cap on the whole run; a question that would pass it is not asked.",
    )
    args = parser.parse_args(argv)
    if args.budget_usd <= 0:
        parser.error("--budget-usd must be positive")
    if not args.stub:
        from verity.config import get_settings

        if not get_settings().has_llm_api_key:
            parser.error("no LLM API key configured; set one or pass --stub")

    questions = load_questions(args.questions)
    run = asyncio.run(run_bench(args.corpus, questions, stub=args.stub, budget_usd=args.budget_usd))
    out = args.out or args.questions.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "resultats.json").write_text(run.model_dump_json(indent=2))
    table = render_table(run)
    (out / "tableau.md").write_text(table)
    sys.stdout.write(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
