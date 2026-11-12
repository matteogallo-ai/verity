"""Replay a support rule on recorded bench runs, without calling a model.

    uv run python -m tools.prospection.calibrate_support RESULTATS.json [...] [--rule figures]

Reads each run's answers and the drafts it refused, with the quotes the citation
check verified, applies ``verity.agent.support.check_support`` under ``--rule``, and
tallies what would change:

- a refused draft that now passes the gate and grades right (``NT -> juste``) or
  wrong (``nouvelle F``); the bench's grid grades it as if shipped;
- an answer that now fails the gate: right before (``juste -> NT``), or wrong
  before (``F rattrapée``).

What it cannot see: the confidence scorer, which runs after the gate and may still
refuse a draft that passes it, and a new draft prompt (``figures`` also asks the
draft to quote each figure), which changes the drafts themselves. A refused draft is
replayable only from runs that recorded its verified quotes.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from tools.prospection.bench import CitedQuote, Question, grade
from verity.agent.support import SupportRule, check_support
from verity.types import Chunk, Citation, RetrievalHit, RetrieverKind

# Only these refusals come from the gate under replay; a scorer refusal stays one.
_GATE_REASONS = frozenset({"low_citation_coverage", "unquoted_figure"})


class Outcome(StrEnum):
    UNCHANGED = "inchangé"
    NT_TO_RIGHT = "NT -> juste"
    NEW_F = "nouvelle F"
    STILL_REFUSED = "toujours refusé"
    RIGHT_TO_NT = "juste -> NT"
    F_CAUGHT = "F rattrapée"
    NOT_REPLAYABLE = "non rejouable"


@dataclass(frozen=True)
class Row:
    question_id: int
    code_before: str
    outcome: Outcome
    new_reason: str | None
    coverage: float | None
    missing_figures: tuple[str, ...]


def _gate(
    answer: str,
    quotes: list[tuple[str, str | None]],
    question: str,
    *,
    rule: SupportRule,
    settings: dict[str, Any],
) -> tuple[str | None, float, tuple[str, ...]]:
    hits, citations = [], []
    for quote, section in quotes:
        chunk = Chunk(
            document_id=uuid4(),
            text=quote,
            ordinal=0,
            char_start=0,
            char_end=len(quote),
            section=section,
        )
        hits.append(RetrievalHit(chunk=chunk, score=0.0, kind=RetrieverKind.RERANKED, rank=0))
        citations.append(
            Citation(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                quote=quote,
                char_start=0,
                char_end=len(quote),
            )
        )
    check = check_support(
        answer,
        tuple(citations),
        hits,
        min_coverage=float(settings.get("min_citation_coverage", 0.5)),
        # The cited-relevance gate was off in every bench run, and it is not under study.
        min_rerank_score=-math.inf,
        language=settings.get("language", "en"),
        rule=rule,
        question=question,
    )
    reason = str(check.reason) if check.reason else None
    return reason, check.coverage, check.missing_figures


def _replay_one(result: dict[str, Any], *, rule: SupportRule, settings: dict[str, Any]) -> Row:
    question = Question.model_validate(result["question"])
    code = result["graded"]["code"]
    decision = result.get("decision") or {}
    if not result["refused"]:
        quotes = [(c["quote"], c.get("section")) for c in result["citations"]]
        reason, coverage, missing = _gate(
            result["answer"], quotes, question.question, rule=rule, settings=settings
        )
        if reason is None:
            outcome = Outcome.UNCHANGED
        else:
            outcome = Outcome.F_CAUGHT if code == "F" else Outcome.RIGHT_TO_NT
        return Row(question.id, code, outcome, reason, coverage, missing)
    if decision.get("reason") not in _GATE_REASONS:
        return Row(question.id, code, Outcome.UNCHANGED, decision.get("reason"), None, ())
    if decision.get("draft_citations") is None or not decision.get("draft"):
        return Row(question.id, code, Outcome.NOT_REPLAYABLE, None, None, ())
    verified = [
        (c["quote"], c.get("section")) for c in decision["draft_citations"] if c["verified"]
    ]
    if not verified:
        return Row(question.id, code, Outcome.STILL_REFUSED, "no_citation", None, ())
    draft = decision["draft"]
    reason, coverage, missing = _gate(
        draft, verified, question.question, rule=rule, settings=settings
    )
    if reason is not None:
        return Row(question.id, code, Outcome.STILL_REFUSED, reason, coverage, missing)
    shipped = grade(
        question,
        draft,
        refused=False,
        error=None,
        citations=tuple(CitedQuote(quote=q, section=s) for q, s in verified),
    )
    outcome = Outcome.NEW_F if shipped.code == "F" else Outcome.NT_TO_RIGHT
    return Row(question.id, code, outcome, None, coverage, missing)


def replay_run(run: dict[str, Any], *, rule: SupportRule) -> list[Row]:
    settings = run.get("run_settings", {})
    return [_replay_one(r, rule=rule, settings=settings) for r in run["results"]]


def tally(rows: list[Row]) -> collections.Counter[Outcome]:
    counts: collections.Counter[Outcome] = collections.Counter({o: 0 for o in Outcome})
    counts.update(r.outcome for r in rows)
    return counts


_COLUMNS = (
    Outcome.NT_TO_RIGHT,
    Outcome.NEW_F,
    Outcome.RIGHT_TO_NT,
    Outcome.F_CAUGHT,
    Outcome.STILL_REFUSED,
    Outcome.NOT_REPLAYABLE,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("runs", nargs="+", type=Path, help="resultats.json of live bench runs")
    parser.add_argument("--rule", choices=["coverage", "figures"], default="figures")
    args = parser.parse_args(argv)
    print("| Run | Questions | " + " | ".join(o.value for o in _COLUMNS) + " |")
    print("|---|---:|" + "---:|" * len(_COLUMNS))
    total: collections.Counter[Outcome] = collections.Counter()
    details = []
    for path in args.runs:
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("mode") != "live":
            continue
        rows = replay_run(run, rule=args.rule)
        counts = tally(rows)
        total.update(counts)
        name = f"{path.parent.parent.name}/{path.parent.name}"
        print(f"| {name} | {len(rows)} | " + " | ".join(str(counts[o]) for o in _COLUMNS) + " |")
        details += [
            f"- {name} q{r.question_id} ({r.code_before}): {r.outcome.value}"
            + (f", {r.new_reason}" if r.new_reason else "")
            + (f", chiffres non cités {list(r.missing_figures)}" if r.missing_figures else "")
            for r in rows
            if r.outcome not in (Outcome.UNCHANGED,)
        ]
    print("| **Total** | | " + " | ".join(f"**{total[o]}**" for o in _COLUMNS) + " |")
    print()
    print("\n".join(details))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
