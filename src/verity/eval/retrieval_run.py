"""Compute the retrieval scorecard: run HybridRetriever on the answerable questions
and macro-average per-example metrics.

The scorecard is *real* and *measured*: local embeddings + local reranker + a git SHA
resolved from the current repo. No LLM is invoked here — the numbers are reproducible
bit-for-bit given the same corpus, dataset, and code.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from verity.eval.retrieval_metric import (
    BinaryRetrievalMetric,
    PerExampleScore,
    macro_average,
)
from verity.retrieval.base import Retriever
from verity.types import EvalExample, RetrievalMetrics


def load_dataset(path: Path) -> list[EvalExample]:
    """Read a JSONL dataset and return the answerable-and-labeled subset only.

    Out-of-scope questions (empty ``relevant_chunk_ids``) are filtered *here*, so
    every caller of this function sees the exact set that retrieval metrics are
    defined over. See ``docs/evaluation-methodology.md`` — refusal calibration is
    the eval track for the unanswerable half.
    """
    return [ex for ex in load_full_dataset(path) if ex.answerable and ex.relevant_chunk_ids]


def load_full_dataset(path: Path) -> list[EvalExample]:
    """Read a JSONL dataset and return **every** example, answerable or not.

    Used by the S4 harness (refusal calibration needs the out-of-scope half).
    """
    examples: list[EvalExample] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            examples.append(EvalExample.model_validate(json.loads(raw)))
    return examples


def current_git_sha(default: str = "unknown") -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or default
    except (subprocess.CalledProcessError, FileNotFoundError):
        return default


@dataclass(frozen=True)
class RetrievalScorecard:
    """One retrieval-only scorecard. Written as JSON to
    ``datasets/eval/runs/retrieval_<sha>.json``."""

    git_sha: str
    dataset: str
    n_examples: int
    k: int
    aggregate: RetrievalMetrics
    per_example: list[PerExampleScore]
    created_at: datetime
    backend: str  # "pgvector" or "in-memory" — visible in the persisted artefact

    def to_dict(self) -> dict[str, object]:
        return {
            "git_sha": self.git_sha,
            "dataset": self.dataset,
            "n_examples": self.n_examples,
            "k": self.k,
            "backend": self.backend,
            "created_at": self.created_at.isoformat(),
            "aggregate": self.aggregate.model_dump(),
            "per_example": [
                {"example_id": p.example_id, "metrics": p.metrics.model_dump()}
                for p in self.per_example
            ],
        }


async def run_retrieval_eval(
    retriever: Retriever,
    dataset: list[EvalExample],
    *,
    k: int,
    dataset_name: str,
    backend: str,
) -> RetrievalScorecard:
    metric = BinaryRetrievalMetric()
    per_example: list[PerExampleScore] = []
    for example in dataset:
        hits = await retriever.retrieve(example.question, k=k)
        per_example.append(
            PerExampleScore(
                example_id=example.id,
                metrics=metric.score(example, hits, k=k),
            )
        )
    aggregate = macro_average(per_example, k=k)
    return RetrievalScorecard(
        git_sha=current_git_sha(),
        dataset=dataset_name,
        n_examples=len(dataset),
        k=k,
        aggregate=aggregate,
        per_example=per_example,
        created_at=datetime.now(UTC),
        backend=backend,
    )


def persist_scorecard(scorecard: RetrievalScorecard, runs_dir: Path) -> Path:
    """Write scorecard as JSON. Filename encodes git SHA so runs are comparable."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"retrieval_{scorecard.git_sha}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(scorecard.to_dict(), fh, indent=2, sort_keys=True)
    return path


__all__ = [
    "RetrievalScorecard",
    "current_git_sha",
    "load_dataset",
    "persist_scorecard",
    "run_retrieval_eval",
]
