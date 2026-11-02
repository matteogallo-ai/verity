"""Every committed live run artefact agrees with itself: its stored answer-quality
aggregate is what :mod:`verity.eval.answer_quality` computes from its own audit
records, so the repository holds one figure per metric for each run."""

from __future__ import annotations

from pathlib import Path

import pytest

from verity.eval.answer_quality import answer_quality
from verity.eval.run_store import load_run

LIVE = sorted((Path(__file__).resolve().parents[2] / "datasets/eval/runs/live").glob("*.json"))


@pytest.mark.parametrize("path", LIVE, ids=[p.name for p in LIVE])
def test_stored_aggregate_is_recomputed_from_the_audit_records(path: Path) -> None:
    run = load_run(path)
    q = answer_quality(run.per_audit)
    stored = run.scorecard.answer
    assert stored.faithfulness == pytest.approx(q.faithfulness_macro.value or 0.0)
    assert stored.citation_accuracy == pytest.approx(q.citation_accuracy_macro.value or 0.0)
    assert stored.hallucination_rate == pytest.approx(q.hallucination_rate.value or 0.0)
