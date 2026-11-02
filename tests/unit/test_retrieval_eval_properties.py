"""Property-based tests of rank fusion, retrieval metrics, refusal calibration and the
headline aggregation: the numbers the scorecard publishes."""

from __future__ import annotations

import itertools
from typing import Any
from uuid import UUID

import pytest
from hypothesis import given
from hypothesis import strategies as st

from verity.eval.headline import _compute_headline, _NormalizedInputs
from verity.eval.refusal import RefusalOutcome, build_confusion
from verity.eval.retrieval_metric import (
    PerExampleScore,
    macro_average,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from verity.retrieval.fusion import ReciprocalRankFusion
from verity.types import Chunk, RetrievalHit, RetrievalMetrics, RetrieverKind

_DOC = UUID(int=1)
_IDS = st.integers(0, 15).map(lambda n: UUID(int=n + 100))


def _hit(chunk_id: UUID, rank: int) -> RetrievalHit:
    chunk = Chunk(id=chunk_id, document_id=_DOC, text="t", ordinal=0, char_start=0, char_end=1)
    return RetrievalHit(chunk=chunk, score=0.0, kind=RetrieverKind.DENSE, rank=rank)


# A ranked list never repeats a chunk: stores return each chunk once per query.
_RANKED = st.lists(_IDS, unique=True, max_size=8).map(
    lambda ids: [_hit(cid, r) for r, cid in enumerate(ids)]
)


@given(st.lists(_RANKED, max_size=4), st.integers(-2, 20), st.integers(1, 100))
def test_rrf_scores_are_the_rank_sum_ordered_with_an_id_tiebreak(
    lists: list[list[RetrievalHit]], k: int, rrf_k: int
) -> None:
    fused = ReciprocalRankFusion(rrf_k=rrf_k).fuse(lists, k)

    expected: dict[str, float] = {}
    for hits in lists:
        for rank, h in enumerate(hits):
            expected[str(h.chunk.id)] = expected.get(str(h.chunk.id), 0.0) + 1.0 / (rrf_k + rank)
    assert len(fused) == max(0, min(k, len(expected)))
    assert [h.rank for h in fused] == list(range(len(fused)))
    assert all(h.kind is RetrieverKind.FUSED for h in fused)
    for h in fused:
        assert h.score == pytest.approx(expected[str(h.chunk.id)])
    for a, b in itertools.pairwise(fused):
        assert (a.score, str(b.chunk.id)) >= (b.score, str(a.chunk.id))
    # The k best are kept: nothing left out scores above the last one kept.
    if fused:
        kept = {str(h.chunk.id) for h in fused}
        assert all(s <= fused[-1].score for cid, s in expected.items() if cid not in kept)


@given(st.lists(_RANKED, max_size=4), st.integers(1, 20))
def test_rrf_does_not_depend_on_the_order_of_its_input_lists(
    lists: list[list[RetrievalHit]], k: int
) -> None:
    fusion = ReciprocalRankFusion(rrf_k=60)
    forward = [(h.chunk.id, h.score) for h in fusion.fuse(lists, k)]
    backward = [(h.chunk.id, h.score) for h in fusion.fuse(lists[::-1], k)]
    assert [cid for cid, _ in forward] == [cid for cid, _ in backward]
    assert [s for _, s in forward] == pytest.approx([s for _, s in backward])


_RETRIEVED = st.lists(st.sampled_from("abcdefgh"), unique=True, max_size=8)
_GOLD = st.sets(st.sampled_from("abcdefghij"), max_size=6)


@given(_RETRIEVED, _GOLD, st.integers(-1, 10))
def test_retrieval_metrics_are_shares_with_the_documented_denominators(
    retrieved: list[str], gold: set[str], k: int
) -> None:
    precision = precision_at_k(retrieved, gold, k)
    relevant = sum(1 for cid in retrieved[: max(k, 0)] if cid in gold)
    if k <= 0 or not retrieved:
        assert precision == 0.0
    else:
        assert precision == relevant / min(k, len(retrieved))
    if not gold:
        with pytest.raises(ValueError):
            recall_at_k(retrieved, gold, k)
        with pytest.raises(ValueError):
            ndcg_at_k(retrieved, gold, k)
        return
    recall = recall_at_k(retrieved, gold, k)
    ndcg = ndcg_at_k(retrieved, gold, k)
    assert recall == (relevant / len(gold) if k > 0 and retrieved else 0.0)
    assert 0.0 <= ndcg <= 1.0 + 1e-12
    assert (ndcg == 0.0) == (relevant == 0)


@given(_RETRIEVED, _GOLD.filter(bool), st.integers(1, 9))
def test_recall_never_drops_when_k_grows(retrieved: list[str], gold: set[str], k: int) -> None:
    assert recall_at_k(retrieved, gold, k + 1) >= recall_at_k(retrieved, gold, k)


@given(_GOLD.filter(bool), _RETRIEVED, st.integers(1, 10))
def test_ndcg_is_one_when_the_gold_chunks_come_first(
    gold: set[str], tail: list[str], k: int
) -> None:
    retrieved = sorted(gold) + [c for c in tail if c not in gold]
    assert ndcg_at_k(retrieved, gold, k) == pytest.approx(1.0)


_UNIT = st.floats(0, 1)
_METRICS = st.builds(
    RetrievalMetrics, precision_at_k=_UNIT, recall_at_k=_UNIT, ndcg_at_k=_UNIT, k=st.just(5)
)


@given(st.lists(_METRICS, max_size=6))
def test_macro_average_lies_between_the_per_example_extremes(
    metrics: list[RetrievalMetrics],
) -> None:
    rows = [PerExampleScore(example_id=str(i), metrics=m) for i, m in enumerate(metrics)]
    avg = macro_average(rows, k=5)
    for field in ("precision_at_k", "recall_at_k", "ndcg_at_k"):
        values = [getattr(m, field) for m in metrics] or [0.0]
        assert min(values) - 1e-9 <= getattr(avg, field) <= max(values) + 1e-9


_OUTCOME = st.tuples(st.booleans(), st.booleans())


@given(st.lists(_OUTCOME, max_size=20))
def test_confusion_matrix_puts_each_example_in_exactly_one_cell(
    pairs: list[tuple[bool, bool]],
) -> None:
    outcomes = [RefusalOutcome(str(i), a, r) for i, (a, r) in enumerate(pairs)]
    for o in outcomes:
        assert [o.tp, o.fp, o.fn, o.tn].count(True) == 1
    c = build_confusion(outcomes)
    assert (c.tp, c.fp, c.fn, c.tn) == (
        sum(1 for a, r in pairs if r and not a),
        sum(1 for a, r in pairs if r and a),
        sum(1 for a, r in pairs if not r and not a),
        sum(1 for a, r in pairs if not r and a),
    )
    assert 0.0 <= c.refusal_precision <= 1.0 and 0.0 <= c.refusal_recall <= 1.0
    if c.tp + c.fp:
        assert c.refusal_precision == c.tp / (c.tp + c.fp)
    if c.tp + c.fn:
        assert c.refusal_recall == c.tp / (c.tp + c.fn)


_AUDIT = st.fixed_dictionaries(
    {
        "expected_answerable": st.booleans(),
        "refused": st.booleans(),
        "judge_claims_is_none": st.booleans(),
        "faithfulness": _UNIT,
        "citation_accuracy": _UNIT,
        "hallucination_rate": _UNIT,
    }
)


def _inputs(audit: list[dict[str, Any]], agent_usd: float, judge_usd: float) -> _NormalizedInputs:
    refusal = [
        {
            "example_id": str(i),
            "expected_answerable": a["expected_answerable"],
            "observed_refused": a["refused"],
        }
        for i, a in enumerate(audit)
    ]
    return _NormalizedInputs(
        scorecard_git_sha="sha",
        scorecard_created_at_iso="2026-01-01T00:00:00+00:00",
        scorecard_dataset="d",
        n_examples=len(audit),
        retrieval_k=5,
        retrieval_precision=0.0,
        retrieval_recall=0.0,
        retrieval_ndcg=0.0,
        latency_p50=0.0,
        latency_p95=0.0,
        latency_p99=0.0,
        cold_start_ms=None,
        cost_per_query_usd=0.0,
        agent_cost_usd=agent_usd,
        judge_cost_usd=judge_usd,
        embedding_model="m",
        agent_summary="a",
        judge_summary="j",
        per_refusal_records=refusal,
        per_audit_records=audit,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@given(st.lists(_AUDIT, max_size=12), st.floats(0, 10), st.floats(0, 10))
def test_headline_metrics_are_means_over_their_declared_scope(
    audit: list[dict[str, Any]], agent_usd: float, judge_usd: float
) -> None:
    headline = _compute_headline(_inputs(audit, agent_usd, judge_usd), dataset_sha=None)
    sizes = headline["sample_sizes"]
    assert sizes["n_answered"] + sizes["n_refused"] == len(audit)

    judged = [a for a in audit if not a["judge_claims_is_none"]]
    at_risk = [a for a in audit if a["expected_answerable"] and not a["refused"]]
    quality = headline["answer_quality"]
    for name, scope in (
        ("faithfulness", judged),
        ("citation_accuracy", judged),
        ("hallucination_rate", at_risk),
    ):
        assert quality[name]["n_denominator"] == len(scope)
        assert quality[name]["value"] == round(_mean([a[name] for a in scope]), 6)
    assert sizes["n_judged"] == len(judged)
    assert sizes["n_answerable_answered"] == len(at_risk)
    assert headline["cost"]["total_usd"] == round(agent_usd + judge_usd, 6)

    confusion = build_confusion(
        RefusalOutcome(str(i), a["expected_answerable"], a["refused"]) for i, a in enumerate(audit)
    )
    calibration = headline["refusal_calibration"]
    assert calibration["out_of_scope_answered"] == {
        "count": confusion.fn,
        "total": confusion.tp + confusion.fn,
    }
    # Both modules compute the refusal pair; they agree wherever it is defined.
    if confusion.tp + confusion.fp:
        assert calibration["refusal_precision"] == confusion.refusal_precision
    if confusion.tp + confusion.fn:
        assert calibration["refusal_recall"] == confusion.refusal_recall
