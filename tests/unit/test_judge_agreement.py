"""scripts/judge_agreement.py: κ by hand, blind export, and the join after annotation."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

from verity.eval.run_store import FileRunStore
from verity.types import (
    AnswerMetrics,
    ClaimVerdict,
    EvalRun,
    LatencyMetrics,
    PerExampleAudit,
    RetrievalHit,
    RetrievalMetrics,
    Scorecard,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "judge_agreement.py"


def _module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("judge_agreement", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their module through sys.modules.
    monkeypatch.setitem(sys.modules, "judge_agreement", module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    for name, file in (
        ("BLIND_CSV", "blind.csv"),
        ("KEY", "key.json"),
        ("FINAL_CSV", "final.csv"),
        ("RESULT", "agreement.json"),
    ):
        monkeypatch.setattr(module, name, tmp_path / file)
    monkeypatch.setattr(module, "OUT_DIR", tmp_path)
    return module


def test_kappa_by_hand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _module(tmp_path, monkeypatch)
    s, u = "supported", "unsupported"
    # 10 pairs: agree 7 (6 s/s, 1 u/u), judge s / human u 2, judge u / human s 1.
    pairs = [(s, s)] * 6 + [(u, u)] + [(s, u)] * 2 + [(u, s)]
    p_o = 0.7
    p_e = (8 / 10) * (7 / 10) + (2 / 10) * (3 / 10)
    assert mod.cohen_kappa(pairs) == pytest.approx((p_o - p_e) / (1 - p_e))
    assert mod.cohen_kappa([(s, s)] * 5) is None
    assert mod.cohen_kappa([]) is None


def _run(hit: RetrievalHit) -> EvalRun:
    verdicts = (
        ClaimVerdict(claim="Notice is 30 days.", supported=True, citation_ok=True),
        ClaimVerdict(claim="Notice is by fax.", supported=False, citation_ok=False),
    )
    metrics = AnswerMetrics(
        faithfulness=0.5,
        citation_accuracy=0.5,
        hallucination_rate=1.0,
        refusal_precision=0.0,
        refusal_recall=0.0,
    )
    return EvalRun(
        scorecard=Scorecard(
            git_sha="abc1234",
            created_at=datetime(2026, 11, 2, tzinfo=UTC),
            dataset="dev",
            n_examples=1,
            retrieval=RetrievalMetrics(precision_at_k=0, recall_at_k=0, ndcg_at_k=0, k=8),
            answer=metrics,
            latency=LatencyMetrics(p50_ms=0, p95_ms=0, p99_ms=0),
            cost_per_query_usd=0.0,
        ),
        embedding_model="e",
        judge_model="j",
        per_audit=(
            PerExampleAudit(
                example_id="q-001",
                question="?",
                expected_answerable=True,
                answer_text="a",
                refused=False,
                hits_used=(hit,),
                judge_claims=verdicts,
                answer_metrics=metrics,
            ),
        ),
    )


def test_export_is_blind_and_kappa_joins_the_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hit: RetrievalHit
) -> None:
    mod = _module(tmp_path, monkeypatch)
    store = FileRunStore(tmp_path / "runs")
    run = _run(hit)
    store.save(run)
    assert mod.main(["export", str(store.path_for("abc1234", "dev"))]) == 0

    with (tmp_path / "blind.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["claim_id"] for r in rows] == ["eval_abc1234_dev/q-001/0", "eval_abc1234_dev/q-001/1"]
    assert "judge_verdict" not in rows[0]
    assert hit.chunk.text in rows[0]["evidence"]

    assert mod.main(["kappa"]) == 1  # not annotated yet
    (tmp_path / "key.json").write_text(json.dumps({"verdicts": {}}))
    assert mod.main(["kappa"]) == 1  # a claim the key does not know
    assert mod.main(["export", str(store.path_for("abc1234", "dev"))]) == 0
    rows[0]["human_verdict"], rows[1]["human_verdict"] = "supported", "supported"
    with (tmp_path / "blind.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert mod.main(["kappa"]) == 0
    result = json.loads((tmp_path / "agreement.json").read_text())
    assert result["n_claims"] == 2 and result["agreement"] == 0.5
    with (tmp_path / "final.csv").open(newline="", encoding="utf-8") as fh:
        final = list(csv.DictReader(fh))
    assert [r["judge_verdict"] for r in final] == ["supported", "unsupported"]
