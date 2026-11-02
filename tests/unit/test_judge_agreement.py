"""scripts/judge_agreement.py: blind export of real + perturbed claims, the judge's
sensitivity on the perturbed ones, κ on the mixed set."""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

from verity.eval.run_store import FileRunStore
from verity.types import (
    Answer,
    AnswerMetrics,
    Chunk,
    Citation,
    ClaimVerdict,
    EvalExample,
    EvalRun,
    LatencyMetrics,
    PerExampleAudit,
    RetrievalHit,
    RetrievalMetrics,
    RetrieverKind,
    Scorecard,
    UsageStats,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "judge_agreement.py"
S, U = "supported", "unsupported"
DOC = uuid.UUID(int=999)

# One claim per answer, each open to one kind of perturbation (and the first two to
# several), plus answers whose claims stay real.
CLAIMS = [
    ("Notice is 30 days.", True),
    ("The policy took effect in March 2024.", True),
    ("Backups are encrypted.", True),
    ("Northwind signs the order form.", True),
    ("Fees are due in 45 days.", True),
    ("The audit took place in June 2023.", True),
    ("Support is staffed around the clock.", True),
    ("Notice is by fax.", False),
    ("Acme hosts the data.", True),
]


def _module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("judge_agreement", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their module through sys.modules.
    monkeypatch.setitem(sys.modules, "judge_agreement", module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(
        module, "CSV_FILES", (tmp_path / "out" / "m1.csv", tmp_path / "out" / "m2.csv")
    )
    for name in ("KEY", "REJUDGED", "FINAL_CSV", "RESULT"):
        monkeypatch.setattr(module, name, tmp_path / "out" / f"{name.lower()}.json")
    entities = tmp_path / "entities.json"
    entities.write_text(json.dumps({"_doc": "x", "company": ["Acme", "Northwind"]}))
    monkeypatch.setattr(module, "ENTITIES", entities)
    monkeypatch.setattr(module, "N_PER_KIND", 1)
    monkeypatch.setattr(module, "N_REAL", 4)
    return module


def _metrics() -> AnswerMetrics:
    return AnswerMetrics(
        faithfulness=1.0,
        citation_accuracy=1.0,
        hallucination_rate=0.0,
        refusal_precision=0.0,
        refusal_recall=0.0,
    )


def _audit(i: int, claim: str, supported: bool) -> PerExampleAudit:
    chunk = Chunk(
        id=uuid.UUID(int=i + 1),
        document_id=DOC,
        text=f"Extract {i}: the contract terms. Notice is 30 days.",
        ordinal=i,
        char_start=0,
        char_end=50,
    )
    return PerExampleAudit(
        example_id=f"q-{i:03d}",
        question=f"Question {i}?",
        expected_answerable=True,
        answer_text=claim,
        refused=False,
        citations=(
            Citation(
                chunk_id=chunk.id,
                document_id=DOC,
                quote="the contract terms",
                char_start=0,
                char_end=1,
            ),
            Citation(
                chunk_id=chunk.id,
                document_id=DOC,
                quote="Notice is 30 days.",
                char_start=0,
                char_end=1,
            ),
        ),
        hits_used=(RetrievalHit(chunk=chunk, score=0.9, kind=RetrieverKind.RERANKED, rank=0),),
        judge_claims=(ClaimVerdict(claim=claim, supported=supported, citation_ok=True),),
        answer_metrics=_metrics(),
    )


def _save_run(tmp_path: Path) -> Path:
    run = EvalRun(
        scorecard=Scorecard(
            git_sha="abc1234",
            created_at=datetime(2026, 11, 2, tzinfo=UTC),
            dataset="dev",
            n_examples=len(CLAIMS),
            retrieval=RetrievalMetrics(precision_at_k=0, recall_at_k=0, ndcg_at_k=0, k=8),
            answer=_metrics(),
            latency=LatencyMetrics(p50_ms=0, p95_ms=0, p99_ms=0),
            cost_per_query_usd=0.0,
        ),
        embedding_model="e",
        judge_model="j",
        per_audit=tuple(_audit(i, c, s) for i, (c, s) in enumerate(CLAIMS)),
    )
    store = FileRunStore(tmp_path / "runs")
    store.save(run)
    return store.path_for("abc1234", "dev")


def _rows(mod: ModuleType) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in mod.CSV_FILES:
        with path.open(newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def test_kappa_by_hand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _module(tmp_path, monkeypatch)
    # 10 pairs: agree 7 (6 s/s, 1 u/u), judge s / human u 2, judge u / human s 1.
    pairs = [(S, S)] * 6 + [(U, U)] + [(S, U)] * 2 + [(U, S)]
    p_o = 0.7
    p_e = (8 / 10) * (7 / 10) + (2 / 10) * (3 / 10)
    assert mod.cohen_kappa(pairs) == pytest.approx((p_o - p_e) / (1 - p_e))
    assert mod.cohen_kappa([(S, S)] * 5) is None
    assert mod.cohen_kappa([]) is None


def test_cited_passage_is_the_citation_closest_to_the_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _module(tmp_path, monkeypatch)
    citations = _audit(0, "x", True).citations
    assert mod.cited_passage("Notice is 45 days.", citations) == "Notice is 30 days."
    assert mod.cited_passage("Notice is 45 days.", ()) == "(the answer cited no passage)"


def test_plan_keeps_perturbed_answers_out_of_the_real_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _module(tmp_path, monkeypatch)
    claims = mod.collect([_save_run(tmp_path)])
    perturbed, real = mod.plan(claims, mod.load_entities())
    assert sorted(str(p.kind) for _, p in perturbed) == ["date", "entity", "negation", "number"]
    sources = {c.answer_id for c, _ in perturbed}
    assert len(sources) == 4  # one perturbation per answer
    assert not sources & {c.answer_id for c in real}
    assert len(real) == 4 and any(c.judge_verdict == U for c in real)  # unsupported kept
    assert mod.plan(claims, mod.load_entities()) == (perturbed, real)  # seeded


def test_plan_fails_when_a_kind_cannot_be_filled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _module(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "N_PER_KIND", 5)
    with pytest.raises(ValueError, match="perturbation"):
        mod.plan(mod.collect([_save_run(tmp_path)]), mod.load_entities())


def test_export_writes_two_blind_files_cited_passage_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _module(tmp_path, monkeypatch)
    assert mod.main(["export", str(_save_run(tmp_path))]) == 0
    files = []
    for path in mod.CSV_FILES:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == [
                "item_id",
                "claim",
                "cited_passage",
                "extracts",
                "human_verdict",
            ]
            files.append(list(reader))
    assert [len(f) for f in files] == [4, 4]
    assert [r["item_id"] for r in files[0]] == ["A01", "A02", "A03", "A04"]
    key = json.loads(mod.KEY.read_text())["items"]
    rows = files[0] + files[1]
    assert sorted(key) == sorted(r["item_id"] for r in rows)
    assert sum(1 for v in key.values() if v["origin"] == "perturbed") == 4
    for r in rows:
        item = key[r["item_id"]]
        assert r["claim"] == item["claim"]
        assert "Extract" in r["extracts"] and r["human_verdict"] == ""
        if item["origin"] == "perturbed":
            assert r["claim"] != item["original"] and item["judge_verdict"] is None


class _FakeJudge:
    """Says "unsupported" when the claim holds a word from ``catch``; costs $0.01."""

    model = "fake-judge"

    def __init__(self, catch: tuple[str, ...], malformed: tuple[str, ...] = ()) -> None:
        self.catch, self.malformed = catch, malformed
        self.calls = 0
        self.last_usage: UsageStats | None = None
        self.last_claims: tuple[ClaimVerdict, ...] | None = None

    async def judge(self, example: EvalExample, answer: Answer) -> AnswerMetrics:
        self.calls += 1
        self.last_usage = UsageStats(cost_usd=0.01, llm_calls=1)
        if any(m in answer.text for m in self.malformed):
            self.last_claims = ()
        else:
            caught = any(w in answer.text for w in self.catch)
            self.last_claims = (
                ClaimVerdict(claim=answer.text, supported=not caught, citation_ok=True),
            )
        return _metrics()


def test_judge_perturbed_detects_counts_malformed_and_stops_at_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _module(tmp_path, monkeypatch)
    assert mod.main(["export", str(_save_run(tmp_path))]) == 0
    judge = _FakeJudge(catch=("not",), malformed=("Acme",))
    result = asyncio.run(mod.judge_perturbed(judge, budget_usd=1.0))
    # Item ids are opaque: check the verdicts by claim text.
    by_text = {
        json.loads(mod.KEY.read_text())["items"][i]["claim"]: v["verdict"]
        for i, v in result["verdicts"].items()
    }
    assert len(by_text) == 4 and result["cost_usd"] == pytest.approx(0.04)
    for text, verdict in by_text.items():
        if "Acme" in text:
            assert verdict is None  # malformed output: no verdict, not a miss
        elif " not " in text:
            assert verdict == U
        else:
            assert verdict == S

    capped = _FakeJudge(catch=())
    result = asyncio.run(mod.judge_perturbed(capped, budget_usd=0.015))
    assert capped.calls == 2 and len(result["verdicts"]) == 2


def test_kappa_reports_sensitivity_and_mixed_agreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _module(tmp_path, monkeypatch)
    assert mod.main(["export", str(_save_run(tmp_path))]) == 0
    assert mod.main(["kappa"]) == 1  # perturbed claims not judged yet
    key = json.loads(mod.KEY.read_text())["items"]
    perturbed = sorted(i for i, v in key.items() if v["origin"] == "perturbed")
    # The judge catches 3 of the 4 perturbed claims.
    mod.REJUDGED.write_text(
        json.dumps({"verdicts": {i: {"verdict": U if n else S} for n, i in enumerate(perturbed)}})
    )
    assert mod.main(["kappa"]) == 1  # not annotated yet
    # The human marks every perturbed claim unsupported and every real one supported.
    for path in mod.CSV_FILES:
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            r["human_verdict"] = U if key[r["item_id"]]["origin"] == "perturbed" else S
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    assert mod.main(["kappa"]) == 0
    result = json.loads(mod.RESULT.read_text())
    sens = result["sensitivity"]
    assert (sens["judge_detected"]["k"], sens["judge_detected"]["n"]) == (3, 4)
    assert (
        sens["judge_detected"]["ci95_wilson"][0] < 0.75 < sens["judge_detected"]["ci95_wilson"][1]
    )
    assert sens["human_unsupported"]["k"] == 4
    mixed = result["agreement_mixed"]
    # 8 pairs: the real unsupported claim disagrees (judge u, human s), one perturbed
    # claim is missed (judge s, human u); 6 agree.
    assert mixed["n_claims"] == 8 and mixed["observed"]["k"] == 6
    assert mixed["confusion"]["judge_supported/human_unsupported"] == 1
    assert mixed["cohen_kappa"] is not None
    with mod.FINAL_CSV.open(newline="", encoding="utf-8") as fh:
        final = list(csv.DictReader(fh))
    assert {r["origin"] for r in final} == {"real", "perturbed"}


def test_plan_never_perturbs_an_answer_with_an_unsupported_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every unsupported claim goes into the real sample: a perturbation from the same
    # answer would push it out. The unsupported answer here also holds a claim open to
    # every kind, so some seed would pick it.
    mod = _module(tmp_path, monkeypatch)
    claims = mod.collect([_save_run(tmp_path)])
    mixed = _audit(42, "Notice is by fax.", False).model_copy(
        update={
            "judge_claims": (
                ClaimVerdict(claim="Notice is by fax.", supported=False, citation_ok=True),
                ClaimVerdict(
                    claim="Northwind gave notice in March 2024 within 30 days.",
                    supported=True,
                    citation_ok=True,
                ),
            )
        }
    )
    claims += [
        mod.Claim(f"run/q-042/{j}", v.claim, S if v.supported else U, "run", mixed)
        for j, v in enumerate(mixed.judge_claims or ())
    ]
    for seed in range(30):
        _, real = mod.plan(claims, mod.load_entities(), seed=seed)
        assert sum(1 for c in real if c.judge_verdict == U) == 2, seed
