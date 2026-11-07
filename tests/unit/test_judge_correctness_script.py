"""scripts/judge_correctness.py: what it grades, its cap, and its committed outputs. No network."""

from __future__ import annotations

import asyncio

from verity.eval.correctness import LLMCorrectnessJudge
from verity.types import Completion, Message, Provider, UsageStats


class _Scripted:
    clients: tuple[object, ...] = ()

    def __init__(self, text: str) -> None:
        self.text = text

    async def complete(self, messages: list[Message], *, max_tokens: int = 1024) -> Completion:
        return Completion(
            text=self.text,
            provider=Provider.ANTHROPIC,
            model="claude-opus-5-5",
            usage=UsageStats(cost_usd=0.01, llm_calls=1),
        )


def _script():  # type: ignore[no-untyped-def]
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "judge_correctness.py"
    spec = importlib.util.spec_from_file_location("judge_correctness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["judge_correctness"] = module
    spec.loader.exec_module(module)
    return module


def test_the_test_run_grades_its_answered_answerable_questions_only() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    pairs = _script().to_grade(
        root / "datasets/eval/runs/live/eval_969e1d5_test.json", root / "datasets/eval/test.jsonl"
    )
    assert len(pairs) == 31  # the 31 of the hallucination rate's denominator
    assert all(a.expected_answerable and not a.refused for a, _ in pairs)


def test_a_capped_grade_keeps_what_it_paid_for() -> None:
    from verity.llm.budget import SpendCappedClient

    script = _script()
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    pairs = script.to_grade(
        root / "datasets/eval/runs/live/eval_969e1d5_test.json", root / "datasets/eval/test.jsonl"
    )[:3]

    capped = SpendCappedClient(_Scripted('{"verdict": "correct", "reason": "ok"}'), cap_usd=0.015)  # type: ignore[arg-type]
    verdicts, incomplete = asyncio.run(script._grade(LLMCorrectnessJudge(capped), pairs))
    # The scripted client has no price, so the cap only sees reported spend: 0.01 per call.
    assert incomplete == "budget_exceeded" and len(verdicts) == 2


def test_summary_counts_verdicts_and_the_number_disagreements() -> None:
    record = {
        "run_artefact": "r.json",
        "judge_model": "m",
        "cost_usd": 0.03,
        "incomplete_reason": None,
        "n_to_grade": 3,
        "verdicts": [
            {"example_id": "q1", "verdict": "correct"},
            {"example_id": "q2", "verdict": "incorrect"},
            {"example_id": "q3", "verdict": "partial"},
        ],
    }
    s = _script().summarise(record, {"q1": True, "q2": True, "q3": None})
    assert (s["correct"]["k"], s["correct"]["n"]) == (1, 3)
    assert s["not_correct_ids"] == ["q2", "q3"]
    assert s["numbers"] == {"n_with_numbers": 2, "numbers_match": 2, "disagree_ids": ["q2"]}


def test_the_verdict_file_does_not_store_the_number_check() -> None:
    """The number check is recomputed at render time from the artefact, so a fix to it
    never needs the paid verdict file rewritten."""
    import inspect

    assert "numbers_match" not in inspect.getsource(_script()._grade)


def test_the_committed_correctness_block_matches_its_verdict_files() -> None:
    assert _script().main(["render", "--check"]) == 0


def test_a_provider_failure_keeps_the_verdicts_already_paid_for() -> None:
    """An outage mid-run writes the paid verdicts as an incomplete file instead of losing them."""
    from pathlib import Path

    from verity.llm.base import ProviderUnavailableError

    script = _script()
    root = Path(__file__).resolve().parents[2]
    pairs = script.to_grade(
        root / "datasets/eval/runs/live/eval_969e1d5_test.json", root / "datasets/eval/test.jsonl"
    )[:3]

    class _FailsSecond(_Scripted):
        calls = 0

        async def complete(self, messages: list[Message], *, max_tokens: int = 1024) -> Completion:
            type(self).calls += 1
            if type(self).calls == 2:
                raise ProviderUnavailableError("down")
            return await super().complete(messages, max_tokens=max_tokens)

    judge = LLMCorrectnessJudge(_FailsSecond('{"verdict": "correct", "reason": "ok"}'))  # type: ignore[arg-type]
    verdicts, incomplete = asyncio.run(script._grade(judge, pairs))
    assert len(verdicts) == 1
    assert incomplete == "provider_error:ProviderUnavailableError"
