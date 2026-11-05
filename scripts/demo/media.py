"""Spend cap and ffmpeg edit of the demo recording, kept free of Playwright so the
fast lane can test them.

The recorder (record.py) logs when each answer was requested and when it rendered.
The edit plays the rest at real speed and speeds up only those waits, by a factor
the recording itself captions while the wait is on screen; the real latency of each
answer stays readable in its usage footer.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

# The cap of one recording session, over every live answer it requests.
CAP_USD = 0.50
# The bound is priced for this model; the recorder refuses to run against another.
AGENT_MODEL = "claude-sonnet-4-6"
# Decomposer + synthesizer + scorer max_tokens (256 + 512 + 256, pinned by
# tests/unit/test_budget.py::test_every_agent_call_bounds_its_output_tokens).
MAX_OUTPUT_TOKENS = 1024
# 5x the largest input of the 162 answers of record (12,162 tokens). An assumption,
# not a provider limit; the overlay turns SDK retries off so one call bills once.
MAX_INPUT_TOKENS = 60_000
PER_ANSWER_BOUND_USD = 0.20
# The API admits an answer while the session's spend is under this budget, and the
# recorder sends one question at a time, so at most one answer can start below it
# and finish above it: total spend < SERVER_BUDGET_USD + PER_ANSWER_BOUND_USD = CAP_USD.
# The API enforces the cap; the recorder only checks the total afterwards.
SERVER_BUDGET_USD = round(CAP_USD - PER_ANSWER_BOUND_USD, 2)

WAIT_SPEED_MP4 = 3.0
WAIT_SPEED_GIF = 8.0


class SpendCapError(RuntimeError):
    """The recorded answers cost more than CAP_USD."""


def check_spend(costs_usd: Sequence[float], cap_usd: float = CAP_USD) -> float:
    """The total spend of the session; raises when it exceeds the cap."""
    total = float(sum(costs_usd))
    if total > cap_usd:
        raise SpendCapError(f"demo spend ${total:.4f} exceeds the ${cap_usd:.2f} cap")
    return total


@dataclass(frozen=True)
class Wait:
    """One answer being produced: seconds from the start of the recording."""

    start: float
    end: float


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speed: float


def segments(span: tuple[float, float], waits: Sequence[Wait], wait_speed: float) -> list[Segment]:
    """Cut ``span`` of the recording into real-speed pieces and sped-up waits.

    Waits are clipped to the span; overlapping or unordered waits are a recorder bug.
    """
    lo, hi = span
    if not lo < hi:
        raise ValueError(f"empty span {span}")
    if wait_speed < 1.0:
        raise ValueError("wait_speed must be >= 1")
    ordered = sorted(waits, key=lambda w: w.start)
    for a, b in itertools.pairwise(ordered):
        if b.start < a.end:
            raise ValueError(f"overlapping waits {a} and {b}")
    out: list[Segment] = []
    cursor = lo
    for wait in ordered:
        start, end = max(wait.start, lo), min(wait.end, hi)
        if end <= start:
            continue
        if start > cursor:
            out.append(Segment(cursor, start, 1.0))
        out.append(Segment(start, end, wait_speed))
        cursor = end
    if cursor < hi:
        out.append(Segment(cursor, hi, 1.0))
    return out


def duration(parts: Sequence[Segment]) -> float:
    """Length of the edit, in seconds."""
    return sum((s.end - s.start) / s.speed for s in parts)


def filter_graph(parts: Sequence[Segment], tail: str) -> str:
    """An ffmpeg filter_complex: trim and retime each part, concatenate, then ``tail``."""
    if not parts:
        raise ValueError("no segments")
    chains = [
        f"[0:v]trim=start={s.start:.3f}:end={s.end:.3f},setpts=(PTS-STARTPTS)/{s.speed:g}[v{i}]"
        for i, s in enumerate(parts)
    ]
    labels = "".join(f"[v{i}]" for i in range(len(parts)))
    return ";".join([*chains, f"{labels}concat=n={len(parts)}:v=1:a=0,{tail}"])


MP4_TAIL = "fps=30,format=yuv420p[out]"
GIF_TAIL = (
    "fps=12,scale=900:-1:flags=lanczos,split[a][b];"
    "[a]palettegen=max_colors=128:stats_mode=diff[p];"
    "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle[out]"
)
