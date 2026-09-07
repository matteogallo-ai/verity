"""Aggregate the in-memory OTel spans into structured metrics + Prometheus text.

Two consumers today:
- ``GET /metrics`` — JSON aggregates for the dashboard and any external UI.
- ``GET /metrics/prom`` — minimal Prometheus text format for scraping.

Both read the same source of truth: :class:`OTelTracer.exporter`. When the
process has no tracer (or a NoOp one), every aggregate is zero — the endpoints
still respond 200, so the dashboard is safe to load in any state.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    pass


@dataclass(frozen=True)
class StageAggregate:
    stage: str
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float


@dataclass(frozen=True)
class SessionMetrics:
    n_requests: int
    n_refusals: int
    refusal_rate: float
    total_cost_usd: float
    stages: list[StageAggregate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_requests": self.n_requests,
            "n_refusals": self.n_refusals,
            "refusal_rate": self.refusal_rate,
            "total_cost_usd": self.total_cost_usd,
            "stages": [
                {
                    "stage": s.stage,
                    "n": s.n,
                    "p50_ms": s.p50_ms,
                    "p95_ms": s.p95_ms,
                    "p99_ms": s.p99_ms,
                    "mean_ms": s.mean_ms,
                }
                for s in self.stages
            ],
        }


def aggregate_spans(spans: Iterable[Any]) -> SessionMetrics:
    """Compute per-stage latency percentiles + cost + refusal-rate over spans.

    Refusal rate is derived from ``agent.confidence`` spans (attribute
    ``refused``). Total cost sums ``usage.cost_usd`` across every span that
    carries it.
    """
    latencies: dict[str, list[float]] = {}
    total_cost = 0.0
    n_requests = 0
    n_refusals = 0

    for span in spans:
        name = _span_name(span)
        attrs = _span_attributes(span)

        if "latency_ms" in attrs:
            latencies.setdefault(name, []).append(float(attrs["latency_ms"]))

        if "usage.cost_usd" in attrs:
            total_cost += float(attrs["usage.cost_usd"])

        if name == "agent.confidence":
            n_requests += 1
            if bool(attrs.get("refused", False)):
                n_refusals += 1

    stages = [
        _stage_aggregate(stage, sorted(values)) for stage, values in sorted(latencies.items())
    ]
    refusal_rate = (n_refusals / n_requests) if n_requests else 0.0
    return SessionMetrics(
        n_requests=n_requests,
        n_refusals=n_refusals,
        refusal_rate=refusal_rate,
        total_cost_usd=round(total_cost, 6),
        stages=stages,
    )


def to_prometheus(metrics: SessionMetrics) -> str:
    """Emit a minimal Prometheus text-format document."""
    lines: list[str] = []
    lines.append("# HELP verity_requests_total Number of agent.answer requests observed.")
    lines.append("# TYPE verity_requests_total counter")
    lines.append(f"verity_requests_total {metrics.n_requests}")
    lines.append("# HELP verity_refusals_total Number of refused answers observed.")
    lines.append("# TYPE verity_refusals_total counter")
    lines.append(f"verity_refusals_total {metrics.n_refusals}")
    lines.append("# HELP verity_refusal_rate Refusals divided by total requests (0..1).")
    lines.append("# TYPE verity_refusal_rate gauge")
    lines.append(f"verity_refusal_rate {metrics.refusal_rate:.6f}")
    lines.append("# HELP verity_cost_usd_total Total USD cost observed in this session.")
    lines.append("# TYPE verity_cost_usd_total counter")
    lines.append(f"verity_cost_usd_total {metrics.total_cost_usd:.6f}")
    lines.append("# HELP verity_stage_latency_ms Per-stage latency in milliseconds.")
    lines.append("# TYPE verity_stage_latency_ms gauge")
    for s in metrics.stages:
        for quantile, value in (("0.5", s.p50_ms), ("0.95", s.p95_ms), ("0.99", s.p99_ms)):
            lines.append(
                f'verity_stage_latency_ms{{stage="{s.stage}",quantile="{quantile}"}} {value:.3f}'
            )
    return "\n".join(lines) + "\n"


def _span_name(span: Any) -> str:
    name = getattr(span, "name", None)
    return str(name) if name is not None else "unknown"


def _span_attributes(span: Any) -> dict[str, Any]:
    attrs = getattr(span, "attributes", None) or {}
    try:
        return dict(attrs)
    except TypeError:  # pragma: no cover — defensive
        return {}


def _stage_aggregate(stage: str, sorted_values: Sequence[float]) -> StageAggregate:
    n = len(sorted_values)
    if n == 0:
        return StageAggregate(stage=stage, n=0, p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, mean_ms=0.0)
    return StageAggregate(
        stage=stage,
        n=n,
        p50_ms=_pct(sorted_values, 50),
        p95_ms=_pct(sorted_values, 95),
        p99_ms=_pct(sorted_values, 99),
        mean_ms=round(sum(sorted_values) / n, 3),
    )


def _pct(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(float(values[0]), 3)
    rank = (pct / 100.0) * (len(values) - 1)
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    frac = rank - low
    return round(float(values[low] + (values[high] - values[low]) * frac), 3)


__all__ = ["SessionMetrics", "StageAggregate", "aggregate_spans", "to_prometheus"]
