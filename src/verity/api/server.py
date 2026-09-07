"""FastAPI app that exposes ``/metrics`` (JSON) + ``/metrics/prom`` (text) +
``/dashboard`` (server-rendered HTML).

The dashboard is intentionally hand-rendered HTML (no build system, no
framework) — a full Next.js UI is scheduled for S6. This keeps S5 focused on
"the data is real, the endpoints work, a reviewer can *see* something".

Two data sources feed the endpoints:

- **In-memory session spans** — the :class:`OTelTracer.exporter` accumulates
  spans from every request served by this process. Fresh every restart.
- **Persisted eval runs** — the JSON scorecards under
  ``datasets/eval/runs/eval_*.json``, read via :class:`FileRunStore`.

Nothing sensitive is exposed: only aggregates (counts, percentiles, cost),
never chunk texts or raw prompts.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from verity import __version__
from verity.api.aggregator import SessionMetrics, aggregate_spans, to_prometheus
from verity.eval import FileRunStore


def create_app(
    *,
    tracer: Any = None,
    runs_dir: Path = Path("datasets/eval/runs"),
    dataset: str = "questions",
) -> FastAPI:
    """Build the FastAPI app.

    ``tracer`` may be a :class:`~verity.observability.tracer.OTelTracer`
    (session spans populate ``/metrics``) or ``None`` (all session aggregates
    read zero — endpoints still respond 200). ``runs_dir`` + ``dataset``
    control which persisted eval runs the dashboard surfaces.
    """
    app = FastAPI(
        title="verity",
        version=__version__,
        docs_url="/docs",
        redoc_url=None,
    )
    store = FileRunStore(runs_dir)

    def _session_metrics() -> SessionMetrics:
        if tracer is None or not hasattr(tracer, "get_finished_spans"):
            return aggregate_spans([])
        return aggregate_spans(tracer.get_finished_spans())

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "version": __version__})

    @app.get("/metrics")
    def metrics() -> JSONResponse:
        return JSONResponse(_session_metrics().to_dict())

    @app.get("/metrics/prom", response_class=PlainTextResponse)
    def metrics_prom() -> Response:
        return PlainTextResponse(
            to_prometheus(_session_metrics()),
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        session = _session_metrics()
        runs = store.history(dataset, limit=10)
        return HTMLResponse(_render_dashboard(session=session, runs=runs, dataset=dataset))

    return app


# --------------------------------------------------------------------------------------
# Dashboard rendering
# --------------------------------------------------------------------------------------


def _render_dashboard(*, session: SessionMetrics, runs: list, dataset: str) -> str:  # type: ignore[type-arg]
    """Hand-rendered dashboard HTML — no template engine, no dependency.

    Two panels: (1) live in-memory session metrics from the current process,
    (2) persisted eval-run history (cost/latency/quality over time). Every
    value is an aggregate — no raw chunk text or prompt is ever rendered.
    """
    stage_rows = (
        "".join(
            f"<tr><td>{html.escape(s.stage)}</td><td class='n'>{s.n}</td>"
            f"<td class='n'>{s.p50_ms:.1f}</td><td class='n'>{s.p95_ms:.1f}</td>"
            f"<td class='n'>{s.p99_ms:.1f}</td><td class='n'>{s.mean_ms:.1f}</td></tr>"
            for s in session.stages
        )
        or "<tr><td colspan=6 class='muted'>No spans yet — issue a request to populate.</td></tr>"
    )

    run_rows = (
        "".join(
            f"<tr><td>{html.escape(r.scorecard.git_sha)}</td>"
            f"<td class='mono'>{html.escape(r.agent_model)}</td>"
            f"<td class='mono'>{html.escape(r.judge_model)}</td>"
            f"<td class='n'>{r.scorecard.retrieval.ndcg_at_k:.3f}</td>"
            f"<td class='n'>{r.scorecard.answer.refusal_recall:.3f}</td>"
            f"<td class='n'>{r.scorecard.latency.p50_ms:.0f}</td>"
            f"<td class='n'>{r.scorecard.latency.p95_ms:.0f}</td>"
            f"<td class='n'>{r.scorecard.cost_per_query_usd:.4f}</td>"
            f"<td class='n'>{_fmt_cold(r.scorecard.cold_start_ms)}</td></tr>"
            for r in runs
        )
        or f"<tr><td colspan=9 class='muted'>No persisted runs found for dataset '{html.escape(dataset)}'.</td></tr>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Verity dashboard</title>
  <style>
    body {{ font-family: -apple-system, sans-serif; margin: 2rem; color: #222; }}
    h1 {{ margin: 0 0 .5rem; }}
    h2 {{ margin: 2rem 0 .5rem; border-bottom: 1px solid #ddd; padding-bottom: .25rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: .5rem 0; }}
    th, td {{ padding: .35rem .6rem; border-bottom: 1px solid #eee; text-align: left; }}
    th {{ background: #fafafa; font-size: .85rem; text-transform: uppercase; letter-spacing: .04em; color: #666; }}
    td.n {{ text-align: right; font-variant-numeric: tabular-nums; }}
    td.mono {{ font-family: ui-monospace, Menlo, monospace; font-size: .85rem; }}
    td.muted {{ color: #888; text-align: center; padding: 1rem; }}
    .kpis {{ display: flex; gap: 1rem; margin: 1rem 0; }}
    .kpi {{ flex: 1; padding: 1rem; background: #f7f7f9; border-radius: .5rem; }}
    .kpi .label {{ font-size: .75rem; text-transform: uppercase; color: #888; }}
    .kpi .value {{ font-size: 1.5rem; font-weight: bold; }}
    footer {{ margin-top: 3rem; font-size: .8rem; color: #888; }}
  </style>
</head>
<body>
  <h1>Verity dashboard <small style='color:#888;font-weight:normal'>v{html.escape(__version__)}</small></h1>
  <p class="muted">Live session metrics + persisted eval-run history. No raw content shown.</p>

  <h2>Live session</h2>
  <div class="kpis">
    <div class="kpi"><div class="label">Requests</div><div class="value">{session.n_requests}</div></div>
    <div class="kpi"><div class="label">Refusals</div><div class="value">{session.n_refusals}</div></div>
    <div class="kpi"><div class="label">Refusal rate</div><div class="value">{session.refusal_rate:.1%}</div></div>
    <div class="kpi"><div class="label">Session cost (USD)</div><div class="value">${session.total_cost_usd:.4f}</div></div>
  </div>

  <h2>Per-stage latency (this session)</h2>
  <table>
    <thead><tr><th>stage</th><th>n</th><th>p50 ms</th><th>p95 ms</th><th>p99 ms</th><th>mean ms</th></tr></thead>
    <tbody>{stage_rows}</tbody>
  </table>

  <h2>Persisted eval runs — dataset '{html.escape(dataset)}'</h2>
  <table>
    <thead><tr><th>git sha</th><th>agent</th><th>judge</th><th>nDCG@k</th>
      <th>refusal recall</th><th>p50 ms</th><th>p95 ms</th><th>cost/query</th><th>cold start ms</th></tr></thead>
    <tbody>{run_rows}</tbody>
  </table>

  <footer>Endpoints: <a href="/metrics">/metrics</a> (JSON) · <a href="/metrics/prom">/metrics/prom</a> · <a href="/health">/health</a></footer>
</body></html>"""


def _fmt_cold(cold_ms: float | None) -> str:
    return "-" if cold_ms is None else f"{cold_ms:.0f}"


__all__ = ["create_app"]
