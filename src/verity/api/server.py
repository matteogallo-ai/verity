"""FastAPI app that exposes ``/metrics`` (JSON) + ``/metrics/prom`` (text) +
``/dashboard`` (server-rendered HTML).

The dashboard is hand-rendered HTML (no build system, no framework) so the
operational view works without the Next.js UI in ``web/``.

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
import re
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from verity import __version__
from verity.api.aggregator import SessionMetrics, aggregate_spans, to_prometheus
from verity.api.router import router as api_router
from verity.api.state import AppRuntime, default_agent_mode
from verity.config import Settings, get_settings
from verity.eval import FileRunStore
from verity.observability.tracer import NoOpTracer

_log = structlog.get_logger("verity.api")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# Multipart framing around one file part; generous so a legal upload is never refused.
_MULTIPART_OVERHEAD_BYTES = 16 * 1024
# JSON envelope around a question at 4 UTF-8 bytes per character, plus headroom.
_JSON_OVERHEAD_BYTES = 4 * 1024


def create_app(
    *,
    tracer: Any = None,
    runs_dir: Path = Path("datasets/eval/runs"),
    dataset: str = "questions",
    corpus_dir: Path = Path("datasets/corpus"),
    agent_mode: str | None = None,
    cors_origins: tuple[str, ...] = ("http://localhost:3000",),
    settings: Settings | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    ``tracer`` may be a :class:`~verity.observability.tracer.OTelTracer`
    (session spans populate ``/metrics``) or ``None`` (all session aggregates
    read zero — endpoints still respond 200). ``runs_dir`` + ``dataset``
    control which persisted eval runs the dashboard surfaces. ``corpus_dir``
    is preloaded into the shared in-memory store on first request so the
    demo UI can ask questions immediately. ``settings`` defaults to the process
    settings.
    """
    settings = settings or get_settings()
    app = FastAPI(
        title="verity",
        version=__version__,
        docs_url="/docs",
        redoc_url=None,
    )
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
    runtime_tracer = tracer if tracer is not None else NoOpTracer()
    resolved_mode = agent_mode or default_agent_mode()
    if resolved_mode not in {"stub", "live"}:
        raise ValueError(f"agent_mode must be 'stub' or 'live', got {resolved_mode!r}")
    runtime = AppRuntime(
        mode="stub" if resolved_mode == "stub" else "live",
        corpus_dir=corpus_dir,
        tracer=runtime_tracer,
    )
    app.state.runtime = runtime
    app.state.settings = settings
    _install_request_context(app, settings)
    app.include_router(api_router)

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


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", ""))


def _body_limit(path: str, settings: Settings) -> int:
    if path == "/ingest":
        return settings.api_max_upload_bytes + _MULTIPART_OVERHEAD_BYTES
    return settings.api_max_question_chars * 4 + _JSON_OVERHEAD_BYTES


def _install_request_context(app: FastAPI, settings: Settings) -> None:
    """Request ID on every response and log line; generic error bodies.

    The client gets a status, a short message and the request ID; the detail
    (exception, traceback, parser message) is in the structured log under that ID.
    """

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await _guarded(request, call_next, settings)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            {"detail": exc.detail, "request_id": _request_id(request)},
            status_code=exc.status_code,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default body echoes the rejected input; location and reason suffice.
        errors = [{"loc": e.get("loc"), "msg": e.get("msg")} for e in exc.errors()]
        return JSONResponse({"detail": errors, "request_id": _request_id(request)}, status_code=422)


async def _guarded(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
    settings: Settings,
) -> Response:
    request_id = _request_id(request)
    if request.method == "POST":
        declared = request.headers.get("content-length")
        if declared is None or not declared.isdigit():
            return JSONResponse(
                {"detail": "Content-Length required.", "request_id": request_id},
                status_code=411,
            )
        if int(declared) > _body_limit(request.url.path, settings):
            return JSONResponse(
                {"detail": "Request body too large.", "request_id": request_id},
                status_code=413,
            )
    try:
        return await call_next(request)
    except Exception as exc:
        # Last handler before the client: any uncaught error becomes a generic 500;
        # the traceback is logged under the request ID the client receives.
        _log.error("unhandled_error", path=request.url.path, exc_info=exc)
        return JSONResponse(
            {"detail": "Internal error.", "request_id": request_id}, status_code=500
        )


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
