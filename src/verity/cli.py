"""Verity command-line interface: version, ingest, db upgrade, ask, retrieve, serve, and the
eval commands (retrieval, run, compare). Every command is implemented."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import typer
from rich.console import Console
from rich.table import Table

from verity import __version__
from verity.config import get_settings

if TYPE_CHECKING:
    from verity.eval.harness import HarnessResult
    from verity.retrieval.base import Retriever
    from verity.runtime import StoreKind
    from verity.types import Answer, Chunk, EvalExample, RetrievalHit

app = typer.Typer(
    name="verity",
    help="Evaluated, observable RAG agent platform for enterprise documents.",
    no_args_is_help=True,
    add_completion=False,
)
eval_app = typer.Typer(
    name="eval",
    help="Run and compare the evaluation harness.",
    no_args_is_help=True,
)
db_app = typer.Typer(
    name="db",
    help="Database lifecycle (migrations, health).",
    no_args_is_help=True,
)
app.add_typer(eval_app)
app.add_typer(db_app)

console = Console()

_SCAFFOLD = "[yellow]scaffold[/yellow]"


@app.command()
def version() -> None:
    """Print the Verity version."""
    console.print(f"verity {__version__}")


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="File, directory, or URL to ingest.")],
    dry_run: Annotated[
        bool,
        typer.Option(help="Parse + chunk + embed but do not write to the vector store."),
    ] = False,
) -> None:
    """Ingest documents into the index.

    Runs parse → chunk → embed → store. A document whose text, chunks and embedding
    model match the index is skipped; an edited one has its chunks replaced. Given a
    directory, documents whose file left it are removed. Errors per source are logged
    and skipped so a single bad file does not sink the whole batch; a store schema
    error (wrong embedding dimension, migrations not applied) stops the run with exit
    code 2.
    """
    # Local imports keep `verity version` and `verity db upgrade` cheap when the
    # heavy ingestion deps (docling, torch) are not needed for the invoked command.
    from verity.ingestion import (
        IngestionPipeline,
        LocalEmbedder,
        StructureAwareChunker,
        create_default_parser,
    )
    from verity.ingestion.pipeline import IngestionResult, discover_sources
    from verity.retrieval import PgVectorStore
    from verity.retrieval.base import StoreSchemaError, StoreUnavailableError
    from verity.runtime import prune_missing

    uris = discover_sources(str(path))
    if not uris:
        console.print(f"[yellow]No supported sources found under {path}[/yellow]")
        raise typer.Exit(code=0)

    settings = get_settings()
    parser = create_default_parser()
    chunker = StructureAwareChunker()
    embedder = LocalEmbedder()
    sink = None if dry_run else PgVectorStore(settings.database_url.get_secret_value())

    console.print(
        f"[cyan]Ingesting[/cyan] {len(uris)} source(s) with model "
        f"[bold]{embedder.model}[/bold]"
        + (" [yellow](dry run — no writes)[/yellow]" if dry_run else "")
    )
    pipeline = IngestionPipeline(parser=parser, chunker=chunker, embedder=embedder, sink=sink)

    async def _run() -> tuple[IngestionResult, list[str]]:
        result = await pipeline.ingest(uris)
        removed = (
            await prune_missing(sink, path, uris) if sink is not None and path.is_dir() else []
        )
        return result, removed

    try:
        result, removed = asyncio.run(_run())
    except (StoreSchemaError, StoreUnavailableError) as exc:
        console.print(f"[red]Ingestion stopped:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    table = Table(title="Ingestion summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("documents", str(result.n_documents))
    if sink is not None:
        # A dry run has no store to compare with or remove from.
        table.add_row("unchanged (not embedded)", str(result.unchanged))
        table.add_row("removed (file gone)", str(len(removed)))
    table.add_row("chunks", str(result.n_chunks))
    table.add_row("failures", str(len(result.failures)))
    table.add_row("elapsed", f"{result.elapsed_s:.2f}s")
    table.add_row("embedding model", embedder.model)
    console.print(table)

    for f in result.failures:
        console.print(f"[red]failed[/red] {f.uri} — {f.error}")

    if result.failures and result.n_documents == 0:
        raise typer.Exit(code=1)


@db_app.command("upgrade")
def db_upgrade() -> None:
    """Apply pending database migrations (idempotent)."""
    import psycopg

    from verity.db import apply_pending, list_applied
    from verity.db.redact import redact_conninfo

    database_url = get_settings().database_url.get_secret_value()
    try:
        applied = apply_pending(database_url)
        total = list_applied(database_url)
    except psycopg.Error as exc:
        # Printed redacted and not chained: libpq may quote the password.
        reason = redact_conninfo(str(exc).strip(), database_url)
        console.print(f"[red]Migration failed:[/red] {type(exc).__name__}: {reason}")
        raise typer.Exit(code=2) from None
    if applied:
        console.print(f"[green]Applied {len(applied)} migration(s):[/green] {', '.join(applied)}")
    else:
        console.print("[green]Database is up to date.[/green]")
    console.print(f"[dim]schema_migrations rows: {len(total)}[/dim]")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to ask the corpus.")],
    corpus: Annotated[
        Path, typer.Option(help="Corpus directory to ingest into the store first (idempotent).")
    ] = Path("datasets/corpus"),
    store: Annotated[
        str | None,
        typer.Option(
            help="Vector store, 'pgvector' or 'memory'. Default: VERITY_STORE (pgvector)."
        ),
    ] = None,
    stub: Annotated[
        bool,
        typer.Option(
            help=(
                "Use a scripted stub LLM (same one the tests use) — no keys, offline. "
                "Only knows the four demo themes."
            )
        ),
    ] = False,
    budget_usd: Annotated[
        float | None,
        typer.Option(
            help=(
                "Spending cap in USD for this answer. Before each LLM call the spend so "
                "far plus that call's estimated upper cost must fit, else the answer stops "
                "(exit 4). Default: VERITY_ASK_BUDGET_USD."
            ),
        ),
    ] = None,
) -> None:
    """Answer a question with citations. Refuses when the evidence is weak.

    Exit codes: 3 the answer failed (provider or store); 4 the spending cap stopped it.
    """
    from verity.agent import create_default_agent
    from verity.agent.scripted_stub import build_stub_router
    from verity.llm import create_default_routing_client
    from verity.llm.budget import SpendCappedClient

    cap = budget_usd if budget_usd is not None else get_settings().ask_budget_usd
    if cap <= 0:
        raise typer.BadParameter("--budget-usd must be positive")

    if not stub:
        _require_llm_api_key()

    retriever, _, _ = asyncio.run(_build_retriever(corpus=corpus, store=store, no_rerank=False))
    router = build_stub_router() if stub else create_default_routing_client()
    client = SpendCappedClient(router, cap_usd=cap)
    agent = create_default_agent(retriever=retriever, client=client)
    answer = asyncio.run(agent.answer(question))
    _print_answer(answer)


def resolve_runs_dir(runs_dir: Path, *, judge_mode: str, agent_stub: bool) -> Path:
    """Select the effective persist directory for an eval run.

    **Rule**: any run whose agent OR judge is a real LLM (i.e. spends at
    least one dollar) nests into a ``live/`` subdirectory of ``runs_dir``.
    Stub-only runs stay at the plain ``runs_dir``. This physically
    separates the payable audit trail from freely-overwritable CI churn.

    Truth table — pinned by ``tests/unit/test_stub_live_path_separation``:

    ============  ============  ==============  =========================
    judge_mode    agent_stub    spends money?   effective path
    ============  ============  ==============  =========================
    ``"stub"``    ``True``      no              ``runs_dir/``
    ``"stub"``    ``False``     agent           ``runs_dir/live/``
    ``"live"``    ``True``      judge           ``runs_dir/live/``
    ``"live"``    ``False``     both            ``runs_dir/live/``
    ============  ============  ==============  =========================

    The rule is defined once here so the CLI, the CLI tests, and any other
    caller share the same source of truth — no drift between the runtime
    behaviour and its regression assertions.
    """
    spends_money = (judge_mode == "live") or (not agent_stub)
    return runs_dir / "live" if spends_money else runs_dir


MISSING_KEY_MESSAGE = (
    "No LLM API key configured — set VERITY_ANTHROPIC_API_KEY "
    "(or VERITY_OPENAI_API_KEY), or run with --stub."
)


def _require_llm_api_key() -> None:
    """Fail fast with an actionable message when no provider key is configured.

    Refusing to fall back to the stub silently is deliberate: the caller
    asked for a real answer, so we surface the misconfiguration instead of quietly
    serving them a scripted response. The message text is exposed as
    :data:`MISSING_KEY_MESSAGE` so tests can assert on it directly without depending
    on Rich's terminal-width-driven panel wrapping.
    """
    settings = get_settings()
    if settings.has_llm_api_key:
        return
    raise typer.BadParameter(MISSING_KEY_MESSAGE)


@app.command()
def retrieve(
    query: Annotated[str, typer.Argument(help="Question to retrieve evidence for.")],
    k: Annotated[int, typer.Option(help="Number of hits to display.")] = 5,
    corpus: Annotated[
        Path,
        typer.Option(help="Corpus directory to ingest into the store first (idempotent)."),
    ] = Path("datasets/corpus"),
    store: Annotated[
        str | None,
        typer.Option(
            help="Vector store, 'pgvector' or 'memory'. Default: VERITY_STORE (pgvector)."
        ),
    ] = None,
    no_rerank: Annotated[
        bool,
        typer.Option(help="Skip the cross-encoder rerank stage (dense + sparse + RRF only)."),
    ] = False,
) -> None:
    """Run one hybrid retrieval and print the top hits."""
    hits = asyncio.run(
        _run_retrieve(query=query, k=k, corpus=corpus, store=store, no_rerank=no_rerank)
    )
    if not hits:
        console.print("[yellow]No hits.[/yellow]")
        return

    table = Table(
        title=f"Top {len(hits)} hits · {query!r}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("kind")
    table.add_column("section")
    table.add_column("excerpt", overflow="fold")
    for h in hits:
        excerpt = h.chunk.text.replace("\n", " ").strip()
        if len(excerpt) > 200:
            excerpt = excerpt[:200] + "…"
        table.add_row(
            str(h.rank),
            f"{h.score:.3f}",
            h.kind.value,
            h.chunk.section or "-",
            excerpt,
        )
    console.print(table)


@eval_app.command("retrieval")
def eval_retrieval(
    dataset: Annotated[Path, typer.Option(help="Path to the labeled dataset JSONL.")] = Path(
        "datasets/eval/questions.jsonl"
    ),
    corpus: Annotated[Path, typer.Option(help="Corpus directory to ingest for evaluation.")] = Path(
        "datasets/corpus"
    ),
    k: Annotated[
        int | None,
        typer.Option(help="k used for precision@k / recall@k / nDCG@k (default: rerank_k)."),
    ] = None,
    store: Annotated[
        str | None,
        typer.Option(
            help="Vector store, 'pgvector' or 'memory'. Default: VERITY_STORE (pgvector)."
        ),
    ] = None,
    no_rerank: Annotated[
        bool, typer.Option(help="Score the fused set without the cross-encoder rerank.")
    ] = False,
    runs_dir: Annotated[Path, typer.Option(help="Where to persist the scorecard JSON.")] = Path(
        "datasets/eval/runs"
    ),
    floor_precision: Annotated[
        float | None,
        typer.Option(
            help="Fail with exit-code 2 if macro precision@k falls below this value.",
        ),
    ] = None,
    floor_recall: Annotated[
        float | None,
        typer.Option(help="Fail with exit-code 2 if macro recall@k falls below this value."),
    ] = None,
    floor_ndcg: Annotated[
        float | None,
        typer.Option(help="Fail with exit-code 2 if macro nDCG@k falls below this value."),
    ] = None,
) -> None:
    """Produce the retrieval scorecard: precision@k, recall@k, nDCG@k with a git SHA."""
    from verity.eval import load_dataset, persist_scorecard, run_retrieval_eval

    settings = get_settings()
    effective_k = k if k is not None else settings.rerank_k

    examples = load_dataset(dataset)
    if not examples:
        console.print(
            "[yellow]No answerable+labeled questions in dataset — nothing to score.[/yellow]"
        )
        raise typer.Exit(code=0)

    retriever, backend_name, chunks = asyncio.run(
        _build_retriever(corpus=corpus, store=store, no_rerank=no_rerank)
    )
    examples = [e for e in _gold_from_evidence(examples, chunks) if e.relevant_chunk_ids]

    scorecard = asyncio.run(
        run_retrieval_eval(
            retriever=retriever,
            dataset=examples,
            k=effective_k,
            dataset_name=dataset.stem,
            backend=backend_name,
        )
    )

    console.print(
        f"[bold cyan]Retrieval scorecard[/bold cyan] · sha=[bold]{scorecard.git_sha}[/bold] · "
        f"backend={scorecard.backend} · n={scorecard.n_examples} · k={scorecard.k}"
    )
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("precision@k", f"{scorecard.aggregate.precision_at_k:.3f}")
    table.add_row("recall@k", f"{scorecard.aggregate.recall_at_k:.3f}")
    table.add_row("nDCG@k", f"{scorecard.aggregate.ndcg_at_k:.3f}")
    console.print(table)

    per = Table(
        title="Per-example",
        show_header=True,
        header_style="dim",
    )
    per.add_column("id")
    per.add_column("precision@k", justify="right")
    per.add_column("recall@k", justify="right")
    per.add_column("nDCG@k", justify="right")
    for p in scorecard.per_example:
        per.add_row(
            p.example_id,
            f"{p.metrics.precision_at_k:.3f}",
            f"{p.metrics.recall_at_k:.3f}",
            f"{p.metrics.ndcg_at_k:.3f}",
        )
    console.print(per)

    path = persist_scorecard(scorecard, runs_dir)
    console.print(f"[green]Persisted[/green] {path}")

    failed: list[str] = []
    if floor_precision is not None and scorecard.aggregate.precision_at_k < floor_precision:
        failed.append(
            f"precision@k {scorecard.aggregate.precision_at_k:.3f} < floor {floor_precision:.3f}"
        )
    if floor_recall is not None and scorecard.aggregate.recall_at_k < floor_recall:
        failed.append(f"recall@k {scorecard.aggregate.recall_at_k:.3f} < floor {floor_recall:.3f}")
    if floor_ndcg is not None and scorecard.aggregate.ndcg_at_k < floor_ndcg:
        failed.append(f"nDCG@k {scorecard.aggregate.ndcg_at_k:.3f} < floor {floor_ndcg:.3f}")
    if failed:
        for msg in failed:
            console.print(f"[red]{msg}[/red]")
        raise typer.Exit(code=2)


async def _build_retriever(
    *, corpus: Path, store: str | None, no_rerank: bool
) -> tuple[Retriever, str, list[Chunk]]:
    """Build the retriever on the chosen store, after ingesting ``corpus`` into it.

    Ingesting syncs the store's corpus documents with the directory (unchanged ones
    are not embedded again, edited ones are replaced, removed files leave). Documents
    from other sources, such as API uploads to the same database, stay and are
    retrieved too. Returns ``(retriever, store_name, corpus_chunks)``: the scorecard
    records where the numbers came from, and the eval resolves its gold chunks from
    the chunks of this parse.
    """
    from verity.runtime import build_local_components, build_retriever, ingest_corpus

    kind = _resolve_store(store)
    vector_store, embedder, reranker = build_local_components(store=kind, no_rerank=no_rerank)
    result = await ingest_corpus(corpus_dir=corpus, store=vector_store, embedder=embedder)
    console.print(
        f"[dim]{kind} store: ingested {result.n_documents} docs, {result.n_chunks} chunks "
        f"(elapsed {result.elapsed_s:.2f}s)[/dim]"
    )
    retriever = build_retriever(store=vector_store, embedder=embedder, reranker=reranker)
    return retriever, kind, result.chunks


def _gold_from_evidence(examples: list[EvalExample], chunks: list[Chunk]) -> list[EvalExample]:
    """The examples with their gold chunks resolved from their evidence against this
    parse of the corpus (see ``verity.eval.labels``); exit 2 when a quote is in no
    chunk, since every retrieval figure would then be scored against a wrong gold."""
    from verity.eval.labels import resolve_gold

    resolved, problems = resolve_gold(examples, chunks)
    if problems:
        for problem in problems:
            console.print(f"[red]{problem}[/red]")
        raise typer.Exit(code=2)
    return resolved


def _resolve_store(store: str | None) -> StoreKind:
    resolved = store or get_settings().store
    if resolved not in ("pgvector", "memory"):
        raise typer.BadParameter(f"--store must be 'pgvector' or 'memory', got {resolved!r}")
    return cast("StoreKind", resolved)


async def _run_retrieve(
    *, query: str, k: int, corpus: Path, store: str | None, no_rerank: bool
) -> list[RetrievalHit]:
    retriever, _, _ = await _build_retriever(corpus=corpus, store=store, no_rerank=no_rerank)
    return await retriever.retrieve(query, k=k)


def _print_answer(answer: Answer) -> None:
    """Render an :class:`Answer` for the terminal — refuses visibly when refused."""
    from verity.types import ErrorKind

    if answer.confidence is None:
        kind = answer.error.kind if answer.error else None
        detail = answer.error.detail if answer.error else ""
        console.print(
            f"[bold red]ERROR[/bold red] ({kind.value if kind else 'unknown'}) · "
            f"trace_id=[dim]{answer.trace_id}[/dim]"
        )
        console.print(detail)
        _print_usage(answer)
        raise typer.Exit(code=4 if kind is ErrorKind.BUDGET_EXCEEDED else 3)
    header = (
        "[bold red]REFUSED[/bold red]"
        if answer.confidence.refused
        else "[bold green]ANSWER[/bold green]"
    )
    console.print(
        f"{header} · confidence=[bold]{answer.confidence.score:.2f}[/bold] "
        f"(threshold=[dim]{get_settings().confidence_threshold:.2f}[/dim]) · "
        f"trace_id=[dim]{answer.trace_id}[/dim]"
    )
    console.print()
    console.print(answer.text)
    if answer.confidence.rationale:
        console.print(f"\n[dim]rationale:[/dim] {answer.confidence.rationale}")

    if answer.citations:
        console.print()
        cit = Table(title="Citations", show_header=True, header_style="bold cyan")
        cit.add_column("#", justify="right")
        cit.add_column("chunk_id", overflow="fold")
        cit.add_column("char span")
        cit.add_column("quote", overflow="fold")
        for i, c in enumerate(answer.citations):
            cit.add_row(str(i), str(c.chunk_id), f"{c.char_start}-{c.char_end}", c.quote)
        console.print(cit)

    _print_usage(answer)


def _print_usage(answer: Answer) -> None:
    """Tokens and cost, also of a failed answer: what it spent before it stopped."""
    console.print()
    usage = Table(show_header=False, box=None)
    usage.add_column(style="dim")
    usage.add_column(justify="right")
    usage.add_row("input tokens", str(answer.usage.input_tokens))
    usage.add_row("output tokens", str(answer.usage.output_tokens))
    usage.add_row("llm calls", str(answer.usage.llm_calls))
    usage.add_row("cost usd", f"{answer.usage.cost_usd:.4f}")
    usage.add_row("latency ms", f"{answer.usage.latency_ms:.1f}")
    console.print(usage)


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


@app.command()
def serve(
    host: Annotated[
        str,
        typer.Option(
            help="Bind host. Loopback by default; pass 0.0.0.0 explicitly to expose the "
            "API to the network (see SECURITY.md)."
        ),
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
    runs_dir: Annotated[
        Path, typer.Option(help="Where the dashboard reads persisted eval runs from.")
    ] = Path("datasets/eval/runs"),
    dataset: Annotated[
        str, typer.Option(help="Dataset whose eval history to surface on /dashboard.")
    ] = "questions",
) -> None:
    """Serve /metrics + /dashboard (FastAPI + uvicorn).

    The dashboard is server-rendered HTML with two panels: live session metrics
    (in-memory spans from this process's tracer) and persisted eval-run
    history. The full UI is the Next.js app in ``web/``, which talks to the same
    process through /status, /ask and /ingest.
    """
    import uvicorn

    from verity.api import create_app
    from verity.observability import OTelTracer, configure_logging

    settings = get_settings()
    configure_logging(settings.log_level)
    tracer = OTelTracer(otel_endpoint=settings.otel_endpoint, max_spans=settings.trace_max_spans)
    try:
        app_instance = create_app(tracer=tracer, runs_dir=runs_dir, dataset=dataset)
    except ValueError as exc:
        console.print(f"[red]Cannot start:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    if host not in _LOOPBACK_HOSTS:
        console.print(
            f"[yellow]Bound to {host}: the API is reachable from other machines. "
            "See SECURITY.md before exposing it.[/yellow]"
        )
    console.print(f"[cyan]Serving[/cyan] http://{host}:{port}  (dashboard: /dashboard)")
    uvicorn.run(app_instance, host=host, port=port, log_level=settings.log_level.lower())


@eval_app.command("run")
def eval_run(
    dataset: Annotated[Path, typer.Option(help="Path to the labeled JSONL dataset.")] = Path(
        "datasets/eval/questions.jsonl"
    ),
    corpus: Annotated[Path, typer.Option(help="Corpus directory to ingest.")] = Path(
        "datasets/corpus"
    ),
    judge: Annotated[
        str, typer.Option(help="Faithfulness judge: 'stub' (offline, deterministic) or 'live'.")
    ] = "stub",
    store: Annotated[
        str | None,
        typer.Option(
            help="Vector store, 'pgvector' or 'memory'. Default: VERITY_STORE (pgvector)."
        ),
    ] = None,
    agent_stub: Annotated[
        bool,
        typer.Option(
            help=(
                "Use the scripted stub for the AGENT's LLM (decompose/synthesize/score). "
                "Default: True when --judge stub, required when running without keys."
            )
        ),
    ] = True,
    runs_dir: Annotated[
        Path,
        typer.Option(
            help=(
                "Where to persist the eval run JSON. Stub-provenance runs write "
                "directly here (freely overwritable by CI); live-provenance runs "
                "auto-nest into a ``live/`` subdirectory to keep audit artefacts "
                "physically separate from stub churn."
            ),
        ),
    ] = Path("datasets/eval/runs"),
    ci: Annotated[
        bool,
        typer.Option(
            help="Alias for --judge stub (offline, deterministic). Used by CI and the quickstart.",
        ),
    ] = False,
    warmup: Annotated[
        bool,
        typer.Option(
            help=(
                "Prime lazy components with one throwaway request BEFORE timing so "
                "p50/p95/p99 reflect steady-state serving latency. cold_start_ms of "
                "that priming request is reported separately in the Scorecard."
            )
        ),
    ] = True,
    floor_refusal_recall: Annotated[
        float | None,
        typer.Option(help="Fail if refusal_recall drops below this value."),
    ] = None,
    floor_ndcg: Annotated[
        float | None,
        typer.Option(help="Fail if retrieval nDCG@k drops below this value."),
    ] = None,
    floor_recall: Annotated[
        float | None,
        typer.Option(help="Fail if retrieval recall@k drops below this value."),
    ] = None,
    scorecard_json: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Path to write the enriched publishable scorecard JSON — full "
                "provenance triple + total_cost_usd + per-example refusal + "
                "latency percentiles. Used by scripts/generate_headline_results.py "
                "to render the README's headline table. Independent of --runs-dir."
            )
        ),
    ] = None,
    budget_usd: Annotated[
        float | None,
        typer.Option(
            help=(
                "Hard budget cap in USD. Aborts the run when cumulative "
                "cost_usd across answers exceeds this cap. Intended for the "
                "one-shot live-judge run (spec: ~$8 ceiling, ~$5 expected)."
            )
        ),
    ] = None,
    report_only_floors: Annotated[
        bool,
        typer.Option(
            help=(
                "Print 'below floor' findings but exit 0 anyway. Intended for "
                "the one-shot LIVE run: a floor breach on a run of record must "
                "NOT be presented as 'failure' that invites a re-run (=re-spend). "
                "Stub / CI runs keep hard floors (do NOT set this in CI)."
            )
        ),
    ] = False,
) -> None:
    """Run the full eval harness on the labeled dataset and print a Scorecard.

    ``--judge stub`` (the default) produces a DETERMINISTIC scorecard: retrieval,
    refusal, latency, cost are REAL numbers ; faithfulness / citation_accuracy /
    hallucination_rate come from a scripted mechanical judge and the persisted
    ``judge_model`` field is set to ``stub-judge-v1`` so they are never mistaken
    for a real LLM verdict. Switch to ``--judge live`` (with an API key configured)
    to publish real answer-quality numbers.

    Exit codes: 2 a floor is breached (0 with ``--report-only-floors``); 4 the
    budget is exceeded; 5 a stub run would overwrite a live artefact; 6 at least
    one example errored (provider or store outage), so the run is incomplete.
    """
    from verity.agent.agent import create_default_agent
    from verity.agent.scripted_stub import build_stub_router
    from verity.eval import (
        STUB_JUDGE_MODEL,
        AgentEvaluator,
        BudgetExceededError,
        FileRunStore,
        LLMFaithfulnessJudge,
        StubFaithfulnessJudge,
        current_git_sha,
        load_full_dataset,
    )
    from verity.llm import create_default_routing_client
    from verity.llm.routing import create_judge_client, primary_client_identity

    judge_mode = "stub" if ci else judge.lower()
    if judge_mode not in {"stub", "live"}:
        raise typer.BadParameter("--judge must be 'stub' or 'live'")

    if judge_mode == "live":
        _require_llm_api_key()

    examples = load_full_dataset(dataset)
    if not examples:
        console.print("[yellow]Empty dataset — nothing to score.[/yellow]")
        raise typer.Exit(code=0)

    retriever, backend_name, chunks = asyncio.run(
        _build_retriever(corpus=corpus, store=store, no_rerank=False)
    )
    examples = _gold_from_evidence(examples, chunks)

    # Agent LLM: stub by default (offline), real client if the user turned it off.
    agent_client = build_stub_router() if agent_stub else create_default_routing_client()
    if not agent_stub:
        _require_llm_api_key()
    agent = create_default_agent(retriever=retriever, client=agent_client)
    # Traceability: the refusal numbers in the Scorecard are meaningless without
    # knowing WHICH agent produced the answers they aggregate over. "stub-agent"
    # is the sentinel picked up by the CLI and README as a prudence flag.
    agent_model = (
        "stub-agent" if agent_stub else primary_client_identity(agent_client)[1] or "llm-agent"
    )

    # Judge: mechanical stub by default; real LLM only in --judge live.
    if judge_mode == "stub":
        judge_impl: object = StubFaithfulnessJudge()
        judge_model = STUB_JUDGE_MODEL
    else:
        try:
            judge_client = create_judge_client(agent_client=agent_client)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=2) from exc
        judge_impl = LLMFaithfulnessJudge(judge_client)
        judge_model = getattr(judge_impl, "model", "llm-judge")

    settings = get_settings()
    evaluator = AgentEvaluator(
        agent=agent,
        judge=judge_impl,  # type: ignore[arg-type]
        embedding_model=settings.embedding_model,
        judge_model=judge_model,
        agent_model=agent_model,
        dataset_name=dataset.stem,
        warmup=warmup,
    )
    git_sha = current_git_sha()
    console.print(
        f"[cyan]Running harness[/cyan] · sha=[bold]{git_sha}[/bold] · "
        f"agent=[bold]{agent_model}[/bold] · judge=[bold]{judge_model}[/bold] · "
        f"backend={backend_name} · n={len(examples)}"
    )
    from verity.eval.run_store import StubOverwritesLiveError

    effective_runs_dir = resolve_runs_dir(runs_dir, judge_mode=judge_mode, agent_stub=agent_stub)
    run_store = FileRunStore(effective_runs_dir)
    dataset_sha256 = hashlib.sha256(dataset.read_bytes()).hexdigest()
    # create_default_agent above builds its scorer from these settings.
    threshold = settings.confidence_threshold
    try:
        result = asyncio.run(evaluator.run(examples, git_sha=git_sha, budget_usd_cap=budget_usd))
    except BudgetExceededError as exc:
        console.print(f"[red]{exc}[/red]")
        if exc.result is not None:
            # The examples paid before the abort are written, not thrown away.
            partial_path = run_store.save(
                exc.result.as_eval_run(
                    dataset_sha256=dataset_sha256, confidence_threshold=threshold
                )
            )
            console.print(f"[yellow]Persisted incomplete run[/yellow] {partial_path}")
            _print_total_cost(exc.result)
        raise typer.Exit(code=4) from exc

    _print_scorecard(result)

    run = result.as_eval_run(dataset_sha256=dataset_sha256, confidence_threshold=threshold)
    try:
        path = run_store.save(run)
    except StubOverwritesLiveError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=5) from exc
    console.print(f"[green]Persisted[/green] {path}")

    # JSON write is deliberately UNCONDITIONAL and happens BEFORE any floor
    # evaluation: a floor breached on the one-shot measurement must never
    # prevent the artefact from being written to disk. A paid run that fails a
    # floor and loses its scorecard invites a re-spend, which contradicts the
    # "one run of record" invariant.
    # An incomplete run never becomes the published headline.
    if scorecard_json is not None and result.scorecard.incomplete_reason is None:
        from verity.eval import write_headline_scorecard
        from verity.eval.headline import StubOverwritesLiveScorecardError

        try:
            headline_path = write_headline_scorecard(
                run, scorecard_json, source_artefact=_display_path(path)
            )
        except StubOverwritesLiveScorecardError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=5) from exc
        console.print(f"[green]Wrote headline scorecard[/green] {headline_path}")

    _print_total_cost(result)

    if result.scorecard.errored_example_ids:
        # Metrics cover fewer questions than the scorecard's n: the run is
        # incomplete and must not pass as a run of record, even with report-only
        # floors. Its artefact above is kept for its paid examples.
        console.print(
            f"[red]{len(result.scorecard.errored_example_ids)} example(s) errored: "
            "the run is incomplete; do not publish it.[/red]"
        )
        raise typer.Exit(code=6)

    failed: list[str] = []
    if (
        floor_refusal_recall is not None
        and result.scorecard.answer.refusal_recall < floor_refusal_recall
    ):
        failed.append(
            f"refusal_recall {result.scorecard.answer.refusal_recall:.3f} "
            f"< floor {floor_refusal_recall:.3f}"
        )
    if floor_ndcg is not None and result.scorecard.retrieval.ndcg_at_k < floor_ndcg:
        failed.append(f"nDCG@k {result.scorecard.retrieval.ndcg_at_k:.3f} < floor {floor_ndcg:.3f}")
    if floor_recall is not None and result.scorecard.retrieval.recall_at_k < floor_recall:
        failed.append(
            f"recall@k {result.scorecard.retrieval.recall_at_k:.3f} < floor {floor_recall:.3f}"
        )
    if failed:
        if report_only_floors:
            # Report-only: log the finding (so the operator sees the
            # breach in the run of record) but exit 0. Semantics for the LIVE
            # one-shot: the JSON is committed above, the below-floor observation
            # is recorded in the console log, no re-run is invited.
            for msg in failed:
                console.print(f"[yellow]below floor (report-only): {msg}[/yellow]")
        else:
            for msg in failed:
                console.print(f"[red]{msg}[/red]")
            raise typer.Exit(code=2)


@eval_app.command("compare")
def eval_compare(
    dataset: Annotated[
        str, typer.Option(help="Dataset name whose run history to compare.")
    ] = "questions",
    runs_dir: Annotated[Path, typer.Option(help="Where run JSONs live.")] = Path(
        "datasets/eval/runs"
    ),
    limit: Annotated[int, typer.Option(help="Number of recent runs to inspect.")] = 20,
) -> None:
    """Diff the latest eval run against the most recent COMPATIBLE prior run.

    Two runs are compatible iff they share the same dataset, the same
    ``agent_model``, and the same ``judge_model``. Comparing a stub-agent run
    against a live-agent run (or a stub-judge run against a real-judge run) would
    surface deltas that reflect the *provenance change*, not a real regression —
    so those pairs are skipped and reported as "nothing comparable".

    Exit codes:

    - **0** — no regression detected. Includes the "fewer than two compatible
      runs" case ("nothing to compare"), so a fresh checkout with a single
      persisted scorecard exits cleanly.
    - **3** — at least one metric regressed beyond the numerical epsilon (worse
      than the metric's "better" direction).
    """
    from verity.eval import FileRunStore, are_compatible, diff_runs

    store = FileRunStore(runs_dir)
    history = store.history(dataset, limit=limit)
    if len(history) < 2:
        console.print(
            f"[yellow]Nothing to compare ({len(history)} run(s) for dataset "
            f"{dataset!r}); need ≥2.[/yellow]"
        )
        raise typer.Exit(code=0)

    current = history[0]
    previous = next(
        (run for run in history[1:] if are_compatible(current, run)),
        None,
    )
    if previous is None:
        # There are older runs, but none share the current run's (agent, judge).
        console.print(
            f"[yellow]Nothing comparable: current run "
            f"(agent={current.agent_model}, judge={current.judge_model}) "
            f"has no prior run with the same provenance in dataset {dataset!r}.[/yellow]"
        )
        raise typer.Exit(code=0)

    console.print(
        f"[bold cyan]Compare[/bold cyan] {previous.scorecard.git_sha} → {current.scorecard.git_sha} "
        f"(agent={current.agent_model}, judge={current.judge_model})"
    )

    deltas = diff_runs(previous, current)
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("metric")
    table.add_column("prev", justify="right")
    table.add_column("curr", justify="right")
    table.add_column("Δ", justify="right")
    for d in deltas:
        style = "red" if d.is_regression else "green" if abs(d.delta) > 1e-6 else "dim"
        marker = "↓" if d.is_regression else ("↑" if abs(d.delta) > 1e-6 else "=")
        table.add_row(
            d.metric,
            f"{d.previous:.4f}",
            f"{d.current:.4f}",
            f"[{style}]{marker} {d.delta:+.4f}[/{style}]",
        )
    console.print(table)

    regressions = [d.metric for d in deltas if d.is_regression]
    if regressions:
        console.print(f"[red]Regressions on:[/red] {', '.join(regressions)}")
        raise typer.Exit(code=3)


def _display_path(path: Path) -> str:
    """``path`` relative to the working directory when inside it, else absolute."""
    resolved = path.resolve()
    cwd = Path.cwd()
    return (resolved.relative_to(cwd) if resolved.is_relative_to(cwd) else resolved).as_posix()


def _print_total_cost(result: HarnessResult) -> None:
    """Everything the run paid for, errored examples included, for the operator to
    check before the artefact is committed."""
    agent = sum(a.usage.cost_usd for a in result.answers) + sum(
        e.usage.cost_usd for e in result.errored
    )
    console.print(
        f"[dim]total observed cost:[/dim] "
        f"[bold]${agent + result.judge_cost_usd:.4f}[/bold] "
        f"(agent ${agent:.4f}, judge ${result.judge_cost_usd:.4f})"
    )


def _print_scorecard(result) -> None:  # type: ignore[no-untyped-def]
    """Render a full Scorecard with an explicit stub-vs-real annotation.

    The refusal numbers are mechanically deterministic (a confusion matrix with
    zero LLM calls), but they measure the refusal behaviour of whichever agent
    produced the Answers. When the agent is the scripted stub, the numbers
    describe the stub, not Verity's behaviour with a real model. The header and
    the refusal rows carry that agent provenance so no reader can quote a number
    without seeing the caveat.
    """
    from verity.eval import STUB_JUDGE_MODEL

    sc = result.scorecard
    judge_is_stub = result.judge_model == STUB_JUDGE_MODEL
    agent_is_stub = result.agent_model == "stub-agent"

    header = Table(show_header=False, box=None)
    header.add_column(style="dim")
    header.add_column()
    header.add_row("git_sha", sc.git_sha)
    header.add_row("dataset", sc.dataset)
    header.add_row("n_examples", str(sc.n_examples))
    header.add_row("embedding_model", result.embedding_model)
    header.add_row(
        "agent_model",
        f"{result.agent_model} [yellow](scripted stub — NOT a real LLM agent)[/yellow]"
        if agent_is_stub
        else result.agent_model,
    )
    header.add_row(
        "judge_model",
        f"{result.judge_model} [yellow](mechanical stub — NOT a real LLM judge)[/yellow]"
        if judge_is_stub
        else result.judge_model,
    )
    console.print(header)

    retrieval = Table(
        title="Retrieval [green](REAL — deterministic, no LLM)[/green]",
        show_header=True,
        header_style="bold cyan",
    )
    retrieval.add_column("metric")
    retrieval.add_column("value", justify="right")
    retrieval.add_row("precision@k", f"{sc.retrieval.precision_at_k:.3f}")
    retrieval.add_row("recall@k", f"{sc.retrieval.recall_at_k:.3f}")
    retrieval.add_row("nDCG@k", f"{sc.retrieval.ndcg_at_k:.3f}")
    console.print(retrieval)

    answer_title = (
        "Answer quality [yellow](faithfulness/citation/hallucination = STUB — "
        "not a real LLM judge)[/yellow]"
        if judge_is_stub
        else "Answer quality [green](REAL — LLM judge)[/green]"
    )
    answer = Table(title=answer_title, show_header=True, header_style="bold cyan")
    answer.add_column("metric")
    answer.add_column("value", justify="right")
    answer.add_column("kind", justify="right")
    stub_note = "STUB" if judge_is_stub else "REAL"
    # Refusal maths is deterministic, but the *observed* refuses come from the agent
    # → tag the agent explicitly on each refusal row so no reader can quote a
    # refusal number without knowing which agent produced it.
    refusal_tag = (
        "REAL calc / agent=stub-agent"
        if agent_is_stub
        else f"REAL calc / agent={result.agent_model}"
    )
    # One definition with the published headline: verity.eval.answer_quality.
    from verity.eval.answer_quality import Ratio, answer_quality

    def _value(ratio: Ratio) -> str:
        return "n/a" if ratio.value is None else f"{ratio.value:.3f}"

    q = answer_quality(result.per_audit)
    for name, macro, micro in (
        ("faithfulness", q.faithfulness_macro, q.faithfulness_micro),
        ("citation_accuracy", q.citation_accuracy_macro, q.citation_accuracy_micro),
    ):
        answer.add_row(f"{name} (answered, n={macro.denominator})", _value(macro), stub_note)
        answer.add_row(f"{name} micro (claims, n={micro.denominator})", _value(micro), stub_note)
    answer.add_row(
        f"hallucination_rate (answerable_and_answered, n={q.hallucination_rate.denominator})",
        _value(q.hallucination_rate),
        stub_note,
    )
    if q.n_judged_without_claims:
        answer.add_row("judged without a usable claim", str(q.n_judged_without_claims), stub_note)
    answer.add_row("refusal_precision", f"{sc.answer.refusal_precision:.3f}", refusal_tag)
    answer.add_row("refusal_recall", f"{sc.answer.refusal_recall:.3f}", refusal_tag)
    # Companion to hallucination_rate — the count of out-of-scope examples
    # the agent DID NOT refuse. Computed from per_refusal for parity with the
    # JSON export.
    oos_answered = sum(
        1 for p in result.per_refusal if not p.expected_answerable and not p.observed_refused
    )
    oos_total = sum(1 for p in result.per_refusal if not p.expected_answerable)
    answer.add_row("out_of_scope_answered", f"{oos_answered}/{oos_total}", refusal_tag)
    console.print(answer)

    ops_title_suffix = (
        " [dim](percentiles are steady-state; cold_start reported separately)[/dim]"
        if sc.cold_start_ms is not None
        else " [yellow](no warmup — percentiles include cold-start outlier)[/yellow]"
    )
    ops = Table(
        title=f"Operational [green](REAL — measured)[/green]{ops_title_suffix}",
        show_header=True,
        header_style="bold cyan",
    )
    ops.add_column("metric")
    ops.add_column("value", justify="right")
    ops.add_row("latency p50 ms", f"{sc.latency.p50_ms:.1f}")
    ops.add_row("latency p95 ms", f"{sc.latency.p95_ms:.1f}")
    ops.add_row("latency p99 ms", f"{sc.latency.p99_ms:.1f}")
    if sc.cold_start_ms is not None:
        ops.add_row("cold start ms", f"{sc.cold_start_ms:.1f}")
    ops.add_row("cost / query USD", f"{sc.cost_per_query_usd:.4f}")
    console.print(ops)
    if sc.errored_example_ids:
        console.print(
            f"[bold red]{len(sc.errored_example_ids)} example(s) errored[/bold red] "
            "(provider or store failure; excluded from every metric above): "
            + ", ".join(sc.errored_example_ids)
        )

    # Refusal per-example (headline feature — always shown explicitly).
    refusal_title_tag = (
        "[yellow](stubbed agent — refusal calibration reflects the stub, not real Verity)[/yellow]"
        if agent_is_stub
        else "[green](real agent)[/green]"
    )
    refusal = Table(
        title=f"Refusal outcomes · agent={result.agent_model} {refusal_title_tag}",
        show_header=True,
        header_style="bold cyan",
    )
    refusal.add_column("id")
    refusal.add_column("expected")
    refusal.add_column("observed refused")
    refusal.add_column("verdict")
    for p in result.per_refusal:
        expected = "answerable" if p.expected_answerable else "out-of-scope"
        if p.observed_refused and not p.expected_answerable:
            verdict = "[green]TP (correct refusal)[/green]"
        elif p.observed_refused and p.expected_answerable:
            verdict = "[yellow]FP (uselessly timid)[/yellow]"
        elif not p.observed_refused and not p.expected_answerable:
            # "FN" here is the confusion-matrix cell (unrefused + out-of-scope),
            # NOT a verdict on the answer text (the judge has its own per-claim
            # verdicts under ``per_audit``). Renamed away from "hallucinated"
            # to stop conflating the two — see PerExampleAudit.
            verdict = "[red]FN (didn't refuse out-of-scope)[/red]"
        else:
            verdict = "[green]TN (correctly answered)[/green]"
        refusal.add_row(p.example_id, expected, str(p.observed_refused), verdict)
    console.print(refusal)


if __name__ == "__main__":
    app()
