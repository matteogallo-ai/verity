"""Verity command-line interface.

The command surface is defined now so it is stable across the build-out; the bodies
are filled stage by stage. Commands that are not yet implemented say so honestly and
exit 0 (they are scaffold gates, not failures) — this keeps CI green without faking
results.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

from verity import __version__
from verity.config import get_settings

if TYPE_CHECKING:
    from verity.retrieval.base import Retriever
    from verity.types import Answer, RetrievalHit

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

    Runs parse → chunk → embed → upsert. Errors per source are logged and skipped so
    a single bad file does not sink the whole batch.
    """
    # Local imports keep `verity version` and `verity db upgrade` cheap when the
    # heavy ingestion deps (docling, torch) are not needed for the invoked command.
    from verity.ingestion import (
        IngestionPipeline,
        LocalEmbedder,
        StructureAwareChunker,
        create_default_parser,
    )
    from verity.ingestion.pipeline import discover_sources
    from verity.retrieval import PgVectorStore

    uris = discover_sources(str(path))
    if not uris:
        console.print(f"[yellow]No supported sources found under {path}[/yellow]")
        raise typer.Exit(code=0)

    settings = get_settings()
    parser = create_default_parser()
    chunker = StructureAwareChunker()
    embedder = LocalEmbedder()
    sink = None if dry_run else PgVectorStore(settings.database_url)

    console.print(
        f"[cyan]Ingesting[/cyan] {len(uris)} source(s) with model "
        f"[bold]{embedder.model}[/bold]"
        + (" [yellow](dry run — no writes)[/yellow]" if dry_run else "")
    )
    pipeline = IngestionPipeline(parser=parser, chunker=chunker, embedder=embedder, sink=sink)
    result = asyncio.run(pipeline.ingest(uris))

    table = Table(title="Ingestion summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("documents", str(result.n_documents))
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
    from verity.db import apply_pending, list_applied

    settings = get_settings()
    applied = apply_pending(settings.database_url)
    total = list_applied(settings.database_url)
    if applied:
        console.print(f"[green]Applied {len(applied)} migration(s):[/green] {', '.join(applied)}")
    else:
        console.print("[green]Database is up to date.[/green]")
    console.print(f"[dim]schema_migrations rows: {len(total)}[/dim]")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to ask the corpus.")],
    corpus: Annotated[
        Path, typer.Option(help="Corpus directory to ingest into an in-memory store.")
    ] = Path("datasets/corpus"),
    in_memory: Annotated[
        bool, typer.Option(help="Use an in-memory store (no Postgres required). Default: True.")
    ] = True,
    stub: Annotated[
        bool,
        typer.Option(
            help=(
                "Use a scripted stub LLM (same one the tests use) — no keys, offline. "
                "Only knows the four demo themes."
            )
        ),
    ] = False,
) -> None:
    """Answer a question with citations. Refuses honestly when the evidence is weak."""
    from verity.agent import create_default_agent
    from verity.agent.scripted_stub import build_stub_router
    from verity.llm import create_default_routing_client

    if not stub:
        _require_llm_api_key()

    retriever, _ = asyncio.run(
        _build_retriever(corpus=corpus, in_memory=in_memory, no_rerank=False)
    )
    client = build_stub_router() if stub else create_default_routing_client()
    agent = create_default_agent(retriever=retriever, client=client)
    answer = asyncio.run(agent.answer(question))
    _print_answer(answer)


MISSING_KEY_MESSAGE = (
    "No LLM API key configured — set VERITY_ANTHROPIC_API_KEY "
    "(or VERITY_OPENAI_API_KEY), or run with --stub."
)


def _require_llm_api_key() -> None:
    """Fail fast with an actionable message when no provider key is configured.

    Refusing to fall back to the stub silently is the honest behaviour: the caller
    asked for a real answer, so we surface the misconfiguration instead of quietly
    serving them a scripted response. The message text is exposed as
    :data:`MISSING_KEY_MESSAGE` so tests can assert on it directly without depending
    on Rich's terminal-width-driven panel wrapping.
    """
    settings = get_settings()
    if settings.anthropic_api_key or settings.openai_api_key:
        return
    raise typer.BadParameter(MISSING_KEY_MESSAGE)


@app.command()
def retrieve(
    query: Annotated[str, typer.Argument(help="Question to retrieve evidence for.")],
    k: Annotated[int, typer.Option(help="Number of hits to display.")] = 5,
    corpus: Annotated[
        Path,
        typer.Option(help="Corpus directory to ingest into an in-memory store."),
    ] = Path("datasets/corpus"),
    in_memory: Annotated[
        bool,
        typer.Option(
            help=(
                "Use an in-memory store populated from --corpus instead of hitting "
                "pgvector. Default: True when no DB is available."
            )
        ),
    ] = True,
    no_rerank: Annotated[
        bool,
        typer.Option(help="Skip the cross-encoder rerank stage (dense + sparse + RRF only)."),
    ] = False,
) -> None:
    """Run one hybrid retrieval and print the top hits."""
    hits = asyncio.run(
        _run_retrieve(query=query, k=k, corpus=corpus, in_memory=in_memory, no_rerank=no_rerank)
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
    in_memory: Annotated[
        bool,
        typer.Option(help="Use an in-memory store (no Postgres required). Default: True."),
    ] = True,
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

    retriever, backend_name = asyncio.run(
        _build_retriever(corpus=corpus, in_memory=in_memory, no_rerank=no_rerank)
    )

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
    *, corpus: Path, in_memory: bool, no_rerank: bool
) -> tuple[Retriever, str]:
    """Instantiate the retriever + populate the store when in-memory.

    Returns ``(retriever, backend_name)`` so the scorecard records where the
    numbers came from.
    """
    from verity.ingestion import (
        IngestionPipeline,
        LocalEmbedder,
        StructureAwareChunker,
        create_default_parser,
    )
    from verity.ingestion.pipeline import discover_sources
    from verity.retrieval import (
        CrossEncoderReranker,
        HybridRetriever,
        InMemoryVectorStore,
        PgVectorStore,
    )

    embedder = LocalEmbedder()
    reranker = None if no_rerank else CrossEncoderReranker()

    if in_memory:
        store = InMemoryVectorStore()
        parser = create_default_parser()
        chunker = StructureAwareChunker()
        pipeline = IngestionPipeline(parser=parser, chunker=chunker, embedder=embedder, sink=store)
        uris = discover_sources(str(corpus))
        result = await pipeline.ingest(uris)
        console.print(
            f"[dim]in-memory store: {result.n_documents} docs, {result.n_chunks} chunks "
            f"(elapsed {result.elapsed_s:.2f}s)[/dim]"
        )
        return HybridRetriever(store=store, embedder=embedder, reranker=reranker), "in-memory"

    pg_store = PgVectorStore()
    return HybridRetriever(store=pg_store, embedder=embedder, reranker=reranker), "pgvector"


async def _run_retrieve(
    *, query: str, k: int, corpus: Path, in_memory: bool, no_rerank: bool
) -> list[RetrievalHit]:
    retriever, _ = await _build_retriever(corpus=corpus, in_memory=in_memory, no_rerank=no_rerank)
    return await retriever.retrieve(query, k=k)


def _print_answer(answer: Answer) -> None:
    """Render an :class:`Answer` for the terminal — refuses visibly when refused."""
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


def _first_client_model(client: object) -> str | None:
    """Return the first registered client's ``.model`` on a RoutingClient, or None."""
    clients = getattr(client, "clients", None)
    if not clients:
        return None
    model = getattr(clients[0], "model", None)
    return model if isinstance(model, str) else None


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind host.")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
) -> None:
    """Run the API + dashboard. [implemented in S5/S6]"""
    console.print(f"{_SCAFFOLD} serve → http://{host}:{port}  (API lands in S5)")


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
    in_memory: Annotated[
        bool, typer.Option(help="Use in-memory store instead of pgvector. Default: True.")
    ] = True,
    agent_stub: Annotated[
        bool,
        typer.Option(
            help=(
                "Use the scripted stub for the AGENT's LLM (decompose/synthesize/score). "
                "Default: True when --judge stub, required when running without keys."
            )
        ),
    ] = True,
    runs_dir: Annotated[Path, typer.Option(help="Where to persist the eval run JSON.")] = Path(
        "datasets/eval/runs"
    ),
    ci: Annotated[
        bool,
        typer.Option(
            help="Legacy alias for --judge stub — kept for backwards compat with S0 CI.",
        ),
    ] = False,
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
) -> None:
    """Run the full eval harness on the labeled dataset and print a Scorecard.

    ``--judge stub`` (the default) produces a DETERMINISTIC scorecard: retrieval,
    refusal, latency, cost are REAL numbers ; faithfulness / citation_accuracy /
    hallucination_rate come from a scripted mechanical judge and the persisted
    ``judge_model`` field is set to ``stub-judge-v1`` so they are never mistaken
    for a real LLM verdict. Switch to ``--judge live`` (with an API key configured)
    to publish real answer-quality numbers.
    """
    from verity.agent.agent import create_default_agent
    from verity.agent.scripted_stub import build_stub_router
    from verity.eval import (
        STUB_JUDGE_MODEL,
        AgentEvaluator,
        FileRunStore,
        LLMFaithfulnessJudge,
        StubFaithfulnessJudge,
        current_git_sha,
        load_full_dataset,
    )
    from verity.llm import create_default_routing_client

    judge_mode = "stub" if ci else judge.lower()
    if judge_mode not in {"stub", "live"}:
        raise typer.BadParameter("--judge must be 'stub' or 'live'")

    if judge_mode == "live":
        _require_llm_api_key()

    examples = load_full_dataset(dataset)
    if not examples:
        console.print("[yellow]Empty dataset — nothing to score.[/yellow]")
        raise typer.Exit(code=0)

    retriever, backend_name = asyncio.run(
        _build_retriever(corpus=corpus, in_memory=in_memory, no_rerank=False)
    )

    # Agent LLM: stub by default (offline), real client if the user turned it off.
    agent_client = build_stub_router() if agent_stub else create_default_routing_client()
    if not agent_stub:
        _require_llm_api_key()
    agent = create_default_agent(retriever=retriever, client=agent_client)
    # Traceability: the refusal chiffres in the Scorecard are meaningless without
    # knowing WHICH agent produced the answers they aggregate over. "stub-agent"
    # is the sentinel picked up by the CLI and README as a prudence flag.
    agent_model = "stub-agent" if agent_stub else _first_client_model(agent_client) or "llm-agent"

    # Judge: mechanical stub by default; real LLM only in --judge live.
    if judge_mode == "stub":
        judge_impl: object = StubFaithfulnessJudge()
        judge_model = STUB_JUDGE_MODEL
    else:
        judge_client = create_default_routing_client()
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
    )
    git_sha = current_git_sha()
    console.print(
        f"[cyan]Running harness[/cyan] · sha=[bold]{git_sha}[/bold] · "
        f"agent=[bold]{agent_model}[/bold] · judge=[bold]{judge_model}[/bold] · "
        f"backend={backend_name} · n={len(examples)}"
    )
    result = asyncio.run(evaluator.run(examples, git_sha=git_sha))

    _print_scorecard(result)

    store = FileRunStore(runs_dir)
    run = result.as_eval_run()
    store.save(run)
    path = store._path_for(run.scorecard.git_sha, run.scorecard.dataset)
    console.print(f"[green]Persisted[/green] {path}")

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
    limit: Annotated[int, typer.Option(help="Number of recent runs to display.")] = 5,
) -> None:
    """Diff the two most recent eval runs and highlight regressions."""
    from verity.eval import FileRunStore, diff_runs

    store = FileRunStore(runs_dir)
    history = store.history(dataset, limit=limit)
    if len(history) < 2:
        console.print(
            f"[yellow]Not enough runs to compare ({len(history)} found); need ≥2.[/yellow]"
        )
        raise typer.Exit(code=0)

    current, previous = history[0], history[1]
    console.print(
        f"[bold cyan]Compare[/bold cyan] {previous.scorecard.git_sha} → {current.scorecard.git_sha} "
        f"(judge_model was {previous.judge_model!r} → now {current.judge_model!r})"
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


def _print_scorecard(result) -> None:  # type: ignore[no-untyped-def]
    """Render a full Scorecard with an explicit stub-vs-real annotation.

    The refusal chiffres are *mechanically* deterministic (a confusion matrix
    with zero LLM calls), but they measure the calibration of whichever agent
    produced the Answers. When the agent is the scripted stub, the numbers
    reflect the stub's calibration, not Verity's real behaviour in production.
    The header + the refusal rows carry that agent-provenance so no reader can
    lift a chiffre without seeing the caveat.
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
    # refusal chiffre without knowing which agent produced it.
    refusal_tag = (
        "REAL calc / agent=stub-agent"
        if agent_is_stub
        else f"REAL calc / agent={result.agent_model}"
    )
    answer.add_row("faithfulness", f"{sc.answer.faithfulness:.3f}", stub_note)
    answer.add_row("citation_accuracy", f"{sc.answer.citation_accuracy:.3f}", stub_note)
    answer.add_row("hallucination_rate", f"{sc.answer.hallucination_rate:.3f}", stub_note)
    answer.add_row("refusal_precision", f"{sc.answer.refusal_precision:.3f}", refusal_tag)
    answer.add_row("refusal_recall", f"{sc.answer.refusal_recall:.3f}", refusal_tag)
    console.print(answer)

    ops = Table(
        title="Operational [green](REAL — measured)[/green]",
        show_header=True,
        header_style="bold cyan",
    )
    ops.add_column("metric")
    ops.add_column("value", justify="right")
    ops.add_row("latency p50 ms", f"{sc.latency.p50_ms:.1f}")
    ops.add_row("latency p95 ms", f"{sc.latency.p95_ms:.1f}")
    ops.add_row("latency p99 ms", f"{sc.latency.p99_ms:.1f}")
    ops.add_row("cost / query USD", f"{sc.cost_per_query_usd:.4f}")
    console.print(ops)

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
            verdict = "[red]FN (hallucinated on out-of-scope)[/red]"
        else:
            verdict = "[green]TN (correctly answered)[/green]"
        refusal.add_row(p.example_id, expected, str(p.observed_refused), verdict)
    console.print(refusal)


if __name__ == "__main__":
    app()
