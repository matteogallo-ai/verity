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
    from verity.types import RetrievalHit

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


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind host.")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
) -> None:
    """Run the API + dashboard. [implemented in S5/S6]"""
    console.print(f"{_SCAFFOLD} serve → http://{host}:{port}  (API lands in S5)")


@eval_app.command("run")
def eval_run(
    dataset: Annotated[Path, typer.Option(help="Path to the labeled dataset.")] = Path(
        "datasets/eval"
    ),
    ci: Annotated[
        bool, typer.Option(help="Deterministic mode: local embeddings + stubbed judge.")
    ] = False,
) -> None:
    """Run the full eval harness and print a scorecard. [implemented in S4]"""
    mode = "CI (deterministic)" if ci else "full (real judge)"
    console.print(f"{_SCAFFOLD} eval run → dataset={dataset} mode={mode}")
    console.print(
        "The scorecard is produced by the harness in S4 — no numbers are fabricated here."
    )


@eval_app.command("compare")
def eval_compare(
    dataset: Annotated[
        str, typer.Option(help="Dataset name whose run history to compare.")
    ] = "default",
    limit: Annotated[int, typer.Option(help="Number of recent runs to compare.")] = 10,
) -> None:
    """Compare recent eval runs to surface regressions. [implemented in S4]"""
    console.print(f"{_SCAFFOLD} eval compare → dataset={dataset} limit={limit}")


if __name__ == "__main__":
    app()
