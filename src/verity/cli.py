"""Verity command-line interface.

The command surface is defined now so it is stable across the build-out; the bodies
are filled stage by stage. Commands that are not yet implemented say so honestly and
exit 0 (they are scaffold gates, not failures) — this keeps CI green without faking
results.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from verity import __version__
from verity.config import get_settings

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
