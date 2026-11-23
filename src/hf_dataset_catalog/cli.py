from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Annotated, Any

import typer

from hf_dataset_catalog.config import Settings, load_settings
from hf_dataset_catalog.exporter import APPROVED_COLUMNS, CatalogExporter
from hf_dataset_catalog.hf_client import HuggingFaceSource
from hf_dataset_catalog.http import RetryingJsonClient
from hf_dataset_catalog.papers import (
    OpenAlexProvider,
    PaperCandidate,
    PaperClue,
    PaperResolver,
    SemanticScholarProvider,
)
from hf_dataset_catalog.pipeline import CatalogPipeline, DatasetSource, Resolver
from hf_dataset_catalog.storage import CatalogStore

app = typer.Typer(
    name="hf-catalog",
    help="Catalog public Hugging Face dataset metadata without downloading files.",
    no_args_is_help=True,
)


class LimitedSource:
    def __init__(self, source: DatasetSource, limit: int) -> None:
        self.source = source
        self.limit = limit

    def iter_datasets(
        self,
        *,
        fetched_at: Any = None,
        on_rejected: Callable[[str, str], None] | None = None,
    ) -> Iterable[dict[str, Any]]:
        return islice(
            self.source.iter_datasets(
                fetched_at=fetched_at,
                on_rejected=on_rejected,
            ),
            self.limit,
        )


class SkippedPaperResolver:
    async def resolve(self, clue: PaperClue) -> PaperCandidate:
        return PaperCandidate(
            title=clue.title,
            doi=clue.doi,
            arxiv_id=clue.arxiv_id,
            match_method="skipped",
        )


@dataclass
class PipelineRuntime:
    pipeline: CatalogPipeline
    clients: list[RetryingJsonClient]

    def close(self) -> None:
        async def close_all() -> None:
            for client in self.clients:
                await client.aclose()

        asyncio.run(close_all())


def build_runtime(
    settings: Settings,
    *,
    max_datasets: int | None = None,
    skip_papers: bool = False,
) -> PipelineRuntime:
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    source: DatasetSource = HuggingFaceSource(token=token)
    if max_datasets is not None:
        source = LimitedSource(source, max_datasets)

    clients: list[RetryingJsonClient] = []
    resolver: Resolver
    if skip_papers:
        resolver = SkippedPaperResolver()
    else:
        semantic_key = (
            settings.semantic_scholar_api_key.get_secret_value()
            if settings.semantic_scholar_api_key
            else None
        )
        semantic_client = RetryingJsonClient(
            source="semantic_scholar",
            max_retries=settings.max_retries,
            request_interval_seconds=settings.request_interval_seconds,
            headers={"x-api-key": semantic_key} if semantic_key else None,
        )
        openalex_client = RetryingJsonClient(
            source="openalex",
            max_retries=settings.max_retries,
            request_interval_seconds=settings.request_interval_seconds,
        )
        clients.extend([semantic_client, openalex_client])
        resolver = PaperResolver(
            semantic_scholar=SemanticScholarProvider(semantic_client),
            openalex=OpenAlexProvider(
                openalex_client,
                api_key=(
                    settings.openalex_api_key.get_secret_value()
                    if settings.openalex_api_key
                    else None
                ),
            ),
            confirm_threshold=settings.confirm_threshold,
            review_threshold=settings.review_threshold,
        )
    return PipelineRuntime(
        pipeline=CatalogPipeline(settings, source=source, resolver=resolver),
        clients=clients,
    )


@app.command()
def collect(
    config: Annotated[
        Path | None,
        typer.Option("--config", help="YAML configuration path."),
    ] = None,
    max_datasets: Annotated[
        int | None,
        typer.Option(
            "--max-datasets",
            min=1,
            help="Smoke-test limit only; omitted for full enumeration.",
        ),
    ] = None,
    skip_papers: Annotated[
        bool,
        typer.Option("--skip-papers", help="Skip external paper provider requests."),
    ] = False,
) -> None:
    """Collect public metadata; dataset and repository files are never downloaded."""
    settings = load_settings(config)
    runtime = build_runtime(
        settings,
        max_datasets=max_datasets,
        skip_papers=skip_papers,
    )
    try:
        run_id = runtime.pipeline.collect()
    except Exception as error:
        typer.echo(f"Collection failed: {error}", err=True)
        raise typer.Exit(1) from error
    finally:
        runtime.close()
    typer.echo(f"run_id={run_id}")
    typer.echo(f"output={settings.output_dir / run_id}")


@app.command()
def resume(
    run_id: Annotated[str, typer.Argument(help="Collection run ID.")],
    config: Annotated[Path | None, typer.Option("--config")] = None,
    skip_papers: Annotated[bool, typer.Option("--skip-papers")] = False,
) -> None:
    """Resume a run from its durable stage checkpoints."""
    settings = load_settings(config)
    runtime = build_runtime(settings, skip_papers=skip_papers)
    try:
        runtime.pipeline.resume(run_id)
    except Exception as error:
        typer.echo(f"Resume failed: {error}", err=True)
        raise typer.Exit(1) from error
    finally:
        runtime.close()
    typer.echo(f"run_id={run_id}")
    typer.echo(f"output={settings.output_dir / run_id}")


@app.command("export")
def export_command(
    run_id: Annotated[str, typer.Argument(help="Collection run ID.")],
    config: Annotated[Path | None, typer.Option("--config")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    state_db: Annotated[Path | None, typer.Option("--state-db")] = None,
) -> None:
    """Recreate validated deliverables from DuckDB state."""
    overrides: dict[str, Any] = {}
    if output_dir:
        overrides["output_dir"] = output_dir
    if state_db:
        overrides["state_db"] = state_db
    settings = load_settings(config, overrides)
    store = CatalogStore(settings.state_db)
    store.initialize()
    _require_run(store, run_id)
    try:
        output = CatalogExporter(store, settings.output_dir).export(run_id)
    except Exception as error:
        typer.echo(f"Export failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"output={output}")


@app.command()
def validate(
    run_id: Annotated[str, typer.Argument(help="Collection run ID.")],
    config: Annotated[Path | None, typer.Option("--config")] = None,
    state_db: Annotated[Path | None, typer.Option("--state-db")] = None,
) -> None:
    """Validate stored schemas, keys, rankings, selection, and URLs."""
    settings = load_settings(
        config,
        {"state_db": state_db} if state_db else None,
    )
    store = CatalogStore(settings.state_db)
    store.initialize()
    _require_run(store, run_id)
    tables = {
        name: store.table_arrow(name, run_id).select(columns)
        for name, columns in APPROVED_COLUMNS.items()
    }
    try:
        CatalogExporter(store, settings.output_dir).validate(run_id, tables)
    except Exception as error:
        typer.echo(f"Validation failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo("valid")


@app.command()
def status(
    run_id: Annotated[str | None, typer.Argument(help="Collection run ID.")] = None,
    latest: Annotated[bool, typer.Option("--latest", help="Show the latest run.")] = False,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: table or run-id."),
    ] = "table",
    state_db: Annotated[
        Path,
        typer.Option("--state-db", help="DuckDB state file."),
    ] = Path("data/state/catalog.duckdb"),
) -> None:
    """Show run status and checkpoint progress."""
    if output_format not in {"table", "run-id"}:
        typer.echo("--format must be table or run-id", err=True)
        raise typer.Exit(2)
    store = CatalogStore(state_db)
    store.initialize()
    if run_id and latest:
        typer.echo("Use either RUN_ID or --latest, not both.", err=True)
        raise typer.Exit(2)
    rows = store.query_dicts(
        """
        SELECT * FROM collection_runs
        WHERE (? IS NULL OR collection_run_id = ?)
        ORDER BY started_at DESC
        LIMIT 1
        """,
        [run_id, run_id],
    )
    if not rows:
        typer.echo(f"Unknown run: {run_id or 'latest'}", err=True)
        raise typer.Exit(1)
    row = rows[0]
    selected_run = str(row["collection_run_id"])
    if output_format == "run-id":
        typer.echo(selected_run)
        return
    checkpoint_count = store.query_dicts(
        """
        SELECT count(*) AS count FROM collection_checkpoints
        WHERE collection_run_id = ? AND checkpoint_key = 'complete'
        """,
        [selected_run],
    )[0]["count"]
    failure_count = store.query_dicts(
        "SELECT count(*) AS count FROM collection_failures WHERE collection_run_id = ?",
        [selected_run],
    )[0]["count"]
    typer.echo(f"run_id: {selected_run}")
    typer.echo(f"status: {row['status']}")
    typer.echo(f"current_stage: {row['current_stage'] or '-'}")
    typer.echo(f"completed_stages: {checkpoint_count}")
    typer.echo(f"failures: {failure_count}")


def _require_run(store: CatalogStore, run_id: str) -> None:
    rows = store.query_dicts(
        "SELECT collection_run_id FROM collection_runs WHERE collection_run_id = ?",
        [run_id],
    )
    if not rows:
        typer.echo(f"Unknown run: {run_id}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
