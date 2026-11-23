from __future__ import annotations

import csv
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow.parquet as pq

from hf_dataset_catalog.config import Settings
from hf_dataset_catalog.exporter import APPROVED_COLUMNS, CatalogExporter
from hf_dataset_catalog.hf_client import write_raw_batch
from hf_dataset_catalog.models import (
    DatasetRecord,
    FileRecord,
    MatchStatus,
    MetricRecord,
    PaperRecord,
    RankingRecord,
)
from hf_dataset_catalog.storage import CatalogStore

NOW = datetime(2026, 11, 23, tzinfo=UTC)


def master(dataset_id: str) -> DatasetRecord:
    return DatasetRecord(
        dataset_id=dataset_id,
        dataset_name=dataset_id,
        dataset_url=f"https://huggingface.co/datasets/{dataset_id}",
        files_url=f"https://huggingface.co/datasets/{dataset_id}/tree/main",
        viewer_url=f"https://huggingface.co/datasets/{dataset_id}/viewer",
        ranking_date=NOW.date(),
        is_recent=True,
        fetched_at=NOW,
    )


def test_export_preserves_approved_schemas_and_null_citations(tmp_path: Path) -> None:
    settings = Settings(output_dir=tmp_path / "output", state_db=tmp_path / "state.duckdb")
    store = CatalogStore(settings.state_db)
    store.initialize()
    run_id = store.create_run(settings, now=NOW)
    store.upsert_models(
        run_id,
        "hf_datasets_master",
        [master("a"), master("b")],
    )
    store.upsert_models(
        run_id,
        "hf_dataset_papers",
        [
            PaperRecord(
                dataset_id="a",
                paper_id="paper-a",
                citation_count=None,
                match_method="title",
                match_confidence=0.9,
                match_status=MatchStatus.REVIEW,
                fetched_at=NOW,
            )
        ],
    )
    store.upsert_models(
        run_id,
        "hf_dataset_files",
        [
            FileRecord(
                dataset_id="a",
                file_path="data.parquet",
                file_name="data.parquet",
                extension="parquet",
                file_page_url="https://huggingface.co/datasets/a/blob/main/data.parquet",
                resolve_url="https://huggingface.co/datasets/a/resolve/main/data.parquet",
                fetched_at=NOW,
            )
        ],
    )
    store.upsert_metrics(
        run_id,
        [
            MetricRecord(
                dataset_id="a",
                snapshot_date=date(2026, 11, 23),
                downloads=1,
                fetched_at=NOW,
            )
        ],
    )
    store.upsert_rankings(
        run_id,
        [
            RankingRecord(
                dataset_id="a",
                metric="downloads",
                metric_value=1,
                rank=1,
                ranking_date=NOW.date(),
            )
        ],
    )
    raw_path, checksum = write_raw_batch(
        tmp_path / "raw",
        run_id,
        0,
        [{"dataset_id": "a"}, {"dataset_id": "b"}],
    )
    store.register_raw_batch(run_id, "enumerate", 0, raw_path, 2, checksum)

    output = CatalogExporter(store, settings.output_dir).export(run_id)

    for table, columns in APPROVED_COLUMNS.items():
        parquet_path = output / f"{table}.parquet"
        csv_path = output / f"{table}.csv"
        assert parquet_path.exists()
        assert csv_path.exists()
        assert pq.read_schema(parquet_path).names == columns
        with csv_path.open(newline="", encoding="utf-8") as handle:
            assert next(csv.reader(handle)) == columns

    papers = pq.read_table(output / "hf_dataset_papers.parquet").to_pylist()
    assert papers[0]["citation_count"] is None
    assert (output / "hf_datasets_raw.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert (output / "collection_report.json").exists()
    assert not list(output.glob("*.tmp"))
