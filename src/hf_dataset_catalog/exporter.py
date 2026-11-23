from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from hf_dataset_catalog.models import DatasetRecord, FileRecord, MetricRecord, PaperRecord
from hf_dataset_catalog.report import CollectionReport, build_report
from hf_dataset_catalog.storage import CatalogStore

METRIC_COLUMNS = ["collection_run_id", *MetricRecord.model_fields]
APPROVED_COLUMNS = {
    "hf_datasets_master": ["collection_run_id", *DatasetRecord.model_fields],
    "hf_dataset_papers": ["collection_run_id", *PaperRecord.model_fields],
    "hf_dataset_files": ["collection_run_id", *FileRecord.model_fields],
    "hf_dataset_metrics": METRIC_COLUMNS,
}
REQUIRED_MASTER_FIELDS = [
    "dataset_id",
    "dataset_url",
    "files_url",
    "viewer_url",
    "ranking_date",
    "collection_status",
    "fetched_at",
]


class CatalogExporter:
    def __init__(self, store: CatalogStore, output_dir: Path) -> None:
        self.store = store
        self.output_dir = output_dir

    def export(self, run_id: str) -> Path:
        output = self.output_dir / run_id
        output.mkdir(parents=True, exist_ok=True)
        tables = {
            name: self.store.table_arrow(name, run_id).select(columns)
            for name, columns in APPROVED_COLUMNS.items()
        }
        self.validate(run_id, tables)

        for name, table in tables.items():
            self._write_parquet_atomic(table, output / f"{name}.parquet")
            self._write_csv_atomic(table, output / f"{name}.csv")

        self._write_raw_jsonl(run_id, output / "hf_datasets_raw.jsonl")
        self._write_intermediate_tables(run_id, tables["hf_datasets_master"], output)
        report = self._build_report(run_id, tables)
        self._write_json_atomic(
            report.model_dump(mode="json"),
            output / "collection_report.json",
        )
        return output

    def validate(self, run_id: str, tables: dict[str, pa.Table]) -> None:
        master_rows = tables["hf_datasets_master"].to_pylist()
        paper_rows = tables["hf_dataset_papers"].to_pylist()
        file_rows = tables["hf_dataset_files"].to_pylist()
        metric_rows = tables["hf_dataset_metrics"].to_pylist()

        _require_unique(master_rows, ("collection_run_id", "dataset_id"))
        _require_unique(
            paper_rows,
            ("collection_run_id", "dataset_id", "paper_id"),
        )
        _require_unique(
            file_rows,
            ("collection_run_id", "dataset_id", "file_path"),
        )
        _require_unique(metric_rows, ("dataset_id", "snapshot_date"))

        for row in master_rows:
            if row["private"]:
                raise ValueError(f"private dataset cannot be exported: {row['dataset_id']}")
            if not all(row[field] for field in ("dataset_url", "files_url", "viewer_url")):
                raise ValueError(f"dataset has missing required URL: {row['dataset_id']}")
            if not any(
                row[field]
                for field in (
                    "is_recent",
                    "is_top_download",
                    "is_top_download_all_time",
                    "is_top_citation",
                )
            ):
                raise ValueError(f"dataset is not in the selected union: {row['dataset_id']}")

        for row in file_rows:
            if not row["file_page_url"] or not row["resolve_url"]:
                raise ValueError(
                    f"file has missing required URL: {row['dataset_id']}:{row['file_path']}"
                )

        rankings = self.store.ranking_rows(run_id)
        _require_unique(rankings, ("collection_run_id", "metric", "dataset_id"))
        _require_unique(rankings, ("collection_run_id", "metric", "rank"))

    def _write_raw_jsonl(self, run_id: str, final_path: Path) -> None:
        temporary_path = _temporary_path(final_path)
        digest_by_path = {
            Path(row["path"]): str(row["checksum"])
            for row in self.store.raw_batches(run_id)
            if row["stage"] == "enumerate"
        }
        with temporary_path.open("wb") as output:
            for source_path, expected_checksum in digest_by_path.items():
                payload = source_path.read_bytes()
                actual_checksum = sha256(payload).hexdigest()
                if expected_checksum and actual_checksum != expected_checksum:
                    raise ValueError(f"raw batch checksum mismatch: {source_path}")
                output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        temporary_path.replace(final_path)

    def _write_intermediate_tables(
        self,
        run_id: str,
        master: pa.Table,
        output: Path,
    ) -> None:
        recent_rows = [
            {
                "collection_run_id": row["collection_run_id"],
                "dataset_id": row["dataset_id"],
                "created_at": row["created_at"],
                "ranking_date": row["ranking_date"],
            }
            for row in master.to_pylist()
            if row["is_recent"]
        ]
        recent_table = pa.Table.from_pylist(
            recent_rows,
            schema=pa.schema(
                [
                    ("collection_run_id", pa.string()),
                    ("dataset_id", pa.string()),
                    ("created_at", pa.timestamp("us", tz="UTC")),
                    ("ranking_date", pa.date32()),
                ]
            ),
        )
        rankings = self.store.table_arrow("dataset_rankings", run_id)
        ranking_rows = rankings.to_pylist()
        download_table = pa.Table.from_pylist(
            [row for row in ranking_rows if row["metric"] in {"downloads", "downloads_all_time"}],
            schema=rankings.schema,
        )
        citation_table = pa.Table.from_pylist(
            [row for row in ranking_rows if row["metric"] == "max_paper_citation_count"],
            schema=rankings.schema,
        )
        self._write_parquet_atomic(recent_table, output / "dataset_recent.parquet")
        self._write_parquet_atomic(
            download_table,
            output / "dataset_download_top.parquet",
        )
        self._write_parquet_atomic(
            citation_table,
            output / "dataset_citation_top.parquet",
        )

    def _build_report(
        self,
        run_id: str,
        tables: dict[str, pa.Table],
    ) -> CollectionReport:
        master_rows = tables["hf_datasets_master"].to_pylist()
        paper_rows = tables["hf_dataset_papers"].to_pylist()
        failures = self.store.query_dicts(
            "SELECT * FROM collection_failures WHERE collection_run_id = ?",
            [run_id],
        )
        run_rows = self.store.query_dicts(
            "SELECT * FROM collection_runs WHERE collection_run_id = ?",
            [run_id],
        )
        run = run_rows[0] if run_rows else {}
        enumerated_checkpoint = self.store.get_checkpoint(
            run_id,
            "enumerate",
            "record_count",
        )
        enumerated = (
            int(enumerated_checkpoint)
            if isinstance(enumerated_checkpoint, int)
            else sum(
                int(batch["record_count"])
                for batch in self.store.raw_batches(run_id)
                if batch["stage"] == "enumerate"
            )
        )
        clue_datasets = {row["dataset_id"] for row in paper_rows}
        matched_datasets = {
            row["dataset_id"] for row in paper_rows if row["match_status"] == "confirmed"
        }
        required_cells = len(master_rows) * len(REQUIRED_MASTER_FIELDS)
        populated_cells = sum(
            row[field] is not None and row[field] != ""
            for row in master_rows
            for field in REQUIRED_MASTER_FIELDS
        )
        quality_counts: Counter[str] = Counter()
        for row in master_rows:
            quality_counts.update(row["quality_flags"] or [])
        return build_report(
            collection_run_id=run_id,
            started_at=run.get("started_at"),
            finished_at=run.get("finished_at") or datetime.now(UTC),
            enumerated=enumerated,
            recent=sum(bool(row["is_recent"]) for row in master_rows),
            top_download=sum(bool(row["is_top_download"]) for row in master_rows),
            top_download_all_time=sum(bool(row["is_top_download_all_time"]) for row in master_rows),
            top_citation=sum(bool(row["is_top_citation"]) for row in master_rows),
            union_count=len(master_rows),
            paper_clue_datasets=len(clue_datasets),
            paper_matched_datasets=len(matched_datasets),
            api_failures=len(failures),
            required_field_cells=required_cells,
            populated_required_field_cells=populated_cells,
            retry_count=sum(int(row["retry_count"]) for row in failures),
            quality_flag_counts=dict(sorted(quality_counts.items())),
        )

    @staticmethod
    def _write_parquet_atomic(table: pa.Table, final_path: Path) -> None:
        temporary_path = _temporary_path(final_path)
        pq.write_table(table, temporary_path, compression="zstd")
        temporary_path.replace(final_path)

    @staticmethod
    def _write_csv_atomic(table: pa.Table, final_path: Path) -> None:
        temporary_path = _temporary_path(final_path)
        pacsv.write_csv(_csv_compatible(table), temporary_path)
        temporary_path.replace(final_path)

    @staticmethod
    def _write_json_atomic(payload: Any, final_path: Path) -> None:
        temporary_path = _temporary_path(final_path)
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(final_path)


def _csv_compatible(table: pa.Table) -> pa.Table:
    columns = []
    for field, column in zip(table.schema, table.columns, strict=True):
        if (
            pa.types.is_list(field.type)
            or pa.types.is_large_list(field.type)
            or pa.types.is_struct(field.type)
            or pa.types.is_map(field.type)
        ):
            columns.append(
                pa.array(
                    [
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if value is not None
                        else None
                        for value in column.to_pylist()
                    ],
                    type=pa.string(),
                )
            )
        else:
            columns.append(column)
    return pa.Table.from_arrays(columns, names=table.column_names)


def _require_unique(rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        if key in seen:
            raise ValueError(f"duplicate key for {fields}: {key}")
        seen.add(key)


def _temporary_path(final_path: Path) -> Path:
    return final_path.with_suffix(f"{final_path.suffix}.tmp")
