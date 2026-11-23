from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
from pydantic import BaseModel

from hf_dataset_catalog.config import Settings
from hf_dataset_catalog.models import (
    CollectionStatus,
    DatasetRecord,
    FileRecord,
    MetricRecord,
    PaperRecord,
    RankingRecord,
)


class CatalogStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(self.path), read_only=read_only)

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS collection_runs (
                    collection_run_id VARCHAR PRIMARY KEY,
                    started_at TIMESTAMPTZ NOT NULL,
                    finished_at TIMESTAMPTZ,
                    status VARCHAR NOT NULL,
                    current_stage VARCHAR,
                    config_json JSON NOT NULL
                );
                CREATE TABLE IF NOT EXISTS raw_batches (
                    collection_run_id VARCHAR,
                    stage VARCHAR,
                    batch_number BIGINT,
                    path VARCHAR,
                    record_count BIGINT,
                    checksum VARCHAR,
                    created_at TIMESTAMPTZ,
                    PRIMARY KEY (collection_run_id, stage, batch_number)
                );
                CREATE TABLE IF NOT EXISTS collection_checkpoints (
                    collection_run_id VARCHAR,
                    stage VARCHAR,
                    checkpoint_key VARCHAR,
                    checkpoint_value JSON NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (collection_run_id, stage, checkpoint_key)
                );
                CREATE TABLE IF NOT EXISTS collection_failures (
                    collection_run_id VARCHAR,
                    stage VARCHAR,
                    entity_id VARCHAR,
                    error_type VARCHAR NOT NULL,
                    error_message VARCHAR NOT NULL,
                    retry_count INTEGER NOT NULL,
                    occurred_at TIMESTAMPTZ NOT NULL
                );
                CREATE TABLE IF NOT EXISTS hf_datasets_raw (
                    collection_run_id VARCHAR,
                    dataset_id VARCHAR,
                    payload JSON NOT NULL,
                    fetched_at TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (collection_run_id, dataset_id)
                );
                CREATE TABLE IF NOT EXISTS dataset_rankings (
                    collection_run_id VARCHAR,
                    dataset_id VARCHAR,
                    metric VARCHAR,
                    metric_value BIGINT,
                    rank BIGINT NOT NULL,
                    ranking_date DATE NOT NULL,
                    PRIMARY KEY (collection_run_id, metric, dataset_id)
                );
                """
            )
            self._create_domain_tables(connection)

    @staticmethod
    def _create_domain_tables(connection: duckdb.DuckDBPyConnection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS hf_datasets_master (
                collection_run_id VARCHAR,
                dataset_id VARCHAR,
                dataset_name VARCHAR,
                author VARCHAR,
                dataset_url VARCHAR,
                files_url VARCHAR,
                viewer_url VARCHAR,
                created_at TIMESTAMPTZ,
                last_modified TIMESTAMPTZ,
                sha VARCHAR,
                description VARCHAR,
                tags VARCHAR[],
                license VARCHAR,
                license_raw VARCHAR,
                commercial_use_restriction BOOLEAN,
                redistribution_restriction BOOLEAN,
                special_license_terms VARCHAR,
                modalities VARCHAR[],
                use_cases VARCHAR[],
                tasks VARCHAR[],
                domains VARCHAR[],
                data_formats VARCHAR[],
                classification_source VARCHAR[],
                gated VARCHAR,
                private BOOLEAN,
                disabled BOOLEAN,
                likes BIGINT,
                downloads BIGINT,
                downloads_all_time BIGINT,
                downloads_rank BIGINT,
                downloads_all_time_rank BIGINT,
                max_paper_citation_count BIGINT,
                citation_rank BIGINT,
                citation_source VARCHAR,
                top_paper_id VARCHAR,
                top_paper_title VARCHAR,
                paper_match_status VARCHAR,
                associated_paper_count BIGINT,
                file_count BIGINT,
                used_storage_bytes BIGINT,
                total_file_size_bytes BIGINT,
                file_extensions VARCHAR[],
                is_recent BOOLEAN,
                is_top_download BOOLEAN,
                is_top_download_all_time BOOLEAN,
                is_top_citation BOOLEAN,
                ranking_date DATE,
                collection_status VARCHAR,
                quality_flags VARCHAR[],
                fetched_at TIMESTAMPTZ,
                PRIMARY KEY (collection_run_id, dataset_id)
            );
            CREATE TABLE IF NOT EXISTS hf_dataset_papers (
                collection_run_id VARCHAR,
                dataset_id VARCHAR,
                paper_id VARCHAR,
                semantic_scholar_id VARCHAR,
                openalex_id VARCHAR,
                paper_title VARCHAR,
                doi VARCHAR,
                arxiv_id VARCHAR,
                paper_url VARCHAR,
                authors VARCHAR[],
                publication_year INTEGER,
                publication_date DATE,
                external_ids JSON,
                citation_count BIGINT,
                citation_source VARCHAR,
                match_method VARCHAR,
                match_confidence DOUBLE,
                match_status VARCHAR,
                is_top_cited_paper BOOLEAN,
                fetched_at TIMESTAMPTZ,
                PRIMARY KEY (collection_run_id, dataset_id, paper_id)
            );
            CREATE TABLE IF NOT EXISTS hf_dataset_files (
                collection_run_id VARCHAR,
                dataset_id VARCHAR,
                revision VARCHAR,
                file_path VARCHAR,
                file_name VARCHAR,
                extension VARCHAR,
                file_type VARCHAR,
                size_bytes BIGINT,
                is_lfs BOOLEAN,
                file_page_url VARCHAR,
                resolve_url VARCHAR,
                fetched_at TIMESTAMPTZ,
                PRIMARY KEY (collection_run_id, dataset_id, file_path)
            );
            CREATE TABLE IF NOT EXISTS hf_dataset_metrics (
                collection_run_id VARCHAR,
                dataset_id VARCHAR,
                snapshot_date DATE,
                downloads BIGINT,
                downloads_all_time BIGINT,
                likes BIGINT,
                citation_count BIGINT,
                citation_source VARCHAR,
                fetched_at TIMESTAMPTZ,
                PRIMARY KEY (dataset_id, snapshot_date)
            );
            """
        )

    def create_run(self, settings: Settings, now: datetime | None = None) -> str:
        run_id = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{run_id}-{uuid.uuid4().hex[:8]}"
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO collection_runs
                VALUES (?, ?, NULL, ?, NULL, ?)
                """,
                [
                    run_id,
                    now or datetime.now(UTC),
                    CollectionStatus.RUNNING.value,
                    json.dumps(settings.model_dump(mode="json"), sort_keys=True),
                ],
            )
        return run_id

    def complete_run(
        self,
        run_id: str,
        status: CollectionStatus = CollectionStatus.SUCCESS,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE collection_runs
                SET status = ?, finished_at = ?, current_stage = NULL
                WHERE collection_run_id = ?
                """,
                [status.value, datetime.now(UTC), run_id],
            )

    def set_current_stage(self, run_id: str, stage: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE collection_runs
                SET current_stage = ?, status = ?, finished_at = NULL
                WHERE collection_run_id = ?
                """,
                [stage, CollectionStatus.RUNNING.value, run_id],
            )

    def register_raw_batch(
        self,
        run_id: str,
        stage: str,
        batch_number: int,
        path: Path,
        record_count: int,
        checksum: str = "",
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO raw_batches VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (collection_run_id, stage, batch_number) DO NOTHING
                """,
                [
                    run_id,
                    stage,
                    batch_number,
                    str(path),
                    record_count,
                    checksum,
                    datetime.now(UTC),
                ],
            )

    def raw_batch_count(self, run_id: str) -> int:
        with self.connect() as connection:
            value = connection.execute(
                "SELECT count(*) FROM raw_batches WHERE collection_run_id = ?",
                [run_id],
            ).fetchone()
        return int(value[0]) if value else 0

    def raw_batches(self, run_id: str) -> list[dict[str, Any]]:
        return self.query_dicts(
            """
            SELECT * FROM raw_batches
            WHERE collection_run_id = ?
            ORDER BY stage, batch_number
            """,
            [run_id],
        )

    def set_checkpoint(self, run_id: str, stage: str, key: str, value: Any) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO collection_checkpoints VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (collection_run_id, stage, checkpoint_key)
                DO UPDATE SET checkpoint_value = excluded.checkpoint_value,
                              updated_at = excluded.updated_at
                """,
                [run_id, stage, key, json.dumps(value), datetime.now(UTC)],
            )

    def get_checkpoint(self, run_id: str, stage: str, key: str) -> Any | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT checkpoint_value FROM collection_checkpoints
                WHERE collection_run_id = ? AND stage = ? AND checkpoint_key = ?
                """,
                [run_id, stage, key],
            ).fetchone()
        return json.loads(row[0]) if row else None

    def stage_complete(self, run_id: str, stage: str) -> bool:
        return self.get_checkpoint(run_id, stage, "complete") is True

    def mark_stage_complete(self, run_id: str, stage: str) -> None:
        self.set_checkpoint(run_id, stage, "complete", True)

    def record_failure(
        self,
        run_id: str,
        stage: str,
        entity_id: str,
        error: Exception,
        retry_count: int = 0,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO collection_failures VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    run_id,
                    stage,
                    entity_id,
                    type(error).__name__,
                    str(error),
                    retry_count,
                    datetime.now(UTC),
                ],
            )

    def upsert_raw(self, run_id: str, records: Iterable[dict[str, Any]]) -> None:
        rows = [
            [
                run_id,
                record["dataset_id"],
                json.dumps(record["payload"], default=str, sort_keys=True),
                record["fetched_at"],
            ]
            for record in records
        ]
        if not rows:
            return
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO hf_datasets_raw VALUES (?, ?, ?, ?)
                ON CONFLICT (collection_run_id, dataset_id)
                DO UPDATE SET payload = excluded.payload, fetched_at = excluded.fetched_at
                """,
                rows,
            )

    def upsert_metrics(self, run_id: str, records: Iterable[MetricRecord]) -> None:
        rows = [
            [
                run_id,
                record.dataset_id,
                record.snapshot_date,
                record.downloads,
                record.downloads_all_time,
                record.likes,
                record.citation_count,
                record.citation_source,
                record.fetched_at,
            ]
            for record in records
        ]
        if not rows:
            return
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO hf_dataset_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (dataset_id, snapshot_date) DO UPDATE SET
                    collection_run_id = excluded.collection_run_id,
                    downloads = excluded.downloads,
                    downloads_all_time = excluded.downloads_all_time,
                    likes = excluded.likes,
                    citation_count = excluded.citation_count,
                    citation_source = excluded.citation_source,
                    fetched_at = excluded.fetched_at
                """,
                rows,
            )

    def get_metrics(self, dataset_id: str) -> list[dict[str, Any]]:
        return self.query_dicts(
            """
            SELECT * FROM hf_dataset_metrics
            WHERE dataset_id = ? ORDER BY snapshot_date
            """,
            [dataset_id],
        )

    def raw_rows(self, run_id: str) -> list[dict[str, Any]]:
        return self.query_dicts(
            """
            SELECT * FROM hf_datasets_raw
            WHERE collection_run_id = ? ORDER BY dataset_id
            """,
            [run_id],
        )

    def ranking_rows(self, run_id: str) -> list[dict[str, Any]]:
        return self.query_dicts(
            """
            SELECT * FROM dataset_rankings
            WHERE collection_run_id = ? ORDER BY metric, rank
            """,
            [run_id],
        )

    def paper_rows(self, run_id: str) -> list[dict[str, Any]]:
        return self.query_dicts(
            """
            SELECT * FROM hf_dataset_papers
            WHERE collection_run_id = ? ORDER BY dataset_id, paper_id
            """,
            [run_id],
        )

    def master_rows(self, run_id: str) -> list[dict[str, Any]]:
        return self.query_dicts(
            """
            SELECT * FROM hf_datasets_master
            WHERE collection_run_id = ? ORDER BY dataset_id
            """,
            [run_id],
        )

    def upsert_rankings(self, run_id: str, records: Iterable[RankingRecord]) -> None:
        rows = [
            [
                run_id,
                record.dataset_id,
                record.metric,
                record.metric_value,
                record.rank,
                record.ranking_date,
            ]
            for record in records
        ]
        if not rows:
            return
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO dataset_rankings VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (collection_run_id, metric, dataset_id) DO UPDATE SET
                    metric_value = excluded.metric_value,
                    rank = excluded.rank,
                    ranking_date = excluded.ranking_date
                """,
                rows,
            )

    def upsert_models(
        self,
        run_id: str,
        table: str,
        records: Iterable[BaseModel],
    ) -> None:
        allowed: dict[str, type[BaseModel]] = {
            "hf_datasets_master": DatasetRecord,
            "hf_dataset_papers": PaperRecord,
            "hf_dataset_files": FileRecord,
        }
        model_type = allowed.get(table)
        if model_type is None:
            raise ValueError(f"unsupported model table: {table}")
        records_list = list(records)
        if not records_list:
            return
        fields = list(model_type.model_fields)
        columns = ["collection_run_id", *fields]
        placeholders = ", ".join("?" for _ in columns)
        values = [
            [run_id, *[self._db_value(getattr(record, field)) for field in fields]]
            for record in records_list
        ]
        key_columns = {
            "hf_datasets_master": ["collection_run_id", "dataset_id"],
            "hf_dataset_papers": ["collection_run_id", "dataset_id", "paper_id"],
            "hf_dataset_files": ["collection_run_id", "dataset_id", "file_path"],
        }[table]
        update_fields = [field for field in fields if field not in key_columns]
        updates = ", ".join(f"{field} = excluded.{field}" for field in update_fields)
        with self.connect() as connection:
            connection.executemany(
                f"""
                INSERT INTO {table} ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT ({", ".join(key_columns)}) DO UPDATE SET {updates}
                """,
                values,
            )

    @staticmethod
    def _db_value(value: Any) -> Any:
        if isinstance(value, StrEnum):
            return value.value
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True)
        return value

    def query_dicts(self, query: str, parameters: list[Any] | None = None) -> list[dict[str, Any]]:
        with self.connect(read_only=True) as connection:
            result = connection.execute(query, parameters or [])
            columns = [item[0] for item in result.description]
            return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]

    def table_arrow(self, table: str, run_id: str | None = None) -> pa.Table:
        allowed = {
            "hf_datasets_master",
            "hf_dataset_papers",
            "hf_dataset_files",
            "hf_dataset_metrics",
            "dataset_rankings",
        }
        if table not in allowed:
            raise ValueError(f"unsupported export table: {table}")
        query = f"SELECT * FROM {table}"
        parameters: list[Any] = []
        if run_id and table != "hf_dataset_metrics":
            query += " WHERE collection_run_id = ?"
            parameters.append(run_id)
        with self.connect(read_only=True) as connection:
            return connection.execute(query, parameters).to_arrow_table()
