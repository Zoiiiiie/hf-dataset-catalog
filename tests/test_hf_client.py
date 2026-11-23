from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hf_dataset_catalog.hf_client import (
    DATASET_EXPAND_FIELDS,
    HuggingFaceSource,
    adapt_dataset_info,
    adapt_file_info,
    persist_raw_batch,
    write_raw_batch,
)
from hf_dataset_catalog.storage import CatalogStore

FIXED_NOW = datetime(2026, 11, 23, 9, 30, tzinfo=UTC)


class FakeApi:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows
        self.kwargs: dict[str, Any] = {}

    def list_datasets(self, **kwargs: Any) -> list[Any]:
        self.kwargs = kwargs
        return self.rows


def test_adapt_dataset_info_preserves_distinct_download_metrics() -> None:
    info = SimpleNamespace(
        id="org/data",
        downloads=11,
        downloads_all_time=99,
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    record = adapt_dataset_info(info, fetched_at=FIXED_NOW)
    assert record["dataset_id"] == "org/data"
    assert record["downloads"] == 11
    assert record["downloads_all_time"] == 99
    assert record["fetched_at"] == FIXED_NOW


def test_write_raw_batch_is_atomic_and_deterministic(tmp_path: Path) -> None:
    path, checksum = write_raw_batch(tmp_path, "run-1", 3, [{"id": "a"}, {"id": "b"}])
    assert path.name == "00000003.jsonl"
    assert path.parent == tmp_path / "run-1" / "enumerate"
    assert not list(path.parent.glob("*.tmp"))
    assert checksum == sha256(path.read_bytes()).hexdigest()


def test_source_uses_exact_expand_fields_and_filters_private_rows() -> None:
    api = FakeApi(
        [
            SimpleNamespace(id="org/public", private=False),
            SimpleNamespace(id="org/private", private=True),
        ]
    )
    failures: list[tuple[str, str]] = []
    source = HuggingFaceSource(api=api)

    rows = list(
        source.iter_datasets(
            fetched_at=FIXED_NOW,
            on_rejected=lambda dataset_id, reason: failures.append((dataset_id, reason)),
        )
    )

    assert [row["dataset_id"] for row in rows] == ["org/public"]
    assert failures == [("org/private", "private dataset excluded")]
    assert api.kwargs == {"limit": None, "expand": DATASET_EXPAND_FIELDS}


def test_file_adapter_builds_metadata_urls_without_fetching() -> None:
    file = SimpleNamespace(
        rfilename="folder/a file.parquet",
        size=42,
        lfs=SimpleNamespace(size=42),
    )
    record = adapt_file_info("org/data name", "abc123", file, fetched_at=FIXED_NOW)
    assert record.file_path == "folder/a file.parquet"
    assert record.extension == "parquet"
    assert record.file_page_url.endswith(
        "/datasets/org/data%20name/blob/abc123/folder/a%20file.parquet"
    )
    assert record.resolve_url.endswith(
        "/datasets/org/data%20name/resolve/abc123/folder/a%20file.parquet"
    )
    assert record.is_lfs is True


def test_persist_raw_batch_registers_once_and_skips_rewrite(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "state.duckdb")
    store.initialize()
    first = persist_raw_batch(
        store=store,
        output_dir=tmp_path / "raw",
        run_id="run-1",
        batch_number=0,
        records=[{"id": "a"}],
    )
    original = first.path.read_bytes()

    second = persist_raw_batch(
        store=store,
        output_dir=tmp_path / "raw",
        run_id="run-1",
        batch_number=0,
        records=[{"id": "changed"}],
    )

    assert second.skipped is True
    assert second.path.read_bytes() == original
    assert store.raw_batch_count("run-1") == 1
