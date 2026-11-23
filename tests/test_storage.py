from datetime import date
from pathlib import Path

from hf_dataset_catalog.models import MetricRecord
from hf_dataset_catalog.storage import CatalogStore


def test_raw_batch_registration_is_idempotent(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "state.duckdb")
    store.initialize()
    store.register_raw_batch("run-1", "enumerate", 0, tmp_path / "0.jsonl", 2)
    store.register_raw_batch("run-1", "enumerate", 0, tmp_path / "0.jsonl", 2)
    assert store.raw_batch_count("run-1") == 1


def test_metric_snapshot_upserts_same_day(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "state.duckdb")
    store.initialize()
    first = MetricRecord(
        dataset_id="org/data",
        snapshot_date=date(2026, 11, 23),
        downloads=10,
    )
    second = first.model_copy(update={"downloads": 12})
    store.upsert_metrics("run-1", [first, second])
    assert store.get_metrics("org/data")[0]["downloads"] == 12


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "state.duckdb")
    store.initialize()
    store.set_checkpoint("run-1", "enumerate", "cursor", {"batch": 4})
    assert store.get_checkpoint("run-1", "enumerate", "cursor") == {"batch": 4}

