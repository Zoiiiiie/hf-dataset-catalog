from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
from datetime import UTC, date, datetime
from typing import Any

from dateutil.relativedelta import relativedelta

from hf_dataset_catalog.models import RankingRecord, SelectionFlags


def calendar_cutoff(now: datetime, years: int) -> datetime:
    if years < 1:
        raise ValueError("years must be at least 1")
    return now - relativedelta(years=years)


def select_recent(
    rows: Iterable[Mapping[str, Any]],
    *,
    now: datetime,
    years: int,
) -> set[str]:
    cutoff = calendar_cutoff(now, years)
    return {
        str(row["dataset_id"])
        for row in rows
        if isinstance(row.get("created_at"), datetime) and row["created_at"] >= cutoff
    }


def rank_metric(
    rows: Iterable[Mapping[str, Any]],
    metric: str,
    top_n: int,
    ranking_date: date | None = None,
) -> list[RankingRecord]:
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    valid = (row for row in rows if row.get(metric) is not None)
    ordered = sorted(valid, key=lambda row: (-int(row[metric]), str(row["dataset_id"])))
    calculated_on = ranking_date or datetime.now(UTC).date()
    return [
        RankingRecord(
            dataset_id=str(row["dataset_id"]),
            metric=metric,
            metric_value=int(row[metric]),
            rank=index,
            ranking_date=calculated_on,
        )
        for index, row in enumerate(ordered[:top_n], start=1)
    ]


def merge_selection_sets(
    *,
    recent: Set[str],
    downloads: Set[str],
    downloads_all_time: Set[str],
    citations: Set[str],
) -> dict[str, SelectionFlags]:
    dataset_ids = recent | downloads | downloads_all_time | citations
    return {
        dataset_id: SelectionFlags(
            is_recent=dataset_id in recent,
            is_top_download=dataset_id in downloads,
            is_top_download_all_time=dataset_id in downloads_all_time,
            is_top_citation=dataset_id in citations,
        )
        for dataset_id in sorted(dataset_ids)
    }
