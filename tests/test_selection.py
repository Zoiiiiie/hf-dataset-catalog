from datetime import UTC, date, datetime

from hf_dataset_catalog.selection import (
    calendar_cutoff,
    merge_selection_sets,
    rank_metric,
    select_recent,
)


def test_calendar_cutoff_handles_leap_day() -> None:
    now = datetime(2024, 2, 29, 12, tzinfo=UTC)
    assert calendar_cutoff(now, 5) == datetime(2019, 2, 28, 12, tzinfo=UTC)


def test_rank_is_null_last_and_stable_by_dataset_id() -> None:
    rows = [
        {"dataset_id": "b/data", "downloads": 7},
        {"dataset_id": "a/data", "downloads": 7},
        {"dataset_id": "c/data", "downloads": None},
    ]
    ranked = rank_metric(rows, "downloads", top_n=3, ranking_date=date(2026, 11, 23))
    assert [row.dataset_id for row in ranked] == ["a/data", "b/data"]
    assert [row.rank for row in ranked] == [1, 2]


def test_union_preserves_every_membership_reason() -> None:
    result = merge_selection_sets(
        recent={"a", "both"},
        downloads={"b", "both"},
        downloads_all_time={"old"},
        citations={"paper", "both"},
    )
    assert result["both"].model_dump() == {
        "is_recent": True,
        "is_top_download": True,
        "is_top_download_all_time": False,
        "is_top_citation": True,
    }
    assert set(result) == {"a", "b", "both", "old", "paper"}


def test_recent_selection_excludes_missing_and_pre_cutoff_dates() -> None:
    rows = [
        {"dataset_id": "at-cutoff", "created_at": datetime(2019, 2, 28, 12, tzinfo=UTC)},
        {"dataset_id": "old", "created_at": datetime(2019, 2, 28, 11, 59, tzinfo=UTC)},
        {"dataset_id": "missing", "created_at": None},
    ]
    selected = select_recent(rows, now=datetime(2024, 2, 29, 12, tzinfo=UTC), years=5)
    assert selected == {"at-cutoff"}
