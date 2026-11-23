from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CollectionReport(BaseModel):
    collection_run_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    enumerated_count: int
    recent_count: int
    top_download_count: int
    top_download_all_time_count: int
    top_citation_count: int
    union_count: int
    deduplication_count: int
    paper_clue_dataset_count: int
    paper_matched_dataset_count: int
    paper_match_rate: float | None
    api_failure_count: int
    api_failure_rate: float | None
    required_field_cell_count: int
    populated_required_field_cell_count: int
    field_completeness_rate: float | None
    retry_count: int = 0
    quality_flag_counts: dict[str, int] = Field(default_factory=dict)


def build_report(
    *,
    enumerated: int,
    recent: int,
    top_download: int,
    top_download_all_time: int,
    top_citation: int,
    union_count: int,
    paper_clue_datasets: int,
    paper_matched_datasets: int,
    api_failures: int,
    required_field_cells: int,
    populated_required_field_cells: int,
    collection_run_id: str | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    retry_count: int = 0,
    quality_flag_counts: dict[str, int] | None = None,
) -> CollectionReport:
    selection_entries = recent + top_download + top_download_all_time + top_citation
    return CollectionReport(
        collection_run_id=collection_run_id,
        started_at=started_at,
        finished_at=finished_at,
        enumerated_count=enumerated,
        recent_count=recent,
        top_download_count=top_download,
        top_download_all_time_count=top_download_all_time,
        top_citation_count=top_citation,
        union_count=union_count,
        deduplication_count=max(0, selection_entries - union_count),
        paper_clue_dataset_count=paper_clue_datasets,
        paper_matched_dataset_count=paper_matched_datasets,
        paper_match_rate=_ratio(paper_matched_datasets, paper_clue_datasets),
        api_failure_count=api_failures,
        api_failure_rate=_ratio(api_failures, enumerated),
        required_field_cell_count=required_field_cells,
        populated_required_field_cell_count=populated_required_field_cells,
        field_completeness_rate=_ratio(
            populated_required_field_cells,
            required_field_cells,
        ),
        retry_count=retry_count,
        quality_flag_counts=quality_flag_counts or {},
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
