from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class CollectionStatus(StrEnum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    API_ERROR = "API_ERROR"
    FAILED = "FAILED"


class MatchStatus(StrEnum):
    CONFIRMED = "confirmed"
    REVIEW = "review"
    UNMATCHED = "unmatched"
    NO_CLUE = "no_clue"


class QualityFlag(StrEnum):
    PAPER_UNMATCHED = "PAPER_UNMATCHED"
    CARD_MISSING = "CARD_MISSING"
    LICENSE_UNKNOWN = "LICENSE_UNKNOWN"
    METADATA_INCOMPLETE = "METADATA_INCOMPLETE"


class FileType(StrEnum):
    TABLE = "table"
    STRUCTURED = "structured"
    TEXT = "text"
    ARCHIVE = "archive"
    PARQUET = "parquet"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    OTHER = "other"


class RunRecord(BaseModel):
    collection_run_id: str
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    status: CollectionStatus = CollectionStatus.RUNNING
    current_stage: str | None = None
    config_json: dict[str, Any]


class RawDatasetRecord(BaseModel):
    dataset_id: str
    payload: dict[str, Any]
    fetched_at: datetime = Field(default_factory=utc_now)


class MetricRecord(BaseModel):
    dataset_id: str
    snapshot_date: date
    downloads: int | None = None
    downloads_all_time: int | None = None
    likes: int | None = None
    citation_count: int | None = None
    citation_source: str | None = None
    fetched_at: datetime = Field(default_factory=utc_now)


class RankingRecord(BaseModel):
    dataset_id: str
    metric: str
    metric_value: int
    rank: int
    ranking_date: date


class SelectionFlags(BaseModel):
    is_recent: bool = False
    is_top_download: bool = False
    is_top_download_all_time: bool = False
    is_top_citation: bool = False


class DatasetRecord(BaseModel):
    dataset_id: str
    dataset_name: str
    author: str | None = None
    dataset_url: str
    files_url: str
    viewer_url: str
    created_at: datetime | None = None
    last_modified: datetime | None = None
    sha: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    license: str = "Unknown"
    license_raw: str | None = None
    commercial_use_restriction: bool | None = None
    redistribution_restriction: bool | None = None
    special_license_terms: str | None = None
    modalities: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    data_formats: list[str] = Field(default_factory=list)
    classification_source: list[str] = Field(default_factory=list)
    gated: str | bool | None = None
    private: bool = False
    disabled: bool | None = None
    likes: int | None = None
    downloads: int | None = None
    downloads_all_time: int | None = None
    downloads_rank: int | None = None
    downloads_all_time_rank: int | None = None
    max_paper_citation_count: int | None = None
    citation_rank: int | None = None
    citation_source: str | None = None
    top_paper_id: str | None = None
    top_paper_title: str | None = None
    paper_match_status: MatchStatus = MatchStatus.NO_CLUE
    associated_paper_count: int = 0
    file_count: int | None = None
    used_storage_bytes: int | None = None
    total_file_size_bytes: int | None = None
    file_extensions: list[str] = Field(default_factory=list)
    is_recent: bool = False
    is_top_download: bool = False
    is_top_download_all_time: bool = False
    is_top_citation: bool = False
    ranking_date: date
    collection_status: CollectionStatus = CollectionStatus.SUCCESS
    quality_flags: list[str] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=utc_now)


class PaperRecord(BaseModel):
    dataset_id: str
    paper_id: str | None = None
    semantic_scholar_id: str | None = None
    openalex_id: str | None = None
    paper_title: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    paper_url: str | None = None
    authors: list[str] = Field(default_factory=list)
    publication_year: int | None = None
    publication_date: date | None = None
    external_ids: dict[str, Any] | None = None
    citation_count: int | None = None
    citation_source: str | None = None
    match_method: str
    match_confidence: float = Field(ge=0, le=1)
    match_status: MatchStatus
    is_top_cited_paper: bool = False
    fetched_at: datetime = Field(default_factory=utc_now)


class FileRecord(BaseModel):
    dataset_id: str
    revision: str | None = None
    file_path: str
    file_name: str
    extension: str | None = None
    file_type: FileType = FileType.OTHER
    size_bytes: int | None = None
    is_lfs: bool | None = None
    file_page_url: str
    resolve_url: str
    fetched_at: datetime = Field(default_factory=utc_now)
