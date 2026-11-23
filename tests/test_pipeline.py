from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hf_dataset_catalog.config import Settings
from hf_dataset_catalog.models import MatchStatus
from hf_dataset_catalog.papers import PaperCandidate, PaperClue
from hf_dataset_catalog.pipeline import CatalogPipeline


class FakeHuggingFaceSource:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.iteration_count = 0

    def iter_datasets(
        self,
        *,
        fetched_at: datetime | None = None,
        on_rejected: Any = None,
    ) -> list[dict[str, Any]]:
        self.iteration_count += 1
        return [{**row, "fetched_at": fetched_at or datetime.now(UTC)} for row in self.rows]


class FakePaperResolver:
    def __init__(self, citations: dict[str, int]) -> None:
        self.citations = citations
        self.calls = 0

    async def resolve(self, clue: PaperClue) -> PaperCandidate:
        self.calls += 1
        count = self.citations.get(clue.doi or "")
        if count is None:
            return PaperCandidate(title=clue.title)
        return PaperCandidate(
            paper_id=f"paper-{clue.doi}",
            title="Confirmed paper",
            doi=clue.doi,
            citation_count=count,
            citation_source="semantic_scholar",
            match_method="doi",
            match_confidence=1,
            match_status=MatchStatus.CONFIRMED,
            effective_citation_count=count,
        )


def dataset(
    dataset_id: str,
    *,
    created_at: str,
    downloads: int,
    downloads_all_time: int = 0,
    doi: str | None = None,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "author": None,
        "card_data": {"license": "mit"} if doi else {},
        "citation": f"doi:{doi}" if doi else None,
        "created_at": datetime.fromisoformat(created_at).replace(tzinfo=UTC),
        "description": "Synthetic API fixture",
        "disabled": False,
        "downloads": downloads,
        "downloads_all_time": downloads_all_time,
        "gated": False,
        "last_modified": datetime(2026, 1, 2, tzinfo=UTC),
        "likes": 1,
        "main_size": None,
        "paperswithcode_id": None,
        "private": False,
        "sha": "abc123",
        "siblings": [{"rfilename": "data/train.parquet", "size": 10, "lfs": None}],
        "tags": ["task_categories:text-classification"],
        "used_storage": 10,
        "dataset_url": f"https://huggingface.co/datasets/{dataset_id}",
        "files_url": f"https://huggingface.co/datasets/{dataset_id}/tree/abc123",
        "viewer_url": f"https://huggingface.co/datasets/{dataset_id}/viewer",
    }


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        output_dir=tmp_path / "output",
        state_db=tmp_path / "state.duckdb",
        recent_years=5,
        top_downloads=1,
        top_downloads_all_time=1,
        top_citations=1,
        batch_size=2,
        request_interval_seconds=0,
    )


def test_pipeline_builds_or_union_and_resumes_without_duplicates(tmp_path: Path) -> None:
    source = FakeHuggingFaceSource(
        [
            dataset("recent", created_at="2026-01-01", downloads=1),
            dataset(
                "popular",
                created_at="2018-01-01",
                downloads=100,
                downloads_all_time=1000,
            ),
            dataset(
                "cited",
                created_at="2018-01-01",
                downloads=0,
                doi="10.1000/example",
            ),
        ]
    )
    resolver = FakePaperResolver({"10.1000/example": 200})
    pipeline = CatalogPipeline(make_settings(tmp_path), source=source, resolver=resolver)

    run_id = pipeline.collect(now=datetime(2026, 11, 23, tzinfo=UTC))
    pipeline.resume(run_id)
    rows = pipeline.store.master_rows(run_id)

    assert {row["dataset_id"] for row in rows} == {"recent", "popular", "cited"}
    assert len(rows) == 3
    assert source.iteration_count == 1
    assert resolver.calls == 1
    cited = next(row for row in rows if row["dataset_id"] == "cited")
    assert cited["is_top_citation"] is True
    assert cited["max_paper_citation_count"] == 200


def test_pipeline_derives_partial_quality_flags(tmp_path: Path) -> None:
    row = dataset("incomplete", created_at="2026-01-01", downloads=1)
    row["card_data"] = None
    pipeline = CatalogPipeline(
        make_settings(tmp_path),
        source=FakeHuggingFaceSource([row]),
        resolver=FakePaperResolver({}),
    )

    run_id = pipeline.collect(now=datetime(2026, 11, 23, tzinfo=UTC))
    result = pipeline.store.master_rows(run_id)[0]

    assert result["collection_status"] == "PARTIAL"
    assert "CARD_MISSING" in result["quality_flags"]
    assert "LICENSE_UNKNOWN" in result["quality_flags"]
