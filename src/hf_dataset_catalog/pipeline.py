from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from dateutil.parser import isoparse

from hf_dataset_catalog.classification import classify_dataset, classify_license
from hf_dataset_catalog.config import Settings
from hf_dataset_catalog.hf_client import adapt_file_info, dataset_urls, persist_raw_batch
from hf_dataset_catalog.models import (
    CollectionStatus,
    DatasetRecord,
    MatchStatus,
    MetricRecord,
    PaperRecord,
    QualityFlag,
    SelectionFlags,
)
from hf_dataset_catalog.papers import (
    PaperCandidate,
    PaperClue,
    candidate_to_record,
    extract_paper_clues,
    select_top_paper,
)
from hf_dataset_catalog.selection import merge_selection_sets, rank_metric, select_recent
from hf_dataset_catalog.storage import CatalogStore


class DatasetSource(Protocol):
    def iter_datasets(
        self,
        *,
        fetched_at: datetime | None = None,
        on_rejected: Callable[[str, str], None] | None = None,
    ) -> Iterable[dict[str, Any]]: ...


class Resolver(Protocol):
    async def resolve(self, clue: PaperClue) -> PaperCandidate: ...


class CatalogPipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        source: DatasetSource,
        resolver: Resolver,
        store: CatalogStore | None = None,
    ) -> None:
        self.settings = settings
        self.source = source
        self.resolver = resolver
        self.store = store or CatalogStore(settings.state_db)
        self.store.initialize()

    def collect(self, now: datetime | None = None) -> str:
        run_id = self.store.create_run(self.settings, now=now)
        return self.resume(run_id)

    def resume(self, run_id: str) -> str:
        try:
            self.enumerate(run_id)
            self.select_downloads_and_recent(run_id)
            asyncio.run(self.resolve_papers(run_id))
            self.select_citations_and_merge(run_id)
            self.enrich(run_id)
            self.export(run_id)
        except Exception:
            self.store.complete_run(run_id, CollectionStatus.FAILED)
            raise
        self.store.complete_run(run_id)
        return run_id

    def enumerate(self, run_id: str) -> None:
        self._run_stage_once(run_id, "enumerate", self._enumerate_impl)

    def select_downloads_and_recent(self, run_id: str) -> None:
        self._run_stage_once(run_id, "select", self._select_impl)

    async def resolve_papers(self, run_id: str) -> None:
        if self.store.stage_complete(run_id, "papers"):
            return
        self.store.set_current_stage(run_id, "papers")
        await self._resolve_papers_impl(run_id)
        self.store.mark_stage_complete(run_id, "papers")

    def select_citations_and_merge(self, run_id: str) -> None:
        self._run_stage_once(run_id, "merge", self._merge_impl)

    def enrich(self, run_id: str) -> None:
        self._run_stage_once(run_id, "enrich", self._enrich_impl)

    def export(self, run_id: str) -> Path:
        self._run_stage_once(run_id, "export", self._export_impl)
        return self.settings.output_dir / run_id

    def _run_stage_once(
        self,
        run_id: str,
        stage: str,
        operation: Callable[[str], None],
    ) -> None:
        if self.store.stage_complete(run_id, stage):
            return
        self.store.set_current_stage(run_id, stage)
        operation(run_id)
        self.store.mark_stage_complete(run_id, stage)

    def _enumerate_impl(self, run_id: str) -> None:
        fetched_at = self._run_datetime(run_id)
        batch: list[dict[str, Any]] = []
        batch_number = 0
        total = 0

        def reject(dataset_id: str, reason: str) -> None:
            self.store.record_failure(
                run_id,
                "enumerate",
                dataset_id,
                ValueError(reason),
            )

        for record in self.source.iter_datasets(fetched_at=fetched_at, on_rejected=reject):
            batch.append(record)
            if len(batch) >= self.settings.batch_size:
                self._persist_enumeration_batch(run_id, batch_number, batch)
                total += len(batch)
                batch = []
                batch_number += 1
        if batch:
            self._persist_enumeration_batch(run_id, batch_number, batch)
            total += len(batch)
        self.store.set_checkpoint(run_id, "enumerate", "record_count", total)

    def _persist_enumeration_batch(
        self,
        run_id: str,
        batch_number: int,
        records: list[dict[str, Any]],
    ) -> None:
        result = persist_raw_batch(
            store=self.store,
            output_dir=self.settings.output_dir / "_raw",
            run_id=run_id,
            batch_number=batch_number,
            records=records,
        )
        if result.skipped:
            return
        self.store.upsert_raw(
            run_id,
            [
                {
                    "dataset_id": record["dataset_id"],
                    "payload": record,
                    "fetched_at": _as_datetime(record.get("fetched_at"))
                    or datetime.now(UTC),
                }
                for record in records
            ],
        )

    def _select_impl(self, run_id: str) -> None:
        now = self._run_datetime(run_id)
        rows = self._raw_payloads(run_id)
        recent = select_recent(
            (
                {
                    "dataset_id": row["dataset_id"],
                    "created_at": _as_datetime(row.get("created_at")),
                }
                for row in rows
            ),
            now=now,
            years=self.settings.recent_years,
        )
        downloads_ranked = rank_metric(
            rows,
            "downloads",
            self.settings.top_downloads,
            ranking_date=now.date(),
        )
        all_time_ranked = rank_metric(
            rows,
            "downloads_all_time",
            self.settings.top_downloads_all_time,
            ranking_date=now.date(),
        )
        self.store.upsert_rankings(run_id, [*downloads_ranked, *all_time_ranked])
        self.store.upsert_metrics(
            run_id,
            [
                MetricRecord(
                    dataset_id=str(row["dataset_id"]),
                    snapshot_date=now.date(),
                    downloads=_optional_int(row.get("downloads")),
                    downloads_all_time=_optional_int(row.get("downloads_all_time")),
                    likes=_optional_int(row.get("likes")),
                    fetched_at=_as_datetime(row.get("fetched_at")) or now,
                )
                for row in rows
            ],
        )
        self.store.set_checkpoint(run_id, "select", "recent_ids", sorted(recent))
        self.store.set_checkpoint(
            run_id,
            "select",
            "download_ids",
            [record.dataset_id for record in downloads_ranked],
        )
        self.store.set_checkpoint(
            run_id,
            "select",
            "download_all_time_ids",
            [record.dataset_id for record in all_time_ranked],
        )

    async def _resolve_papers_impl(self, run_id: str) -> None:
        for dataset in self._raw_payloads(run_id):
            dataset_id = str(dataset["dataset_id"])
            clues = extract_paper_clues(dataset)
            records: dict[str, PaperRecord] = {}
            for clue in clues:
                try:
                    candidate = await self.resolver.resolve(clue)
                except Exception as error:
                    self.store.record_failure(run_id, "papers", dataset_id, error)
                    continue
                relation_id = candidate.paper_id or _clue_relation_id(clue)
                candidate = candidate.model_copy(update={"paper_id": relation_id})
                record = candidate_to_record(dataset_id, candidate)
                records.setdefault(relation_id, record)

            top = select_top_paper(records.values())
            if top:
                records[top.paper_id or ""] = top.model_copy(update={"is_top_cited_paper": True})
            self.store.upsert_models(run_id, "hf_dataset_papers", records.values())

    def _merge_impl(self, run_id: str) -> None:
        ranking_date = self._run_datetime(run_id).date()
        maximums: dict[str, int] = {}
        for row in self.store.paper_rows(run_id):
            if row["match_status"] != MatchStatus.CONFIRMED.value:
                continue
            count = _optional_int(row.get("citation_count"))
            if count is not None:
                dataset_id = str(row["dataset_id"])
                maximums[dataset_id] = max(maximums.get(dataset_id, count), count)

        citation_ranked = rank_metric(
            (
                {"dataset_id": dataset_id, "max_paper_citation_count": count}
                for dataset_id, count in maximums.items()
            ),
            "max_paper_citation_count",
            self.settings.top_citations,
            ranking_date=ranking_date,
        )
        self.store.upsert_rankings(run_id, citation_ranked)
        flags = merge_selection_sets(
            recent=set(self._checkpoint_list(run_id, "select", "recent_ids")),
            downloads=set(self._checkpoint_list(run_id, "select", "download_ids")),
            downloads_all_time=set(
                self._checkpoint_list(run_id, "select", "download_all_time_ids")
            ),
            citations={record.dataset_id for record in citation_ranked},
        )
        self.store.set_checkpoint(
            run_id,
            "merge",
            "selection_flags",
            {dataset_id: value.model_dump() for dataset_id, value in flags.items()},
        )

    def _enrich_impl(self, run_id: str) -> None:
        now = self._run_datetime(run_id)
        raw_by_id = {
            str(row["dataset_id"]): row
            for row in self._raw_payloads(run_id)
        }
        flags = {
            dataset_id: SelectionFlags.model_validate(value)
            for dataset_id, value in self._checkpoint_mapping(
                run_id,
                "merge",
                "selection_flags",
            ).items()
        }
        rankings = {
            (str(row["dataset_id"]), str(row["metric"])): int(row["rank"])
            for row in self.store.ranking_rows(run_id)
        }
        papers_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for paper in self.store.paper_rows(run_id):
            papers_by_dataset[str(paper["dataset_id"])].append(paper)

        master_records: list[DatasetRecord] = []
        file_records = []
        for dataset_id, membership in flags.items():
            raw = raw_by_id[dataset_id]
            siblings = [
                item for item in _as_sequence(raw.get("siblings")) if isinstance(item, Mapping)
            ]
            files = [
                adapt_file_info(
                    dataset_id,
                    _optional_string(raw.get("sha")),
                    item,
                    fetched_at=_as_datetime(raw.get("fetched_at")) or now,
                )
                for item in siblings
            ]
            file_records.extend(files)

            tags = [str(tag) for tag in _as_sequence(raw.get("tags"))]
            card_data = _as_mapping(raw.get("card_data"))
            classification = classify_dataset(
                tags=tags,
                file_paths=[record.file_path for record in files],
                card_data=card_data,
            )
            license_result = classify_license(card_data=card_data, tags=tags)
            paper_rows = papers_by_dataset.get(dataset_id, [])
            paper_status = _aggregate_match_status(paper_rows)
            top_paper = next(
                (paper for paper in paper_rows if paper["is_top_cited_paper"]),
                None,
            )
            quality_flags = self._quality_flags(
                raw=raw,
                license_name=license_result.license,
                has_paper_clues=bool(extract_paper_clues(raw)),
                paper_status=paper_status,
            )
            extensions = sorted(
                {record.extension.upper() for record in files if record.extension}
            )
            known_sizes = [record.size_bytes for record in files if record.size_bytes is not None]
            urls = dataset_urls(dataset_id, _optional_string(raw.get("sha")))
            master_records.append(
                DatasetRecord(
                    dataset_id=dataset_id,
                    dataset_name=dataset_id.rsplit("/", maxsplit=1)[-1],
                    author=_optional_string(raw.get("author")),
                    dataset_url=_optional_string(raw.get("dataset_url"))
                    or urls["dataset_url"],
                    files_url=_optional_string(raw.get("files_url")) or urls["files_url"],
                    viewer_url=_optional_string(raw.get("viewer_url"))
                    or urls["viewer_url"],
                    created_at=_as_datetime(raw.get("created_at")),
                    last_modified=_as_datetime(raw.get("last_modified")),
                    sha=_optional_string(raw.get("sha")),
                    description=_optional_string(raw.get("description")),
                    tags=tags,
                    license=license_result.license,
                    license_raw=license_result.license_raw,
                    commercial_use_restriction=(
                        license_result.commercial_use_restriction
                    ),
                    redistribution_restriction=(
                        license_result.redistribution_restriction
                    ),
                    special_license_terms=license_result.special_license_terms,
                    modalities=classification.modalities,
                    use_cases=classification.use_cases,
                    tasks=classification.tasks,
                    domains=classification.domains,
                    data_formats=classification.data_formats,
                    classification_source=classification.classification_source,
                    gated=raw.get("gated"),
                    private=bool(raw.get("private", False)),
                    disabled=_optional_bool(raw.get("disabled")),
                    likes=_optional_int(raw.get("likes")),
                    downloads=_optional_int(raw.get("downloads")),
                    downloads_all_time=_optional_int(raw.get("downloads_all_time")),
                    downloads_rank=rankings.get((dataset_id, "downloads")),
                    downloads_all_time_rank=rankings.get(
                        (dataset_id, "downloads_all_time")
                    ),
                    max_paper_citation_count=(
                        _optional_int(top_paper.get("citation_count"))
                        if top_paper
                        else None
                    ),
                    citation_rank=rankings.get(
                        (dataset_id, "max_paper_citation_count")
                    ),
                    citation_source=(
                        _optional_string(top_paper.get("citation_source"))
                        if top_paper
                        else None
                    ),
                    top_paper_id=(
                        _optional_string(top_paper.get("paper_id")) if top_paper else None
                    ),
                    top_paper_title=(
                        _optional_string(top_paper.get("paper_title"))
                        if top_paper
                        else None
                    ),
                    paper_match_status=paper_status,
                    associated_paper_count=len(paper_rows),
                    file_count=len(files),
                    used_storage_bytes=_optional_int(raw.get("used_storage")),
                    total_file_size_bytes=sum(known_sizes) if known_sizes else None,
                    file_extensions=extensions,
                    **membership.model_dump(),
                    ranking_date=now.date(),
                    collection_status=(
                        CollectionStatus.PARTIAL
                        if quality_flags
                        else CollectionStatus.SUCCESS
                    ),
                    quality_flags=quality_flags,
                    fetched_at=_as_datetime(raw.get("fetched_at")) or now,
                )
            )
        self.store.upsert_models(run_id, "hf_dataset_files", file_records)
        self.store.upsert_models(run_id, "hf_datasets_master", master_records)

    def _export_impl(self, run_id: str) -> None:
        from hf_dataset_catalog.exporter import CatalogExporter

        CatalogExporter(self.store, self.settings.output_dir).export(run_id)

    def _run_datetime(self, run_id: str) -> datetime:
        rows = self.store.query_dicts(
            "SELECT started_at FROM collection_runs WHERE collection_run_id = ?",
            [run_id],
        )
        if not rows:
            raise ValueError(f"unknown collection run: {run_id}")
        value = rows[0]["started_at"]
        parsed = _as_datetime(value)
        if parsed is None:
            raise ValueError(f"run has invalid started_at: {run_id}")
        return parsed

    def _raw_payloads(self, run_id: str) -> list[dict[str, Any]]:
        payloads = []
        for row in self.store.raw_rows(run_id):
            payload = _as_mapping(row["payload"])
            payloads.append({"dataset_id": row["dataset_id"], **payload})
        return payloads

    def _checkpoint_list(self, run_id: str, stage: str, key: str) -> list[str]:
        value = self.store.get_checkpoint(run_id, stage, key)
        if not isinstance(value, list):
            return []
        return [str(item) for item in value]

    def _checkpoint_mapping(
        self,
        run_id: str,
        stage: str,
        key: str,
    ) -> dict[str, Any]:
        value = self.store.get_checkpoint(run_id, stage, key)
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _quality_flags(
        *,
        raw: Mapping[str, Any],
        license_name: str,
        has_paper_clues: bool,
        paper_status: MatchStatus,
    ) -> list[str]:
        flags: list[QualityFlag] = []
        card_data = raw.get("card_data")
        if not isinstance(card_data, Mapping) or not card_data:
            flags.append(QualityFlag.CARD_MISSING)
        if license_name == "Unknown":
            flags.append(QualityFlag.LICENSE_UNKNOWN)
        if has_paper_clues and paper_status != MatchStatus.CONFIRMED:
            flags.append(QualityFlag.PAPER_UNMATCHED)
        if _as_datetime(raw.get("created_at")) is None:
            flags.append(QualityFlag.METADATA_INCOMPLETE)
        return [flag.value for flag in flags]


def _aggregate_match_status(rows: Iterable[Mapping[str, Any]]) -> MatchStatus:
    statuses = {str(row["match_status"]) for row in rows}
    if MatchStatus.CONFIRMED.value in statuses:
        return MatchStatus.CONFIRMED
    if MatchStatus.REVIEW.value in statuses:
        return MatchStatus.REVIEW
    if statuses:
        return MatchStatus.UNMATCHED
    return MatchStatus.NO_CLUE


def _clue_relation_id(clue: PaperClue) -> str:
    return f"clue-{sha256(clue.cache_key().encode()).hexdigest()[:20]}"


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(decoded) if isinstance(decoded, Mapping) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> list[Any]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return list(decoded) if isinstance(decoded, list) else []
    return list(value) if isinstance(value, (list, tuple)) else []


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return isoparse(value)
        except ValueError:
            return None
    return None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None
