from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, date, datetime
from enum import Enum
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from huggingface_hub import HfApi
from huggingface_hub.hf_api import ExpandDatasetProperty_T

from hf_dataset_catalog.models import FileRecord, FileType
from hf_dataset_catalog.storage import CatalogStore

HF_BASE_URL = "https://huggingface.co"

DATASET_EXPAND_FIELDS: list[ExpandDatasetProperty_T] = [
    "author",
    "cardData",
    "citation",
    "createdAt",
    "description",
    "disabled",
    "downloads",
    "downloadsAllTime",
    "gated",
    "lastModified",
    "likes",
    "mainSize",
    "paperswithcode_id",
    "private",
    "sha",
    "siblings",
    "tags",
]

DATASET_ATTRIBUTES = {
    "author": ("author",),
    "card_data": ("card_data", "cardData"),
    "citation": ("citation",),
    "created_at": ("created_at", "createdAt"),
    "description": ("description",),
    "disabled": ("disabled",),
    "downloads": ("downloads",),
    "downloads_all_time": ("downloads_all_time", "downloadsAllTime"),
    "gated": ("gated",),
    "last_modified": ("last_modified", "lastModified"),
    "likes": ("likes",),
    "main_size": ("main_size", "mainSize"),
    "paperswithcode_id": ("paperswithcode_id",),
    "private": ("private",),
    "sha": ("sha",),
    "siblings": ("siblings",),
    "tags": ("tags",),
    "used_storage": ("used_storage", "usedStorage"),
}

FILE_TYPES = {
    ".arrow": FileType.TABLE,
    ".avro": FileType.STRUCTURED,
    ".csv": FileType.TABLE,
    ".flac": FileType.AUDIO,
    ".gif": FileType.IMAGE,
    ".jpeg": FileType.IMAGE,
    ".jpg": FileType.IMAGE,
    ".json": FileType.STRUCTURED,
    ".jsonl": FileType.STRUCTURED,
    ".mp3": FileType.AUDIO,
    ".mp4": FileType.VIDEO,
    ".parquet": FileType.PARQUET,
    ".png": FileType.IMAGE,
    ".tsv": FileType.TABLE,
    ".txt": FileType.TEXT,
    ".wav": FileType.AUDIO,
    ".webm": FileType.VIDEO,
    ".webp": FileType.IMAGE,
    ".xml": FileType.STRUCTURED,
    ".yaml": FileType.STRUCTURED,
    ".yml": FileType.STRUCTURED,
    ".zip": FileType.ARCHIVE,
}


@dataclass(frozen=True)
class BatchWriteResult:
    path: Path
    checksum: str
    record_count: int
    skipped: bool = False


class HuggingFaceSource:
    def __init__(self, api: Any | None = None, token: str | None = None) -> None:
        self.api = api or HfApi(token=token)

    def iter_datasets(
        self,
        *,
        fetched_at: datetime | None = None,
        on_rejected: Callable[[str, str], None] | None = None,
    ) -> Iterator[dict[str, Any]]:
        timestamp = fetched_at or datetime.now(UTC)
        rows = self.api.list_datasets(limit=None, expand=DATASET_EXPAND_FIELDS)
        for info in rows:
            record = adapt_dataset_info(info, fetched_at=timestamp)
            if record["private"] is True:
                if on_rejected:
                    on_rejected(record["dataset_id"], "private dataset excluded")
                continue
            yield record


def adapt_dataset_info(info: Any, *, fetched_at: datetime) -> dict[str, Any]:
    raw = _object_mapping(info)
    dataset_id = _first_value(info, raw, ("id", "dataset_id"))
    if not dataset_id:
        raise ValueError("Hugging Face dataset row has no id")

    record: dict[str, Any] = {
        "dataset_id": str(dataset_id),
        "fetched_at": fetched_at,
    }
    for target, candidates in DATASET_ATTRIBUTES.items():
        record[target] = _first_value(info, raw, candidates)

    record.update(dataset_urls(str(dataset_id), record.get("sha")))
    return record


def dataset_urls(dataset_id: str, revision: str | None = None) -> dict[str, str]:
    encoded_id = quote(dataset_id)
    encoded_revision = quote(revision or "main", safe="")
    root = f"{HF_BASE_URL}/datasets/{encoded_id}"
    return {
        "dataset_url": root,
        "files_url": f"{root}/tree/{encoded_revision}",
        "viewer_url": f"{root}/viewer",
    }


def adapt_file_info(
    dataset_id: str,
    revision: str | None,
    file_info: Any,
    *,
    fetched_at: datetime,
) -> FileRecord:
    raw = _object_mapping(file_info)
    file_path_value = _first_value(file_info, raw, ("rfilename", "path", "file_path"))
    if not file_path_value:
        raise ValueError(f"file row for {dataset_id} has no path")

    file_path = str(file_path_value)
    extension_with_dot = PurePosixPath(file_path.lower()).suffix
    encoded_id = quote(dataset_id)
    encoded_revision = quote(revision or "main", safe="")
    encoded_path = quote(file_path)
    lfs = _first_value(file_info, raw, ("lfs",))
    return FileRecord(
        dataset_id=dataset_id,
        revision=revision,
        file_path=file_path,
        file_name=PurePosixPath(file_path).name,
        extension=extension_with_dot.removeprefix(".") or None,
        file_type=FILE_TYPES.get(extension_with_dot, FileType.OTHER),
        size_bytes=_optional_int(_first_value(file_info, raw, ("size",))),
        is_lfs=True if lfs is not None else None,
        file_page_url=(
            f"{HF_BASE_URL}/datasets/{encoded_id}/blob/{encoded_revision}/{encoded_path}"
        ),
        resolve_url=(
            f"{HF_BASE_URL}/datasets/{encoded_id}/resolve/{encoded_revision}/{encoded_path}"
        ),
        fetched_at=fetched_at,
    )


def write_raw_batch(
    output_dir: Path,
    run_id: str,
    batch_number: int,
    records: Sequence[Mapping[str, Any]],
    *,
    stage: str = "enumerate",
) -> tuple[Path, str]:
    batch_dir = output_dir / run_id / stage
    batch_dir.mkdir(parents=True, exist_ok=True)
    final_path = batch_dir / f"{batch_number:08d}.jsonl"
    temporary_path = final_path.with_suffix(f"{final_path.suffix}.tmp")

    digest = sha256()
    with temporary_path.open("wb") as handle:
        for record in records:
            line = (
                json.dumps(
                    record,
                    default=_json_default,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            handle.write(line)
            digest.update(line)
        handle.flush()
        os.fsync(handle.fileno())

    temporary_path.replace(final_path)
    return final_path, digest.hexdigest()


def persist_raw_batch(
    *,
    store: CatalogStore,
    output_dir: Path,
    run_id: str,
    batch_number: int,
    records: Sequence[Mapping[str, Any]],
    stage: str = "enumerate",
) -> BatchWriteResult:
    existing = next(
        (
            batch
            for batch in store.raw_batches(run_id)
            if batch["stage"] == stage and batch["batch_number"] == batch_number
        ),
        None,
    )
    if existing:
        return BatchWriteResult(
            path=Path(existing["path"]),
            checksum=str(existing["checksum"]),
            record_count=int(existing["record_count"]),
            skipped=True,
        )

    path, checksum = write_raw_batch(
        output_dir,
        run_id,
        batch_number,
        records,
        stage=stage,
    )
    store.register_raw_batch(
        run_id,
        stage,
        batch_number,
        path,
        len(records),
        checksum,
    )
    return BatchWriteResult(
        path=path,
        checksum=checksum,
        record_count=len(records),
    )


def _object_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        converted = asdict(value)
        if isinstance(converted, Mapping):
            return converted
    attributes = getattr(value, "__dict__", None)
    return attributes if isinstance(attributes, Mapping) else {}


def _first_value(
    value: Any,
    raw: Mapping[str, Any],
    candidates: Iterable[str],
) -> Any:
    for candidate in candidates:
        if candidate in raw and raw[candidate] is not None:
            return raw[candidate]
        attribute = getattr(value, candidate, None)
        if attribute is not None:
            return attribute
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
