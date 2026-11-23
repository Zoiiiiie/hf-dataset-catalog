# Hugging Face Dataset Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable Python CLI that catalogs public Hugging Face dataset metadata, selects the PRD-defined union, resolves associated paper citations, and exports normalized Parquet/CSV/JSONL deliverables without downloading repository file contents.

**Architecture:** Use a `src`-layout Python package with dependency-injected API clients. DuckDB stores normalized run state, rankings, checkpoints, failures, and metrics; raw Hugging Face responses are written as atomic JSONL batches. Pure transformation modules handle ranking, union membership, paper normalization, classification, and reports so that network behavior and business rules can be tested separately.

**Tech Stack:** Python 3.11+, `huggingface_hub`, `duckdb`, `pyarrow`, `pydantic`, `pydantic-settings`, `httpx`, `tenacity`, `typer`, `PyYAML`, `rapidfuzz`, `python-dateutil`, `pytest`, `respx`, `ruff`, `mypy`

---

## File Map

- `pyproject.toml`: package metadata, dependencies, CLI entry point, and tool configuration.
- `config/default.yaml`: PRD defaults for time window, rankings, batches, retries, and confidence thresholds.
- `src/hf_dataset_catalog/config.py`: typed YAML/environment configuration.
- `src/hf_dataset_catalog/models.py`: canonical records and enums shared across stages.
- `src/hf_dataset_catalog/storage.py`: DuckDB schema, run state, raw batch manifest, upserts, and exports.
- `src/hf_dataset_catalog/http.py`: retry-aware async JSON client.
- `src/hf_dataset_catalog/hf_client.py`: Hugging Face enumeration and repository metadata adaptation.
- `src/hf_dataset_catalog/papers.py`: paper clue extraction, identifier normalization, provider matching, and citation selection.
- `src/hf_dataset_catalog/selection.py`: calendar cutoff, stable rankings, and OR-union flags.
- `src/hf_dataset_catalog/classification.py`: deterministic modality, use-case, task, domain, format, and license rules.
- `src/hf_dataset_catalog/report.py`: collection-quality metrics.
- `src/hf_dataset_catalog/exporter.py`: schema-controlled Parquet, CSV, JSONL, and report output.
- `src/hf_dataset_catalog/pipeline.py`: stage orchestration and resume behavior.
- `src/hf_dataset_catalog/cli.py`: `collect`, `resume`, `export`, `validate`, and `status`.
- `tests/`: focused unit and integration tests with mocked APIs.
- `.github/workflows/collect.yml`: manual and scheduled execution example.
- `README.md`: installation, credentials, commands, outputs, and operational limits.

### Task 1: Package Scaffold and Typed Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `config/default.yaml`
- Create: `src/hf_dataset_catalog/__init__.py`
- Create: `src/hf_dataset_catalog/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing configuration tests**

```python
from pathlib import Path

from hf_dataset_catalog.config import Settings


def test_default_settings_match_prd(tmp_path: Path) -> None:
    settings = Settings(output_dir=tmp_path)
    assert settings.recent_years == 5
    assert settings.top_downloads == 10_000
    assert settings.top_downloads_all_time == 10_000
    assert settings.top_citations == 10_000
    assert settings.confirm_threshold == 0.95
    assert settings.review_threshold == 0.80


def test_review_threshold_must_not_exceed_confirm_threshold() -> None:
    with pytest.raises(ValueError):
        Settings(review_threshold=0.96, confirm_threshold=0.95)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `pytest tests/test_config.py -q`

Expected: collection fails because `hf_dataset_catalog.config` does not exist.

- [ ] **Step 3: Implement package metadata and settings**

Create a Pydantic `Settings` model with:

```python
class Settings(BaseModel):
    output_dir: Path = Path("data/output")
    state_db: Path = Path("data/state/catalog.duckdb")
    recent_years: int = Field(default=5, ge=1)
    top_downloads: int = Field(default=10_000, ge=1)
    top_downloads_all_time: int = Field(default=10_000, ge=1)
    top_citations: int = Field(default=10_000, ge=1)
    batch_size: int = Field(default=500, ge=1)
    max_retries: int = Field(default=6, ge=0)
    request_interval_seconds: float = Field(default=0.1, ge=0)
    confirm_threshold: float = Field(default=0.95, ge=0, le=1)
    review_threshold: float = Field(default=0.80, ge=0, le=1)
    hf_token: SecretStr | None = None
    semantic_scholar_api_key: SecretStr | None = None
    openalex_api_key: SecretStr | None = None

    @model_validator(mode="after")
    def validate_thresholds(self) -> "Settings":
        if self.review_threshold > self.confirm_threshold:
            raise ValueError("review_threshold must be <= confirm_threshold")
        return self
```

Expose `load_settings(path: Path | None, overrides: dict[str, Any]) -> Settings`,
merge YAML with environment secrets, and register
`hf-catalog = "hf_dataset_catalog.cli:app"` in `pyproject.toml`.

- [ ] **Step 4: Run tests and static checks**

Run: `pytest tests/test_config.py -q && ruff check src tests`

Expected: all tests and lint checks pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml config/default.yaml src/hf_dataset_catalog tests/test_config.py
git commit -m "feat: scaffold catalog package and configuration"
```

### Task 2: Canonical Models and DuckDB State

**Files:**
- Create: `src/hf_dataset_catalog/models.py`
- Create: `src/hf_dataset_catalog/storage.py`
- Create: `tests/test_storage.py`

- [ ] **Step 1: Write failing storage tests**

```python
def test_raw_batch_registration_is_idempotent(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "state.duckdb")
    store.initialize()
    store.register_raw_batch("run-1", "enumerate", 0, tmp_path / "0.jsonl", 2)
    store.register_raw_batch("run-1", "enumerate", 0, tmp_path / "0.jsonl", 2)
    assert store.raw_batch_count("run-1") == 1


def test_metric_snapshot_upserts_same_day(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "state.duckdb")
    store.initialize()
    first = MetricRecord(dataset_id="org/data", snapshot_date=date(2026, 11, 23), downloads=10)
    second = first.model_copy(update={"downloads": 12})
    store.upsert_metrics("run-1", [first, second])
    assert store.get_metrics("org/data")[0]["downloads"] == 12
```

- [ ] **Step 2: Verify the tests fail**

Run: `pytest tests/test_storage.py -q`

Expected: imports fail because models and storage are absent.

- [ ] **Step 3: Implement canonical records**

Define string enums for collection status, match status, quality flags, and
file type. Define Pydantic records for:

```python
class RawDatasetRecord(BaseModel):
    dataset_id: str
    payload: dict[str, Any]
    fetched_at: datetime


class MetricRecord(BaseModel):
    dataset_id: str
    snapshot_date: date
    downloads: int | None = None
    downloads_all_time: int | None = None
    likes: int | None = None
    citation_count: int | None = None
    citation_source: str | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

Also define complete `DatasetRecord`, `PaperRecord`, `FileRecord`,
`RankingRecord`, and `RunRecord` models matching the approved design schema.

- [ ] **Step 4: Implement the DuckDB schema and idempotent methods**

`CatalogStore.initialize()` executes versioned schema migrations. The first
migration creates the exact approved columns from the design document and
these required keys:

```sql
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
```

Create `hf_datasets_master`, `hf_dataset_papers`, `hf_dataset_files`, and
`hf_dataset_metrics` with every field and type listed in design sections
6.1-6.4. Their primary keys are respectively `(collection_run_id,
dataset_id)`, `(collection_run_id, dataset_id, paper_id)`,
`(collection_run_id, dataset_id, file_path)`, and `(dataset_id,
snapshot_date)`. Implement transactions for run creation, batch registration,
checkpoint updates, failures, model upserts, status summaries, and
table-to-Arrow reads.

- [ ] **Step 5: Run focused and full tests**

Run: `pytest tests/test_storage.py -q && pytest -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/hf_dataset_catalog/models.py src/hf_dataset_catalog/storage.py tests/test_storage.py
git commit -m "feat: add canonical schemas and DuckDB state"
```

### Task 3: Selection and Classification Rules

**Files:**
- Create: `src/hf_dataset_catalog/selection.py`
- Create: `src/hf_dataset_catalog/classification.py`
- Create: `tests/test_selection.py`
- Create: `tests/test_classification.py`

- [ ] **Step 1: Write failing cutoff and ranking tests**

```python
def test_calendar_cutoff_handles_leap_day() -> None:
    now = datetime(2024, 2, 29, 12, tzinfo=timezone.utc)
    assert calendar_cutoff(now, 5) == datetime(2019, 2, 28, 12, tzinfo=timezone.utc)


def test_rank_is_null_last_and_stable_by_dataset_id() -> None:
    rows = [
        {"dataset_id": "b/data", "downloads": 7},
        {"dataset_id": "a/data", "downloads": 7},
        {"dataset_id": "c/data", "downloads": None},
    ]
    ranked = rank_metric(rows, "downloads", top_n=3)
    assert [row.dataset_id for row in ranked] == ["a/data", "b/data"]


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
```

- [ ] **Step 2: Write failing classification tests**

```python
def test_multivalue_classification_uses_tags_and_extensions() -> None:
    result = classify_dataset(
        tags=["task_categories:image-classification", "language:en"],
        file_paths=["data/train.parquet", "images/0001.jpg"],
        card_data={},
    )
    assert result.modalities == ["Image", "Tabular"]
    assert result.data_formats == ["JPG", "PARQUET"]
    assert "tags" in result.classification_source
    assert "file_extensions" in result.classification_source


def test_public_access_does_not_imply_commercial_permission() -> None:
    result = classify_license(card_data={}, tags=[])
    assert result.license == "Unknown"
    assert result.commercial_use_restriction is None
```

- [ ] **Step 3: Verify both test modules fail**

Run: `pytest tests/test_selection.py tests/test_classification.py -q`

Expected: module import failures.

- [ ] **Step 4: Implement pure selection functions**

```python
def calendar_cutoff(now: datetime, years: int) -> datetime:
    return now - relativedelta(years=years)


def rank_metric(
    rows: Iterable[Mapping[str, Any]],
    metric: str,
    top_n: int,
    ranking_date: date | None = None,
) -> list[RankingRecord]:
    valid = (row for row in rows if row.get(metric) is not None)
    ordered = sorted(valid, key=lambda row: (-int(row[metric]), str(row["dataset_id"])))
    return [
        RankingRecord(
            dataset_id=row["dataset_id"],
            metric=metric,
            metric_value=row[metric],
            rank=index,
            ranking_date=ranking_date or datetime.now(timezone.utc).date(),
        )
        for index, row in enumerate(ordered[:top_n], start=1)
    ]
```

Implement exact recent filtering and `merge_selection_sets` with four booleans.

- [ ] **Step 5: Implement deterministic classification**

Create ordered mappings from HF tag prefixes and file extensions to the PRD
modalities, use cases, tasks, domains, and formats. Deduplicate and sort values
for reproducibility. Keep the evidence sources. License output is based only on
structured `license` fields/tags and explicit restriction fields.

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_selection.py tests/test_classification.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/hf_dataset_catalog/selection.py src/hf_dataset_catalog/classification.py tests/test_selection.py tests/test_classification.py
git commit -m "feat: implement selection and classification rules"
```

### Task 4: Hugging Face Enumeration and Atomic Raw Batches

**Files:**
- Create: `src/hf_dataset_catalog/hf_client.py`
- Create: `tests/test_hf_client.py`

- [ ] **Step 1: Write failing adapter and raw-batch tests**

```python
def test_adapt_dataset_info_preserves_distinct_download_metrics() -> None:
    info = SimpleNamespace(
        id="org/data",
        downloads=11,
        downloads_all_time=99,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    record = adapt_dataset_info(info, fetched_at=FIXED_NOW)
    assert record["downloads"] == 11
    assert record["downloads_all_time"] == 99


def test_write_raw_batch_is_atomic_and_deterministic(tmp_path: Path) -> None:
    path, checksum = write_raw_batch(
        tmp_path, "run-1", 3, [{"id": "a"}, {"id": "b"}]
    )
    assert path.name == "00000003.jsonl"
    assert not list(path.parent.glob("*.tmp"))
    assert checksum == sha256(path.read_bytes()).hexdigest()
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_hf_client.py -q`

Expected: `hf_client` import fails.

- [ ] **Step 3: Implement the Hugging Face source**

`HuggingFaceSource.iter_datasets()` calls:

```python
self.api.list_datasets(
    limit=None,
    expand=[
        "author", "cardData", "citation", "createdAt", "description",
        "disabled", "downloads", "downloadsAllTime", "gated",
        "lastModified", "likes", "mainSize", "paperswithcode_id",
        "private", "sha", "siblings", "tags", "usedStorage",
    ],
)
```

Adapt SDK objects through `asdict()` when available and a controlled attribute
mapping otherwise. Construct dataset, files, viewer, file-page, and resolve
URLs without requesting those URLs. Reject private rows from normalized
selection while preserving a failure/audit event.

- [ ] **Step 4: Implement atomic batching**

Write UTF-8 JSONL to `<batch>.jsonl.tmp`, flush and `fsync`, calculate SHA-256,
then use `Path.replace()` to publish `<batch>.jsonl`. Register the batch only
after publication. Skip the API-side batch write when the deterministic batch
key is already in the manifest.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_hf_client.py -q && pytest -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/hf_dataset_catalog/hf_client.py tests/test_hf_client.py
git commit -m "feat: enumerate Hugging Face metadata in atomic batches"
```

### Task 5: Retry Client and Academic Paper Resolution

**Files:**
- Create: `src/hf_dataset_catalog/http.py`
- Create: `src/hf_dataset_catalog/papers.py`
- Create: `tests/test_http.py`
- Create: `tests/test_papers.py`

- [ ] **Step 1: Write failing identifier and confidence tests**

```python
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://doi.org/10.1000/XYZ", "10.1000/xyz"),
        ("doi:10.1000/xyz.", "10.1000/xyz"),
    ],
)
def test_normalize_doi(raw: str, expected: str) -> None:
    assert normalize_doi(raw) == expected


def test_review_match_does_not_contribute_citations() -> None:
    candidate = PaperCandidate(title="Example", match_confidence=0.91, citation_count=50)
    result = apply_match_thresholds(candidate, confirm=0.95, review=0.80)
    assert result.match_status == MatchStatus.REVIEW
    assert result.effective_citation_count is None


def test_select_top_paper_does_not_sum_citations() -> None:
    papers = [
        PaperRecord(dataset_id="d", paper_id="1", citation_count=10, match_status="confirmed"),
        PaperRecord(dataset_id="d", paper_id="2", citation_count=20, match_status="confirmed"),
    ]
    assert select_top_paper(papers).paper_id == "2"
```

- [ ] **Step 2: Write failing retry test**

Use `respx` to return 429 with `Retry-After: 0`, then 200. Assert
`RetryingJsonClient.get_json()` returns the successful payload and records one
retry.

- [ ] **Step 3: Verify failure**

Run: `pytest tests/test_http.py tests/test_papers.py -q`

Expected: modules are missing.

- [ ] **Step 4: Implement the HTTP client**

Implement async GET/POST JSON methods using `httpx.AsyncClient`. Retry network
errors, 429, and 5xx responses with bounded exponential backoff and jitter.
Honor numeric `Retry-After`. Do not retry other 4xx responses. Return a typed
response carrying payload, status, retry count, source, and fetch time.

- [ ] **Step 5: Implement paper clue extraction**

Parse only structured values already returned by the HF API:

```python
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
ARXIV_PATTERN = re.compile(r"(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)


def extract_paper_clues(dataset: Mapping[str, Any]) -> list[PaperClue]:
    values = [
        dataset.get("citation"),
        dataset.get("paperswithcode_id"),
        dataset.get("card_data"),
    ]
    return deduplicate_clues(parse_structured_values(values))
```

Do not fetch README or repository file content.

- [ ] **Step 6: Implement provider resolvers**

Use Semantic Scholar identifier detail endpoints first. Use paper search for
title matching only when no exact identifier resolves. Use OpenAlex DOI/arXiv
lookup and title search as fallback. Normalize both provider responses to
`PaperCandidate`. Calculate title similarity with RapidFuzz, require author
evidence when available, apply thresholds, cache by normalized clue, and retain
review candidates with null effective citation counts.

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_http.py tests/test_papers.py -q && pytest -q`

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/hf_dataset_catalog/http.py src/hf_dataset_catalog/papers.py tests/test_http.py tests/test_papers.py
git commit -m "feat: resolve academic papers with retry-aware providers"
```

### Task 6: Pipeline Orchestration and Resume

**Files:**
- Create: `src/hf_dataset_catalog/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write a failing end-to-end mocked pipeline test**

```python
def test_pipeline_builds_or_union_and_resumes_without_duplicates(tmp_path: Path) -> None:
    source = FakeHuggingFaceSource(
        [
            dataset("recent", created_at="2026-01-01", downloads=1),
            dataset("popular", created_at="2018-01-01", downloads=100),
            dataset("cited", created_at="2018-01-01", downloads=0, doi="10.1/x"),
        ]
    )
    resolver = FakePaperResolver({"10.1/x": confirmed_paper(citations=200)})
    pipeline = CatalogPipeline(test_settings(tmp_path), source, resolver)
    run_id = pipeline.collect(now=datetime(2026, 11, 23, tzinfo=timezone.utc))
    pipeline.resume(run_id)
    rows = pipeline.store.master_rows(run_id)
    assert {row["dataset_id"] for row in rows} == {"recent", "popular", "cited"}
    assert len(rows) == 3
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_pipeline.py -q`

Expected: `pipeline` module is missing.

- [ ] **Step 3: Implement explicit stages**

Create these `CatalogPipeline` methods with the following orchestration:

```python
def collect(self, now: datetime | None = None) -> str:
    run_id = self.store.create_run(self.settings, now=now)
    return self.resume(run_id)


def resume(self, run_id: str) -> str:
    self.enumerate(run_id)
    self.select_downloads_and_recent(run_id)
    asyncio.run(self.resolve_papers(run_id))
    self.select_citations_and_merge(run_id)
    self.enrich(run_id)
    self.export(run_id)
    self.store.complete_run(run_id)
    return run_id


def enumerate(self, run_id: str) -> None:
    self._run_stage_once(run_id, "enumerate", self._enumerate_impl)


def select_downloads_and_recent(self, run_id: str) -> None:
    self._run_stage_once(run_id, "select", self._select_impl)


async def resolve_papers(self, run_id: str) -> None:
    if not self.store.stage_complete(run_id, "papers"):
        await self._resolve_papers_impl(run_id)
        self.store.mark_stage_complete(run_id, "papers")


def select_citations_and_merge(self, run_id: str) -> None:
    self._run_stage_once(run_id, "merge", self._merge_impl)


def enrich(self, run_id: str) -> None:
    self._run_stage_once(run_id, "enrich", self._enrich_impl)


def export(self, run_id: str) -> Path:
    self._run_stage_once(run_id, "export", self._export_impl)
    return self.settings.output_dir / run_id
```

Each stage checks its checkpoint, writes records transactionally, records
failures per dataset or paper clue, and marks completion only after durable
output. Citation resolution runs only for datasets with structured clues.
Detailed enrichment and file normalization run only for the final union.

- [ ] **Step 4: Implement quality statuses**

Derive flags including `PAPER_UNMATCHED`, `CARD_MISSING`,
`LICENSE_UNKNOWN`, and `METADATA_INCOMPLETE`. A dataset is `SUCCESS` when no
flags exist, `PARTIAL` when fields are missing, and `API_ERROR` only when its
required HF metadata request failed.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_pipeline.py -q && pytest -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/hf_dataset_catalog/pipeline.py tests/test_pipeline.py
git commit -m "feat: orchestrate resumable catalog collection"
```

### Task 7: Exports, Validation, and Collection Report

**Files:**
- Create: `src/hf_dataset_catalog/report.py`
- Create: `src/hf_dataset_catalog/exporter.py`
- Create: `tests/test_exporter.py`
- Create: `tests/test_report.py`

- [ ] **Step 1: Write failing report tests**

```python
def test_report_distinguishes_paper_coverage_from_collection_failure() -> None:
    report = build_report(
        enumerated=100,
        recent=40,
        top_download=10,
        top_download_all_time=10,
        top_citation=5,
        union_count=52,
        paper_clue_datasets=20,
        paper_matched_datasets=10,
        api_failures=2,
        required_field_cells=500,
        populated_required_field_cells=450,
    )
    assert report.paper_match_rate == 0.5
    assert report.api_failure_rate == 0.02
    assert report.field_completeness_rate == 0.9
```

- [ ] **Step 2: Write failing export-schema test**

Export a two-row in-memory store and assert every approved master, paper,
file, and metrics column exists in both Parquet and CSV. Assert null citation
counts remain null after a Parquet round trip.

- [ ] **Step 3: Verify failure**

Run: `pytest tests/test_exporter.py tests/test_report.py -q`

Expected: report/exporter modules are missing.

- [ ] **Step 4: Implement reports**

Define a typed `CollectionReport` with all PRD minimum metrics plus retries and
quality-flag counts. Rates use explicit denominators and return null when the
denominator is zero.

- [ ] **Step 5: Implement atomic exports**

Read DuckDB tables as Arrow, project columns in declared order, and write:

```python
pq.write_table(table, temporary_parquet, compression="zstd")
pacsv.write_csv(table, temporary_csv)
temporary_parquet.replace(final_parquet)
temporary_csv.replace(final_csv)
```

Build final raw JSONL by concatenating manifest batches in batch-number order
to a temporary file, then atomically replace the final file. Validate primary
keys, membership flags, required URLs, rank uniqueness, and no private rows
before publishing.

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_exporter.py tests/test_report.py -q && pytest -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/hf_dataset_catalog/report.py src/hf_dataset_catalog/exporter.py tests/test_exporter.py tests/test_report.py
git commit -m "feat: export validated catalog deliverables"
```

### Task 8: CLI, Documentation, and Automation

**Files:**
- Create: `src/hf_dataset_catalog/cli.py`
- Create: `tests/test_cli.py`
- Create: `README.md`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `.github/workflows/collect.yml`

- [ ] **Step 1: Write failing CLI tests**

```python
def test_collect_help_lists_safe_scope() -> None:
    result = runner.invoke(app, ["collect", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.stdout
    assert "--max-datasets" in result.stdout


def test_status_unknown_run_is_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["status", "--state-db", str(tmp_path / "x.duckdb"), "missing"])
    assert result.exit_code != 0
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_cli.py -q`

Expected: CLI module is missing.

- [ ] **Step 3: Implement CLI commands**

Use Typer to expose:

```text
hf-catalog collect [--config PATH] [--max-datasets N] [--skip-papers]
hf-catalog resume RUN_ID [--config PATH]
hf-catalog export RUN_ID [--output-dir PATH]
hf-catalog validate RUN_ID
hf-catalog status [RUN_ID] [--latest] [--format table|run-id]
```

`--max-datasets` is an explicit smoke-test control and is never used by the
default/full path. Print the run ID and output directory on success. Return
nonzero on invalid schema or an unrecoverable stage failure.

- [ ] **Step 4: Document operation**

README includes installation with `uv` and `pip`, optional tokens, public-only
scope, no-file-download guarantee, all commands, output schemas, full-run cost
expectations, resuming, GitHub Actions, and a small smoke-test command:

```bash
hf-catalog collect --max-datasets 20 --skip-papers
```

Ignore `.env`, `data/state`, `data/raw`, and generated output data. Keep empty
directories out of Git.

- [ ] **Step 5: Add automation**

Create a workflow with `workflow_dispatch` and a monthly cron. Install with
`uv sync --frozen`, run `hf-catalog collect`, and upload the generated
deliverables as an Actions artifact. Read provider tokens only from GitHub
Secrets. Set `timeout-minutes` and `concurrency` to avoid overlapping runs.

- [ ] **Step 6: Run tests and checks**

Run: `pytest -q && ruff check src tests && mypy src`

Expected: all tests and checks pass.

- [ ] **Step 7: Commit**

```bash
git add src/hf_dataset_catalog/cli.py tests/test_cli.py README.md .env.example .gitignore .github/workflows/collect.yml
git commit -m "feat: add catalog CLI and automation"
```

### Task 9: Online Smoke Test and Release Verification

**Files:**
- Create: `tests/test_live_smoke.py`
- Modify: `README.md`

- [ ] **Step 1: Add opt-in live test**

```python
@pytest.mark.live
def test_live_hf_sample_has_required_api_fields() -> None:
    source = HuggingFaceSource(Settings())
    row = next(source.iter_datasets(limit=1))
    assert row["dataset_id"]
    assert row["dataset_url"].startswith("https://huggingface.co/datasets/")
    assert "downloads" in row
    assert "downloads_all_time" in row
```

Skip unless `RUN_LIVE_TESTS=1`.

- [ ] **Step 2: Run all offline verification**

Run:

```bash
pytest -q
ruff check src tests
ruff format --check src tests
mypy src
python -m build
```

Expected: every command exits zero.

- [ ] **Step 3: Run the live API compatibility test**

Run: `RUN_LIVE_TESTS=1 pytest tests/test_live_smoke.py -q -m live`

Expected: one live test passes without downloading repository files.

- [ ] **Step 4: Run a bounded end-to-end collection**

Run:

```bash
hf-catalog collect --max-datasets 20 --skip-papers
hf-catalog validate "$(hf-catalog status --latest --format run-id)"
```

Expected: all required deliverables exist, validation passes, and the report
states `hf_enumerated_count = 20`.

- [ ] **Step 5: Inspect repository scope**

Run: `git status --short && git diff --check && git log --oneline --decorate -10`

Expected: no generated data is tracked, no whitespace errors, and commits map
to the implementation tasks.

- [ ] **Step 6: Commit verification additions**

```bash
git add tests/test_live_smoke.py README.md
git commit -m "test: verify live Hugging Face API compatibility"
```

## Final Push

After GitHub authentication is available:

```bash
gh repo create zoiiiiie/hf-dataset-catalog --public --source=. --remote=origin --push
```

Before creating the repository, confirm that this exact repository name is
unused. If it already exists, add its URL as `origin` and push the current
branch without overwriting remote history.
