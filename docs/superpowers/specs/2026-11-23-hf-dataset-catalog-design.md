# Hugging Face High-Value Dataset Catalog Design

## 1. Objective

Build a repeatable Python pipeline that inventories public Hugging Face
datasets and creates a normalized, queryable metadata catalog. A dataset is
selected when it belongs to at least one of these sets:

- A: created within the configured year window, five years by default;
- B: Top N by recent downloads or Top N by all-time downloads;
- C: Top N by the citation count of its best-matched source paper.

The final population is `A union B union C`, deduplicated by `dataset_id`.
Selection flags and rankings remain independently visible.

## 2. Scope

### Included

- Enumerate all publicly visible Hugging Face dataset repositories.
- Collect repository page URLs and metadata exposed by official APIs.
- Collect repository file-list metadata such as paths, extensions, and sizes.
- Extract structured paper clues exposed by the Hugging Face API.
- Resolve papers through Semantic Scholar first and OpenAlex second.
- Preserve raw API responses, normalized tables, metric history, checkpoints,
  quality flags, and a collection report.
- Export Parquet, CSV, and JSONL files suitable for analysis or table import.

### Excluded

- Downloading dataset contents.
- Reading or downloading repository files, including README content.
- Inspecting file headers.
- Bypassing gated dataset access.
- Enumerating private datasets.
- Treating a citation block as a measured citation count.
- Uploading generated data to a hosted database or Feishu Base in V1.

## 3. Architecture

The application is a Python 3.11+ command-line package. DuckDB is the local
state and processing store. API payloads are written to atomic, run-scoped
JSONL batch files before normalization so that transformations can be repeated
without recollection.

The pipeline has six independently resumable stages:

1. `enumerate`: stream all public dataset metadata from `huggingface_hub`.
2. `select`: calculate the recent and two download-ranked candidate sets.
3. `papers`: extract structured paper clues and resolve citation metadata.
4. `merge`: calculate the citation-ranked set and union all selection sets.
5. `enrich`: normalize final-set metadata and file-list metadata.
6. `export`: validate and write deliverables plus the collection report.

Full-site enumeration uses only fields required for selection and paper-clue
discovery. Per-repository enrichment is restricted to the union population.
This prevents one detail request per repository across the entire Hub.

## 4. Data Sources

### Hugging Face

Use `huggingface_hub.HfApi.list_datasets(limit=None, expand=[...])`.
Requested properties include `author`, `cardData`, `citation`, `createdAt`,
`description`, `disabled`, `downloads`, `downloadsAllTime`, `gated`,
`lastModified`, `likes`, `mainSize`, `paperswithcode_id`, `private`, `sha`,
`siblings`, `tags`, and `usedStorage`.

The client iterator is consumed incrementally. Every configurable batch is
committed to raw JSONL and DuckDB before processing continues.

### Semantic Scholar

Resolve exact identifiers before text:

1. DOI
2. arXiv ID
3. Semantic Scholar paper ID
4. normalized title plus author evidence
5. normalized title only

Exact identifier matches have confidence `1.0`. Text matches at or above
`0.95` are confirmed automatically. Matches in `[0.80, 0.95)` are retained as
review candidates but do not contribute to citation ranking. Lower scores are
unmatched.

### OpenAlex

OpenAlex is the fallback for unresolved DOI, arXiv, or title queries. Citation
counts from different providers are never added together. Each confirmed paper
record has one selected citation source and preserves provider identifiers.

## 5. Time and Ranking Rules

- The recent cutoff is the exact calendar anniversary in UTC, not
  `365 * years`.
- Missing `created_at` never qualifies for set A.
- Null metric values do not qualify as zero; sorting uses a temporary null-last
  key without altering stored values.
- Ties are stable: metric descending, then `dataset_id` ascending.
- Each rank records `ranking_date`.
- A dataset enters set C when any confirmed associated paper produces its
  maximum confirmed citation count and the dataset ranks within Top N.
- An unmatched paper has `citation_count = NULL`, never zero.

## 6. Storage Model

DuckDB stores stage tables, checkpoints, failures, a raw-batch manifest, and
historical metric snapshots. Each raw batch is written to a temporary file,
atomically renamed, and registered by a deterministic batch key. Resume skips
registered batches. Export concatenates manifest batches into the final JSONL.
A `collection_run_id` connects all tables and reports.

### 6.1 `hf_datasets_master`

| Field | Type | Meaning |
|---|---|---|
| collection_run_id | string | Collection run identifier |
| dataset_id | string | Stable Hugging Face repository ID |
| dataset_name | string | Repository name |
| author | string nullable | User or organization |
| dataset_url | string | Dataset landing page |
| files_url | string | Repository file browser page |
| viewer_url | string | Dataset viewer page |
| created_at | timestamp nullable | Initial repository creation time |
| last_modified | timestamp nullable | Latest repository update |
| sha | string nullable | Collected repository revision |
| description | string nullable | API-provided description |
| tags | list<string> | Original tags |
| license | string | Normalized license or `Unknown` |
| license_raw | string nullable | Original license declaration |
| commercial_use_restriction | boolean nullable | Explicit restriction only |
| redistribution_restriction | boolean nullable | Explicit restriction only |
| special_license_terms | string nullable | Other explicit terms |
| modalities | list<string> | Standardized modalities |
| use_cases | list<string> | Standardized intended uses |
| tasks | list<string> | Standardized tasks |
| domains | list<string> | Standardized domains |
| data_formats | list<string> | Formats inferred from tags and file names |
| classification_source | list<string> | Evidence used for classifications |
| gated | string nullable | Hugging Face gated state |
| private | boolean | Expected false; retained for audit |
| disabled | boolean nullable | Repository disabled state |
| likes | integer nullable | Likes at collection time |
| downloads | integer nullable | Current download statistic |
| downloads_all_time | integer nullable | All-time download statistic |
| downloads_rank | integer nullable | Rank by current downloads |
| downloads_all_time_rank | integer nullable | Rank by all-time downloads |
| max_paper_citation_count | integer nullable | Highest confirmed paper citation count |
| citation_rank | integer nullable | Dataset citation rank |
| citation_source | string nullable | Provider for the maximum count |
| top_paper_id | string nullable | Highest-cited confirmed paper |
| top_paper_title | string nullable | Highest-cited confirmed paper title |
| paper_match_status | string | confirmed, review, unmatched, or no_clue |
| associated_paper_count | integer | Count of retained paper relationships |
| file_count | integer nullable | Number of listed files |
| used_storage_bytes | integer nullable | API-provided repository storage |
| total_file_size_bytes | integer nullable | Sum of known file sizes |
| file_extensions | list<string> | Distinct normalized extensions |
| is_recent | boolean | Membership in set A |
| is_top_download | boolean | Membership in recent-download set B |
| is_top_download_all_time | boolean | Membership in all-time-download set B |
| is_top_citation | boolean | Membership in set C |
| ranking_date | date | Ranking calculation date |
| collection_status | string | SUCCESS, PARTIAL, or API_ERROR |
| quality_flags | list<string> | Zero or more quality conditions |
| fetched_at | timestamp | Hugging Face fetch time |

`selection_reasons` is omitted because it duplicates the four membership
flags. Current metrics are intentionally duplicated from the historical
metrics table to make the master export self-contained.

### 6.2 `hf_dataset_papers`

| Field | Type |
|---|---|
| collection_run_id | string |
| dataset_id | string |
| paper_id | string nullable |
| semantic_scholar_id | string nullable |
| openalex_id | string nullable |
| paper_title | string nullable |
| doi | string nullable |
| arxiv_id | string nullable |
| paper_url | string nullable |
| authors | list<string> |
| publication_year | integer nullable |
| publication_date | date nullable |
| external_ids | json nullable |
| citation_count | integer nullable |
| citation_source | string nullable |
| match_method | string |
| match_confidence | double |
| match_status | string |
| is_top_cited_paper | boolean |
| fetched_at | timestamp |

Paper deduplication prioritizes normalized DOI, then provider paper ID, then
arXiv ID. Titles alone never form an unconditional match.

### 6.3 `hf_dataset_files`

| Field | Type |
|---|---|
| collection_run_id | string |
| dataset_id | string |
| revision | string nullable |
| file_path | string |
| file_name | string |
| extension | string nullable |
| file_type | string |
| size_bytes | integer nullable |
| is_lfs | boolean nullable |
| file_page_url | string |
| resolve_url | string |
| fetched_at | timestamp |

URLs are recorded only. The pipeline never requests `resolve_url`.

### 6.4 `hf_dataset_metrics`

| Field | Type |
|---|---|
| collection_run_id | string |
| dataset_id | string |
| snapshot_date | date |
| downloads | integer nullable |
| downloads_all_time | integer nullable |
| likes | integer nullable |
| citation_count | integer nullable |
| citation_source | string nullable |
| fetched_at | timestamp |

The natural key is `(dataset_id, snapshot_date)`. Re-running on the same day
updates that day's snapshot instead of creating a duplicate.

### 6.5 Intermediate Tables

- `dataset_recent`
- `dataset_download_top`
- `dataset_citation_top`
- `collection_failures`
- `collection_checkpoints`

Intermediate ranking tables retain `dataset_id`, metric value, rank, ranking
date, and run ID.

## 7. Classification

Classification is deterministic and rules-based in V1:

- modalities: Text, Image, Audio, Video, Multimodal, Tabular,
  Structured Data, Other;
- use cases: Pretraining, Supervised Fine-tuning, Instruction Tuning,
  Preference Optimization, RLHF, Evaluation, Benchmark,
  Retrieval Augmented Generation, Other.

Rules use tags, card data fields, and file extensions. They do not inspect file
contents. Multiple values are allowed. Evidence labels are retained in
`classification_source`.

License restrictions are populated only from explicit structured declarations.
Public visibility never implies commercial or redistribution permission.

## 8. Reliability

- Retry HTTP 429 using `Retry-After` when present.
- Retry transient 5xx and network failures with exponential backoff and jitter.
- Cap retries through configuration.
- Atomically publish each raw batch, register its deterministic key, and then
  persist the corresponding normalized records and stage checkpoint.
- Record non-recoverable failures without dropping the dataset from later
  processing.
- Resume an interrupted run from its stage checkpoint.
- Use atomic temporary-file replacement for final exports.
- Keep secrets in environment variables:
  `HF_TOKEN`, `SEMANTIC_SCHOLAR_API_KEY`, and `OPENALEX_API_KEY`.

Semantic Scholar and OpenAlex requests use bounded concurrency and configurable
minimum intervals. Provider responses are cached by normalized identifier.

## 9. Commands and Configuration

The CLI exposes:

- `hf-catalog collect`: run all stages or selected stages;
- `hf-catalog resume --run-id ...`: continue an interrupted run;
- `hf-catalog export --run-id ...`: recreate deliverables from DuckDB;
- `hf-catalog validate --run-id ...`: run data-quality checks;
- `hf-catalog status --run-id ...`: show checkpoint and failure summaries.

YAML configuration controls output paths, recent-year window, all three Top N
values, batch size, retries, request intervals, confidence thresholds, and
provider settings. Defaults are five years and 10,000 for each ranking.

## 10. Deliverables

Each completed run writes:

- `hf_datasets_master.parquet` and `.csv`;
- `hf_dataset_papers.parquet` and `.csv`;
- `hf_dataset_files.parquet` and `.csv`;
- `hf_dataset_metrics.parquet` and `.csv`;
- `hf_datasets_raw.jsonl`;
- `dataset_recent.parquet`;
- `dataset_download_top.parquet`;
- `dataset_citation_top.parquet`;
- `collection_report.json`.

The report contains start/end times, full enumeration count, each set size,
union size, deduplication count, paper match rate, API failure rate, field
completeness, retry counts, and unresolved quality-flag counts.

## 11. Testing and Acceptance

Unit tests cover exact anniversary calculation, null-aware stable ranking,
OR-union membership flags, DOI/arXiv normalization, paper confidence
thresholds, classification rules, URL construction, checkpoint resume, and
report metrics.

Integration tests use recorded synthetic API payloads, never synthetic
research results. A configurable online smoke test collects a small live sample
and verifies API compatibility without asserting volatile metric values.

Acceptance requires:

- no artificial limit in the full enumeration path;
- all four membership flags and all three ranks are reproducible;
- no dataset content or repository file content is downloaded;
- unknown values remain null rather than becoming zero;
- interrupted runs resume without duplicate raw or normalized records;
- every required output validates against its declared schema;
- the same configuration can be rerun to append or update metric history.
