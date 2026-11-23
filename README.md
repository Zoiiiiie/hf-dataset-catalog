# Hugging Face Dataset Catalog

A resumable, metadata-only catalog of high-value public Hugging Face datasets.
The selected population is the union of:

- datasets created within the configured calendar-year window;
- Top N datasets by current or all-time downloads;
- Top N datasets by the highest confirmed source-paper citation count.

The collector does **not** download dataset contents, README files, repository
files, or constructed `resolve` URLs. It only records metadata and URLs exposed
by official APIs. Private repositories are excluded, and gated access is never
bypassed.

## Install

Python 3.11 or newer is required.

```bash
uv sync --extra dev
uv run hf-catalog --help
```

Or install with pip:

```bash
python -m pip install -e .
hf-catalog --help
```

Copy `.env.example` values into your shell or secret manager as needed.
`HF_TOKEN` is optional for public metadata. Semantic Scholar and OpenAlex keys
are optional but improve quota and reliability.

## Smoke Test

Use an explicit limit for a quick API compatibility check:

```bash
uv run hf-catalog collect --max-datasets 20 --skip-papers
```

`--max-datasets` is never applied by default. Omitting it requests the complete
public dataset listing and can take hours, generate substantial local state,
and consume academic-provider quotas.

## Commands

```bash
# Full collection using built-in defaults
uv run hf-catalog collect

# Collection using a YAML configuration
uv run hf-catalog collect --config config/default.yaml

# Resume durable stage checkpoints
uv run hf-catalog resume RUN_ID --config config/default.yaml

# Recreate deliverables without recollecting
uv run hf-catalog export RUN_ID --output-dir data/output

# Validate schemas, keys, rankings, membership, and URLs
uv run hf-catalog validate RUN_ID

# Inspect one run or print the latest run ID
uv run hf-catalog status RUN_ID
uv run hf-catalog status --latest --format run-id
```

The stages are `enumerate`, `select`, `papers`, `merge`, `enrich`, and
`export`. A stage is skipped on resume only after its completion checkpoint is
durably stored.

## Configuration

See [`config/default.yaml`](config/default.yaml). The defaults use a five-year
calendar window and Top 10,000 for each ranking. Secrets are read only from:

- `HF_TOKEN`
- `SEMANTIC_SCHOLAR_API_KEY`
- `OPENALEX_API_KEY`

Unknown metric and citation values remain null. Citation counts from providers
are never added together. Text matches below the confirmation threshold are
retained for review but do not contribute to citation ranking.

## Outputs

Each run is written under `data/output/RUN_ID/`:

- `hf_datasets_master.parquet` and `.csv`
- `hf_dataset_papers.parquet` and `.csv`
- `hf_dataset_files.parquet` and `.csv`
- `hf_dataset_metrics.parquet` and `.csv`
- `hf_datasets_raw.jsonl`
- `dataset_recent.parquet`
- `dataset_download_top.parquet`
- `dataset_citation_top.parquet`
- `collection_report.json`

The master, paper, and file schemas follow the approved PRD fields. CSV list
columns are JSON-encoded strings; Parquet retains native list and null types.
Raw API batches, DuckDB state, and final files are published atomically.

## Automation

`.github/workflows/collect.yml` supports manual dispatch and a monthly run. Add
provider values as GitHub Actions secrets. The workflow prevents overlapping
collections, enforces a timeout, and uploads the run output as an artifact.

Local state and generated datasets are intentionally ignored by Git.

## Verification

The default test suite is offline. Run the opt-in public API compatibility
check with:

```bash
RUN_LIVE_TESTS=1 uv run pytest tests/test_live_smoke.py -q -m live
```

This test reads one dataset metadata row and does not request repository file
content.
