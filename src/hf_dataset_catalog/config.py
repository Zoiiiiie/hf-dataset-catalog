from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator


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
    def validate_thresholds(self) -> Settings:
        if self.review_threshold > self.confirm_threshold:
            raise ValueError("review_threshold must be <= confirm_threshold")
        return self


SECRET_ENV_VARS = {
    "HF_TOKEN": "hf_token",
    "SEMANTIC_SCHOLAR_API_KEY": "semantic_scholar_api_key",
    "OPENALEX_API_KEY": "openalex_api_key",
}


def load_settings(
    path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    values: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError("configuration root must be a mapping")
        values.update(loaded or {})
    for env_name, field_name in SECRET_ENV_VARS.items():
        if secret := os.getenv(env_name):
            values[field_name] = secret
    values.update(overrides or {})
    return Settings.model_validate(values)
