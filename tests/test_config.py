from pathlib import Path

import pytest
from pydantic import ValidationError

from hf_dataset_catalog.config import Settings, load_settings


def test_default_settings_match_prd(tmp_path: Path) -> None:
    settings = Settings(output_dir=tmp_path)
    assert settings.recent_years == 5
    assert settings.top_downloads == 10_000
    assert settings.top_downloads_all_time == 10_000
    assert settings.top_citations == 10_000
    assert settings.confirm_threshold == 0.95
    assert settings.review_threshold == 0.80


def test_review_threshold_must_not_exceed_confirm_threshold() -> None:
    with pytest.raises(ValidationError):
        Settings(review_threshold=0.96, confirm_threshold=0.95)


def test_yaml_and_overrides_are_merged(tmp_path: Path) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text("recent_years: 3\nbatch_size: 20\n", encoding="utf-8")
    settings = load_settings(config, {"batch_size": 10})
    assert settings.recent_years == 3
    assert settings.batch_size == 10

