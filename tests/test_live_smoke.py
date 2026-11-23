import os

import pytest

from hf_dataset_catalog.hf_client import HuggingFaceSource

pytestmark = pytest.mark.live


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="set RUN_LIVE_TESTS=1 to access the public Hugging Face API",
)
def test_live_hf_sample_has_required_api_fields() -> None:
    source = HuggingFaceSource()
    row = next(source.iter_datasets())
    assert row["dataset_id"]
    assert row["dataset_url"].startswith("https://huggingface.co/datasets/")
    assert "downloads" in row
    assert "downloads_all_time" in row
