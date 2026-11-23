from hf_dataset_catalog.report import build_report


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
    assert report.deduplication_count == 13


def test_report_rates_are_null_when_denominator_is_zero() -> None:
    report = build_report(
        enumerated=0,
        recent=0,
        top_download=0,
        top_download_all_time=0,
        top_citation=0,
        union_count=0,
        paper_clue_datasets=0,
        paper_matched_datasets=0,
        api_failures=0,
        required_field_cells=0,
        populated_required_field_cells=0,
    )
    assert report.paper_match_rate is None
    assert report.api_failure_rate is None
    assert report.field_completeness_rate is None
