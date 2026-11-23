from hf_dataset_catalog.classification import classify_dataset, classify_license


def test_multivalue_classification_uses_tags_and_extensions() -> None:
    result = classify_dataset(
        tags=["task_categories:image-classification", "language:en"],
        file_paths=["data/train.parquet", "images/0001.jpg"],
        card_data={},
    )
    assert result.modalities == ["Image", "Tabular"]
    assert result.data_formats == ["JPG", "PARQUET"]
    assert result.tasks == ["Image Classification"]
    assert "tags" in result.classification_source
    assert "file_extensions" in result.classification_source


def test_public_access_does_not_imply_commercial_permission() -> None:
    result = classify_license(card_data={}, tags=[])
    assert result.license == "Unknown"
    assert result.commercial_use_restriction is None
    assert result.redistribution_restriction is None


def test_license_uses_only_explicit_structured_restrictions() -> None:
    result = classify_license(
        card_data={
            "license": "cc-by-nc-4.0",
            "commercial_use_restriction": True,
            "redistribution_restriction": False,
            "special_license_terms": "Attribution required",
        },
        tags=["public"],
    )
    assert result.license == "CC-BY-NC-4.0"
    assert result.license_raw == "cc-by-nc-4.0"
    assert result.commercial_use_restriction is True
    assert result.redistribution_restriction is False
    assert result.special_license_terms == "Attribution required"


def test_use_case_and_domain_rules_are_deterministic() -> None:
    result = classify_dataset(
        tags=[
            "task_categories:question-answering",
            "task_ids:open-domain-qa",
            "multilinguality:multilingual",
            "language:en",
        ],
        file_paths=["records.jsonl"],
        card_data={"pretty_name": "Evaluation benchmark"},
    )
    assert result.use_cases == ["Benchmark", "Evaluation", "Retrieval Augmented Generation"]
    assert result.tasks == ["Question Answering"]
    assert result.domains == ["Multilingual"]
    assert result.data_formats == ["JSONL"]
