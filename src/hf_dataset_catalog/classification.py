from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    modalities: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    data_formats: list[str] = Field(default_factory=list)
    classification_source: list[str] = Field(default_factory=list)


class LicenseResult(BaseModel):
    license: str = "Unknown"
    license_raw: str | None = None
    commercial_use_restriction: bool | None = None
    redistribution_restriction: bool | None = None
    special_license_terms: str | None = None


FORMAT_BY_EXTENSION = {
    ".arrow": "ARROW",
    ".avro": "AVRO",
    ".csv": "CSV",
    ".flac": "FLAC",
    ".gif": "GIF",
    ".jpeg": "JPEG",
    ".jpg": "JPG",
    ".json": "JSON",
    ".jsonl": "JSONL",
    ".mp3": "MP3",
    ".mp4": "MP4",
    ".parquet": "PARQUET",
    ".png": "PNG",
    ".tar": "TAR",
    ".tsv": "TSV",
    ".txt": "TXT",
    ".wav": "WAV",
    ".webm": "WEBM",
    ".webp": "WEBP",
    ".xml": "XML",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".zip": "ZIP",
}

MODALITY_BY_EXTENSION = {
    ".arrow": "Tabular",
    ".avro": "Structured Data",
    ".csv": "Tabular",
    ".flac": "Audio",
    ".gif": "Image",
    ".jpeg": "Image",
    ".jpg": "Image",
    ".json": "Structured Data",
    ".jsonl": "Structured Data",
    ".mp3": "Audio",
    ".mp4": "Video",
    ".parquet": "Tabular",
    ".png": "Image",
    ".tsv": "Tabular",
    ".txt": "Text",
    ".wav": "Audio",
    ".webm": "Video",
    ".webp": "Image",
    ".xml": "Structured Data",
    ".yaml": "Structured Data",
    ".yml": "Structured Data",
}

TASK_NAMES = {
    "audio-classification": "Audio Classification",
    "automatic-speech-recognition": "Automatic Speech Recognition",
    "image-classification": "Image Classification",
    "image-to-text": "Image to Text",
    "object-detection": "Object Detection",
    "question-answering": "Question Answering",
    "summarization": "Summarization",
    "text-classification": "Text Classification",
    "text-generation": "Text Generation",
    "token-classification": "Token Classification",
    "translation": "Translation",
    "video-classification": "Video Classification",
    "visual-question-answering": "Visual Question Answering",
}

MODALITY_BY_TASK_PREFIX = {
    "audio-": "Audio",
    "image-": "Image",
    "object-detection": "Image",
    "text-": "Text",
    "token-": "Text",
    "video-": "Video",
}

USE_CASE_TAG_RULES = {
    "benchmark": "Benchmark",
    "evaluation": "Evaluation",
    "instruction-tuning": "Instruction Tuning",
    "preference-optimization": "Preference Optimization",
    "pretraining": "Pretraining",
    "reinforcement-learning-from-human-feedback": "RLHF",
    "rlhf": "RLHF",
    "supervised-fine-tuning": "Supervised Fine-tuning",
}

DOMAIN_TAG_RULES = {
    "code": "Code",
    "finance": "Finance",
    "legal": "Legal",
    "medical": "Medical",
    "multilingual": "Multilingual",
    "scientific": "Scientific",
}


def _tag_value(tag: str) -> tuple[str | None, str]:
    prefix, separator, value = tag.strip().lower().partition(":")
    return (prefix if separator else None, value if separator else prefix)


def _extension(path: str) -> str:
    return PurePosixPath(path.lower()).suffix


def classify_dataset(
    *,
    tags: Sequence[str],
    file_paths: Sequence[str],
    card_data: Mapping[str, Any],
) -> ClassificationResult:
    modalities: set[str] = set()
    use_cases: set[str] = set()
    tasks: set[str] = set()
    domains: set[str] = set()
    formats: set[str] = set()
    sources: set[str] = set()

    for tag in tags:
        prefix, value = _tag_value(tag)
        if prefix in {"task_categories", "task_ids"}:
            task = TASK_NAMES.get(value)
            if task:
                tasks.add(task)
            for task_prefix, modality in MODALITY_BY_TASK_PREFIX.items():
                if value == task_prefix or value.startswith(task_prefix):
                    modalities.add(modality)
            if value in {"open-domain-qa", "retrieval-augmented-generation"}:
                use_cases.add("Retrieval Augmented Generation")
        if prefix == "multilinguality" and value == "multilingual":
            domains.add("Multilingual")
        if value in USE_CASE_TAG_RULES:
            use_cases.add(USE_CASE_TAG_RULES[value])
        if value in DOMAIN_TAG_RULES:
            domains.add(DOMAIN_TAG_RULES[value])

    if modalities or use_cases or tasks or domains:
        sources.add("tags")

    for path in file_paths:
        extension = _extension(path)
        if extension in FORMAT_BY_EXTENSION:
            formats.add(FORMAT_BY_EXTENSION[extension])
        if extension in MODALITY_BY_EXTENSION:
            modalities.add(MODALITY_BY_EXTENSION[extension])
    if formats:
        sources.add("file_extensions")

    card_text = " ".join(
        str(card_data.get(key, "")).lower()
        for key in ("pretty_name", "dataset_name", "task_name", "intended_use")
    )
    for token, use_case in USE_CASE_TAG_RULES.items():
        if token.replace("-", " ") in card_text:
            use_cases.add(use_case)
    if "benchmark" in card_text:
        use_cases.add("Benchmark")
    if "evaluation" in card_text:
        use_cases.add("Evaluation")
    if any(source in card_text for source in ("benchmark", "evaluation")):
        sources.add("card_data")

    if len(modalities & {"Text", "Image", "Audio", "Video"}) > 1:
        modalities.add("Multimodal")

    return ClassificationResult(
        modalities=sorted(modalities),
        use_cases=sorted(use_cases),
        tasks=sorted(tasks),
        domains=sorted(domains),
        data_formats=sorted(formats),
        classification_source=sorted(sources),
    )


def classify_license(
    *,
    card_data: Mapping[str, Any],
    tags: Sequence[str],
) -> LicenseResult:
    raw_license = card_data.get("license")
    if isinstance(raw_license, list):
        raw_license = ", ".join(str(value) for value in raw_license)
    if not isinstance(raw_license, str) or not raw_license.strip():
        raw_license = next(
            (
                value
                for prefix, value in (_tag_value(tag) for tag in tags)
                if prefix == "license" and value
            ),
            None,
        )

    normalized = raw_license.strip().upper() if raw_license else "Unknown"
    return LicenseResult(
        license=normalized,
        license_raw=raw_license,
        commercial_use_restriction=_explicit_bool(
            card_data.get("commercial_use_restriction")
        ),
        redistribution_restriction=_explicit_bool(
            card_data.get("redistribution_restriction")
        ),
        special_license_terms=_optional_text(card_data.get("special_license_terms")),
    )


def _explicit_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
