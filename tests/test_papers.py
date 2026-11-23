from __future__ import annotations

import pytest

from hf_dataset_catalog.models import MatchStatus, PaperRecord
from hf_dataset_catalog.papers import (
    PaperCandidate,
    PaperClue,
    PaperResolver,
    apply_match_thresholds,
    extract_paper_clues,
    normalize_arxiv_id,
    normalize_doi,
    select_top_paper,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://doi.org/10.1000/XYZ", "10.1000/xyz"),
        ("doi:10.1000/xyz.", "10.1000/xyz"),
    ],
)
def test_normalize_doi(raw: str, expected: str) -> None:
    assert normalize_doi(raw) == expected


def test_normalize_arxiv_id_removes_url_and_version() -> None:
    assert normalize_arxiv_id("https://arxiv.org/abs/2401.01234v2") == "2401.01234"


def test_review_match_does_not_contribute_citations() -> None:
    candidate = PaperCandidate(title="Example", match_confidence=0.91, citation_count=50)
    result = apply_match_thresholds(candidate, confirm=0.95, review=0.80)
    assert result.match_status == MatchStatus.REVIEW
    assert result.effective_citation_count is None


def test_select_top_paper_does_not_sum_citations() -> None:
    papers = [
        PaperRecord(
            dataset_id="d",
            paper_id="1",
            citation_count=10,
            match_method="doi",
            match_confidence=1,
            match_status=MatchStatus.CONFIRMED,
        ),
        PaperRecord(
            dataset_id="d",
            paper_id="2",
            citation_count=20,
            match_method="arxiv",
            match_confidence=1,
            match_status=MatchStatus.CONFIRMED,
        ),
    ]
    assert select_top_paper(papers).paper_id == "2"


def test_extract_paper_clues_uses_structured_api_values() -> None:
    clues = extract_paper_clues(
        {
            "citation": "@article{x, doi={10.1000/XYZ}, title={Example Dataset}}",
            "paperswithcode_id": "example-dataset",
            "card_data": {"arxiv": "2401.01234v2", "authors": ["A. Author"]},
            "description": "This unstructured field contains 10.9999/ignored.",
        }
    )
    assert any(clue.doi == "10.1000/xyz" for clue in clues)
    assert any(clue.arxiv_id == "2401.01234" for clue in clues)
    assert any(clue.title == "Example Dataset" for clue in clues)
    assert all(clue.doi != "10.9999/ignored" for clue in clues)


class StubProvider:
    def __init__(
        self,
        *,
        exact: PaperCandidate | None = None,
        title: PaperCandidate | None = None,
    ) -> None:
        self.exact = exact
        self.title = title
        self.calls: list[str] = []

    async def resolve_exact(self, clue: PaperClue) -> PaperCandidate | None:
        self.calls.append("exact")
        return self.exact

    async def search_title(self, clue: PaperClue) -> PaperCandidate | None:
        self.calls.append("title")
        return self.title


@pytest.mark.asyncio
async def test_resolver_checks_all_exact_sources_before_title_search() -> None:
    semantic = StubProvider()
    openalex = StubProvider(
        exact=PaperCandidate(
            paper_id="W1",
            title="Example",
            citation_count=7,
            citation_source="openalex",
            match_method="doi",
            match_confidence=1,
        )
    )
    resolver = PaperResolver(
        semantic_scholar=semantic,
        openalex=openalex,
        confirm_threshold=0.95,
        review_threshold=0.80,
    )

    result = await resolver.resolve(PaperClue(doi="10.1000/example"))

    assert result.match_status == MatchStatus.CONFIRMED
    assert result.effective_citation_count == 7
    assert semantic.calls == ["exact"]
    assert openalex.calls == ["exact"]


@pytest.mark.asyncio
async def test_resolver_caches_by_normalized_clue() -> None:
    semantic = StubProvider(
        exact=PaperCandidate(
            paper_id="S1",
            title="Example",
            citation_count=3,
            citation_source="semantic_scholar",
            match_method="doi",
            match_confidence=1,
        )
    )
    resolver = PaperResolver(semantic_scholar=semantic, openalex=StubProvider())

    first = await resolver.resolve(PaperClue(doi="10.1000/EXAMPLE"))
    second = await resolver.resolve(PaperClue(doi="https://doi.org/10.1000/example"))

    assert first == second
    assert semantic.calls == ["exact"]
