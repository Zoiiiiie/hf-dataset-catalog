from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from typing import Any, Protocol
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field, field_validator
from rapidfuzz import fuzz

from hf_dataset_catalog.http import RetryingJsonClient
from hf_dataset_catalog.models import MatchStatus, PaperRecord

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
ARXIV_PATTERN = re.compile(r"(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
TITLE_PATTERN = re.compile(r"title\s*=\s*[{'\"]+([^{}'\"]+)", re.IGNORECASE)
AUTHOR_PATTERN = re.compile(r"author\s*=\s*[{'\"]+([^{}'\"]+)", re.IGNORECASE)


class PaperClue(BaseModel):
    doi: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    paperswithcode_id: str | None = None
    source: str | None = None

    @field_validator("doi", mode="before")
    @classmethod
    def normalize_doi_field(cls, value: Any) -> str | None:
        return normalize_doi(str(value)) if value else None

    @field_validator("arxiv_id", mode="before")
    @classmethod
    def normalize_arxiv_field(cls, value: Any) -> str | None:
        return normalize_arxiv_id(str(value)) if value else None

    @field_validator("title", mode="before")
    @classmethod
    def normalize_title_field(cls, value: Any) -> str | None:
        return _clean_text(value)

    @field_validator("authors", mode="before")
    @classmethod
    def normalize_authors_field(cls, value: Any) -> list[str]:
        if not value:
            return []
        values = re.split(r"\s+and\s+|[,;]", value) if isinstance(value, str) else value
        cleaned = {_clean_text(item) for item in values}
        return sorted(item for item in cleaned if item is not None)

    def cache_key(self) -> str:
        parts = [
            self.doi or "",
            self.arxiv_id or "",
            (self.semantic_scholar_id or "").lower(),
            _normalized_text(self.title),
            "|".join(_normalized_text(author) for author in self.authors),
        ]
        return "\x1f".join(parts)


class PaperCandidate(BaseModel):
    paper_id: str | None = None
    semantic_scholar_id: str | None = None
    openalex_id: str | None = None
    title: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    paper_url: str | None = None
    authors: list[str] = Field(default_factory=list)
    publication_year: int | None = None
    publication_date: date | None = None
    external_ids: dict[str, Any] | None = None
    citation_count: int | None = None
    citation_source: str | None = None
    match_method: str = "none"
    match_confidence: float = Field(default=0, ge=0, le=1)
    match_status: MatchStatus = MatchStatus.UNMATCHED
    effective_citation_count: int | None = None


class PaperProvider(Protocol):
    async def resolve_exact(self, clue: PaperClue) -> PaperCandidate | None: ...

    async def search_title(self, clue: PaperClue) -> PaperCandidate | None: ...


class PaperResolver:
    def __init__(
        self,
        *,
        semantic_scholar: PaperProvider,
        openalex: PaperProvider,
        confirm_threshold: float = 0.95,
        review_threshold: float = 0.80,
    ) -> None:
        self.semantic_scholar = semantic_scholar
        self.openalex = openalex
        self.confirm_threshold = confirm_threshold
        self.review_threshold = review_threshold
        self._cache: dict[str, PaperCandidate] = {}

    async def resolve(self, clue: PaperClue) -> PaperCandidate:
        key = clue.cache_key()
        if key in self._cache:
            return self._cache[key]

        candidate: PaperCandidate | None = None
        if clue.doi or clue.arxiv_id or clue.semantic_scholar_id:
            candidate = await self.semantic_scholar.resolve_exact(clue)
            if candidate is None:
                candidate = await self.openalex.resolve_exact(clue)

        if candidate is None and clue.title:
            candidate = await self.semantic_scholar.search_title(clue)
            if candidate is None:
                candidate = await self.openalex.search_title(clue)

        if candidate is None:
            candidate = PaperCandidate(title=clue.title)
        result = apply_match_thresholds(
            candidate,
            confirm=self.confirm_threshold,
            review=self.review_threshold,
        )
        self._cache[key] = result
        return result


class SemanticScholarProvider:
    base_url = "https://api.semanticscholar.org/graph/v1"
    fields = (
        "paperId,title,authors,year,publicationDate,externalIds,"
        "citationCount,url"
    )

    def __init__(self, client: RetryingJsonClient) -> None:
        self.client = client

    async def resolve_exact(self, clue: PaperClue) -> PaperCandidate | None:
        identifiers = [
            (f"DOI:{clue.doi}", "doi") if clue.doi else None,
            (f"ARXIV:{clue.arxiv_id}", "arxiv") if clue.arxiv_id else None,
            (clue.semantic_scholar_id, "semantic_scholar_id")
            if clue.semantic_scholar_id
            else None,
        ]
        for item in identifiers:
            if item is None:
                continue
            identifier, method = item
            try:
                response = await self.client.get_json(
                    f"{self.base_url}/paper/{quote(identifier, safe='')}",
                    params={"fields": self.fields},
                )
            except httpx.HTTPStatusError as error:
                if error.response.status_code == 404:
                    continue
                raise
            payload = _mapping(response.payload)
            if payload:
                return semantic_scholar_candidate(
                    payload,
                    match_method=method,
                    match_confidence=1,
                )
        return None

    async def search_title(self, clue: PaperClue) -> PaperCandidate | None:
        if not clue.title:
            return None
        response = await self.client.get_json(
            f"{self.base_url}/paper/search",
            params={"query": clue.title, "limit": 5, "fields": self.fields},
        )
        payload = _mapping(response.payload)
        candidates = [
            semantic_scholar_candidate(_mapping(item), match_method="title")
            for item in _sequence(payload.get("data"))
        ]
        return _best_text_candidate(clue, candidates)


class OpenAlexProvider:
    base_url = "https://api.openalex.org"

    def __init__(self, client: RetryingJsonClient) -> None:
        self.client = client

    async def resolve_exact(self, clue: PaperClue) -> PaperCandidate | None:
        filters = [
            (f"doi:{clue.doi}", "doi") if clue.doi else None,
            (f"locations.landing_page_url:https://arxiv.org/abs/{clue.arxiv_id}", "arxiv")
            if clue.arxiv_id
            else None,
        ]
        for item in filters:
            if item is None:
                continue
            filter_value, method = item
            response = await self.client.get_json(
                f"{self.base_url}/works",
                params={"filter": filter_value, "per-page": 1},
            )
            first = _first_result(response.payload)
            if first:
                return openalex_candidate(first, match_method=method, match_confidence=1)
        return None

    async def search_title(self, clue: PaperClue) -> PaperCandidate | None:
        if not clue.title:
            return None
        response = await self.client.get_json(
            f"{self.base_url}/works",
            params={"search": clue.title, "per-page": 5},
        )
        payload = _mapping(response.payload)
        candidates = [
            openalex_candidate(_mapping(item), match_method="title")
            for item in _sequence(payload.get("results"))
        ]
        return _best_text_candidate(clue, candidates)


def normalize_doi(raw: str) -> str | None:
    value = raw.strip()
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value, flags=re.I)
    match = DOI_PATTERN.search(value)
    if not match:
        return None
    return match.group(0).rstrip(".,;:)]}").lower()


def normalize_arxiv_id(raw: str) -> str | None:
    value = re.sub(
        r"^(?:https?://arxiv\.org/(?:abs|pdf)/|arxiv:\s*)",
        "",
        raw.strip(),
        flags=re.I,
    )
    match = ARXIV_PATTERN.search(value)
    if not match:
        return None
    return re.sub(r"v\d+$", "", match.group(1), flags=re.I).lower()


def extract_paper_clues(dataset: Mapping[str, Any]) -> list[PaperClue]:
    clues: list[PaperClue] = []
    citation = dataset.get("citation")
    if isinstance(citation, str):
        clues.extend(_clues_from_text(citation, source="citation", parse_title=True))

    paperswithcode_id = dataset.get("paperswithcode_id")
    if isinstance(paperswithcode_id, str) and paperswithcode_id.strip():
        clues.append(
            PaperClue(
                paperswithcode_id=paperswithcode_id.strip(),
                source="paperswithcode_id",
            )
        )

    card_data = dataset.get("card_data")
    if isinstance(card_data, Mapping):
        clues.extend(_clues_from_card_data(card_data))
    return deduplicate_clues(clues)


def deduplicate_clues(clues: Iterable[PaperClue]) -> list[PaperClue]:
    deduplicated: dict[str, PaperClue] = {}
    for clue in clues:
        if not any(
            (
                clue.doi,
                clue.arxiv_id,
                clue.semantic_scholar_id,
                clue.title,
                clue.paperswithcode_id,
            )
        ):
            continue
        deduplicated.setdefault(clue.cache_key() + f"\x1f{clue.paperswithcode_id or ''}", clue)
    return list(deduplicated.values())


def apply_match_thresholds(
    candidate: PaperCandidate,
    *,
    confirm: float,
    review: float,
) -> PaperCandidate:
    if candidate.match_confidence >= confirm:
        status = MatchStatus.CONFIRMED
        effective_count = candidate.citation_count
    elif candidate.match_confidence >= review:
        status = MatchStatus.REVIEW
        effective_count = None
    else:
        status = MatchStatus.UNMATCHED
        effective_count = None
    return candidate.model_copy(
        update={
            "match_status": status,
            "effective_citation_count": effective_count,
        }
    )


def select_top_paper(papers: Iterable[PaperRecord]) -> PaperRecord | None:
    confirmed = [
        paper
        for paper in papers
        if paper.match_status == MatchStatus.CONFIRMED and paper.citation_count is not None
    ]
    if not confirmed:
        return None
    return min(
        confirmed,
        key=lambda paper: (
            -int(paper.citation_count or 0),
            paper.paper_id or "",
        ),
    )


def candidate_to_record(dataset_id: str, candidate: PaperCandidate) -> PaperRecord:
    return PaperRecord(
        dataset_id=dataset_id,
        paper_id=candidate.paper_id,
        semantic_scholar_id=candidate.semantic_scholar_id,
        openalex_id=candidate.openalex_id,
        paper_title=candidate.title,
        doi=candidate.doi,
        arxiv_id=candidate.arxiv_id,
        paper_url=candidate.paper_url,
        authors=candidate.authors,
        publication_year=candidate.publication_year,
        publication_date=candidate.publication_date,
        external_ids=candidate.external_ids,
        citation_count=candidate.effective_citation_count,
        citation_source=candidate.citation_source,
        match_method=candidate.match_method,
        match_confidence=candidate.match_confidence,
        match_status=candidate.match_status,
    )


def semantic_scholar_candidate(
    payload: Mapping[str, Any],
    *,
    match_method: str,
    match_confidence: float = 0,
) -> PaperCandidate:
    external_ids = _mapping(payload.get("externalIds"))
    paper_id = _optional_string(payload.get("paperId"))
    return PaperCandidate(
        paper_id=paper_id,
        semantic_scholar_id=paper_id,
        title=_optional_string(payload.get("title")),
        doi=normalize_doi(str(external_ids["DOI"])) if external_ids.get("DOI") else None,
        arxiv_id=(
            normalize_arxiv_id(str(external_ids["ArXiv"]))
            if external_ids.get("ArXiv")
            else None
        ),
        paper_url=_optional_string(payload.get("url")),
        authors=[
            str(author["name"])
            for author in (_mapping(item) for item in _sequence(payload.get("authors")))
            if author.get("name")
        ],
        publication_year=_optional_int(payload.get("year")),
        publication_date=_optional_date(payload.get("publicationDate")),
        external_ids=dict(external_ids) or None,
        citation_count=_optional_int(payload.get("citationCount")),
        citation_source="semantic_scholar",
        match_method=match_method,
        match_confidence=match_confidence,
    )


def openalex_candidate(
    payload: Mapping[str, Any],
    *,
    match_method: str,
    match_confidence: float = 0,
) -> PaperCandidate:
    ids = _mapping(payload.get("ids"))
    openalex_id = _short_openalex_id(payload.get("id") or ids.get("openalex"))
    authors = []
    for item in _sequence(payload.get("authorships")):
        author = _mapping(_mapping(item).get("author"))
        if author.get("display_name"):
            authors.append(str(author["display_name"]))
    doi_value = payload.get("doi") or ids.get("doi")
    return PaperCandidate(
        paper_id=openalex_id,
        openalex_id=openalex_id,
        title=_optional_string(payload.get("display_name") or payload.get("title")),
        doi=normalize_doi(str(doi_value)) if doi_value else None,
        paper_url=_optional_string(payload.get("id")),
        authors=authors,
        publication_year=_optional_int(payload.get("publication_year")),
        publication_date=_optional_date(payload.get("publication_date")),
        external_ids=dict(ids) or None,
        citation_count=_optional_int(payload.get("cited_by_count")),
        citation_source="openalex",
        match_method=match_method,
        match_confidence=match_confidence,
    )


def _clues_from_text(text: str, *, source: str, parse_title: bool) -> list[PaperClue]:
    clues = [
        PaperClue(doi=match.group(0), source=source) for match in DOI_PATTERN.finditer(text)
    ]
    clues.extend(
        PaperClue(arxiv_id=match.group(1), source=source)
        for match in ARXIV_PATTERN.finditer(text)
    )
    if parse_title:
        title_match = TITLE_PATTERN.search(text)
        author_match = AUTHOR_PATTERN.search(text)
        if title_match:
            authors = (
                re.split(r"\s+and\s+|[,;]", author_match.group(1)) if author_match else []
            )
            clues.append(
                PaperClue(
                    title=title_match.group(1),
                    authors=authors,
                    source=source,
                )
            )
    return clues


def _clues_from_card_data(card_data: Mapping[str, Any]) -> list[PaperClue]:
    clues: list[PaperClue] = []
    for key, value in card_data.items():
        normalized_key = str(key).lower().replace("-", "_")
        if isinstance(value, Mapping):
            clues.extend(_clues_from_card_data(value))
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                if isinstance(item, Mapping):
                    clues.extend(_clues_from_card_data(item))
                elif isinstance(item, str):
                    clues.extend(_clues_from_text(item, source="card_data", parse_title=False))
            continue
        if not isinstance(value, str):
            continue
        clues.extend(_clues_from_text(value, source="card_data", parse_title=False))
        if normalized_key in {"paper_title", "title"}:
            clues.append(PaperClue(title=value, source="card_data"))
        elif normalized_key in {"semantic_scholar_id", "s2_paper_id"}:
            clues.append(PaperClue(semantic_scholar_id=value.strip(), source="card_data"))
    return clues


def _best_text_candidate(
    clue: PaperClue,
    candidates: Iterable[PaperCandidate],
) -> PaperCandidate | None:
    scored: list[PaperCandidate] = []
    for candidate in candidates:
        if not candidate.title or not clue.title:
            continue
        title_score = fuzz.ratio(_normalized_text(clue.title), _normalized_text(candidate.title))
        confidence = title_score / 100
        if clue.authors and candidate.authors:
            author_score = max(
                fuzz.ratio(_normalized_text(expected), _normalized_text(actual))
                for expected in clue.authors
                for actual in candidate.authors
            )
            if author_score < 80:
                confidence = min(confidence, 0.79)
            else:
                confidence = (confidence * 0.9) + ((author_score / 100) * 0.1)
        scored.append(candidate.model_copy(update={"match_confidence": confidence}))
    if not scored:
        return None
    return max(
        scored,
        key=lambda candidate: (
            candidate.match_confidence,
            candidate.citation_count if candidate.citation_count is not None else -1,
            candidate.paper_id or "",
        ),
    )


def _first_result(payload: Any) -> Mapping[str, Any] | None:
    results = _sequence(_mapping(payload).get("results"))
    return _mapping(results[0]) if results else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _normalized_text(value: str | None) -> str:
    return re.sub(r"\W+", " ", (value or "").casefold()).strip()


def _optional_string(value: Any) -> str | None:
    return _clean_text(value)


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _short_openalex_id(value: Any) -> str | None:
    text = _optional_string(value)
    return text.rsplit("/", maxsplit=1)[-1] if text else None
