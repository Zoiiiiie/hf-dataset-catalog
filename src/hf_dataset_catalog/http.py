from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx


@dataclass(frozen=True)
class JsonResponse:
    payload: Any
    status_code: int
    retry_count: int
    source: str
    fetched_at: datetime


class RetryingJsonClient:
    def __init__(
        self,
        *,
        source: str,
        client: httpx.AsyncClient | None = None,
        max_retries: int = 6,
        backoff_base_seconds: float = 0.5,
        jitter_seconds: float = 0.25,
        request_interval_seconds: float = 0,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.source = source
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.jitter_seconds = jitter_seconds
        self.request_interval_seconds = request_interval_seconds
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(headers=headers, timeout=30)
        self._interval_lock = asyncio.Lock()
        self._last_request_at: float | None = None

    async def __aenter__(self) -> RetryingJsonClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> JsonResponse:
        return await self._request("GET", url, params=params, headers=headers)

    async def post_json(
        self,
        url: str,
        *,
        payload: Any,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> JsonResponse:
        return await self._request(
            "POST",
            url,
            params=params,
            headers=headers,
            json=payload,
        )

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> JsonResponse:
        retry_count = 0
        while True:
            await self._wait_for_interval()
            try:
                response = await self._client.request(method, url, **kwargs)
            except httpx.TransportError:
                if retry_count >= self.max_retries:
                    raise
                await asyncio.sleep(self._backoff_delay(retry_count))
                retry_count += 1
                continue

            if self._is_retryable(response) and retry_count < self.max_retries:
                delay = self._retry_after(response)
                if delay is None:
                    delay = self._backoff_delay(retry_count)
                await asyncio.sleep(delay)
                retry_count += 1
                continue

            response.raise_for_status()
            return JsonResponse(
                payload=response.json(),
                status_code=response.status_code,
                retry_count=retry_count,
                source=self.source,
                fetched_at=datetime.now(UTC),
            )

    async def _wait_for_interval(self) -> None:
        if self.request_interval_seconds <= 0:
            return
        loop = asyncio.get_running_loop()
        async with self._interval_lock:
            now = loop.time()
            if self._last_request_at is not None:
                wait = self.request_interval_seconds - (now - self._last_request_at)
                if wait > 0:
                    await asyncio.sleep(wait)
            self._last_request_at = loop.time()

    @staticmethod
    def _is_retryable(response: httpx.Response) -> bool:
        return response.status_code == 429 or 500 <= response.status_code < 600

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        value = response.headers.get("Retry-After")
        if value is None:
            return None
        try:
            return max(0, float(value))
        except ValueError:
            return None

    def _backoff_delay(self, retry_count: int) -> float:
        exponential = self.backoff_base_seconds * (2**retry_count)
        return float(exponential + random.uniform(0, self.jitter_seconds))
