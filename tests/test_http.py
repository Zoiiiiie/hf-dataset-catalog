import httpx
import pytest
import respx

from hf_dataset_catalog.http import RetryingJsonClient


@pytest.mark.asyncio
async def test_retry_after_429_then_return_success() -> None:
    async with httpx.AsyncClient() as http_client:
        client = RetryingJsonClient(
            source="test",
            client=http_client,
            max_retries=2,
            backoff_base_seconds=0,
            jitter_seconds=0,
        )
        with respx.mock:
            route = respx.get("https://example.test/data").mock(
                side_effect=[
                    httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "busy"}),
                    httpx.Response(200, json={"ok": True}),
                ]
            )
            response = await client.get_json("https://example.test/data")

    assert response.payload == {"ok": True}
    assert response.retry_count == 1
    assert response.status_code == 200
    assert route.call_count == 2


@pytest.mark.asyncio
async def test_non_retryable_400_is_raised_once() -> None:
    async with httpx.AsyncClient() as http_client:
        client = RetryingJsonClient(source="test", client=http_client, max_retries=3)
        with respx.mock:
            route = respx.get("https://example.test/bad").mock(
                return_value=httpx.Response(400, json={"error": "bad request"})
            )
            with pytest.raises(httpx.HTTPStatusError):
                await client.get_json("https://example.test/bad")

    assert route.call_count == 1
