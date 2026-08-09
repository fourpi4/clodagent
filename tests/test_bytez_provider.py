"""
Tests for BytezProvider's model-fallback resilience: if the configured
model is unavailable on the account's catalog, the provider should
transparently try the next candidate rather than failing the whole request.
Uses httpx.MockTransport so no real network calls are made.
"""
import json

import httpx
import pytest

from app.providers.base import ChatMessage, ProviderError
from app.providers.bytez import BytezProvider

NOT_FOUND_BODY = json.dumps(
    {"error": "Model does not exist or has yet to be added to the Bytez catalog.", "output": None}
)


def _completion_body(text: str) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }


@pytest.fixture(autouse=True)
def _patch_async_client(monkeypatch):
    """Redirect every httpx.AsyncClient() construction in bytez.py through a per-test handler."""
    handler_holder: dict = {}

    real_async_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        handler = handler_holder.get("handler")
        if handler is not None:
            kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("app.providers.bytez.httpx.AsyncClient", factory)
    return handler_holder


async def test_falls_back_to_next_candidate_when_default_missing(_patch_async_client):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body["model"])
        if body["model"] == "working/model":
            return httpx.Response(200, json=_completion_body("hello from fallback"))
        return httpx.Response(404, text=NOT_FOUND_BODY)

    _patch_async_client["handler"] = handler
    provider = BytezProvider(
        api_key="test-key",
        default_model="broken/model",
        timeout=5,
        fallback_models=["also-broken/model", "working/model"],
    )

    result = await provider.chat([ChatMessage(role="user", content="hi")])

    assert result.content == "hello from fallback"
    assert calls == ["broken/model", "also-broken/model", "working/model"]


async def test_caches_working_model_after_first_success(_patch_async_client):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body["model"])
        if body["model"] == "working/model":
            return httpx.Response(200, json=_completion_body("ok"))
        return httpx.Response(404, text=NOT_FOUND_BODY)

    _patch_async_client["handler"] = handler
    provider = BytezProvider(
        api_key="test-key",
        default_model="broken/model",
        timeout=5,
        fallback_models=["working/model"],
    )

    await provider.chat([ChatMessage(role="user", content="hi")])
    calls.clear()
    await provider.chat([ChatMessage(role="user", content="hi again")])

    assert calls == ["working/model"]  # no re-probing of the broken default


async def test_explicit_model_override_is_not_substituted(_patch_async_client):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body["model"])
        return httpx.Response(404, text=NOT_FOUND_BODY)

    _patch_async_client["handler"] = handler
    provider = BytezProvider(
        api_key="test-key",
        default_model="broken/model",
        timeout=5,
        fallback_models=["working/model"],
    )

    with pytest.raises(ProviderError):
        await provider.chat([ChatMessage(role="user", content="hi")], model="explicit/choice")

    assert calls == ["explicit/choice"]  # never substitutes a caller-specified model


async def test_auth_error_is_not_masked_by_fallback(_patch_async_client):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body["model"])
        return httpx.Response(401, json={"error": "Unauthorized"})

    _patch_async_client["handler"] = handler
    provider = BytezProvider(
        api_key="bad-key",
        default_model="broken/model",
        timeout=5,
        fallback_models=["working/model"],
    )

    with pytest.raises(ProviderError, match="401"):
        await provider.chat([ChatMessage(role="user", content="hi")])

    assert calls == ["broken/model"]  # stops immediately, does not burn through fallbacks on auth errors


async def test_raises_clear_error_when_all_candidates_fail(_patch_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text=NOT_FOUND_BODY)

    _patch_async_client["handler"] = handler
    provider = BytezProvider(
        api_key="test-key",
        default_model="broken/model",
        timeout=5,
        fallback_models=["also-broken/model"],
    )

    with pytest.raises(ProviderError, match="None of the configured Bytez models are available"):
        await provider.chat([ChatMessage(role="user", content="hi")])
