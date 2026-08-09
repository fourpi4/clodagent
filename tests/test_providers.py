"""
Provider response-mapping tests (Groq, OpenRouter, Gemini, Bytez) and
ProviderRouter fallback behavior. Everything is mocked HTTP via
httpx.MockTransport — no real API keys or network calls required.
"""
import json

import httpx
import pytest

from app.providers.base import ChatMessage, ProviderError
from app.providers.gemini import GeminiProvider
from app.providers.groq import GroqProvider
from app.providers.openrouter import OpenRouterProvider
from tests.conftest import make_multi_provider_router


def _openai_completion(text: str = "hello", tool_calls=None) -> dict:
    message = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = ""
    return {
        "id": "x", "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
    }


# --- Groq -------------------------------------------------------------------

async def test_groq_maps_response_to_chat_result(mock_httpx):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=_openai_completion("Hi from Groq"))

    mock_httpx["handler"] = handler
    provider = GroqProvider(api_key="test-groq-key", default_model="llama-3.3-70b-versatile")
    result = await provider.chat([ChatMessage(role="user", content="hi")])

    assert result.content == "Hi from Groq"
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer test-groq-key"


async def test_groq_maps_tool_calls(mock_httpx):
    tool_calls = [{"id": "c1", "type": "function", "function": {"name": "add", "arguments": "{}"}}]
    mock_httpx["handler"] = lambda r: httpx.Response(200, json=_openai_completion(tool_calls=tool_calls))
    provider = GroqProvider(api_key="k", default_model="m")
    result = await provider.chat([ChatMessage(role="user", content="hi")])
    assert result.tool_calls == tool_calls
    assert result.finish_reason == "tool_calls"


# --- OpenRouter ---------------------------------------------------------------

async def test_openrouter_maps_response(mock_httpx):
    mock_httpx["handler"] = lambda r: httpx.Response(200, json=_openai_completion("Hi from OpenRouter"))
    provider = OpenRouterProvider(api_key="k", default_model="meta-llama/llama-3.3-70b-instruct:free")
    result = await provider.chat([ChatMessage(role="user", content="hi")])
    assert result.content == "Hi from OpenRouter"


async def test_openrouter_capability_mismatch_marked_retryable(mock_httpx):
    body = {"error": "This model does not support tool use / function calling"}
    mock_httpx["handler"] = lambda r: httpx.Response(400, json=body)
    provider = OpenRouterProvider(api_key="k", default_model="some/free-model")
    from app.providers.base import ToolSpec

    with pytest.raises(ProviderError) as excinfo:
        await provider.chat(
            [ChatMessage(role="user", content="hi")],
            tools=[ToolSpec(name="x", description="x", input_schema={"type": "object", "properties": {}})],
        )
    assert excinfo.value.retryable is True


# --- Gemini -------------------------------------------------------------------

def _gemini_response(text=None, function_call=None) -> dict:
    parts = []
    if text:
        parts.append({"text": text})
    if function_call:
        parts.append({"functionCall": function_call})
    return {"candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}]}


async def test_gemini_maps_text_response(mock_httpx):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key_header"] = request.headers.get("x-goog-api-key")
        return httpx.Response(200, json=_gemini_response(text="Hi from Gemini"))

    mock_httpx["handler"] = handler
    provider = GeminiProvider(api_key="gk", default_model="gemini-2.0-flash")
    result = await provider.chat([ChatMessage(role="user", content="hi")])

    assert result.content == "Hi from Gemini"
    assert "gemini-2.0-flash:generateContent" in seen["url"]
    assert seen["key_header"] == "gk"


async def test_gemini_maps_function_call_to_tool_calls(mock_httpx):
    fc = {"name": "add", "id": "call-1", "args": {"a": 1, "b": 2}}
    mock_httpx["handler"] = lambda r: httpx.Response(200, json=_gemini_response(function_call=fc))
    provider = GeminiProvider(api_key="gk", default_model="gemini-2.0-flash")
    result = await provider.chat([ChatMessage(role="user", content="hi")])

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["function"]["name"] == "add"
    assert json.loads(result.tool_calls[0]["function"]["arguments"]) == {"a": 1, "b": 2}
    assert result.finish_reason == "tool_calls"


async def test_gemini_translates_system_message_and_tool_result(mock_httpx):
    """Verifies our ChatMessage -> Gemini contents/systemInstruction translation shape."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_gemini_response(text="ok"))

    mock_httpx["handler"] = handler
    provider = GeminiProvider(api_key="gk", default_model="gemini-2.0-flash")
    await provider.chat([
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="", tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "add", "arguments": "{\"a\":1}"}}
        ]),
        ChatMessage(role="tool", tool_call_id="c1", name="add", content='{"result": 3}'),
    ])

    body = captured["body"]
    assert body["systemInstruction"]["parts"][0]["text"] == "You are helpful."
    roles = [c["role"] for c in body["contents"]]
    assert roles == ["user", "model", "user"]  # tool result merged into a "user" content block
    assert body["contents"][-1]["parts"][0]["functionResponse"]["name"] == "add"


# --- ProviderRouter fallback ---------------------------------------------------

class _StubProvider:
    name = "stub"

    def __init__(self, behavior):
        self._behavior = behavior  # callable(messages) -> ChatResult | raises ProviderError

    async def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=None, model=None, provider=None):
        return self._behavior()

    async def stream(self, messages, *, tools=None, temperature=0.2, max_tokens=None, model=None, provider=None):
        yield ""

    async def list_models(self):
        return []


def _ok(content="ok"):
    from app.providers.base import ChatResult
    return lambda: ChatResult(content=content)


def _fail(**kwargs):
    def raiser():
        raise ProviderError("boom", **kwargs)
    return raiser


async def test_provider_router_falls_back_on_retryable_error(monkeypatch):
    monkeypatch.setattr("app.providers.router.asyncio.sleep", lambda *_: _instant())
    primary = _StubProvider(_fail(status_code=503, retryable=True))
    fallback = _StubProvider(_ok("from fallback"))
    router, _ = make_multi_provider_router({"primary": primary, "fallback": fallback}, default_provider="primary")

    result = await router.chat([ChatMessage(role="user", content="hi")])
    assert result.content == "from fallback"


async def test_provider_router_429_triggers_fallback_and_respects_retry_after(monkeypatch):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("app.providers.router.asyncio.sleep", fake_sleep)
    primary = _StubProvider(_fail(status_code=429, retryable=True, retry_after=2.0))
    fallback = _StubProvider(_ok("from fallback"))
    router, _ = make_multi_provider_router({"primary": primary, "fallback": fallback}, default_provider="primary")

    result = await router.chat([ChatMessage(role="user", content="hi")])
    assert result.content == "from fallback"
    assert sleeps == [2.0]


async def test_provider_router_401_does_not_fall_back(monkeypatch):
    monkeypatch.setattr("app.providers.router.asyncio.sleep", lambda *_: _instant())
    primary = _StubProvider(_fail(status_code=401, retryable=False))
    fallback = _StubProvider(_ok("should not be used"))
    router, _ = make_multi_provider_router({"primary": primary, "fallback": fallback}, default_provider="primary")

    with pytest.raises(ProviderError) as excinfo:
        await router.chat([ChatMessage(role="user", content="hi")])
    assert excinfo.value.status_code == 401


async def _instant():
    return None
