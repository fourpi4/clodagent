"""
Bytez LLM provider.

Bytez (https://bytez.com) exposes an OpenAI-compatible Chat Completions API
covering 200k+ open and closed source models behind a single API key:

    POST {BYTEZ_BASE_URL}/models/v2/openai/v1/chat/completions
    GET  {BYTEZ_BASE_URL}/models/v2/list/models

Auth header is `Authorization: <BYTEZ_API_KEY>` (no "Bearer " prefix).

Docs: https://docs.bytez.com/http-reference/examples/openai-compliant/chatCompletionsExample
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Optional

import httpx

from app.providers.base import ChatMessage, ChatResult, LLMProvider, ModelInfo, ProviderError, ToolSpec

logger = logging.getLogger(__name__)


def _messages_to_openai(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        entry: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.name:
            entry["name"] = m.name
        if m.tool_call_id:
            entry["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            entry["tool_calls"] = m.tool_calls
        out.append(entry)
    return out


def _tools_to_openai(tools: Optional[list[ToolSpec]]) -> Optional[list[dict[str, Any]]]:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


class BytezProvider(LLMProvider):
    """LLMProvider implementation backed by the Bytez inference API."""

    name = "bytez"

    def __init__(
        self,
        api_key: str,
        default_model: str,
        base_url: str = "https://api.bytez.com",
        timeout: int = 60,
    ) -> None:
        if not api_key:
            raise ProviderError(
                "BYTEZ_API_KEY is not set. Add it to your .env file (see .env.example)."
            )
        self._api_key = api_key
        self._default_model = default_model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": self._api_key, "Content-Type": "application/json"}

    def _endpoint(self, path: str) -> str:
        return f"{self._base_url}{path}"

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: Optional[list[ToolSpec]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": _messages_to_openai(messages),
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        openai_tools = _tools_to_openai(tools)
        if openai_tools:
            payload["tools"] = openai_tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.post(
                    self._endpoint("/models/v2/openai/v1/chat/completions"),
                    headers=self._headers(),
                    json=payload,
                )
            except httpx.HTTPError as exc:
                raise ProviderError(f"Bytez request failed: {exc}") from exc

        if resp.status_code != 200:
            raise ProviderError(f"Bytez API error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"Unexpected Bytez response shape: {data}") from exc

        return ChatResult(
            content=message.get("content") or "",
            tool_calls=message.get("tool_calls") or [],
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: Optional[list[ToolSpec]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": _messages_to_openai(messages),
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        openai_tools = _tools_to_openai(tools)
        if openai_tools:
            payload["tools"] = openai_tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    self._endpoint("/models/v2/openai/v1/chat/completions"),
                    headers=self._headers(),
                    json=payload,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        raise ProviderError(f"Bytez API error {resp.status_code}: {body[:500]!r}")
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        chunk = line[len("data:"):].strip()
                        if chunk == "[DONE]":
                            break
                        try:
                            event = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue
                        delta = event.get("choices", [{}])[0].get("delta", {})
                        text = delta.get("content")
                        if text:
                            yield text
            except httpx.HTTPError as exc:
                raise ProviderError(f"Bytez streaming request failed: {exc}") from exc

    async def list_models(self) -> list[ModelInfo]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.get(
                    self._endpoint("/models/v2/list/models"),
                    headers=self._headers(),
                )
            except httpx.HTTPError as exc:
                raise ProviderError(f"Bytez request failed: {exc}") from exc

        if resp.status_code != 200:
            raise ProviderError(f"Bytez API error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        if data.get("error"):
            raise ProviderError(f"Bytez list models error: {data['error']}")
        items = data.get("output", [])
        models: list[ModelInfo] = []
        for item in items or []:
            if isinstance(item, str):
                models.append(ModelInfo(id=item, provider="bytez"))
            elif isinstance(item, dict):
                models.append(
                    ModelInfo(
                        id=item.get("modelId") or item.get("id") or item.get("name", "unknown"),
                        provider="bytez",
                        description=item.get("description", ""),
                        task=item.get("task", ""),
                    )
                )
        return models
