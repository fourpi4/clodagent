"""
Shared implementation for any provider exposing an OpenAI-compatible Chat
Completions API (Groq, OpenRouter, Mistral, and — partially — Bytez).

Concrete providers only need to supply: name, api_key, default_model,
base_url, and (optionally) a custom auth header. Message/tool conversion,
streaming SSE parsing, and error classification are implemented once here.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Optional

import httpx

from app.providers.base import (
    ChatMessage,
    ChatResult,
    LLMProvider,
    ModelInfo,
    ProviderError,
    ToolSpec,
    is_retryable_status,
)

logger = logging.getLogger(__name__)


def messages_to_openai(messages: list[ChatMessage]) -> list[dict[str, Any]]:
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


def tools_to_openai(tools: Optional[list[ToolSpec]]) -> Optional[list[dict[str, Any]]]:
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


def _retry_after_seconds(response: httpx.Response) -> Optional[float]:
    header = response.headers.get("Retry-After")
    if not header:
        return None
    try:
        return float(header)
    except ValueError:
        return None


def raise_for_response(provider_name: str, resp: httpx.Response) -> None:
    if resp.status_code == 200:
        return
    body = resp.text[:500]
    raise ProviderError(
        f"{provider_name} API error {resp.status_code}: {body}",
        status_code=resp.status_code,
        retryable=is_retryable_status(resp.status_code),
        retry_after=_retry_after_seconds(resp) if resp.status_code == 429 else None,
    )


class OpenAICompatibleProvider(LLMProvider):
    """Concrete LLMProvider for any strictly-OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        default_model: str,
        base_url: str,
        timeout: int = 60,
        chat_path: str = "/chat/completions",
        models_path: str = "/models",
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer ",
        extra_headers: Optional[dict[str, str]] = None,
        env_hint: str = "",
    ) -> None:
        if not api_key:
            raise ProviderError(f"{name}: API key is not set. {env_hint}".strip())
        self.name = name
        self._api_key = api_key
        self._default_model = default_model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._chat_path = chat_path
        self._models_path = models_path
        self._auth_header = auth_header
        self._auth_prefix = auth_prefix
        self._extra_headers = extra_headers or {}

    def _headers(self) -> dict[str, str]:
        return {
            self._auth_header: f"{self._auth_prefix}{self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }

    def _endpoint(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _build_payload(
        self,
        messages: list[ChatMessage],
        *,
        tools: Optional[list[ToolSpec]],
        temperature: float,
        max_tokens: Optional[int],
        model: Optional[str],
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": messages_to_openai(messages),
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        openai_tools = tools_to_openai(tools)
        if openai_tools:
            payload["tools"] = openai_tools
            payload["tool_choice"] = "auto"
        return payload

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: Optional[list[ToolSpec]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,  # noqa: ARG002 - accepted for LLMProvider interface compat, unused here
    ) -> ChatResult:
        payload = self._build_payload(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens, model=model, stream=False
        )
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.post(self._endpoint(self._chat_path), headers=self._headers(), json=payload)
            except httpx.HTTPError as exc:
                raise ProviderError(f"{self.name} request failed: {exc}", retryable=True) from exc

        raise_for_response(self.name, resp)
        data = resp.json()
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"Unexpected {self.name} response shape: {data}") from exc

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
        provider: Optional[str] = None,  # noqa: ARG002 - accepted for LLMProvider interface compat, unused here
    ) -> AsyncIterator[str]:
        payload = self._build_payload(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens, model=model, stream=True
        )
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                async with client.stream(
                    "POST", self._endpoint(self._chat_path), headers=self._headers(), json=payload
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        raise ProviderError(
                            f"{self.name} API error {resp.status_code}: {body[:500]!r}",
                            status_code=resp.status_code,
                            retryable=is_retryable_status(resp.status_code),
                        )
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
                raise ProviderError(f"{self.name} streaming request failed: {exc}", retryable=True) from exc

    async def list_models(self) -> list[ModelInfo]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.get(self._endpoint(self._models_path), headers=self._headers())
            except httpx.HTTPError as exc:
                raise ProviderError(f"{self.name} request failed: {exc}", retryable=True) from exc

        raise_for_response(self.name, resp)
        data = resp.json()
        items = data.get("data", data) if isinstance(data, dict) else data
        models: list[ModelInfo] = []
        for item in items or []:
            if isinstance(item, str):
                models.append(ModelInfo(id=item, provider=self.name))
            elif isinstance(item, dict):
                models.append(
                    ModelInfo(
                        id=item.get("id", "unknown"),
                        provider=self.name,
                        description=item.get("description", ""),
                        task=item.get("object", ""),
                    )
                )
        return models
