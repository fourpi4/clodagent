"""
Bytez LLM provider.

Bytez (https://bytez.com) exposes an OpenAI-compatible Chat Completions API
covering 200k+ open and closed source models behind a single API key:

    POST {BYTEZ_BASE_URL}/models/v2/openai/v1/chat/completions
    GET  {BYTEZ_BASE_URL}/models/v2/list/models

Auth header is `Authorization: <BYTEZ_API_KEY>` (no "Bearer " prefix) for
this OpenAI-compatible surface. Docs:
https://docs.bytez.com/http-reference/examples/openai-compliant/chatCompletionsExample

Resilience note: Bytez's catalog is enormous (220k+ models) and individual
model IDs get renamed, deprecated, or are temporarily unavailable — and at
various points their own list/search endpoints have themselves returned
empty results even for IDs featured in their own official docs. Rather than
hard-failing the whole agent whenever the *configured* model is unavailable,
this provider transparently falls back through a short list of well-known
candidate model IDs (configurable via BYTEZ_MODEL_CANDIDATES), caches
whichever one first succeeds, and reuses it for the rest of the process
lifetime. This only kicks in for Bytez's specific "model does not exist"
response — auth errors, rate limits, and network failures are never masked
by this fallback and propagate immediately.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Optional

import httpx

from app.providers.base import ChatMessage, ChatResult, LLMProvider, ModelInfo, ProviderError, ToolSpec

logger = logging.getLogger(__name__)

# Well-known, widely-mirrored open model IDs, roughly smallest/cheapest
# first. These are only ever used if the configured BYTEZ_MODEL is
# unavailable on the account's catalog; the first one that actually works
# gets cached and reused.
DEFAULT_FALLBACK_MODELS: list[str] = [
    "openai-community/gpt2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
]

_NOT_FOUND_MARKERS = ("does not exist", "yet to be added", "not been added")


class ModelNotFoundError(ProviderError):
    """Raised internally when Bytez reports a specific model ID as unavailable."""

    def __init__(self, model: str, status_code: int, body: str) -> None:
        self.model = model
        self.status_code = status_code
        super().__init__(f"Bytez model '{model}' unavailable ({status_code}): {body}")


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


def _looks_like_model_not_found(status_code: int, body: str) -> bool:
    if status_code == 404:
        return True
    lowered = body.lower()
    return any(marker in lowered for marker in _NOT_FOUND_MARKERS)


class BytezProvider(LLMProvider):
    """LLMProvider implementation backed by the Bytez inference API."""

    name = "bytez"

    def __init__(
        self,
        api_key: str,
        default_model: str,
        base_url: str = "https://api.bytez.com",
        timeout: int = 60,
        fallback_models: Optional[list[str]] = None,
    ) -> None:
        if not api_key:
            raise ProviderError(
                "BYTEZ_API_KEY is not set. Add it to your .env file (see .env.example)."
            )
        self._api_key = api_key
        self._default_model = default_model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._fallback_models = fallback_models if fallback_models is not None else DEFAULT_FALLBACK_MODELS
        # Cache of the first model ID that has been confirmed to actually work,
        # so we don't re-probe candidates on every single call.
        self._confirmed_model: Optional[str] = None

    def _headers(self) -> dict[str, str]:
        return {"Authorization": self._api_key, "Content-Type": "application/json"}

    def _endpoint(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _candidate_order(self, explicit_model: Optional[str]) -> list[str]:
        if explicit_model:
            # Caller asked for a specific model by name (e.g. via an AgentProfile
            # override) — respect that exactly, no silent substitution.
            return [explicit_model]
        if self._confirmed_model:
            return [self._confirmed_model]
        ordered = [self._default_model] + [m for m in self._fallback_models if m != self._default_model]
        return ordered

    async def _post_completion(self, client: httpx.AsyncClient, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        request_payload = {**payload, "model": model}
        try:
            resp = await client.post(
                self._endpoint("/models/v2/openai/v1/chat/completions"),
                headers=self._headers(),
                json=request_payload,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"Bytez request failed: {exc}") from exc

        if resp.status_code != 200:
            body = resp.text[:500]
            if _looks_like_model_not_found(resp.status_code, body):
                raise ModelNotFoundError(model, resp.status_code, body)
            raise ProviderError(f"Bytez API error {resp.status_code}: {body}")

        return resp.json()

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

        candidates = self._candidate_order(model)
        last_error: Optional[Exception] = None

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for candidate in candidates:
                try:
                    data = await self._post_completion(client, candidate, payload)
                except ModelNotFoundError as exc:
                    logger.info("Bytez model '%s' unavailable, trying next candidate", candidate)
                    last_error = exc
                    continue

                if not model and candidate != self._confirmed_model:
                    self._confirmed_model = candidate
                    logger.info("Bytez: using model '%s' for this session", candidate)

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

        assert last_error is not None
        raise ProviderError(
            f"None of the configured Bytez models are available on this account "
            f"(tried: {', '.join(candidates)}). Last error: {last_error}"
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
        resolved_model = model or self._confirmed_model or self._default_model
        payload: dict[str, Any] = {
            "model": resolved_model,
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

        if not models:
            # The catalog list endpoint can legitimately return nothing (empty
            # account catalog, transient backend issue, etc). Surface the
            # fallback candidates instead of an empty list so the UI/API
            # caller always has something actionable to try.
            models = [
                ModelInfo(
                    id=self._confirmed_model or self._default_model,
                    provider="bytez",
                    description="Configured default model (catalog list returned empty)",
                    task="unknown",
                )
            ] + [
                ModelInfo(id=m, provider="bytez", description="Untested fallback candidate", task="unknown")
                for m in self._fallback_models
            ]
        return models
