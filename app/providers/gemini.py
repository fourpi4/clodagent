"""
Google Gemini provider, via the official REST API (no SDK dependency):
https://ai.google.dev/gemini-api/docs/generate-content/function-calling

Gemini's wire format differs substantially from OpenAI's:
  - a separate top-level `systemInstruction` instead of a system-role message
  - `contents[].role` is "user" or "model" (never "assistant"/"tool")
  - tool calls are `functionCall` parts on a "model" content entry
  - tool results are `functionResponse` parts on a "user" content entry

This provider is entirely responsible for translating our internal
ChatMessage/ToolSpec/ChatResult shapes to/from Gemini's — Agent Core and
every other part of the system only ever see the normal LLMProvider
contract, unchanged.
"""
from __future__ import annotations

import json
import logging
import uuid
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


def _build_contents_and_system(messages: list[ChatMessage]) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for m in messages:
        if m.role == "system":
            if m.content:
                system_parts.append(m.content)
            continue

        if m.role == "assistant":
            role = "model"
            parts: list[dict[str, Any]] = []
            if m.content:
                parts.append({"text": m.content})
            for call in m.tool_calls or []:
                function = call.get("function", {})
                try:
                    args = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                parts.append({"functionCall": {"name": function.get("name", ""), "id": call.get("id", ""), "args": args}})
        elif m.role == "tool":
            role = "user"
            try:
                response_obj = json.loads(m.content) if m.content else {}
                if not isinstance(response_obj, dict):
                    response_obj = {"result": response_obj}
            except json.JSONDecodeError:
                response_obj = {"result": m.content}
            parts = [{"functionResponse": {"name": m.name or "", "id": m.tool_call_id or "", "response": response_obj}}]
        else:  # "user"
            role = "user"
            parts = [{"text": m.content}]

        # Gemini expects strictly alternating user/model turns; merge
        # consecutive same-role entries (e.g. several tool results in a row)
        # into one content block instead of sending several in sequence.
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": role, "parts": parts})

    system_instruction = {"parts": [{"text": "\n".join(system_parts)}]} if system_parts else None
    return system_instruction, contents


def _tools_to_gemini(tools: Optional[list[ToolSpec]]) -> Optional[list[dict[str, Any]]]:
    if not tools:
        return None
    return [
        {
            "functionDeclarations": [
                {"name": t.name, "description": t.description, "parameters": t.input_schema} for t in tools
            ]
        }
    ]


def _parts_to_chat_result(candidate: dict[str, Any], raw: dict[str, Any]) -> ChatResult:
    parts = candidate.get("content", {}).get("parts", [])
    text_chunks: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for part in parts:
        if "text" in part:
            text_chunks.append(part["text"])
        elif "functionCall" in part:
            fc = part["functionCall"]
            call_id = fc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": fc.get("name", ""), "arguments": json.dumps(fc.get("args", {}))},
                }
            )
    finish_reason = "tool_calls" if tool_calls else str(candidate.get("finishReason", "STOP")).lower()
    return ChatResult(content="".join(text_chunks), tool_calls=tool_calls, finish_reason=finish_reason, raw=raw)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        default_model: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: int = 60,
    ) -> None:
        if not api_key:
            raise ProviderError(
                "GEMINI_API_KEY is not set (get a free key at https://aistudio.google.com/apikey).",
                retryable=False,
            )
        self._api_key = api_key
        self._default_model = default_model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}

    def _model_endpoint(self, model: str, method: str) -> str:
        model_path = model if model.startswith("models/") else f"models/{model}"
        return f"{self._base_url}/{model_path}:{method}"

    def _build_payload(
        self,
        messages: list[ChatMessage],
        *,
        tools: Optional[list[ToolSpec]],
        temperature: float,
        max_tokens: Optional[int],
    ) -> dict[str, Any]:
        system_instruction, contents = _build_contents_and_system(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        gemini_tools = _tools_to_gemini(tools)
        if gemini_tools:
            payload["tools"] = gemini_tools
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
        resolved_model = model or self._default_model
        payload = self._build_payload(messages, tools=tools, temperature=temperature, max_tokens=max_tokens)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.post(
                    self._model_endpoint(resolved_model, "generateContent"), headers=self._headers(), json=payload
                )
            except httpx.HTTPError as exc:
                raise ProviderError(f"Gemini request failed: {exc}", retryable=True) from exc

        if resp.status_code != 200:
            body = resp.text[:500]
            raise ProviderError(
                f"Gemini API error {resp.status_code}: {body}",
                status_code=resp.status_code,
                retryable=is_retryable_status(resp.status_code),
            )

        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise ProviderError(f"Gemini returned no candidates: {data}")
        return _parts_to_chat_result(candidates[0], data)

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
        resolved_model = model or self._default_model
        payload = self._build_payload(messages, tools=tools, temperature=temperature, max_tokens=max_tokens)
        url = f"{self._model_endpoint(resolved_model, 'streamGenerateContent')}?alt=sse"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                async with client.stream("POST", url, headers=self._headers(), json=payload) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        raise ProviderError(
                            f"Gemini API error {resp.status_code}: {body[:500]!r}",
                            status_code=resp.status_code,
                            retryable=is_retryable_status(resp.status_code),
                        )
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        chunk = line[len("data:"):].strip()
                        if not chunk:
                            continue
                        try:
                            event = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue
                        for candidate in event.get("candidates", []):
                            for part in candidate.get("content", {}).get("parts", []):
                                text = part.get("text")
                                if text:
                                    yield text
            except httpx.HTTPError as exc:
                raise ProviderError(f"Gemini streaming request failed: {exc}", retryable=True) from exc

    async def list_models(self) -> list[ModelInfo]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.get(f"{self._base_url}/models", headers=self._headers())
            except httpx.HTTPError as exc:
                raise ProviderError(f"Gemini request failed: {exc}", retryable=True) from exc

        if resp.status_code != 200:
            raise ProviderError(
                f"Gemini API error {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
                retryable=is_retryable_status(resp.status_code),
            )

        data = resp.json()
        models: list[ModelInfo] = []
        for item in data.get("models", []):
            methods = item.get("supportedGenerationMethods", [])
            if methods and "generateContent" not in methods:
                continue
            model_id = item.get("name", "").removeprefix("models/")
            models.append(ModelInfo(id=model_id, provider="gemini", description=item.get("displayName", "")))
        return models
