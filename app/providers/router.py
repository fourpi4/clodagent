"""
ProviderRouter: a single LLMProvider-shaped facade over every registered
provider, implementing automatic fallback.

Fallback is allowed for: timeout, 429/rate limit, temporary 5xx, provider
unavailable, model unavailable. Fallback is NEVER used to silently paper
over: invalid API key (401/403), malformed request/tool schema (400/422) —
those abort the call immediately so the misconfiguration is visible.

Every provider switch is reported (without secrets) via an optional
`on_event` callback so the caller (Executor -> SSE stream) can surface
`provider_selected` / `provider_fallback` events to the UI.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from app.providers.base import ChatMessage, ChatResult, LLMProvider, ModelInfo, ProviderError, ToolSpec
from app.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)

EventCallback = Optional[Callable[[dict[str, Any]], Awaitable[None]]]

_MAX_RETRY_AFTER_SLEEP = 5.0


async def _noop_event(_event: dict[str, Any]) -> None:
    return None


def _classify_reason(exc: ProviderError) -> str:
    if exc.status_code == 429:
        return "rate_limit"
    if exc.status_code == 404:
        return "model_unavailable"
    if exc.status_code in (500, 502, 503, 504):
        return "server_error"
    if exc.status_code is None:
        return "timeout_or_unavailable"
    return "error"


class NoProviderAvailableError(ProviderError):
    pass


class ProviderRouter(LLMProvider):
    name = "router"

    def __init__(self, registry: ProviderRegistry, default_provider: str, fallback_order: list[str]) -> None:
        self._registry = registry
        self._default_provider = default_provider
        self._fallback_order = fallback_order

    def _candidate_names(self, explicit_provider: Optional[str]) -> list[str]:
        if explicit_provider:
            return [explicit_provider]
        order = [self._default_provider] + list(self._fallback_order)
        seen: set[str] = set()
        result = []
        for name in order:
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: Optional[list[ToolSpec]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        on_event: EventCallback = None,
    ) -> ChatResult:
        emit = on_event or _noop_event
        candidates = self._candidate_names(provider)
        last_exc: Optional[ProviderError] = None

        for index, name in enumerate(candidates):
            entry = self._registry.get_entry(name)
            if entry is None or not entry.configured or entry.provider is None:
                continue

            call_model = model if index == 0 else None  # a pinned model name is provider-specific
            try:
                result = await entry.provider.chat(
                    messages, tools=tools, temperature=temperature, max_tokens=max_tokens, model=call_model
                )
            except ProviderError as exc:
                if not exc.retryable:
                    logger.warning("Provider '%s' failed with a non-retryable error: %s", name, exc)
                    raise
                reason = _classify_reason(exc)
                logger.info("Provider '%s' failed (%s), trying next candidate", name, reason)
                if exc.retry_after:
                    await asyncio.sleep(min(exc.retry_after, _MAX_RETRY_AFTER_SLEEP))
                last_exc = exc
                next_name = candidates[index + 1] if index + 1 < len(candidates) else None
                await emit({"type": "provider_fallback", "from": name, "to": next_name, "reason": reason})
                continue

            await emit({"type": "provider_selected", "provider": name})
            return result

        if last_exc is not None:
            raise NoProviderAvailableError(
                f"No configured provider could complete the request (tried: {candidates}). Last error: {last_exc}",
                retryable=False,
            )
        raise NoProviderAvailableError(
            f"No provider is configured (tried: {candidates}). Set at least one *_API_KEY in your .env.",
            retryable=False,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: Optional[list[ToolSpec]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> AsyncIterator[str]:
        candidates = self._candidate_names(provider)
        for index, name in enumerate(candidates):
            entry = self._registry.get_entry(name)
            if entry is None or not entry.configured or entry.provider is None:
                continue
            call_model = model if index == 0 else None
            try:
                async for chunk in entry.provider.stream(
                    messages, tools=tools, temperature=temperature, max_tokens=max_tokens, model=call_model
                ):
                    yield chunk
                return
            except ProviderError as exc:
                if not exc.retryable:
                    raise
                logger.info("Provider '%s' streaming failed (%s), trying next candidate", name, _classify_reason(exc))
                continue
        raise NoProviderAvailableError(f"No configured provider could stream a response (tried: {candidates})")

    async def list_models(self) -> list[ModelInfo]:
        models: list[ModelInfo] = []
        for name in self._candidate_names(None):
            entry = self._registry.get_entry(name)
            if entry is None or not entry.configured or entry.provider is None:
                continue
            try:
                models.extend(await entry.provider.list_models())
            except ProviderError as exc:
                logger.warning("list_models failed for provider '%s': %s", name, exc)
        return models
