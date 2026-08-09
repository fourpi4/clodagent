"""
OpenRouter provider — unified access to 400+ models, including a rotating
set of free (":free" suffix) models. Fully OpenAI-compatible:
https://openrouter.ai/docs/quickstart

    Base URL: https://openrouter.ai/api/v1
    Auth:     Authorization: Bearer <OPENROUTER_API_KEY>

Not every free model supports tool/function calling. When OpenRouter (or
the upstream model) rejects a request specifically because of tool-calling
capability, we mark that failure retryable so ProviderRouter can fall
through to the next configured provider instead of treating it as fatal.
"""
from __future__ import annotations

from typing import Optional

from app.providers.base import ChatMessage, ChatResult, ProviderError, ToolSpec
from app.providers.openai_compatible import OpenAICompatibleProvider

_CAPABILITY_MISMATCH_MARKERS = (
    "tool",
    "function calling",
    "function_call",
    "does not support",
    "not supported",
)


def _looks_like_capability_mismatch(message: str) -> bool:
    lowered = message.lower()
    return "tool" in lowered and any(marker in lowered for marker in _CAPABILITY_MISMATCH_MARKERS)


class OpenRouterProvider(OpenAICompatibleProvider):
    def __init__(
        self, api_key: str, default_model: str, base_url: str = "https://openrouter.ai/api/v1", timeout: int = 60
    ) -> None:
        super().__init__(
            name="openrouter",
            api_key=api_key,
            default_model=default_model,
            base_url=base_url,
            timeout=timeout,
            env_hint="Set OPENROUTER_API_KEY (get a free key at https://openrouter.ai/keys).",
        )

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
        try:
            return await super().chat(
                messages, tools=tools, temperature=temperature, max_tokens=max_tokens, model=model
            )
        except ProviderError as exc:
            if tools and not exc.retryable and _looks_like_capability_mismatch(str(exc)):
                raise ProviderError(
                    f"{exc} (treated as a tool-calling capability mismatch on model "
                    f"'{model or self._default_model}' — falling back)",
                    status_code=exc.status_code,
                    retryable=True,
                ) from exc
            raise
