"""
Provider-agnostic LLM interface.

Any concrete provider (Bytez today, others later) implements this contract.
The Agent Core only ever talks to `LLMProvider`, never to a concrete SDK, so
adding a new model backend never requires touching agent/core.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ChatResult:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    id: str
    provider: str
    description: str = ""
    task: str = ""


class LLMProvider(ABC):
    """Contract every model backend must satisfy."""

    name: str = "base"

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: Optional[list[ToolSpec]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> ChatResult:
        """
        Single (non-streaming) chat completion, optionally tool-aware.

        `provider` is a hint meaningful only to a routing/multi-provider
        implementation (see providers/router.py) — pins the call to one
        named provider instead of using the default+fallback order. Concrete
        single-backend providers accept and ignore it.
        """
        raise NotImplementedError

    @abstractmethod
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
        """Yield incremental text chunks as they are produced."""
        raise NotImplementedError
        yield ""  # pragma: no cover - makes this an async generator

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Return the models this provider can currently serve."""
        raise NotImplementedError


class ProviderError(RuntimeError):
    """
    Raised when a provider call fails (network, auth, malformed response).

    Carries enough metadata for ProviderRouter to decide whether a fallback
    to the next provider is appropriate:
      - retryable=True  -> timeout, 429, temporary 5xx, model/provider unavailable
      - retryable=False -> invalid API key (401/403), malformed request (400/422)
    `retry_after` is populated from a 429 response's Retry-After header when present.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        retryable: bool = False,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after


_RETRYABLE_STATUS_CODES = {404, 408, 429, 500, 502, 503, 504}
_NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 422}


def is_retryable_status(status_code: Optional[int]) -> bool:
    if status_code is None:
        return True  # network error / timeout, no HTTP status at all
    if status_code in _NON_RETRYABLE_STATUS_CODES:
        return False
    return status_code in _RETRYABLE_STATUS_CODES
