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
    ) -> ChatResult:
        """Single (non-streaming) chat completion, optionally tool-aware."""
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
    ) -> AsyncIterator[str]:
        """Yield incremental text chunks as they are produced."""
        raise NotImplementedError
        yield ""  # pragma: no cover - makes this an async generator

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Return the models this provider can currently serve."""
        raise NotImplementedError


class ProviderError(RuntimeError):
    """Raised when a provider call fails (network, auth, malformed response)."""
