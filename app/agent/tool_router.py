"""
Tool routing: turns an LLM's tool_calls (OpenAI function-calling format)
into ToolRegistry executions, respecting per-run call budgets and timeouts.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from app.providers.base import ToolSpec
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class RoutedToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    result: ToolResult


class ToolRouter:
    def __init__(self, registry: ToolRegistry, *, tool_timeout: float, max_retries: int) -> None:
        self._registry = registry
        self._tool_timeout = tool_timeout
        self._max_retries = max_retries

    def available_specs(self, allowed_names: Optional[list[str]] = None) -> list[ToolSpec]:
        specs = []
        for described in self._registry.list_tools():
            if allowed_names and described["name"] not in allowed_names:
                continue
            specs.append(
                ToolSpec(
                    name=described["name"],
                    description=described["description"],
                    input_schema=described["input_schema"],
                )
            )
        return specs

    async def dispatch(self, tool_call: dict[str, Any]) -> RoutedToolCall:
        call_id = tool_call.get("id", "")
        function = tool_call.get("function", {})
        name = function.get("name", "")
        raw_args = function.get("arguments", "{}")
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except json.JSONDecodeError:
            return RoutedToolCall(
                id=call_id, name=name, arguments={},
                result=ToolResult(ok=False, error=f"Malformed tool arguments JSON: {raw_args!r}"),
            )

        last_result: Optional[ToolResult] = None
        for attempt in range(self._max_retries + 1):
            last_result = await self._registry.execute_tool(name, arguments, timeout=self._tool_timeout)
            if last_result.ok:
                break
            logger.info("Tool '%s' attempt %d failed: %s", name, attempt + 1, last_result.error)

        assert last_result is not None
        return RoutedToolCall(id=call_id, name=name, arguments=arguments, result=last_result)
