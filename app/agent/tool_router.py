"""
Tool routing: turns an LLM's tool_calls (OpenAI function-calling format)
into ToolRegistry executions, respecting per-run call budgets, timeouts,
and retry-safety.

Confirmation gating (requires_confirmation) happens *upstream* in the
Executor, before dispatch() is ever called — by the time a tool call
reaches this router it has already been cleared for execution.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from app.providers.base import ToolSpec
from app.tools.base import Tool, ToolError, ToolResult
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

    def available_specs(
        self,
        allowed_names: Optional[list[str]] = None,
        allowed_mcp_servers: Optional[list[str]] = None,
    ) -> list[ToolSpec]:
        """
        `allowed_names` filters by exact tool name (AgentProfile.tools; empty/None = no name restriction).

        `allowed_mcp_servers` filters MCP-backed tools (named "mcp__<server>__<tool>") by their source
        server (AgentProfile.mcp_servers). Safe default: if empty/None, NO mcp__-prefixed tools are ever
        exposed — a profile only sees MCP tools from servers it explicitly lists.
        """
        specs = []
        for described in self._registry.list_tools():
            name = described["name"]
            if allowed_names and name not in allowed_names:
                continue
            if name.startswith("mcp__"):
                parts = name.split("__", 2)
                server = parts[1] if len(parts) >= 2 else None
                if not allowed_mcp_servers or server not in allowed_mcp_servers:
                    continue
            specs.append(
                ToolSpec(name=name, description=described["description"], input_schema=described["input_schema"])
            )
        return specs

    def lookup(self, name: str) -> Optional[Tool]:
        """Looks up a registered tool by name, or None if it doesn't exist."""
        try:
            return self._registry.get_tool(name)
        except ToolError:
            return None

    @staticmethod
    def parse_arguments(tool_call: dict[str, Any]) -> tuple[str, str, dict[str, Any], Optional[str]]:
        """Returns (call_id, tool_name, arguments, parse_error)."""
        call_id = tool_call.get("id", "")
        function = tool_call.get("function", {})
        name = function.get("name", "")
        raw_args = function.get("arguments", "{}")
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except json.JSONDecodeError:
            return call_id, name, {}, f"Malformed tool arguments JSON: {raw_args!r}"
        return call_id, name, arguments, None

    async def dispatch(self, tool_call: dict[str, Any]) -> RoutedToolCall:
        call_id, name, arguments, parse_error = self.parse_arguments(tool_call)
        if parse_error:
            return RoutedToolCall(id=call_id, name=name, arguments={}, result=ToolResult(ok=False, error=parse_error))

        # Never blindly retry a tool with side effects unless it is
        # explicitly marked idempotent/retry-safe (e.g. a read-only lookup).
        # Unknown tools also get exactly one attempt — retrying won't make a
        # hallucinated tool name exist.
        tool = self.lookup(name)
        allow_retry = tool is not None and tool.retry_safe
        attempts = (self._max_retries + 1) if allow_retry else 1

        last_result: Optional[ToolResult] = None
        for attempt in range(attempts):
            last_result = await self._registry.execute_tool(name, arguments, timeout=self._tool_timeout)
            if last_result.ok:
                break
            logger.info("Tool '%s' attempt %d/%d failed: %s", name, attempt + 1, attempts, last_result.error)

        assert last_result is not None
        return RoutedToolCall(id=call_id, name=name, arguments=arguments, result=last_result)
