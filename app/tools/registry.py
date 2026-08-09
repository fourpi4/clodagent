"""
Central registry the Agent Core and API use to discover and invoke tools.

Tools can come from three places, all normalized to the same `Tool` interface:
  - built-in tools (web_tools, github_tools, filesystem_tools)
  - MCP servers (via integrations/mcp.py -> McpToolAdapter)
  - third-party adapters (integrations/*_adapter.py)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.tools.base import Tool, ToolError, ToolInputError, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register_tool(self, tool: Tool, *, overwrite: bool = False) -> None:
        if tool.name in self._tools and not overwrite:
            raise ToolError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def remove_tool(self, name: str) -> None:
        if name not in self._tools:
            raise ToolError(f"Tool '{name}' is not registered")
        del self._tools[name]
        logger.info("Removed tool: %s", name)

    def list_tools(self) -> list[dict]:
        return [t.describe() for t in self._tools.values()]

    def get_tool(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"Tool '{name}' is not registered")
        return tool

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    async def execute_tool(
        self,
        name: str,
        arguments: dict,
        *,
        timeout: Optional[float] = None,
    ) -> ToolResult:
        try:
            tool = self.get_tool(name)
        except ToolError:
            # The model hallucinated a tool name that doesn't exist. This must
            # never crash the run — hand the model a clear error so it can
            # self-correct on the next step.
            return ToolResult(ok=False, error=f"Unknown tool: '{name}'")

        try:
            tool.validate(arguments)
        except ToolInputError as exc:
            return ToolResult(ok=False, error=str(exc))

        try:
            if timeout:
                return await asyncio.wait_for(tool.execute(arguments), timeout=timeout)
            return await tool.execute(arguments)
        except asyncio.TimeoutError:
            return ToolResult(ok=False, error=f"Tool '{name}' timed out after {timeout}s")
        except ToolError as exc:
            return ToolResult(ok=False, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - tool execution boundary, must not crash the agent
            logger.exception("Tool '%s' raised an unexpected error", name)
            return ToolResult(ok=False, error=f"Unexpected tool error: {exc}")
