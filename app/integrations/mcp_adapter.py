"""
Glue between McpClientManager (app/integrations/mcp.py) and the ToolRegistry.

This is the "MCP-compatible tools" integration referenced in the plugin
system: any MCP server declared in mcp_servers.json becomes a first-class
set of Tools with zero core-code changes.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.integrations.mcp import McpClientManager
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


async def register_mcp_tools(
    manager: McpClientManager, registry: ToolRegistry, *, only_servers: Optional[list[str]] = None
) -> int:
    """Discovers tools on enabled MCP servers (optionally filtered) and registers them. Returns count registered."""
    adapters = await manager.discover_all_tools(only_servers=only_servers)
    for adapter in adapters:
        registry.register_tool(adapter, overwrite=True)
    logger.info("Registered %d MCP-backed tools", len(adapters))
    return len(adapters)
