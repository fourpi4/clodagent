"""
MCP (Model Context Protocol) client manager.

Connects to MCP servers declared in `mcp_servers.json`, discovers their
tools, and exposes each one as a normal `Tool` so it can be registered in
the same ToolRegistry as built-in tools. Uses the official `mcp` Python SDK
(https://github.com/modelcontextprotocol/python-sdk) — no protocol
reimplementation.

Supported transports:
  - "stdio": launches a local process (command + args + env)
  - "sse":   connects to a remote server over Server-Sent Events (url)

Each server config also carries an explicit `timeout` (seconds) so a slow
or hung MCP server can never stall the agent loop indefinitely.
"""
from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)

try:
    from mcp.client.sse import sse_client
except ImportError:  # pragma: no cover - sse extra not installed
    sse_client = None


@dataclass
class McpServerConfig:
    name: str
    transport: str  # "stdio" | "sse"
    enabled: bool = True
    timeout: float = 30.0
    # stdio
    command: Optional[str] = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # sse
    url: Optional[str] = None

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "McpServerConfig":
        return cls(
            name=name,
            transport=data.get("transport", "stdio"),
            enabled=data.get("enabled", True),
            timeout=float(data.get("timeout", 30.0)),
            command=data.get("command"),
            args=data.get("args", []),
            env=data.get("env", {}),
            url=data.get("url"),
        )


class McpToolAdapter(Tool):
    """Wraps a single tool exposed by an MCP server as a local Tool."""

    def __init__(self, manager: "McpClientManager", server_name: str, mcp_tool: Any) -> None:
        self._manager = manager
        self._server_name = server_name
        self.name = f"mcp__{server_name}__{mcp_tool.name}"
        self.description = f"[MCP:{server_name}] {mcp_tool.description or mcp_tool.name}"
        self.input_schema = mcp_tool.inputSchema or {"type": "object", "properties": {}}

    async def execute(self, arguments: dict) -> ToolResult:
        return await self._manager.call_tool(self._server_name, self._raw_name(), arguments)

    def _raw_name(self) -> str:
        # strip the "mcp__<server>__" prefix back off to get the tool's real MCP name
        prefix = f"mcp__{self._server_name}__"
        return self.name[len(prefix):]


class McpConnectionError(RuntimeError):
    pass


class McpClientManager:
    """
    Owns the set of configured MCP servers. Opens a fresh session per call
    (simple, crash-safe MVP strategy) rather than holding long-lived
    background connections — see README Phase 2 notes for connection pooling.
    """

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        self._servers: dict[str, McpServerConfig] = {}
        self._tools_cache: dict[str, list[Any]] = {}
        self.load_config()

    def load_config(self) -> None:
        if not self._config_path.exists():
            self._servers = {}
            return
        raw = json.loads(self._config_path.read_text(encoding="utf-8"))
        self._servers = {
            name: McpServerConfig.from_dict(name, cfg) for name, cfg in raw.get("mcpServers", {}).items()
        }

    def save_config(self) -> None:
        data = {
            "mcpServers": {
                s.name: {
                    "transport": s.transport,
                    "enabled": s.enabled,
                    "timeout": s.timeout,
                    "command": s.command,
                    "args": s.args,
                    "env": s.env,
                    "url": s.url,
                }
                for s in self._servers.values()
            }
        }
        self._config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def list_servers(self) -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "transport": s.transport,
                "enabled": s.enabled,
                "timeout": s.timeout,
                "tools_discovered": len(self._tools_cache.get(s.name, [])),
            }
            for s in self._servers.values()
        ]

    def set_enabled(self, name: str, enabled: bool) -> None:
        if name not in self._servers:
            raise McpConnectionError(f"Unknown MCP server '{name}'")
        self._servers[name].enabled = enabled
        self.save_config()

    def add_server(self, config: McpServerConfig) -> None:
        self._servers[config.name] = config
        self.save_config()

    def _session_cm(self, server: McpServerConfig):
        if server.transport == "stdio":
            if not server.command:
                raise McpConnectionError(f"MCP server '{server.name}' has transport=stdio but no command set")
            params = StdioServerParameters(command=server.command, args=server.args, env=server.env or None)
            return stdio_client(params)
        if server.transport == "sse":
            if sse_client is None:
                raise McpConnectionError("SSE transport requested but 'mcp[sse]' extra is not installed")
            if not server.url:
                raise McpConnectionError(f"MCP server '{server.name}' has transport=sse but no url set")
            return sse_client(server.url)
        raise McpConnectionError(f"Unsupported MCP transport '{server.transport}'")

    async def connect_and_list_tools(self, server_name: str) -> list[Any]:
        server = self._servers.get(server_name)
        if server is None:
            raise McpConnectionError(f"Unknown MCP server '{server_name}'")
        if not server.enabled:
            raise McpConnectionError(f"MCP server '{server_name}' is disabled")

        async with AsyncExitStack() as stack:
            try:
                read, write = await stack.enter_async_context(self._session_cm(server))
                session: ClientSession = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                result = await session.list_tools()
            except Exception as exc:  # noqa: BLE001
                raise McpConnectionError(f"Failed to connect to MCP server '{server_name}': {exc}") from exc

        self._tools_cache[server_name] = result.tools
        return result.tools

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> ToolResult:
        server = self._servers.get(server_name)
        if server is None:
            return ToolResult(ok=False, error=f"Unknown MCP server '{server_name}'")
        if not server.enabled:
            return ToolResult(ok=False, error=f"MCP server '{server_name}' is disabled")

        try:
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(self._session_cm(server))
                session: ClientSession = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
        except Exception as exc:  # noqa: BLE001
            logger.exception("MCP tool call failed: %s/%s", server_name, tool_name)
            return ToolResult(ok=False, error=f"MCP call failed: {exc}")

        text_parts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
        return ToolResult(ok=not result.isError, output="\n".join(text_parts) or str(result.content))

    async def discover_all_tools(self) -> list[McpToolAdapter]:
        """Connects to every enabled server and returns adapters for all their tools."""
        adapters: list[McpToolAdapter] = []
        for server in self._servers.values():
            if not server.enabled:
                continue
            try:
                tools = await self.connect_and_list_tools(server.name)
            except McpConnectionError as exc:
                logger.warning("Skipping MCP server '%s': %s", server.name, exc)
                continue
            adapters.extend(McpToolAdapter(self, server.name, t) for t in tools)
        return adapters
