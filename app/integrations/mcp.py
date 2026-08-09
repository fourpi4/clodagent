"""
MCP (Model Context Protocol) client manager — built on the current
official Python SDK (`mcp` >= 2.0, https://github.com/modelcontextprotocol/python-sdk).

Connects to MCP servers declared in `mcp_servers.json`, discovers their
tools, and exposes each one as a normal `Tool` so it can be registered in
the same ToolRegistry as built-in tools.

Supported transports:
  - "stdio":           launches a local process (command + args + env)
  - "streamable_http":  connects to a remote server over Streamable HTTP (the
                        current recommended remote transport)
  - "sse":              legacy Server-Sent Events transport, kept only for
                        backward compatibility with older server configs

Security (stdio is the only transport that can execute arbitrary local
commands, so it gets extra gates):
  - Servers loaded from the trusted `mcp_servers.json` config file at
    startup are implicitly trusted and confirmed.
  - A new stdio server registered later via the REST API is only accepted
    if either its command is in MCP_ALLOWED_COMMANDS, or
    ALLOW_DYNAMIC_STDIO_MCP=true — and even then it must be explicitly
    confirmed (POST /api/mcp/{name}/confirm) before its first connection.
  - Only environment variable *names* present in MCP_ALLOWED_ENV_VARS are
    ever forwarded to a dynamically-registered stdio server; everything
    else is dropped.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from mcp import Client, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from app.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)

TRANSPORTS = {"stdio", "streamable_http", "sse"}


class McpConnectionError(RuntimeError):
    pass


class McpPermissionError(McpConnectionError):
    """Raised when a stdio server is rejected by the command allowlist / dynamic-stdio policy."""


@dataclass
class McpServerConfig:
    name: str
    transport: str  # "stdio" | "streamable_http" | "sse"
    enabled: bool = True
    timeout: float = 30.0
    # stdio
    command: Optional[str] = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # streamable_http / sse
    url: Optional[str] = None
    # Security bookkeeping (not persisted from untrusted input — set by the manager)
    trusted: bool = True
    confirmed: bool = True
    dropped_env_vars: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "McpServerConfig":
        transport = data.get("transport", "stdio")
        return cls(
            name=name,
            transport=transport,
            enabled=data.get("enabled", True),
            timeout=float(data.get("timeout", 30.0)),
            command=data.get("command"),
            args=data.get("args", []),
            env=data.get("env", {}),
            url=data.get("url"),
            trusted=True,
            confirmed=True,
        )

    def to_persisted_dict(self) -> dict[str, Any]:
        return {
            "transport": self.transport,
            "enabled": self.enabled,
            "timeout": self.timeout,
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "url": self.url,
        }


class McpToolAdapter(Tool):
    """Wraps a single tool exposed by an MCP server as a local Tool."""

    risk_level = "network"
    side_effects = True
    retry_safe = False  # unknown side-effect profile of a third-party MCP tool — don't assume idempotent
    requires_confirmation = False

    def __init__(self, manager: "McpClientManager", server_name: str, mcp_tool: Any) -> None:
        self._manager = manager
        self._server_name = server_name
        self.name = f"mcp__{server_name}__{mcp_tool.name}"
        self.description = f"[MCP:{server_name}] {mcp_tool.description or mcp_tool.name}"
        self.input_schema = getattr(mcp_tool, "input_schema", None) or {"type": "object", "properties": {}}

    async def execute(self, arguments: dict) -> ToolResult:
        return await self._manager.call_tool(self._server_name, self._raw_name(), arguments)

    def _raw_name(self) -> str:
        prefix = f"mcp__{self._server_name}__"
        return self.name[len(prefix):]


class McpClientManager:
    """
    Owns the set of configured MCP servers. Opens a fresh connection per
    call (simple, crash-safe MVP strategy) rather than holding long-lived
    background connections — see README Phase 2 notes for connection pooling.
    """

    def __init__(
        self,
        config_path: Path,
        *,
        allow_dynamic_stdio: bool = False,
        allowed_commands: Optional[list[str]] = None,
        allowed_env_vars: Optional[list[str]] = None,
    ) -> None:
        self._config_path = config_path
        self._servers: dict[str, McpServerConfig] = {}
        self._tools_cache: dict[str, list[Any]] = {}
        self._allow_dynamic_stdio = allow_dynamic_stdio
        self._allowed_commands = {c.strip() for c in (allowed_commands or [])}
        self._allowed_env_vars = set(allowed_env_vars or [])
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
        data = {"mcpServers": {s.name: s.to_persisted_dict() for s in self._servers.values()}}
        self._config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def list_servers(self) -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "transport": s.transport,
                "enabled": s.enabled,
                "timeout": s.timeout,
                "trusted": s.trusted,
                "confirmed": s.confirmed,
                "dropped_env_vars": s.dropped_env_vars,
                "tools_discovered": len(self._tools_cache.get(s.name, [])),
            }
            for s in self._servers.values()
        ]

    def set_enabled(self, name: str, enabled: bool) -> None:
        if name not in self._servers:
            raise McpConnectionError(f"Unknown MCP server '{name}'")
        self._servers[name].enabled = enabled
        self.save_config()

    def confirm_server(self, name: str) -> None:
        server = self._servers.get(name)
        if server is None:
            raise McpConnectionError(f"Unknown MCP server '{name}'")
        server.confirmed = True

    def _check_command_allowed(self, command: str) -> None:
        basename = command.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if basename in self._allowed_commands or command in self._allowed_commands:
            return
        if self._allow_dynamic_stdio:
            return
        raise McpPermissionError(
            f"stdio command '{command}' is not in MCP_ALLOWED_COMMANDS and ALLOW_DYNAMIC_STDIO_MCP is false. "
            "Add it to the trusted mcp_servers.json config file, add its command to MCP_ALLOWED_COMMANDS, "
            "or set ALLOW_DYNAMIC_STDIO_MCP=true (and confirm it) to allow arbitrary stdio servers."
        )

    def add_server(self, config: McpServerConfig) -> McpServerConfig:
        """Registers a new server via the REST API (untrusted input) — security gates apply here."""
        if config.transport not in TRANSPORTS:
            raise McpConnectionError(f"Unsupported MCP transport '{config.transport}'")

        config.trusted = False
        if config.transport == "stdio":
            if not config.command:
                raise McpConnectionError("stdio transport requires a command")
            self._check_command_allowed(config.command)
            allowed_env = {k: v for k, v in config.env.items() if k in self._allowed_env_vars}
            dropped = sorted(set(config.env) - set(allowed_env))
            config.env = allowed_env
            config.dropped_env_vars = dropped
            if dropped:
                logger.warning("Dropped non-allowlisted env vars for MCP server '%s': %s", config.name, dropped)
            config.confirmed = False  # first stdio run always needs explicit confirmation
        else:
            config.confirmed = True  # remote URL transports don't spawn local processes

        self._servers[config.name] = config
        self.save_config()
        return config

    def _transport_cm(self, server: McpServerConfig):
        if server.transport == "stdio":
            if not server.command:
                raise McpConnectionError(f"MCP server '{server.name}' has transport=stdio but no command set")
            params = StdioServerParameters(command=server.command, args=server.args, env=server.env or None)
            return stdio_client(params)
        if server.transport == "streamable_http":
            if not server.url:
                raise McpConnectionError(f"MCP server '{server.name}' has transport=streamable_http but no url set")
            return streamable_http_client(server.url)
        if server.transport == "sse":
            if not server.url:
                raise McpConnectionError(f"MCP server '{server.name}' has transport=sse but no url set")
            return sse_client(server.url)
        raise McpConnectionError(f"Unsupported MCP transport '{server.transport}'")

    def _require_ready(self, server: McpServerConfig) -> None:
        if not server.enabled:
            raise McpConnectionError(f"MCP server '{server.name}' is disabled")
        if server.transport == "stdio" and not server.confirmed:
            raise McpPermissionError(
                f"MCP server '{server.name}' has not been confirmed yet. "
                f"Call POST /api/mcp/{server.name}/confirm after reviewing its command before first use."
            )

    async def connect_and_list_tools(self, server_name: str) -> list[Any]:
        server = self._servers.get(server_name)
        if server is None:
            raise McpConnectionError(f"Unknown MCP server '{server_name}'")
        self._require_ready(server)

        try:
            async with Client(self._transport_cm(server), read_timeout_seconds=server.timeout) as client:
                result = await client.list_tools()
        except McpConnectionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise McpConnectionError(f"Failed to connect to MCP server '{server_name}': {exc}") from exc

        self._tools_cache[server_name] = result.tools
        return result.tools

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> ToolResult:
        server = self._servers.get(server_name)
        if server is None:
            return ToolResult(ok=False, error=f"Unknown MCP server '{server_name}'")
        try:
            self._require_ready(server)
        except McpConnectionError as exc:
            return ToolResult(ok=False, error=str(exc))

        try:
            async with Client(self._transport_cm(server), read_timeout_seconds=server.timeout) as client:
                result = await client.call_tool(tool_name, arguments)
        except Exception as exc:  # noqa: BLE001
            logger.exception("MCP tool call failed: %s/%s", server_name, tool_name)
            return ToolResult(ok=False, error=f"MCP call failed: {exc}")

        text_parts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
        return ToolResult(ok=not result.is_error, output="\n".join(text_parts) or str(result.content))

    async def discover_all_tools(self, *, only_servers: Optional[list[str]] = None) -> list[McpToolAdapter]:
        """Connects to enabled (optionally filtered) servers and returns adapters for all their tools."""
        adapters: list[McpToolAdapter] = []
        for server in self._servers.values():
            if not server.enabled:
                continue
            if only_servers is not None and server.name not in only_servers:
                continue
            try:
                tools = await self.connect_and_list_tools(server.name)
            except McpConnectionError as exc:
                logger.warning("Skipping MCP server '%s': %s", server.name, exc)
                continue
            adapters.extend(McpToolAdapter(self, server.name, t) for t in tools)
        return adapters
