"""
MCP security + modernization tests:
  - arbitrary stdio commands are blocked by default
  - allowlisted commands are accepted (but still need explicit confirmation)
  - a real local (127.0.0.1-only) Streamable HTTP MCP server round-trips
    list_tools()/call_tool() through McpClientManager — no external network
  - AgentProfile.mcp_servers actually restricts which MCP tools are exposed
"""
import asyncio

import pytest
import uvicorn
from mcp import types
from mcp.server import Server

from app.agent.tool_router import ToolRouter
from app.integrations.mcp import McpClientManager, McpPermissionError, McpServerConfig
from app.tools.base import Tool, ToolResult
from app.tools.registry import ToolRegistry


def _manager(tmp_path, **kwargs) -> McpClientManager:
    config_path = tmp_path / "mcp_servers.json"
    config_path.write_text('{"mcpServers": {}}', encoding="utf-8")
    return McpClientManager(config_path, **kwargs)


def test_arbitrary_stdio_command_blocked_by_default(tmp_path):
    manager = _manager(tmp_path, allow_dynamic_stdio=False, allowed_commands=[])
    config = McpServerConfig(name="evil", transport="stdio", command="rm", args=["-rf", "/"])

    with pytest.raises(McpPermissionError):
        manager.add_server(config)


def test_allowlisted_stdio_command_accepted_but_unconfirmed(tmp_path):
    manager = _manager(tmp_path, allow_dynamic_stdio=False, allowed_commands=["npx"])
    config = McpServerConfig(name="fs", transport="stdio", command="npx", args=["-y", "some-mcp-server"])

    result = manager.add_server(config)

    assert result.trusted is False  # added via API, not the trusted config file
    assert result.confirmed is False  # first stdio run always needs explicit confirmation
    servers = {s["name"]: s for s in manager.list_servers()}
    assert servers["fs"]["confirmed"] is False


def test_dynamic_stdio_allowed_when_flag_set(tmp_path):
    manager = _manager(tmp_path, allow_dynamic_stdio=True, allowed_commands=[])
    config = McpServerConfig(name="anything", transport="stdio", command="python", args=["-m", "http.server"])
    result = manager.add_server(config)
    assert result.confirmed is False  # still needs confirmation even though the command itself was allowed


def test_non_allowlisted_env_vars_are_dropped(tmp_path):
    manager = _manager(tmp_path, allow_dynamic_stdio=True, allowed_env_vars=["SAFE_VAR"])
    config = McpServerConfig(
        name="withenv", transport="stdio", command="python",
        env={"SAFE_VAR": "ok", "AWS_SECRET_ACCESS_KEY": "should-be-dropped"},
    )
    result = manager.add_server(config)
    assert result.env == {"SAFE_VAR": "ok"}
    assert "AWS_SECRET_ACCESS_KEY" in result.dropped_env_vars


async def test_stdio_call_refused_until_confirmed(tmp_path):
    manager = _manager(tmp_path, allow_dynamic_stdio=True)
    manager.add_server(McpServerConfig(name="unconf", transport="stdio", command="python"))

    result = await manager.call_tool("unconf", "whatever", {})
    assert result.ok is False
    assert "confirm" in result.error.lower()


# --- Real local Streamable HTTP MCP server (127.0.0.1 only, no external network) --------

async def _on_list_tools(ctx, params):
    return types.ListToolsResult(tools=[
        types.Tool(name="ping", description="Replies pong", input_schema={"type": "object", "properties": {}})
    ])


async def _on_call_tool(ctx, params):
    if params.name == "ping":
        return types.CallToolResult(content=[types.TextContent(type="text", text="pong")])
    return types.CallToolResult(content=[types.TextContent(type="text", text="unknown")], is_error=True)


@pytest.fixture
async def local_mcp_http_server():
    server = Server("test-http-server", version="0.1.0", on_list_tools=_on_list_tools, on_call_tool=_on_call_tool)
    app = server.streamable_http_app(stateless_http=True)
    config = uvicorn.Config(app, host="127.0.0.1", port=8931, log_level="warning")
    uv_server = uvicorn.Server(config)
    task = asyncio.create_task(uv_server.serve())
    for _ in range(100):
        if uv_server.started:
            break
        await asyncio.sleep(0.02)
    try:
        yield "http://127.0.0.1:8931/mcp"
    finally:
        uv_server.should_exit = True
        await task


async def test_streamable_http_fake_server_list_and_call_tool(tmp_path, local_mcp_http_server):
    manager = _manager(tmp_path)
    manager.add_server(McpServerConfig(name="http-test", transport="streamable_http", url=local_mcp_http_server))
    # Remote URL transports don't spawn local processes, so they're confirmed immediately.
    assert manager._servers["http-test"].confirmed is True  # noqa: SLF001

    tools = await manager.connect_and_list_tools("http-test")
    assert [t.name for t in tools] == ["ping"]

    result = await manager.call_tool("http-test", "ping", {})
    assert result.ok is True
    assert result.output == "pong"


# --- AgentProfile.mcp_servers restriction ---------------------------------------

class _DummyTool(Tool):
    name = "local_tool"
    description = "a local, non-MCP tool"
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(ok=True)


class _FakeMcpTool(Tool):
    def __init__(self, name: str):
        self.name = name
        self.description = "fake mcp tool"
        self.input_schema = {"type": "object", "properties": {}, "required": []}

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(ok=True)


def test_mcp_tools_hidden_by_default_when_profile_has_no_mcp_servers():
    registry = ToolRegistry()
    registry.register_tool(_DummyTool())
    registry.register_tool(_FakeMcpTool("mcp__serverA__do_thing"))
    router = ToolRouter(registry, tool_timeout=5, max_retries=0)

    specs = router.available_specs(None, None)  # profile.mcp_servers is empty/None
    names = {s.name for s in specs}
    assert names == {"local_tool"}  # safe default: no MCP tools exposed


def test_mcp_tools_from_allowed_server_are_exposed():
    registry = ToolRegistry()
    registry.register_tool(_DummyTool())
    registry.register_tool(_FakeMcpTool("mcp__serverA__do_thing"))
    registry.register_tool(_FakeMcpTool("mcp__serverB__do_other_thing"))
    router = ToolRouter(registry, tool_timeout=5, max_retries=0)

    specs = router.available_specs(None, ["serverA"])
    names = {s.name for s in specs}
    assert names == {"local_tool", "mcp__serverA__do_thing"}
