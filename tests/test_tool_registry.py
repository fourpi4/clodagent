import pytest

from app.tools.base import Tool, ToolResult
from app.tools.registry import ToolRegistry


class EchoTool(Tool):
    name = "echo"
    description = "Echoes the given text"
    input_schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(ok=True, output=arguments["text"])


class ExplodingTool(Tool):
    name = "boom"
    description = "Always raises"
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def execute(self, arguments: dict) -> ToolResult:
        raise RuntimeError("kaboom")


def test_register_and_list():
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    names = [t["name"] for t in registry.list_tools()]
    assert names == ["echo"]


def test_register_duplicate_raises():
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    with pytest.raises(Exception):
        registry.register_tool(EchoTool())
    registry.register_tool(EchoTool(), overwrite=True)  # should not raise


def test_remove_and_get_missing():
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    registry.remove_tool("echo")
    with pytest.raises(Exception):
        registry.get_tool("echo")


async def test_execute_tool_success():
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    result = await registry.execute_tool("echo", {"text": "hi"})
    assert result.ok is True
    assert result.output == "hi"


async def test_execute_tool_missing_argument_is_validated():
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    result = await registry.execute_tool("echo", {})
    assert result.ok is False
    assert "Missing required argument" in result.error


async def test_execute_tool_wrong_type_is_validated():
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    result = await registry.execute_tool("echo", {"text": 123})
    assert result.ok is False
    assert "must be of type" in result.error


async def test_execute_tool_catches_unexpected_exception():
    registry = ToolRegistry()
    registry.register_tool(ExplodingTool())
    result = await registry.execute_tool("boom", {})
    assert result.ok is False
    assert "kaboom" in result.error


async def test_execute_tool_timeout():
    import asyncio

    class SlowTool(Tool):
        name = "slow"
        description = "sleeps"
        input_schema = {"type": "object", "properties": {}, "required": []}

        async def execute(self, arguments: dict) -> ToolResult:
            await asyncio.sleep(5)
            return ToolResult(ok=True)

    registry = ToolRegistry()
    registry.register_tool(SlowTool())
    result = await registry.execute_tool("slow", {}, timeout=0.05)
    assert result.ok is False
    assert "timed out" in result.error
