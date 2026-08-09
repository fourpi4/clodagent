"""
When the model returns several tool_calls in one turn but MAX_TOOL_CALLS
runs out partway through the batch, every tool_call must still get a
structured response (success, failure, or an explicit "budget exhausted"
marker) — never left dangling, which would break the conversation for
OpenAI-style APIs on any subsequent call.
"""
import json

from app.agent.core import Agent
from app.agent.memory import AgentMemory
from app.agent.profiles import ProfileStore
from app.memory.long_term import SqliteLongTermMemory
from app.memory.short_term import SqliteShortTermMemory
from app.providers.base import ChatResult, LLMProvider
from app.tools.base import Tool, ToolResult
from app.tools.registry import ToolRegistry
from tests.conftest import make_router


class EchoTool(Tool):
    name = "echo"
    description = "Echoes back its input"
    input_schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    risk_level = "read"
    side_effects = False
    retry_safe = True

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(ok=True, output=arguments["n"])


class BatchToolCallProvider(LLMProvider):
    """Requests 3 tool calls in a single turn, every time it's asked."""

    name = "batch"

    async def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=None, model=None, provider=None):
        if messages[-1].role == "user" and messages[-1].content.startswith("Break the following task"):
            return ChatResult(content="[]")
        return ChatResult(
            content="",
            tool_calls=[
                {"id": f"call_{i}", "type": "function", "function": {"name": "echo", "arguments": json.dumps({"n": i})}}
                for i in range(3)
            ],
            finish_reason="tool_calls",
        )

    async def stream(self, messages, *, tools=None, temperature=0.2, max_tokens=None, model=None, provider=None):
        yield ""

    async def list_models(self):
        return []


async def test_budget_exhausted_mid_batch_gives_every_call_a_structured_response(tmp_path):
    router = make_router(BatchToolCallProvider())
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    memory = AgentMemory(
        short_term=SqliteShortTermMemory(tmp_path / "short.sqlite3"),
        long_term=SqliteLongTermMemory(tmp_path / "long.sqlite3"),
    )
    profile_store = ProfileStore(tmp_path / "profiles.json")
    # Only 2 tool calls allowed for the whole run, but the model asks for 3 at once.
    agent = Agent(router, registry, memory, profile_store, max_agent_steps=5, max_tool_calls=2, tool_timeout=5, max_retries=0)

    result = await agent.run_task("do three things at once")

    assert result.status == "max_tool_calls_reached"
    assert len(result.tool_calls) == 3  # all three accounted for, none silently dropped
    assert result.tool_calls[0].ok is True
    assert result.tool_calls[1].ok is True
    assert result.tool_calls[2].ok is False
    assert "budget" in result.tool_calls[2].error.lower()
