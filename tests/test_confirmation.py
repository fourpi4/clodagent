"""
Human-in-the-loop confirmation flow tests: a tool marked
requires_confirmation must pause the run instead of executing, and only
run after an explicit approve() against the exact saved arguments (deny()
must never execute it). Also covers: unknown-tool safety and
retry-safety for side-effecting tools.
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


class DangerousTool(Tool):
    name = "delete_everything"
    description = "A destructive tool that requires confirmation"
    input_schema = {
        "type": "object",
        "properties": {"target": {"type": "string"}},
        "required": ["target"],
    }
    risk_level = "destructive"
    side_effects = True
    retry_safe = False
    requires_confirmation = True

    def __init__(self):
        self.executed_with: list[dict] = []

    async def execute(self, arguments: dict) -> ToolResult:
        self.executed_with.append(dict(arguments))
        return ToolResult(ok=True, output=f"deleted {arguments['target']}")


class FlakyWriteTool(Tool):
    """Fails every time it's called — used to prove side-effect tools are never retried."""

    name = "flaky_write"
    description = "Always fails"
    input_schema = {"type": "object", "properties": {}, "required": []}
    risk_level = "write"
    side_effects = True
    retry_safe = False

    def __init__(self):
        self.call_count = 0

    async def execute(self, arguments: dict) -> ToolResult:
        self.call_count += 1
        return ToolResult(ok=False, error="always fails")


class SingleToolCallProvider(LLMProvider):
    """Plans trivially, requests exactly one tool call with fixed arguments, then finishes."""

    name = "single-tool-call"

    def __init__(self, tool_name: str, arguments: dict):
        self._tool_name = tool_name
        self._arguments = arguments
        self._stage = 0

    async def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=None, model=None, provider=None):
        if messages[-1].role == "user" and messages[-1].content.startswith("Break the following task"):
            return ChatResult(content="[]")
        self._stage += 1
        if self._stage == 1:
            return ChatResult(
                content="",
                tool_calls=[{
                    "id": "call_1", "type": "function",
                    "function": {"name": self._tool_name, "arguments": json.dumps(self._arguments)},
                }],
                finish_reason="tool_calls",
            )
        return ChatResult(content="Done.", finish_reason="stop")

    async def stream(self, messages, *, tools=None, temperature=0.2, max_tokens=None, model=None, provider=None):
        yield ""

    async def list_models(self):
        return []


def _build_agent(tmp_path, provider, tool, max_retries: int = 1):
    router = make_router(provider)
    registry = ToolRegistry()
    registry.register_tool(tool)
    memory = AgentMemory(
        short_term=SqliteShortTermMemory(tmp_path / "short.sqlite3"),
        long_term=SqliteLongTermMemory(tmp_path / "long.sqlite3"),
    )
    profile_store = ProfileStore(tmp_path / "profiles.json")
    return Agent(
        router, registry, memory, profile_store,
        max_agent_steps=5, max_tool_calls=5, tool_timeout=5, max_retries=max_retries,
    )


async def test_confirmation_required_pauses_without_executing(tmp_path):
    tool = DangerousTool()
    provider = SingleToolCallProvider("delete_everything", {"target": "prod-db"})
    agent = _build_agent(tmp_path, provider, tool)

    result = await agent.run_task("clean up")

    assert result.status == "confirmation_required"
    assert result.pending_confirmation["tool"] == "delete_everything"
    assert result.pending_confirmation["arguments"] == {"target": "prod-db"}
    assert tool.executed_with == []  # never ran


async def test_approve_executes_with_the_exact_saved_arguments(tmp_path):
    tool = DangerousTool()
    provider = SingleToolCallProvider("delete_everything", {"target": "prod-db"})
    agent = _build_agent(tmp_path, provider, tool)

    pending = await agent.run_task("clean up")
    confirmation_id = pending.pending_confirmation["id"]

    result = await agent.approve_confirmation(confirmation_id)

    assert result.status == "success"
    assert tool.executed_with == [{"target": "prod-db"}]  # exact args, unmodified


async def test_deny_never_executes_the_tool(tmp_path):
    tool = DangerousTool()
    provider = SingleToolCallProvider("delete_everything", {"target": "prod-db"})
    agent = _build_agent(tmp_path, provider, tool)

    pending = await agent.run_task("clean up")
    confirmation_id = pending.pending_confirmation["id"]

    result = await agent.deny_confirmation(confirmation_id)

    assert result.status == "success"  # the model gets a chance to react and finish
    assert tool.executed_with == []  # denied — never ran
    assert result.tool_calls[0].ok is False
    assert "denied" in result.tool_calls[0].error.lower()


async def test_confirmation_approve_api_takes_no_argument_overrides():
    """Structural guarantee: Agent.approve_confirmation only accepts an id — there is
    no parameter through which a caller could substitute different arguments."""
    import inspect

    sig = inspect.signature(Agent.approve_confirmation)
    params = [p for p in sig.parameters if p not in ("self", "on_status")]
    assert params == ["confirmation_id"]


async def test_unknown_tool_does_not_crash_the_run(tmp_path):
    provider = SingleToolCallProvider("nonexistent_tool", {})
    registry = ToolRegistry()  # no tools registered at all
    router = make_router(provider)
    memory = AgentMemory(
        short_term=SqliteShortTermMemory(tmp_path / "short.sqlite3"),
        long_term=SqliteLongTermMemory(tmp_path / "long.sqlite3"),
    )
    profile_store = ProfileStore(tmp_path / "profiles.json")
    agent = Agent(router, registry, memory, profile_store, max_agent_steps=5, max_tool_calls=5, tool_timeout=5, max_retries=0)

    result = await agent.run_task("do the impossible")

    assert result.status == "success"  # did not crash
    assert result.tool_calls[0].ok is False
    assert "Unknown tool" in result.tool_calls[0].error


async def test_side_effect_tool_is_never_retried_on_failure(tmp_path):
    tool = FlakyWriteTool()
    provider = SingleToolCallProvider("flaky_write", {})
    # max_retries=3 would normally mean 4 attempts for a retry-safe tool
    agent = _build_agent(tmp_path, provider, tool, max_retries=3)

    await agent.run_task("try the flaky write")

    assert tool.call_count == 1  # not retried despite max_retries=3, because retry_safe=False
