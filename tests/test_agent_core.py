"""
End-to-end Agent Core test using a fake LLMProvider (no network, no real API key needed).
Verifies the plan -> tool selection -> execute -> observe -> reason/replan loop.
"""
import json

from app.agent.core import Agent
from app.agent.memory import AgentMemory
from app.agent.profiles import AgentProfile, ProfileStore
from app.memory.long_term import SqliteLongTermMemory
from app.memory.short_term import SqliteShortTermMemory
from app.providers.base import ChatResult, LLMProvider, ModelInfo
from app.tools.base import Tool, ToolResult
from app.tools.registry import ToolRegistry
from tests.conftest import make_router


class AddTool(Tool):
    name = "add"
    description = "Adds two numbers"
    input_schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }
    risk_level = "read"
    side_effects = False
    retry_safe = True

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(ok=True, output=arguments["a"] + arguments["b"])


class ScriptedProvider(LLMProvider):
    """Fake provider that: 1) returns a plan, 2) calls the 'add' tool once, 3) returns a final answer."""

    name = "scripted"

    def __init__(self) -> None:
        self._call_count = 0

    async def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=None, model=None, provider=None) -> ChatResult:
        self._call_count += 1

        # planner call: prompt asks for a JSON array of steps
        if messages[-1].role == "user" and messages[-1].content.startswith("Break the following task"):
            return ChatResult(content='["Call the add tool with 2 and 3", "Report the sum"]')

        # first executor call -> request a tool call
        if self._call_count == 2:
            return ChatResult(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "add", "arguments": json.dumps({"a": 2, "b": 3})},
                    }
                ],
                finish_reason="tool_calls",
            )

        # second executor call -> final answer, no more tool calls
        return ChatResult(content="The sum is 5.", finish_reason="stop")

    async def stream(self, messages, *, tools=None, temperature=0.2, max_tokens=None, model=None, provider=None):
        yield "unused"

    async def list_models(self):
        return [ModelInfo(id="fake-model", provider="scripted")]


def _build_agent(tmp_path) -> Agent:
    router = make_router(ScriptedProvider())
    registry = ToolRegistry()
    registry.register_tool(AddTool())
    memory = AgentMemory(
        short_term=SqliteShortTermMemory(tmp_path / "short.sqlite3"),
        long_term=SqliteLongTermMemory(tmp_path / "long.sqlite3"),
    )
    profile_store = ProfileStore(tmp_path / "profiles.json")
    return Agent(
        router,
        registry,
        memory,
        profile_store,
        max_agent_steps=5,
        max_tool_calls=5,
        tool_timeout=5,
        max_retries=1,
    )


async def test_agent_runs_plan_tool_call_and_final_answer(tmp_path):
    agent = _build_agent(tmp_path)
    result = await agent.run_task("What is 2 + 3?")

    assert result.status == "success"
    assert result.final_answer == "The sum is 5."
    assert len(result.plan) == 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "add"
    assert result.tool_calls[0].ok is True
    assert result.tool_calls[0].output == 5


async def test_agent_persists_conversation_to_memory(tmp_path):
    agent = _build_agent(tmp_path)
    session_id = agent.memory.start_session("default")
    await agent.run_turn(session_id, "What is 2 + 3?")

    history = agent.memory.conversation(session_id)
    roles = [h["role"] for h in history]
    assert roles == ["user", "assistant"]
    assert history[-1]["content"] == "The sum is 5."


async def test_agent_respects_max_steps(tmp_path):
    class LoopingProvider(LLMProvider):
        name = "looping"

        async def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=None, model=None, provider=None):
            if messages[-1].content.startswith("Break the following task"):
                return ChatResult(content="[]")
            return ChatResult(
                content="",
                tool_calls=[{"id": "x", "type": "function", "function": {"name": "add", "arguments": "{\"a\":1,\"b\":1}"}}],
                finish_reason="tool_calls",
            )

        async def stream(self, messages, *, tools=None, temperature=0.2, max_tokens=None, model=None, provider=None):
            yield ""

        async def list_models(self):
            return []

    router = make_router(LoopingProvider())
    registry = ToolRegistry()
    registry.register_tool(AddTool())
    memory = AgentMemory(
        short_term=SqliteShortTermMemory(tmp_path / "short.sqlite3"),
        long_term=SqliteLongTermMemory(tmp_path / "long.sqlite3"),
    )
    profile_store = ProfileStore(tmp_path / "profiles.json")
    agent = Agent(
        router, registry, memory, profile_store,
        max_agent_steps=3, max_tool_calls=100, tool_timeout=5, max_retries=0,
    )

    result = await agent.run_task("loop forever")
    assert result.status == "max_steps_reached"
    assert result.steps_used == 3


async def test_plan_is_injected_into_conversation_context(tmp_path):
    """The plan must actually influence the loop: a system message with the plan
    progress is present in the conversation sent to the model, not just returned
    in the API response."""

    class PlanCheckingProvider(LLMProvider):
        name = "plan-checking"

        def __init__(self):
            self.saw_plan_in_context = False

        async def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=None, model=None, provider=None):
            if messages[-1].role == "user" and messages[-1].content.startswith("Break the following task"):
                return ChatResult(content='["Step one", "Step two"]')
            # Every subsequent call should have a system message containing the plan.
            if any("Step one" in m.content and m.role == "system" for m in messages):
                self.saw_plan_in_context = True
            return ChatResult(content="done", finish_reason="stop")

        async def stream(self, messages, *, tools=None, temperature=0.2, max_tokens=None, model=None, provider=None):
            yield ""

        async def list_models(self):
            return []

    fake = PlanCheckingProvider()
    router = make_router(fake)
    registry = ToolRegistry()
    memory = AgentMemory(
        short_term=SqliteShortTermMemory(tmp_path / "short.sqlite3"),
        long_term=SqliteLongTermMemory(tmp_path / "long.sqlite3"),
    )
    profile_store = ProfileStore(tmp_path / "profiles.json")
    agent = Agent(router, registry, memory, profile_store, max_agent_steps=5, max_tool_calls=5, tool_timeout=5, max_retries=0)

    result = await agent.run_task("multi-step task")
    assert result.status == "success"
    assert fake.saw_plan_in_context is True
