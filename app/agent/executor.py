"""
Executor: runs the plan -> tool selection -> execute -> observe -> reason/replan
loop, bounded by MAX_AGENT_STEPS / MAX_TOOL_CALLS / TOOL_TIMEOUT / MAX_RETRIES
so the agent can never spin forever.

Only structured, useful data is retained (task, plan, tool calls/results,
status, final answer) — the model's raw hidden reasoning is never stored or
surfaced to the user.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from app.agent.planner import Planner
from app.agent.profiles import AgentProfile
from app.agent.tool_router import ToolRouter
from app.providers.base import ChatMessage, ChatResult, LLMProvider, ProviderError

logger = logging.getLogger(__name__)

StatusCallback = Optional[Callable[[dict[str, Any]], Awaitable[None]]]


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    ok: bool
    output: Any = None
    error: Optional[str] = None


@dataclass
class AgentRunResult:
    task: str
    plan: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    status: str = "success"  # success | max_steps_reached | max_tool_calls_reached | error
    final_answer: str = ""
    steps_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "plan": self.plan,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments, "ok": tc.ok, "output": tc.output, "error": tc.error}
                for tc in self.tool_calls
            ],
            "status": self.status,
            "final_answer": self.final_answer,
            "steps_used": self.steps_used,
        }


async def _noop_status(_event: dict[str, Any]) -> None:
    return None


class Executor:
    def __init__(
        self,
        provider: LLMProvider,
        planner: Planner,
        tool_router: ToolRouter,
        *,
        max_agent_steps: int,
        max_tool_calls: int,
        max_retries: int,
    ) -> None:
        self._provider = provider
        self._planner = planner
        self._tool_router = tool_router
        self._max_agent_steps = max_agent_steps
        self._max_tool_calls = max_tool_calls
        self._max_retries = max_retries

    async def _chat_with_retries(self, *args, **kwargs) -> ChatResult:
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._provider.chat(*args, **kwargs)
            except ProviderError as exc:
                last_exc = exc
                logger.warning("LLM call attempt %d failed: %s", attempt + 1, exc)
        assert last_exc is not None
        raise last_exc

    async def run(
        self,
        task: str,
        messages: list[ChatMessage],
        profile: AgentProfile,
        *,
        on_status: StatusCallback = None,
    ) -> AgentRunResult:
        status_cb = on_status or _noop_status
        max_steps = profile.max_steps or self._max_agent_steps
        allowed_tools = profile.tools or None
        specs = self._tool_router.available_specs(allowed_tools)

        await status_cb({"type": "planning"})
        try:
            plan = await self._planner.create_plan(task, model=profile.model)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Planning failed unexpectedly")
            return AgentRunResult(task=task, status="error", final_answer=f"Planning failed: {exc}")
        await status_cb({"type": "plan_ready", "plan": plan})

        run = AgentRunResult(task=task, plan=plan)
        conversation = list(messages)
        tool_calls_used = 0

        for step in range(max_steps):
            run.steps_used = step + 1
            await status_cb({"type": "generating", "step": step + 1})
            try:
                result = await self._chat_with_retries(
                    conversation,
                    tools=specs if tool_calls_used < self._max_tool_calls else None,
                    temperature=profile.temperature,
                    model=profile.model,
                )
            except ProviderError as exc:
                run.status = "error"
                run.final_answer = f"LLM provider error: {exc}"
                return run

            if not result.tool_calls:
                run.status = "success"
                run.final_answer = result.content
                return run

            if tool_calls_used >= self._max_tool_calls:
                run.status = "max_tool_calls_reached"
                run.final_answer = result.content or (
                    "Reached the maximum number of tool calls for this run before finishing."
                )
                return run

            conversation.append(
                ChatMessage(role="assistant", content=result.content or "", tool_calls=result.tool_calls)
            )

            for tool_call in result.tool_calls:
                if tool_calls_used >= self._max_tool_calls:
                    break
                function_name = tool_call.get("function", {}).get("name", "unknown")
                await status_cb({"type": "tool_call", "name": function_name})

                routed = await self._tool_router.dispatch(tool_call)
                tool_calls_used += 1
                run.tool_calls.append(
                    ToolCallRecord(
                        name=routed.name,
                        arguments=routed.arguments,
                        ok=routed.result.ok,
                        output=routed.result.output,
                        error=routed.result.error,
                    )
                )
                await status_cb(
                    {"type": "tool_result", "name": routed.name, "ok": routed.result.ok}
                )

                observation = routed.result.output if routed.result.ok else f"ERROR: {routed.result.error}"
                conversation.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=routed.id,
                        name=routed.name,
                        content=str(observation)[:8000],
                    )
                )

        run.status = "max_steps_reached"
        run.final_answer = (
            "The agent reached its maximum step limit before producing a final answer. "
            "Try narrowing the request or increasing MAX_AGENT_STEPS."
        )
        return run
