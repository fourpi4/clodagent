"""
Executor: runs the plan -> tool selection -> execute -> observe -> reason/replan
loop, bounded by MAX_AGENT_STEPS / MAX_TOOL_CALLS / TOOL_TIMEOUT / MAX_RETRIES
so the agent can never spin forever.

Only structured, useful data is retained (task, plan, tool calls/results,
status, final answer) — the model's raw hidden reasoning is never stored or
surfaced to the user.

Confirmation gating: if the model requests a tool with
`requires_confirmation=True`, the run pauses (status="confirmation_required")
instead of executing it. The caller must resolve the confirmation (approve
or deny) via ConfirmationStore, then call `resume()` to continue the exact
same run with the exact same saved arguments — the API never lets a client
substitute different arguments between confirmation and execution.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from app.agent.confirmation import ConfirmationStore, PendingConfirmation
from app.agent.planner import Planner
from app.agent.profiles import AgentProfile
from app.agent.run_state import RunState, RunStateStore, ToolCallRecord
from app.agent.tool_router import ToolRouter
from app.providers.base import ChatMessage, ChatResult, LLMProvider, ProviderError

logger = logging.getLogger(__name__)

StatusCallback = Optional[Callable[[dict[str, Any]], Awaitable[None]]]

PLAN_MARKER = "​__plan_progress__"  # zero-width marker so we can find/replace our own injected message


@dataclass
class AgentRunResult:
    task: str
    run_id: str = ""
    plan: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    # success | max_steps_reached | max_tool_calls_reached | confirmation_required | error
    status: str = "success"
    final_answer: str = ""
    steps_used: int = 0
    pending_confirmation: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "run_id": self.run_id,
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
        if self.pending_confirmation:
            out["confirmation"] = self.pending_confirmation
        return out


async def _noop_status(_event: dict[str, Any]) -> None:
    return None


def _plan_progress_text(state: RunState) -> str:
    lines = [PLAN_MARKER, "Plan:"]
    for i, step in enumerate(state.plan):
        if step in state.completed_steps:
            marker = "done"
        elif step in state.failed_steps:
            marker = "failed"
        elif i == state.current_step_index:
            marker = "current"
        else:
            marker = "pending"
        lines.append(f"{i + 1}. [{marker}] {step}")
    lines.append(f"Progress: {len(state.completed_steps)}/{len(state.plan)} steps completed.")
    if state.failed_steps:
        lines.append(f"Failed steps so far: {len(state.failed_steps)}.")
    return "\n".join(lines)


def _confirmation_reason(tool_name: str, risk_level: str, side_effects: bool) -> str:
    return (
        f"Tool '{tool_name}' is marked risk_level='{risk_level}'"
        f"{' with side effects' if side_effects else ''} and requires explicit human approval "
        f"before it can run."
    )


class Executor:
    def __init__(
        self,
        provider: LLMProvider,
        planner: Planner,
        tool_router: ToolRouter,
        confirmations: ConfirmationStore,
        run_states: RunStateStore,
        *,
        max_agent_steps: int,
        max_tool_calls: int,
        max_retries: int,
    ) -> None:
        self._provider = provider
        self._planner = planner
        self._tool_router = tool_router
        self._confirmations = confirmations
        self._run_states = run_states
        self._max_agent_steps = max_agent_steps
        self._max_tool_calls = max_tool_calls
        self._max_retries = max_retries

    async def _chat_with_retries(
        self,
        conversation: list[ChatMessage],
        *,
        specs,
        profile: AgentProfile,
        status_cb: Callable,
    ) -> ChatResult:
        async def on_event(event: dict[str, Any]) -> None:
            await status_cb(event)

        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._provider.chat(
                    conversation,
                    tools=specs,
                    temperature=profile.temperature,
                    model=profile.model,
                    provider=profile.provider,
                    on_event=on_event,
                )
            except ProviderError as exc:
                last_exc = exc
                logger.warning("LLM call attempt %d failed: %s", attempt + 1, exc)
        assert last_exc is not None
        raise last_exc

    def _ensure_plan_message(self, state: RunState) -> None:
        text = _plan_progress_text(state)
        if state.plan_message_index is not None and state.plan_message_index < len(state.conversation):
            state.conversation[state.plan_message_index].content = text
        else:
            state.conversation.append(ChatMessage(role="system", content=text))
            state.plan_message_index = len(state.conversation) - 1

    def _advance_plan(self, state: RunState, *, ok: bool) -> None:
        if not state.plan:
            return
        if state.current_step_index >= len(state.plan):
            return
        step = state.plan[state.current_step_index]
        if ok:
            state.completed_steps.append(step)
            state.current_step_index += 1
        else:
            state.failed_steps.append(step)
        self._ensure_plan_message(state)

    async def _maybe_replan(self, state: RunState, status_cb: Callable) -> None:
        if state.replanned or not state.failed_steps:
            return
        state.replanned = True
        failure_context = f"Step '{state.failed_steps[-1]}' failed."
        new_plan = await self._planner.replan(
            state.task, state.plan, failure_context, model=state.profile.model
        )
        if new_plan != state.plan:
            state.plan = new_plan
            await status_cb({"type": "plan_ready", "plan": new_plan, "replanned": True})
        self._ensure_plan_message(state)

    async def _execute_one(
        self,
        state: RunState,
        tool_call: dict[str, Any],
        status_cb: Callable,
        *,
        forced_decision: Optional[str] = None,
    ) -> tuple[str, Any]:
        """Returns ("done", ToolCallRecord) or ("confirmation_required", PendingConfirmation)."""
        call_id, name, arguments, parse_error = self._tool_router.parse_arguments(tool_call)
        if parse_error:
            record = ToolCallRecord(name=name, arguments={}, ok=False, error=parse_error)
            state.conversation.append(
                ChatMessage(role="tool", tool_call_id=call_id, name=name, content=f"ERROR: {parse_error}")
            )
            return "done", record

        tool = self._tool_router.lookup(name)
        await status_cb({"type": "tool_requested", "name": name})

        if forced_decision is None and tool is not None and tool.requires_confirmation:
            reason = _confirmation_reason(name, tool.risk_level, tool.side_effects)
            confirmation = self._confirmations.create(
                run_id=state.run_id, tool_name=name, arguments=arguments, reason=reason, risk_level=tool.risk_level
            )
            await status_cb({"type": "confirmation_required", "tool": name, "confirmation_id": confirmation.id})
            return "confirmation_required", confirmation

        await status_cb({"type": "tool_started", "name": name})

        if forced_decision == "denied":
            result_ok, result_output, result_error = False, None, "User denied execution of this tool call."
        else:
            routed = await self._tool_router.dispatch(tool_call)
            result_ok, result_output, result_error = routed.result.ok, routed.result.output, routed.result.error

        record = ToolCallRecord(name=name, arguments=arguments, ok=result_ok, output=result_output, error=result_error)
        await status_cb({"type": "tool_finished", "name": name, "ok": result_ok})

        observation = result_output if result_ok else f"ERROR: {result_error}"
        state.conversation.append(
            ChatMessage(role="tool", tool_call_id=call_id, name=name, content=str(observation)[:8000])
        )
        self._advance_plan(state, ok=result_ok)
        if not result_ok:
            await self._maybe_replan(state, status_cb)
        return "done", record

    async def _run_batch(
        self,
        state: RunState,
        tool_calls: list[dict[str, Any]],
        status_cb: Callable,
        *,
        resume_decision: Optional[str] = None,
    ) -> tuple[str, Any]:
        """Returns ("continue", None) | ("confirmation_required", PendingConfirmation) | ("budget_exhausted", None)."""
        i = state.pending_index
        while i < len(tool_calls):
            if state.tool_calls_used >= self._max_tool_calls:
                # Budget ran out mid-batch. Every assistant tool_call still needs a
                # matching tool response so the conversation stays valid for
                # OpenAI-style APIs — never leave dangling tool_calls.
                for j in range(i, len(tool_calls)):
                    call_id, name, args, _ = self._tool_router.parse_arguments(tool_calls[j])
                    state.conversation.append(
                        ChatMessage(
                            role="tool",
                            tool_call_id=call_id,
                            name=name,
                            content="ERROR: tool-call budget exhausted for this run; this call was not executed.",
                        )
                    )
                    state.tool_calls.append(
                        ToolCallRecord(name=name, arguments=args, ok=False, error="Tool-call budget exhausted")
                    )
                state.pending_index = len(tool_calls)
                return "budget_exhausted", None

            forced = resume_decision if i == state.pending_index and resume_decision is not None else None
            outcome, payload = await self._execute_one(state, tool_calls[i], status_cb, forced_decision=forced)

            if outcome == "confirmation_required":
                state.pending_batch = tool_calls
                state.pending_index = i
                self._run_states.save(state)
                return "confirmation_required", payload

            state.tool_calls.append(payload)
            state.tool_calls_used += 1
            i += 1
            state.pending_index = i

        return "continue", None

    async def _step_loop(self, state: RunState, status_cb: Callable) -> AgentRunResult:
        max_steps = state.profile.max_steps or self._max_agent_steps
        specs = self._tool_router.available_specs(state.profile.tools or None, state.profile.mcp_servers or None)

        while state.steps_used < max_steps:
            state.steps_used += 1
            await status_cb({"type": "generating", "step": state.steps_used})
            try:
                result = await self._chat_with_retries(
                    state.conversation,
                    specs=specs if state.tool_calls_used < self._max_tool_calls else None,
                    profile=state.profile,
                    status_cb=status_cb,
                )
            except ProviderError as exc:
                self._run_states.discard(state.run_id)
                return AgentRunResult(
                    task=state.task, run_id=state.run_id, plan=state.plan, tool_calls=state.tool_calls,
                    status="error", final_answer=f"LLM provider error: {exc}", steps_used=state.steps_used,
                )

            if not result.tool_calls:
                self._run_states.discard(state.run_id)
                return AgentRunResult(
                    task=state.task, run_id=state.run_id, plan=state.plan, tool_calls=state.tool_calls,
                    status="success", final_answer=result.content, steps_used=state.steps_used,
                )

            state.conversation.append(
                ChatMessage(role="assistant", content=result.content or "", tool_calls=result.tool_calls)
            )
            state.pending_index = 0
            outcome, payload = await self._run_batch(state, result.tool_calls, status_cb)

            if outcome == "confirmation_required":
                confirmation: PendingConfirmation = payload
                self._run_states.save(state)
                return AgentRunResult(
                    task=state.task, run_id=state.run_id, plan=state.plan, tool_calls=state.tool_calls,
                    status="confirmation_required", steps_used=state.steps_used,
                    pending_confirmation=confirmation.to_dict(),
                )
            if outcome == "budget_exhausted":
                self._run_states.discard(state.run_id)
                return AgentRunResult(
                    task=state.task, run_id=state.run_id, plan=state.plan, tool_calls=state.tool_calls,
                    status="max_tool_calls_reached", steps_used=state.steps_used,
                    final_answer="Reached the maximum number of tool calls for this run before finishing.",
                )
            # outcome == "continue": loop back and let the model reason over the new observations

        self._run_states.discard(state.run_id)
        return AgentRunResult(
            task=state.task, run_id=state.run_id, plan=state.plan, tool_calls=state.tool_calls,
            status="max_steps_reached", steps_used=state.steps_used,
            final_answer=(
                "The agent reached its maximum step limit before producing a final answer. "
                "Try narrowing the request or increasing MAX_AGENT_STEPS."
            ),
        )

    async def run(
        self,
        task: str,
        session_id: str,
        messages: list[ChatMessage],
        profile: AgentProfile,
        *,
        on_status: StatusCallback = None,
    ) -> AgentRunResult:
        status_cb = on_status or _noop_status
        state = self._run_states.create(task, session_id, profile, list(messages))

        await status_cb({"type": "planning"})
        try:
            plan = await self._planner.create_plan(task, model=profile.model)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Planning failed unexpectedly")
            self._run_states.discard(state.run_id)
            return AgentRunResult(task=task, run_id=state.run_id, status="error", final_answer=f"Planning failed: {exc}")
        state.plan = plan
        self._ensure_plan_message(state)
        await status_cb({"type": "plan_ready", "plan": plan})

        return await self._step_loop(state, status_cb)

    async def resume(
        self,
        run_id: str,
        decision: str,
        *,
        on_status: StatusCallback = None,
    ) -> Optional[AgentRunResult]:
        """Continues a paused run after a confirmation has been approved/denied."""
        status_cb = on_status or _noop_status
        state = self._run_states.get(run_id)
        if state is None:
            return None

        outcome, payload = await self._run_batch(
            state, state.pending_batch, status_cb, resume_decision=decision
        )

        if outcome == "confirmation_required":
            confirmation: PendingConfirmation = payload
            self._run_states.save(state)
            return AgentRunResult(
                task=state.task, run_id=state.run_id, plan=state.plan, tool_calls=state.tool_calls,
                status="confirmation_required", steps_used=state.steps_used,
                pending_confirmation=confirmation.to_dict(),
            )
        if outcome == "budget_exhausted":
            self._run_states.discard(state.run_id)
            return AgentRunResult(
                task=state.task, run_id=state.run_id, plan=state.plan, tool_calls=state.tool_calls,
                status="max_tool_calls_reached", steps_used=state.steps_used,
                final_answer="Reached the maximum number of tool calls for this run before finishing.",
            )

        # The batch that was blocked is now fully resolved — continue the outer loop.
        return await self._step_loop(state, status_cb)
