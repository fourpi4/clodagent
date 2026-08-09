"""
Agent Core: the single entrypoint the API layer talks to.

    USER REQUEST -> CONTEXT -> PLANNER -> TOOL SELECTION -> EXECUTOR
        -> OBSERVATION -> REASON/REPLAN -> FINAL RESULT

All wiring (provider, tools, memory, profiles) happens here so nothing
above (API/UI) needs to know about any concrete implementation.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.agent.confirmation import ConfirmationExpiredError, ConfirmationStore
from app.agent.context import ContextBuilder
from app.agent.executor import AgentRunResult, Executor, StatusCallback
from app.agent.memory import AgentMemory
from app.agent.planner import Planner
from app.agent.profiles import AgentProfile, ProfileStore
from app.agent.run_state import RunStateStore
from app.agent.tool_router import ToolRouter
from app.providers.registry import ProviderRegistry
from app.providers.router import ProviderRouter
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        provider_router: ProviderRouter,
        registry: ToolRegistry,
        memory: AgentMemory,
        profile_store: ProfileStore,
        *,
        max_agent_steps: int,
        max_tool_calls: int,
        tool_timeout: float,
        max_retries: int,
        confirmation_ttl_seconds: int = 600,
        provider_registry: Optional[ProviderRegistry] = None,
    ) -> None:
        self.provider_router = provider_router
        self.provider = provider_router  # LLMProvider-compatible facade (chat/stream/list_models)
        self.provider_registry = provider_registry
        self.registry = registry
        self.memory = memory
        self.profiles = profile_store
        self.confirmations = ConfirmationStore(ttl_seconds=confirmation_ttl_seconds)
        self.run_states = RunStateStore()

        self._context_builder = ContextBuilder(memory)
        self._planner = Planner(provider_router)
        self._tool_router = ToolRouter(registry, tool_timeout=tool_timeout, max_retries=max_retries)
        self._executor = Executor(
            provider_router,
            self._planner,
            self._tool_router,
            self.confirmations,
            self.run_states,
            max_agent_steps=max_agent_steps,
            max_tool_calls=max_tool_calls,
            max_retries=max_retries,
        )

    def _resolve_profile(self, profile_name: Optional[str]) -> AgentProfile:
        if profile_name:
            profile = self.profiles.get_profile(profile_name)
            if profile:
                return profile
        return AgentProfile(name="default")

    def _maybe_record_final(self, profile: AgentProfile, session_id: str, result: AgentRunResult) -> None:
        if profile.memory_enabled and result.status != "confirmation_required" and result.final_answer:
            self.memory.record(session_id, "assistant", result.final_answer)

    async def run_turn(
        self,
        session_id: str,
        user_message: str,
        *,
        profile_name: Optional[str] = None,
        on_status: StatusCallback = None,
    ) -> AgentRunResult:
        profile = self._resolve_profile(profile_name)
        messages = self._context_builder.build(profile, session_id, user_message)

        if profile.memory_enabled:
            self.memory.record(session_id, "user", user_message)

        result = await self._executor.run(user_message, session_id, messages, profile, on_status=on_status)
        self._maybe_record_final(profile, session_id, result)
        return result

    async def run_task(
        self,
        task: str,
        *,
        profile_name: Optional[str] = None,
        on_status: StatusCallback = None,
    ) -> AgentRunResult:
        """Headless one-off run (no persistent chat session), used by /api/agent/run."""
        profile = self._resolve_profile(profile_name)
        session_id = self.memory.start_session(profile.name)
        return await self.run_turn(session_id, task, profile_name=profile.name, on_status=on_status)

    async def approve_confirmation(
        self, confirmation_id: str, *, on_status: StatusCallback = None
    ) -> AgentRunResult:
        return await self._resolve_confirmation(confirmation_id, "approved", on_status=on_status)

    async def deny_confirmation(
        self, confirmation_id: str, *, on_status: StatusCallback = None
    ) -> AgentRunResult:
        return await self._resolve_confirmation(confirmation_id, "denied", on_status=on_status)

    async def _resolve_confirmation(
        self, confirmation_id: str, decision: str, *, on_status: StatusCallback = None
    ) -> AgentRunResult:
        if decision == "approved":
            confirmation = self.confirmations.approve(confirmation_id)
        else:
            confirmation = self.confirmations.deny(confirmation_id)

        state = self.run_states.get(confirmation.run_id)
        if state is None:
            raise ConfirmationExpiredError(
                f"Run '{confirmation.run_id}' for confirmation '{confirmation_id}' is no longer active "
                "(process may have restarted, or the run already completed)."
            )

        result = await self._executor.resume(confirmation.run_id, decision, on_status=on_status)
        assert result is not None
        self._maybe_record_final(state.profile, state.session_id, result)
        return result
