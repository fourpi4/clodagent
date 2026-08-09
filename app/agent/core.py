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

from app.agent.context import ContextBuilder
from app.agent.executor import AgentRunResult, Executor, StatusCallback
from app.agent.memory import AgentMemory
from app.agent.planner import Planner
from app.agent.profiles import AgentProfile, ProfileStore
from app.agent.tool_router import ToolRouter
from app.providers.base import LLMProvider
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        memory: AgentMemory,
        profile_store: ProfileStore,
        *,
        max_agent_steps: int,
        max_tool_calls: int,
        tool_timeout: float,
        max_retries: int,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.memory = memory
        self.profiles = profile_store

        self._context_builder = ContextBuilder(memory)
        self._planner = Planner(provider)
        self._tool_router = ToolRouter(registry, tool_timeout=tool_timeout, max_retries=max_retries)
        self._executor = Executor(
            provider,
            self._planner,
            self._tool_router,
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

        result = await self._executor.run(user_message, messages, profile, on_status=on_status)

        if profile.memory_enabled and result.final_answer:
            self.memory.record(session_id, "assistant", result.final_answer)

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
