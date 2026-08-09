"""
Resumable run state.

A run normally executes start-to-finish in one call. But when it hits a
tool that requires human confirmation, execution must pause and be
resumable later from the exact point it stopped — with the exact same tool
arguments, no way for the client to substitute different ones on resume.

This is a single-process, in-memory MVP store (mirrors ConfirmationStore).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from app.agent.profiles import AgentProfile
from app.providers.base import ChatMessage


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    ok: bool
    output: Any = None
    error: Optional[str] = None


@dataclass
class RunState:
    run_id: str
    task: str
    session_id: str
    profile: AgentProfile
    conversation: list[ChatMessage]
    plan: list[str] = field(default_factory=list)
    plan_message_index: Optional[int] = None
    current_step_index: int = 0
    completed_steps: list[str] = field(default_factory=list)
    failed_steps: list[str] = field(default_factory=list)
    replanned: bool = False
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    tool_calls_used: int = 0
    steps_used: int = 0
    # Mid-batch resumability: the raw tool_calls from the last assistant
    # turn, and how far into that batch we've gotten.
    pending_batch: list[dict[str, Any]] = field(default_factory=list)
    pending_index: int = 0
    created_at: float = field(default_factory=time.time)


class RunStateStore:
    def __init__(self) -> None:
        self._states: dict[str, RunState] = {}

    def create(self, task: str, session_id: str, profile: AgentProfile, conversation: list[ChatMessage]) -> RunState:
        state = RunState(
            run_id=str(uuid.uuid4()),
            task=task,
            session_id=session_id,
            profile=profile,
            conversation=conversation,
        )
        self._states[state.run_id] = state
        return state

    def get(self, run_id: str) -> Optional[RunState]:
        return self._states.get(run_id)

    def save(self, state: RunState) -> None:
        self._states[state.run_id] = state

    def discard(self, run_id: str) -> None:
        self._states.pop(run_id, None)
