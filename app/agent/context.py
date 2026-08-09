"""
Builds the message list sent to the LLM for a given turn: system prompt
(from the AgentProfile) + relevant long-term facts + recent conversation
history + the new user message.
"""
from __future__ import annotations

from app.agent.memory import AgentMemory
from app.agent.profiles import AgentProfile
from app.providers.base import ChatMessage

MAX_HISTORY_MESSAGES = 20


class ContextBuilder:
    def __init__(self, memory: AgentMemory) -> None:
        self._memory = memory

    def build(self, profile: AgentProfile, session_id: str, user_message: str) -> list[ChatMessage]:
        messages: list[ChatMessage] = [ChatMessage(role="system", content=profile.system_prompt)]

        if profile.memory_enabled:
            facts = self._memory.relevant_facts(profile.name, user_message, limit=5)
            if facts:
                fact_lines = "\n".join(f"- {f['text']}" for f in facts)
                messages.append(
                    ChatMessage(
                        role="system",
                        content=f"Relevant long-term memory for this agent:\n{fact_lines}",
                    )
                )

            for turn in self._memory.conversation(session_id, limit=MAX_HISTORY_MESSAGES):
                messages.append(ChatMessage(role=turn["role"], content=turn["content"]))

        messages.append(ChatMessage(role="user", content=user_message))
        return messages
