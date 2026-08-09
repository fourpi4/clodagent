"""
AgentMemory: facade combining short-term (conversation) and long-term
(durable facts) memory into what the rest of Agent Core needs.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory


@dataclass
class AgentMemory:
    short_term: ShortTermMemory
    long_term: LongTermMemory

    def start_session(self, agent_profile: str) -> str:
        return self.short_term.create_session(agent_profile)

    def record(self, session_id: str, role: str, content: str, metadata: dict | None = None) -> None:
        self.short_term.add_message(session_id, role, content, metadata)

    def conversation(self, session_id: str, limit: int = 20) -> list[dict]:
        return self.short_term.get_history(session_id, limit=limit)

    def relevant_facts(self, agent_profile: str, query: str, limit: int = 5) -> list[dict]:
        return self.long_term.recall(agent_profile, query, limit=limit)

    def remember_fact(self, agent_profile: str, text: str, tags: list[str] | None = None) -> str:
        return self.long_term.remember(agent_profile, text, tags)
