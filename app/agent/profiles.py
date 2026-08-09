"""
Agent Profiles: named configurations (system prompt, model, tools, ...).

Stored as a single local JSON file (data/agent_profiles.json). This keeps
the MVP dependency-free; swapping to a database table later only touches
this module.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class AgentProfile:
    name: str
    system_prompt: str = "You are a helpful, precise AI agent. Use tools when they help answer accurately."
    provider: Optional[str] = None  # None => ProviderRouter.DEFAULT_PROVIDER + fallback order
    model: Optional[str] = None  # None => the resolved provider's own default model
    temperature: float = 0.2
    tools: list[str] = field(default_factory=list)  # empty list = all registered tools allowed
    mcp_servers: list[str] = field(default_factory=list)
    memory_enabled: bool = True
    max_steps: Optional[int] = None  # None => Settings.MAX_AGENT_STEPS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_PROFILES = [
    AgentProfile(
        name="Research Agent",
        system_prompt=(
            "You are a research agent. Use web_fetch and github_search to gather accurate, "
            "up-to-date information before answering. Cite sources by URL."
        ),
        temperature=0.3,
    ),
    AgentProfile(
        name="Coding Agent",
        system_prompt=(
            "You are a coding agent. Use file_read/file_write/file_list to work inside the sandboxed "
            "workspace. Write correct, minimal code and explain non-obvious decisions briefly."
        ),
        temperature=0.1,
    ),
    AgentProfile(
        name="GitHub Agent",
        system_prompt=(
            "You are a GitHub research agent. Use github_search and github_get_repo to find and evaluate "
            "open-source AI agent projects. Always report license and activity before recommending a repo."
        ),
        temperature=0.2,
    ),
    AgentProfile(
        name="Web Research Agent",
        system_prompt="You are a web research agent. Use web_fetch to verify claims against live sources.",
        temperature=0.3,
    ),
]


class ProfileStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        if not self._path.exists():
            self._save([p.to_dict() for p in DEFAULT_PROFILES])

    def _load(self) -> list[dict[str, Any]]:
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _save(self, data: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def list_profiles(self) -> list[AgentProfile]:
        return [AgentProfile(**p) for p in self._load()]

    def get_profile(self, name: str) -> Optional[AgentProfile]:
        for p in self.list_profiles():
            if p.name == name:
                return p
        return None

    def upsert_profile(self, profile: AgentProfile) -> None:
        data = self._load()
        data = [p for p in data if p["name"] != profile.name]
        data.append(profile.to_dict())
        self._save(data)

    def delete_profile(self, name: str) -> None:
        data = [p for p in self._load() if p["name"] != name]
        self._save(data)
