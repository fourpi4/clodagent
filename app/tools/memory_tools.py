"""
Long-term memory tools exposed to the model: remember / list / forget.

`memory_remember` only ever stores the exact text the model explicitly
chose to pass as an argument — it is never invoked automatically or fed
the raw conversation. As a safety net against accidentally persisting
credentials, it hard-blocks (policy block, not just a confirmation) any
text that looks like it contains a secret/credential.
"""
from __future__ import annotations

import re

from app.memory.long_term import LongTermMemory
from app.tools.base import Tool, ToolResult

_SENSITIVE_MARKERS = (
    "password",
    "passwd",
    "api key",
    "api_key",
    "apikey",
    "secret",
    "private key",
    "access token",
    "auth token",
    "credential",
    "ssn",
    "social security",
    "credit card",
)

# Common high-entropy credential shapes, even without a nearby keyword.
_SECRET_PATTERNS = [
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),  # OpenAI-style secret key
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),  # GitHub tokens
    re.compile(r"\b[A-Za-z0-9_\-]{32,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{20,}\b"),  # JWT-ish
]


def _looks_sensitive(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in _SENSITIVE_MARKERS):
        return True
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


class RememberMemoryTool(Tool):
    name = "memory_remember"
    description = (
        "Save a specific fact to long-term memory for future conversations. Only call this with text "
        "you have explicitly decided is worth remembering — never whole conversation transcripts. "
        "Never pass secrets, passwords, or API keys; such text is refused."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "agent_profile": {"type": "string", "description": "Agent profile this memory belongs to"},
            "text": {"type": "string", "description": "The exact fact to remember, phrased concisely"},
            "tags": {"type": "array", "description": "Optional short tags for later recall"},
        },
        "required": ["agent_profile", "text"],
    }
    risk_level = "write"
    side_effects = True
    retry_safe = False  # retrying a failed write could create a duplicate memory entry
    requires_confirmation = False  # gated by the policy block below instead

    def __init__(self, long_term: LongTermMemory) -> None:
        self._long_term = long_term

    async def execute(self, arguments: dict) -> ToolResult:
        text = arguments["text"]
        if _looks_sensitive(text):
            return ToolResult(
                ok=False,
                error=(
                    "Refused: this text looks like it may contain a secret, password, or credential. "
                    "Do not store credentials in memory — rephrase without the sensitive value."
                ),
            )
        tags = arguments.get("tags") or []
        memory_id = self._long_term.remember(arguments["agent_profile"], text, tags)
        return ToolResult(ok=True, output={"id": memory_id})


class ListMemoryTool(Tool):
    name = "memory_list"
    description = "List previously remembered facts for an agent profile."
    input_schema = {
        "type": "object",
        "properties": {"agent_profile": {"type": "string", "description": "Agent profile to list memories for"}},
        "required": ["agent_profile"],
    }
    risk_level = "read"
    side_effects = False
    retry_safe = True
    requires_confirmation = False

    def __init__(self, long_term: LongTermMemory) -> None:
        self._long_term = long_term

    async def execute(self, arguments: dict) -> ToolResult:
        memories = self._long_term.list_all(arguments.get("agent_profile"))
        return ToolResult(ok=True, output=memories)


class ForgetMemoryTool(Tool):
    name = "memory_forget"
    description = "Delete a previously remembered fact by its memory id."
    input_schema = {
        "type": "object",
        "properties": {"memory_id": {"type": "string", "description": "id returned by memory_remember or memory_list"}},
        "required": ["memory_id"],
    }
    risk_level = "write"
    side_effects = True
    retry_safe = True  # deleting an already-deleted id is a harmless no-op
    requires_confirmation = False

    def __init__(self, long_term: LongTermMemory) -> None:
        self._long_term = long_term

    async def execute(self, arguments: dict) -> ToolResult:
        self._long_term.forget(arguments["memory_id"])
        return ToolResult(ok=True, output={"deleted": arguments["memory_id"]})
