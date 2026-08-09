"""
Planner: produces a short, structured plan before execution starts.

The plan is stored as structured data (list of step strings) for
observability — it is NOT the model's hidden chain-of-thought, just a
brief, user-presentable outline the executor can loosely follow. If the
model doesn't return valid JSON, a single-step fallback plan is used so the
agent never blocks on planning.
"""
from __future__ import annotations

import json
import logging
import re

from app.providers.base import ChatMessage, LLMProvider

logger = logging.getLogger(__name__)

PLAN_PROMPT = (
    "Break the following task into 2-5 short, concrete steps needed to complete it. "
    "Respond with ONLY a JSON array of strings, no prose, no markdown fences. "
    "If the task is trivial, return a single-step array.\n\nTask: {task}"
)

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class Planner:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def create_plan(self, task: str, *, model: str | None = None) -> list[str]:
        try:
            result = await self._provider.chat(
                [ChatMessage(role="user", content=PLAN_PROMPT.format(task=task))],
                temperature=0.0,
                model=model,
            )
            match = _JSON_ARRAY_RE.search(result.content)
            if match:
                steps = json.loads(match.group(0))
                if isinstance(steps, list) and all(isinstance(s, str) for s in steps) and steps:
                    return steps[:5]
        except Exception:  # noqa: BLE001 - planning must never crash the run
            logger.exception("Planning step failed, falling back to single-step plan")

        return [f"Address the task directly: {task}"]
