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

REPLAN_PROMPT = (
    "You are revising a plan that is no longer working. "
    "Original task: {task}\n"
    "Original plan: {prior_plan}\n"
    "What went wrong: {failure_context}\n"
    "Produce a short revised plan (2-5 concrete steps) that accounts for this failure. "
    "Respond with ONLY a JSON array of strings, no prose, no markdown fences."
)

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _extract_steps(content: str, fallback: list[str]) -> list[str]:
    try:
        match = _JSON_ARRAY_RE.search(content)
        if match:
            steps = json.loads(match.group(0))
            if isinstance(steps, list) and all(isinstance(s, str) for s in steps) and steps:
                return steps[:5]
    except Exception:  # noqa: BLE001
        logger.exception("Failed to parse plan JSON, using fallback")
    return fallback


class Planner:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def create_plan(self, task: str, *, model: str | None = None, provider: str | None = None) -> list[str]:
        try:
            result = await self._provider.chat(
                [ChatMessage(role="user", content=PLAN_PROMPT.format(task=task))],
                temperature=0.0,
                model=model,
                provider=provider,
            )
        except Exception:  # noqa: BLE001 - planning must never crash the run
            logger.exception("Planning step failed, falling back to single-step plan")
            return [f"Address the task directly: {task}"]
        return _extract_steps(result.content, [f"Address the task directly: {task}"])

    async def replan(
        self,
        task: str,
        prior_plan: list[str],
        failure_context: str,
        *,
        model: str | None = None,
    ) -> list[str]:
        """
        Produces a short, structured revised plan after an observation shows the
        current plan no longer fits. Returns only a safe list of step strings —
        never the model's private reasoning about *why* it changed course.
        """
        prompt = REPLAN_PROMPT.format(
            task=task,
            prior_plan=json.dumps(prior_plan),
            failure_context=failure_context[:500],
        )
        try:
            result = await self._provider.chat(
                [ChatMessage(role="user", content=prompt)],
                temperature=0.0,
                model=model,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Replanning failed, keeping prior plan")
            return prior_plan
        return _extract_steps(result.content, prior_plan)
