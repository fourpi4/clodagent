"""
CrewAI adapter (https://github.com/crewAIInc/crewAI, MIT license).

CrewAI is an optional dependency (`pip install crewai`), never imported at
package load time. This adapter wraps an already-constructed `Crew`
instance's official async entrypoint (`kickoff_async`) as a single Tool.
"""
from __future__ import annotations

from typing import Any

from app.tools.base import Tool, ToolError, ToolResult


class CrewAIAdapter(Tool):
    """
    Wraps a CrewAI `Crew` object (built with CrewAI's own Agent/Task/Crew API)
    as a Tool, calling its public `kickoff_async(inputs=...)` method.
    """

    def __init__(self, name: str, description: str, crew: Any, input_schema: dict[str, Any] | None = None) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema or {
            "type": "object",
            "properties": {"inputs": {"type": "object", "description": "Crew kickoff inputs dict"}},
            "required": [],
        }
        self._crew = crew
        if not hasattr(crew, "kickoff_async"):
            raise ToolError(
                "CrewAIAdapter requires a CrewAI 'Crew' instance exposing 'kickoff_async' "
                "(pip install crewai; from crewai import Crew, Agent, Task)."
            )

    async def execute(self, arguments: dict) -> ToolResult:
        inputs = arguments.get("inputs", {})
        try:
            result = await self._crew.kickoff_async(inputs=inputs)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, error=f"CrewAI execution failed: {exc}")
        return ToolResult(ok=True, output=str(result))
