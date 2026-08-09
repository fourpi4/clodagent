"""Standard interface every tool (built-in, adapter-backed, or MCP-backed) implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ToolError(RuntimeError):
    """Raised when a tool fails validation or execution."""


class ToolInputError(ToolError):
    """Raised when the arguments passed to a tool do not match its input_schema."""


@dataclass
class ToolResult:
    ok: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """Base class for all tools registered with the ToolRegistry."""

    name: str
    description: str
    # JSON-schema describing accepted arguments, e.g.
    # {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    input_schema: dict[str, Any]
    # Marks tools that perform side effects a user may want to confirm (writes, network calls, etc).
    requires_confirmation: bool = False

    def validate(self, arguments: dict[str, Any]) -> None:
        """Minimal JSON-schema-ish validation: required keys + basic type checks."""
        schema = self.input_schema or {}
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        for key in required:
            if key not in arguments:
                raise ToolInputError(f"Missing required argument '{key}' for tool '{self.name}'")

        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        for key, value in arguments.items():
            spec = properties.get(key)
            if not spec:
                continue
            expected = type_map.get(spec.get("type"))
            if expected and not isinstance(value, expected):
                raise ToolInputError(
                    f"Argument '{key}' for tool '{self.name}' must be of type {spec.get('type')}"
                )

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "requires_confirmation": self.requires_confirmation,
        }
