"""
LangGraph adapter (https://github.com/langchain-ai/langgraph, MIT license).

LangGraph is not vendored or imported at package load time — it is an
optional dependency (`pip install langgraph`). This adapter wraps an
already-compiled LangGraph graph's official async entrypoint (`ainvoke`) as
a single Tool, so a LangGraph workflow can be exposed to the agent exactly
like any built-in tool, with no changes to Agent Core.
"""
from __future__ import annotations

from typing import Any

from app.tools.base import Tool, ToolError, ToolResult


class LangGraphAdapter(Tool):
    """
    Wraps a compiled LangGraph graph (the object returned by `StateGraph(...).compile()`)
    as a Tool. The caller is responsible for building/compiling the graph using
    LangGraph's own API; this adapter only calls its public `ainvoke` method.
    """

    def __init__(
        self,
        name: str,
        description: str,
        compiled_graph: Any,
        input_schema: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema or {
            "type": "object",
            "properties": {"input": {"type": "object", "description": "Initial LangGraph state"}},
            "required": ["input"],
        }
        self._graph = compiled_graph
        if not hasattr(compiled_graph, "ainvoke"):
            raise ToolError(
                "LangGraphAdapter requires a compiled graph exposing an async 'ainvoke' method "
                "(langgraph.graph.StateGraph(...).compile())."
            )

    async def execute(self, arguments: dict) -> ToolResult:
        state = arguments.get("input", arguments)
        try:
            result = await self._graph.ainvoke(state)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, error=f"LangGraph execution failed: {exc}")
        return ToolResult(ok=True, output=result)


def build_langgraph_adapter_from_module(module_path: str, graph_attr: str, name: str, description: str) -> LangGraphAdapter:
    """
    Convenience loader: imports `module_path` and reads a pre-compiled graph
    object off it (e.g. module_path='my_project.graphs', graph_attr='research_graph').
    Raises ImportError with a clear message if langgraph or the module isn't installed.
    """
    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Could not import '{module_path}'. Install the target project and its "
            f"dependencies (including langgraph) first: pip install langgraph"
        ) from exc

    graph = getattr(module, graph_attr)
    return LangGraphAdapter(name=name, description=description, compiled_graph=graph)
