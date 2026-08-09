"""
Generic REST-API adapter.

Use this when a discovered GitHub project exposes a plain REST API (no MCP,
no Python SDK). It turns a single documented endpoint into a Tool without
importing or executing any of the target project's code — only its public
HTTP contract is used.
"""
from __future__ import annotations

from typing import Any, Optional

import httpx

from app.tools.base import Tool, ToolResult


class CustomApiAdapter(Tool):
    """
    Wraps one REST endpoint as a Tool.

    Example:
        CustomApiAdapter(
            name="weather_lookup",
            description="Get current weather for a city",
            input_schema={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            base_url="https://api.example-weather-project.com",
            path_template="/v1/weather/{city}",
            method="GET",
        )
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        base_url: str,
        path_template: str,
        method: str = "GET",
        headers: Optional[dict[str, str]] = None,
        timeout: int = 20,
        requires_confirmation: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.requires_confirmation = requires_confirmation
        self._base_url = base_url.rstrip("/")
        self._path_template = path_template
        self._method = method.upper()
        self._headers = headers or {}
        self._timeout = timeout

    async def execute(self, arguments: dict) -> ToolResult:
        try:
            path = self._path_template.format(**arguments)
        except KeyError as exc:
            return ToolResult(ok=False, error=f"Missing path parameter: {exc}")

        query_params = {
            k: v for k, v in arguments.items() if f"{{{k}}}" not in self._path_template
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.request(
                    self._method,
                    f"{self._base_url}{path}",
                    headers=self._headers,
                    params=query_params if self._method == "GET" else None,
                    json=query_params if self._method != "GET" else None,
                )
        except httpx.HTTPError as exc:
            return ToolResult(ok=False, error=f"Request failed: {exc}")

        try:
            body: Any = resp.json()
        except ValueError:
            body = resp.text[:5000]

        return ToolResult(
            ok=resp.status_code < 400,
            output=body,
            error=None if resp.status_code < 400 else f"HTTP {resp.status_code}",
            metadata={"status_code": resp.status_code},
        )
