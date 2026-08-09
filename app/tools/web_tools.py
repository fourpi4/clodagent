"""Web tools: HTTP fetch. No JS execution, no browser automation — just GET/POST with a strict size cap."""
from __future__ import annotations

import httpx

from app.tools.base import Tool, ToolResult

MAX_RESPONSE_CHARS = 20_000


class WebFetchTool(Tool):
    name = "web_fetch"
    description = (
        "Fetch the contents of a public URL over HTTP(S) and return the response body as text "
        "(truncated to ~20k characters). Use this to read documentation, APIs returning JSON, or web pages."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Fully-qualified http(s) URL to fetch"},
            "method": {"type": "string", "description": "HTTP method, defaults to GET"},
        },
        "required": ["url"],
    }

    def __init__(self, timeout: int = 20) -> None:
        self._timeout = timeout

    async def execute(self, arguments: dict) -> ToolResult:
        url = arguments["url"]
        method = arguments.get("method", "GET").upper()
        if not (url.startswith("http://") or url.startswith("https://")):
            return ToolResult(ok=False, error="url must start with http:// or https://")

        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                resp = await client.request(method, url)
        except httpx.HTTPError as exc:
            return ToolResult(ok=False, error=f"HTTP request failed: {exc}")

        text = resp.text[:MAX_RESPONSE_CHARS]
        return ToolResult(
            ok=resp.status_code < 400,
            output=text,
            error=None if resp.status_code < 400 else f"HTTP {resp.status_code}",
            metadata={"status_code": resp.status_code, "url": str(resp.url)},
        )
