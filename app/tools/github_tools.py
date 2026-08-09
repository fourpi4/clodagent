"""Agent-facing GitHub tools: repo search and repo lookup, built on the same GitHubClient used by github_discovery."""
from __future__ import annotations

from app.integrations.github_discovery import GitHubClient
from app.tools.base import Tool, ToolResult


class GitHubSearchTool(Tool):
    name = "github_search"
    description = "Search public GitHub repositories by keyword. Returns name, stars, description, license, and URL."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "GitHub search query, e.g. 'LLM agent framework'"},
            "limit": {"type": "integer", "description": "Max results to return (default 10)"},
        },
        "required": ["query"],
    }

    def __init__(self, client: GitHubClient) -> None:
        self._client = client

    async def execute(self, arguments: dict) -> ToolResult:
        limit = int(arguments.get("limit", 10))
        items = await self._client.search_repositories(arguments["query"], per_page=min(limit, 30))
        results = [
            {
                "full_name": i["full_name"],
                "url": i["html_url"],
                "stars": i.get("stargazers_count", 0),
                "description": i.get("description") or "",
                "license": (i.get("license") or {}).get("spdx_id"),
                "language": i.get("language"),
            }
            for i in items[:limit]
        ]
        return ToolResult(ok=True, output=results)


class GitHubGetRepoTool(Tool):
    name = "github_get_repo"
    description = "Fetch metadata for a single GitHub repository given its 'owner/repo' full name."
    input_schema = {
        "type": "object",
        "properties": {"full_name": {"type": "string", "description": "'owner/repo', e.g. 'octocat/Hello-World'"}},
        "required": ["full_name"],
    }

    def __init__(self, client: GitHubClient) -> None:
        self._client = client

    async def execute(self, arguments: dict) -> ToolResult:
        try:
            repo = await self._client.get_repo(arguments["full_name"])
        except ValueError as exc:
            return ToolResult(ok=False, error=str(exc))
        return ToolResult(
            ok=True,
            output={
                "full_name": repo["full_name"],
                "url": repo["html_url"],
                "stars": repo.get("stargazers_count", 0),
                "description": repo.get("description") or "",
                "license": (repo.get("license") or {}).get("spdx_id"),
                "language": repo.get("language"),
                "pushed_at": repo.get("pushed_at"),
            },
        )
