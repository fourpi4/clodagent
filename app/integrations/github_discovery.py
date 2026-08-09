"""
GitHub Discovery: finds, scores and classifies open-source AI-agent projects.

Uses the public GitHub REST API (https://docs.github.com/en/rest) directly —
no scraping, no cloning, no running untrusted code. For each candidate repo
it fetches: repo metadata, license, latest release, and contributor count,
then produces a ranked report plus a license-gated integration recommendation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_QUERIES = [
    "AI agent",
    "autonomous agent",
    "LLM agent",
    "agent framework",
    "MCP server",
    "AI tools",
    "agent tools",
    "multi agent",
    "agentic workflow",
]

# SPDX identifiers GitHub commonly reports. Permissive licenses can be
# auto-integrated; everything else (GPL/AGPL/LGPL/unknown/none) requires a
# human to explicitly confirm before any code or dependency is pulled in.
PERMISSIVE_LICENSES = {
    "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "bsd-3-clause-clear",
    "isc", "unlicense", "0bsd", "mpl-2.0", "zlib",
}
COPYLEFT_LICENSES = {"gpl-2.0", "gpl-3.0", "agpl-3.0", "lgpl-2.1", "lgpl-3.0"}


@dataclass
class RepoCandidate:
    name: str
    full_name: str
    url: str
    description: str
    stars: int
    forks: int
    open_issues: int
    language: Optional[str]
    license_spdx: Optional[str]
    license_name: Optional[str]
    pushed_at: Optional[str]
    created_at: Optional[str]
    contributors_count: Optional[int]
    has_wiki: bool
    homepage: Optional[str]
    topics: list[str] = field(default_factory=list)

    # derived signals, filled in by the scorer
    has_api: bool = False
    has_sdk: bool = False
    has_mcp: bool = False
    license_category: str = "unknown"  # permissive | copyleft | unknown
    auto_integrable: bool = False
    days_since_push: Optional[int] = None
    score: float = 0.0
    integration_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "repository": self.url,
            "stars": self.stars,
            "description": self.description,
            "license": self.license_name or "none",
            "license_category": self.license_category,
            "last_update": self.pushed_at,
            "language": self.language,
            "api_available": self.has_api,
            "sdk_available": self.has_sdk,
            "mcp_available": self.has_mcp,
            "contributors": self.contributors_count,
            "forks": self.forks,
            "open_issues": self.open_issues,
            "score": round(self.score, 2),
            "auto_integrable": self.auto_integrable,
            "integration_hint": self.integration_hint,
        }


class GitHubClient:
    """Thin, auth-aware wrapper around the GitHub REST API."""

    def __init__(self, token: str = "", base_url: str = "https://api.github.com", timeout: int = 20) -> None:
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def search_repositories(self, query: str, *, per_page: int = 15) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{self._base_url}/search/repositories",
                headers=self._headers(),
                params={"q": query, "sort": "stars", "order": "desc", "per_page": per_page},
            )
        if resp.status_code != 200:
            logger.warning("GitHub search failed (%s): %s", resp.status_code, resp.text[:300])
            return []
        return resp.json().get("items", [])

    async def get_license(self, full_name: str) -> Optional[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{self._base_url}/repos/{full_name}/license", headers=self._headers())
        if resp.status_code != 200:
            return None
        return resp.json()

    async def get_contributors_count(self, full_name: str) -> Optional[int]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{self._base_url}/repos/{full_name}/contributors",
                headers=self._headers(),
                params={"per_page": 1, "anon": "true"},
            )
        if resp.status_code != 200:
            return None
        link = resp.headers.get("Link", "")
        if 'rel="last"' in link:
            for part in link.split(","):
                if 'rel="last"' in part:
                    try:
                        url = part[part.index("<") + 1: part.index(">")]
                        page = int(url.split("page=")[-1].split("&")[0])
                        return page
                    except (ValueError, IndexError):
                        return None
        return len(resp.json()) if isinstance(resp.json(), list) else None

    async def get_repo(self, full_name: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{self._base_url}/repos/{full_name}", headers=self._headers())
        if resp.status_code != 200:
            raise ValueError(f"Repository '{full_name}' not found or inaccessible ({resp.status_code})")
        return resp.json()

    async def get_readme_text(self, full_name: str) -> str:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{self._base_url}/repos/{full_name}/readme",
                headers={**self._headers(), "Accept": "application/vnd.github.raw+json"},
            )
        if resp.status_code != 200:
            return ""
        return resp.text[:20_000]


def _days_since(iso_ts: Optional[str]) -> Optional[int]:
    if not iso_ts:
        return None
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - dt).days


def _license_category(spdx: Optional[str]) -> str:
    if not spdx or spdx.lower() in {"noassertion", "other"}:
        return "unknown"
    key = spdx.lower()
    if key in PERMISSIVE_LICENSES:
        return "permissive"
    if key in COPYLEFT_LICENSES:
        return "copyleft"
    return "unknown"


def _detect_capabilities(readme: str, topics: list[str], homepage: Optional[str]) -> tuple[bool, bool, bool]:
    """Heuristic detection of API / SDK / MCP availability from README + topics."""
    text = (readme or "").lower()
    topic_set = {t.lower() for t in topics}

    has_api = any(k in text for k in ["rest api", "http api", "api reference", "openapi", "swagger"]) or bool(homepage)
    has_sdk = any(k in text for k in ["pip install", "npm install", "sdk", "client library", "import "]) or "sdk" in topic_set
    has_mcp = "mcp" in topic_set or "model context protocol" in text or "mcp server" in text
    return has_api, has_sdk, has_mcp


def _score(candidate: RepoCandidate) -> float:
    """
    Weighted score. Stars matter, but are deliberately capped in influence so
    that a 40k-star abandoned repo doesn't outrank a well-maintained,
    well-documented, actively developed project with 2k stars.
    """
    star_score = min(candidate.stars, 20_000) / 20_000 * 30  # capped at 30 pts

    recency_score = 0.0
    if candidate.days_since_push is not None:
        if candidate.days_since_push <= 30:
            recency_score = 20
        elif candidate.days_since_push <= 90:
            recency_score = 14
        elif candidate.days_since_push <= 180:
            recency_score = 8
        elif candidate.days_since_push <= 365:
            recency_score = 3

    contributors_score = 0.0
    if candidate.contributors_count:
        contributors_score = min(candidate.contributors_count, 50) / 50 * 15

    docs_score = 10 if candidate.has_wiki or candidate.homepage else 4
    api_score = 10 if candidate.has_api else 0
    sdk_score = 7 if candidate.has_sdk else 0
    mcp_score = 8 if candidate.has_mcp else 0
    license_score = 10 if candidate.license_category == "permissive" else (3 if candidate.license_category == "copyleft" else 0)

    return (
        star_score + recency_score + contributors_score + docs_score
        + api_score + sdk_score + mcp_score + license_score
    )


def _integration_hint(candidate: RepoCandidate) -> str:
    if candidate.has_mcp:
        return "Connect as an MCP server via integrations/mcp.py (McpClientManager.connect)."
    if candidate.has_api:
        return "Wrap its REST API in a custom adapter (integrations/custom_api_adapter.py)."
    if candidate.has_sdk:
        return "Wrap its Python/JS SDK in a dedicated adapter under app/integrations/."
    return "No API/SDK/MCP detected — treat as reference only, do not vendor code directly."


class GitHubDiscovery:
    def __init__(self, client: GitHubClient, allow_gpl_auto: bool = False) -> None:
        self._client = client
        self._allow_gpl_auto = allow_gpl_auto

    async def _build_candidate(self, item: dict[str, Any]) -> RepoCandidate:
        full_name = item["full_name"]
        license_info = item.get("license") or {}
        spdx = license_info.get("spdx_id")
        license_name = license_info.get("name")

        # license endpoint is more reliable than the search payload for some repos
        if not spdx:
            detailed = await self._client.get_license(full_name)
            if detailed:
                spdx = (detailed.get("license") or {}).get("spdx_id")
                license_name = (detailed.get("license") or {}).get("name")

        readme = await self._client.get_readme_text(full_name)
        contributors = await self._client.get_contributors_count(full_name)

        candidate = RepoCandidate(
            name=item["name"],
            full_name=full_name,
            url=item["html_url"],
            description=item.get("description") or "",
            stars=item.get("stargazers_count", 0),
            forks=item.get("forks_count", 0),
            open_issues=item.get("open_issues_count", 0),
            language=item.get("language"),
            license_spdx=spdx,
            license_name=license_name,
            pushed_at=item.get("pushed_at"),
            created_at=item.get("created_at"),
            contributors_count=contributors,
            has_wiki=item.get("has_wiki", False),
            homepage=item.get("homepage") or None,
            topics=item.get("topics", []) or [],
        )
        candidate.days_since_push = _days_since(candidate.pushed_at)
        candidate.license_category = _license_category(candidate.license_spdx)
        candidate.has_api, candidate.has_sdk, candidate.has_mcp = _detect_capabilities(
            readme, candidate.topics, candidate.homepage
        )
        candidate.auto_integrable = candidate.license_category == "permissive" or (
            candidate.license_category == "copyleft" and self._allow_gpl_auto
        )
        candidate.score = _score(candidate)
        candidate.integration_hint = _integration_hint(candidate)
        return candidate

    async def discover(
        self,
        queries: Optional[list[str]] = None,
        *,
        per_query: int = 10,
        limit: int = 25,
    ) -> list[RepoCandidate]:
        queries = queries or DEFAULT_QUERIES
        seen: dict[str, dict[str, Any]] = {}
        for query in queries:
            for item in await self._client.search_repositories(query, per_page=per_query):
                seen[item["full_name"]] = item

        candidates = [await self._build_candidate(item) for item in seen.values()]
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:limit]

    async def analyze(self, full_name: str) -> dict[str, Any]:
        """Detailed integration-plan view for a single repo, used by /api/github/analyze."""
        repo_data = await self._client.get_repo(full_name)
        candidate = await self._build_candidate(repo_data)
        plan = {
            "repository": candidate.to_dict(),
            "license_gate": {
                "category": candidate.license_category,
                "auto_integrable": candidate.auto_integrable,
                "requires_manual_confirmation": not candidate.auto_integrable,
                "note": (
                    "Permissive license — safe for automatic adapter generation."
                    if candidate.license_category == "permissive"
                    else "Copyleft or unknown license — a human must explicitly approve integration; "
                         "no code will be copied automatically."
                ),
            },
            "recommended_integration": candidate.integration_hint,
            "what_would_be_installed": (
                "Only a thin adapter file under app/integrations/ that calls the project's public "
                "API/SDK/MCP endpoint. No source code from the target repository is copied or executed."
            ),
            "external_apis_used": [candidate.homepage] if candidate.homepage else [],
        }
        return plan
