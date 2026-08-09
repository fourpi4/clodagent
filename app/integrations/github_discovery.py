"""
GitHub Discovery: finds, scores and classifies open-source AI-agent projects.

Uses the public GitHub REST API (https://docs.github.com/en/rest) directly —
no scraping, no cloning, no running untrusted code.

Two-phase, rate-limit-aware pipeline:
  1. search_repositories() only — cheap, no per-repo calls — ranked by a
     lightweight score (stars + recency) computed straight from the search
     payload.
  2. Deep analysis (README, license endpoint fallback, contributor count,
     archived/fork/latest-release) runs only for the top-N candidates from
     phase 1, with bounded concurrency so a discovery scan never fires off
     hundreds of sequential requests.

Capability detection (API/SDK/MCP) is heuristic and says so: each signal
reports `available`, a `confidence` score, and the `evidence` string that
produced it, instead of a bare boolean.
"""
from __future__ import annotations

import asyncio
import logging
import time
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

# SPDX identifiers GitHub commonly reports.
#   permissive     -> safe for automatic adapter generation
#   weak_copyleft  -> file-level copyleft (e.g. MPL); not "just permissive", needs a human look
#   copyleft       -> GPL-family; always needs manual approval unless explicitly overridden
#   unknown        -> always needs manual review
PERMISSIVE_LICENSES = {
    "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "bsd-3-clause-clear",
    "isc", "unlicense", "0bsd", "zlib",
}
WEAK_COPYLEFT_LICENSES = {"mpl-2.0", "mpl-1.1", "epl-1.0", "epl-2.0", "cddl-1.0", "cddl-1.1"}
COPYLEFT_LICENSES = {"gpl-2.0", "gpl-3.0", "agpl-3.0", "lgpl-2.1", "lgpl-3.0"}

_DEEP_ANALYSIS_CONCURRENCY = 5
_CACHE_TTL_SECONDS = 300


@dataclass
class CapabilitySignal:
    available: bool
    confidence: float  # 0.0-1.0, heuristic — never treat as ground truth
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {"available": self.available, "confidence": round(self.confidence, 2), "evidence": self.evidence}


_NO_SIGNAL = CapabilitySignal(False, 0.0, "No supporting evidence found in README/topics/homepage.")


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
    contributors_count: Optional[int] = None
    has_wiki: bool = False
    homepage: Optional[str] = None
    topics: list[str] = field(default_factory=list)
    archived: bool = False
    is_fork: bool = False
    latest_release: Optional[str] = None

    # derived signals, filled in by the scorer
    capabilities: dict[str, CapabilitySignal] = field(default_factory=dict)
    license_category: str = "unknown"  # permissive | weak_copyleft | copyleft | unknown
    auto_integrable: bool = False
    days_since_push: Optional[int] = None
    score: float = 0.0
    integration_hint: str = ""
    deep_analyzed: bool = False

    def to_dict(self) -> dict[str, Any]:
        api = self.capabilities.get("api", _NO_SIGNAL)
        sdk = self.capabilities.get("sdk", _NO_SIGNAL)
        mcp = self.capabilities.get("mcp", _NO_SIGNAL)
        return {
            "name": self.name,
            "repository": self.url,
            "stars": self.stars,
            "description": self.description,
            "license": self.license_name or "none",
            "license_category": self.license_category,
            "last_update": self.pushed_at,
            "language": self.language,
            "archived": self.archived,
            "fork": self.is_fork,
            "latest_release": self.latest_release,
            # kept for backward-compatible simple boolean checks
            "api_available": api.available,
            "sdk_available": sdk.available,
            "mcp_available": mcp.available,
            # new confidence-scored form
            "capabilities": {k: v.to_dict() for k, v in self.capabilities.items()},
            "contributors": self.contributors_count,
            "forks": self.forks,
            "open_issues": self.open_issues,
            "score": round(self.score, 2),
            "auto_integrable": self.auto_integrable,
            "integration_hint": self.integration_hint,
            "deep_analyzed": self.deep_analyzed,
        }


class GitHubClient:
    """Thin, auth-aware wrapper around the GitHub REST API, with a small TTL cache and rate-limit awareness."""

    def __init__(self, token: str = "", base_url: str = "https://api.github.com", timeout: int = 20) -> None:
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._cache: dict[str, tuple[float, Any]] = {}
        self.rate_limit_remaining: Optional[int] = None

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _note_rate_limit(self, resp: httpx.Response) -> None:
        remaining = resp.headers.get("X-RateLimit-Remaining")
        if remaining is not None:
            try:
                self.rate_limit_remaining = int(remaining)
            except ValueError:
                pass

    @property
    def rate_limited(self) -> bool:
        return self.rate_limit_remaining is not None and self.rate_limit_remaining < 3

    def _cache_get(self, key: str) -> Any:
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.time() - ts > _CACHE_TTL_SECONDS:
            self._cache.pop(key, None)
            return None
        return value

    def _cache_set(self, key: str, value: Any) -> None:
        self._cache[key] = (time.time(), value)

    async def search_repositories(self, query: str, *, per_page: int = 15) -> list[dict[str, Any]]:
        cache_key = f"search:{query}:{per_page}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{self._base_url}/search/repositories",
                headers=self._headers(),
                params={"q": query, "sort": "stars", "order": "desc", "per_page": per_page},
            )
        self._note_rate_limit(resp)
        if resp.status_code != 200:
            logger.warning("GitHub search failed (%s): %s", resp.status_code, resp.text[:300])
            return []
        items = resp.json().get("items", [])
        self._cache_set(cache_key, items)
        return items

    async def get_license(self, full_name: str) -> Optional[dict[str, Any]]:
        cache_key = f"license:{full_name}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{self._base_url}/repos/{full_name}/license", headers=self._headers())
        self._note_rate_limit(resp)
        if resp.status_code != 200:
            return None
        data = resp.json()
        self._cache_set(cache_key, data)
        return data

    async def get_contributors_count(self, full_name: str) -> Optional[int]:
        cache_key = f"contributors:{full_name}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{self._base_url}/repos/{full_name}/contributors",
                headers=self._headers(),
                params={"per_page": 1, "anon": "true"},
            )
        self._note_rate_limit(resp)
        if resp.status_code != 200:
            return None
        count: Optional[int] = None
        link = resp.headers.get("Link", "")
        if 'rel="last"' in link:
            for part in link.split(","):
                if 'rel="last"' in part:
                    try:
                        url = part[part.index("<") + 1: part.index(">")]
                        count = int(url.split("page=")[-1].split("&")[0])
                    except (ValueError, IndexError):
                        count = None
        if count is None:
            body = resp.json()
            count = len(body) if isinstance(body, list) else None
        self._cache_set(cache_key, count)
        return count

    async def get_latest_release(self, full_name: str) -> Optional[str]:
        cache_key = f"release:{full_name}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached or None
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{self._base_url}/repos/{full_name}/releases/latest", headers=self._headers())
        self._note_rate_limit(resp)
        tag = resp.json().get("tag_name") if resp.status_code == 200 else None
        self._cache_set(cache_key, tag or "")
        return tag

    async def get_repo(self, full_name: str) -> dict[str, Any]:
        cache_key = f"repo:{full_name}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{self._base_url}/repos/{full_name}", headers=self._headers())
        self._note_rate_limit(resp)
        if resp.status_code != 200:
            raise ValueError(f"Repository '{full_name}' not found or inaccessible ({resp.status_code})")
        data = resp.json()
        self._cache_set(cache_key, data)
        return data

    async def get_readme_text(self, full_name: str) -> str:
        cache_key = f"readme:{full_name}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{self._base_url}/repos/{full_name}/readme",
                headers={**self._headers(), "Accept": "application/vnd.github.raw+json"},
            )
        self._note_rate_limit(resp)
        text = resp.text[:20_000] if resp.status_code == 200 else ""
        self._cache_set(cache_key, text)
        return text


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
    if key in WEAK_COPYLEFT_LICENSES:
        return "weak_copyleft"
    if key in COPYLEFT_LICENSES:
        return "copyleft"
    return "unknown"


def _detect_capabilities(readme: str, topics: list[str], homepage: Optional[str]) -> dict[str, CapabilitySignal]:
    """Heuristic, evidence-carrying detection of API / SDK / MCP availability."""
    text = (readme or "").lower()
    topic_set = {t.lower() for t in topics}
    signals: dict[str, CapabilitySignal] = {}

    api_markers = [m for m in ("rest api", "http api", "api reference", "openapi", "swagger") if m in text]
    if api_markers:
        signals["api"] = CapabilitySignal(True, 0.8, f"README mentions: {', '.join(api_markers)}")
    elif homepage:
        signals["api"] = CapabilitySignal(True, 0.3, f"Has a homepage URL ({homepage}) but no explicit API docs found")
    else:
        signals["api"] = _NO_SIGNAL

    sdk_markers = [m for m in ("pip install", "npm install", "client library", "official sdk") if m in text]
    if sdk_markers:
        signals["sdk"] = CapabilitySignal(True, 0.75, f"README mentions: {', '.join(sdk_markers)}")
    elif "sdk" in topic_set:
        signals["sdk"] = CapabilitySignal(True, 0.5, "Repository is tagged with the 'sdk' topic")
    else:
        signals["sdk"] = _NO_SIGNAL

    if "mcp" in topic_set:
        signals["mcp"] = CapabilitySignal(True, 0.9, "Repository is tagged with the 'mcp' topic")
    elif "model context protocol" in text or "mcp server" in text:
        signals["mcp"] = CapabilitySignal(True, 0.7, "README mentions 'Model Context Protocol' / 'MCP server'")
    else:
        signals["mcp"] = _NO_SIGNAL

    return signals


def _lightweight_score(item: dict[str, Any]) -> float:
    """Phase-1 ranking straight from the search payload, before any deep analysis."""
    stars = item.get("stargazers_count", 0)
    star_score = min(stars, 20_000) / 20_000 * 60

    days = _days_since(item.get("pushed_at"))
    recency_score = 0.0
    if days is not None:
        if days <= 30:
            recency_score = 30
        elif days <= 90:
            recency_score = 20
        elif days <= 180:
            recency_score = 10
        elif days <= 365:
            recency_score = 4

    penalty = 15 if item.get("archived") else 0
    return star_score + recency_score - penalty


def _score(candidate: RepoCandidate) -> float:
    """
    Full weighted score (after deep analysis). Stars matter, but are
    deliberately capped in influence so that a 40k-star abandoned repo
    doesn't outrank a well-maintained, well-documented, actively developed
    project with 2k stars.
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
    api_score = 10 * candidate.capabilities.get("api", _NO_SIGNAL).confidence
    sdk_score = 7 * candidate.capabilities.get("sdk", _NO_SIGNAL).confidence
    mcp_score = 8 * candidate.capabilities.get("mcp", _NO_SIGNAL).confidence
    license_score = {"permissive": 10, "weak_copyleft": 5, "copyleft": 3}.get(candidate.license_category, 0)
    archived_penalty = 25 if candidate.archived else 0
    fork_penalty = 5 if candidate.is_fork else 0

    return (
        star_score + recency_score + contributors_score + docs_score
        + api_score + sdk_score + mcp_score + license_score
        - archived_penalty - fork_penalty
    )


def _integration_hint(candidate: RepoCandidate) -> str:
    if candidate.capabilities.get("mcp", _NO_SIGNAL).available:
        return "Connect as an MCP server via integrations/mcp.py (McpClientManager.connect)."
    if candidate.capabilities.get("api", _NO_SIGNAL).available:
        return "Wrap its REST API in a custom adapter (integrations/custom_api_adapter.py)."
    if candidate.capabilities.get("sdk", _NO_SIGNAL).available:
        return "Wrap its Python/JS SDK in a dedicated adapter under app/integrations/."
    return "No API/SDK/MCP detected — treat as reference only, do not vendor code directly."


class GitHubDiscovery:
    def __init__(self, client: GitHubClient, allow_gpl_auto: bool = False) -> None:
        self._client = client
        self._allow_gpl_auto = allow_gpl_auto

    def _shallow_candidate(self, item: dict[str, Any]) -> RepoCandidate:
        license_info = item.get("license") or {}
        return RepoCandidate(
            name=item["name"],
            full_name=item["full_name"],
            url=item["html_url"],
            description=item.get("description") or "",
            stars=item.get("stargazers_count", 0),
            forks=item.get("forks_count", 0),
            open_issues=item.get("open_issues_count", 0),
            language=item.get("language"),
            license_spdx=license_info.get("spdx_id"),
            license_name=license_info.get("name"),
            pushed_at=item.get("pushed_at"),
            created_at=item.get("created_at"),
            has_wiki=item.get("has_wiki", False),
            homepage=item.get("homepage") or None,
            topics=item.get("topics", []) or [],
            archived=item.get("archived", False),
            is_fork=item.get("fork", False),
            score=_lightweight_score(item),
        )

    async def _deep_analyze(self, candidate: RepoCandidate, semaphore: asyncio.Semaphore) -> RepoCandidate:
        async with semaphore:
            if self._client.rate_limited:
                logger.warning("GitHub rate limit nearly exhausted — skipping deep analysis for %s", candidate.full_name)
                return candidate

            if not candidate.license_spdx:
                detailed = await self._client.get_license(candidate.full_name)
                if detailed:
                    candidate.license_spdx = (detailed.get("license") or {}).get("spdx_id")
                    candidate.license_name = (detailed.get("license") or {}).get("name")

            readme = await self._client.get_readme_text(candidate.full_name)
            candidate.contributors_count = await self._client.get_contributors_count(candidate.full_name)
            candidate.latest_release = await self._client.get_latest_release(candidate.full_name)

        candidate.days_since_push = _days_since(candidate.pushed_at)
        candidate.license_category = _license_category(candidate.license_spdx)
        candidate.capabilities = _detect_capabilities(readme, candidate.topics, candidate.homepage)
        candidate.auto_integrable = candidate.license_category == "permissive" or (
            candidate.license_category in ("copyleft", "weak_copyleft") and self._allow_gpl_auto
        )
        candidate.score = _score(candidate)
        candidate.integration_hint = _integration_hint(candidate)
        candidate.deep_analyzed = True
        return candidate

    async def discover(
        self,
        queries: Optional[list[str]] = None,
        *,
        per_query: int = 10,
        limit: int = 25,
        deep_analyze_top: int = 20,
    ) -> list[RepoCandidate]:
        queries = queries or DEFAULT_QUERIES

        # Phase 1: cheap search only, across all queries.
        seen: dict[str, dict[str, Any]] = {}
        for query in queries:
            for item in await self._client.search_repositories(query, per_page=per_query):
                seen[item["full_name"]] = item
            if self._client.rate_limited:
                logger.warning("GitHub rate limit nearly exhausted — stopping further searches early")
                break

        shallow = [self._shallow_candidate(item) for item in seen.values()]
        shallow.sort(key=lambda c: c.score, reverse=True)

        # Phase 2: deep analysis only for the top candidates, bounded concurrency.
        to_analyze = shallow[:deep_analyze_top]
        rest = shallow[deep_analyze_top:]
        semaphore = asyncio.Semaphore(_DEEP_ANALYSIS_CONCURRENCY)
        analyzed = await asyncio.gather(*(self._deep_analyze(c, semaphore) for c in to_analyze))

        combined = list(analyzed) + rest
        combined.sort(key=lambda c: c.score, reverse=True)
        return combined[:limit]

    async def analyze(self, full_name: str) -> dict[str, Any]:
        """Detailed integration-plan view for a single repo, used by /api/github/analyze."""
        repo_data = await self._client.get_repo(full_name)
        candidate = self._shallow_candidate(repo_data)
        semaphore = asyncio.Semaphore(1)
        candidate = await self._deep_analyze(candidate, semaphore)
        plan = {
            "repository": candidate.to_dict(),
            "license_gate": {
                "category": candidate.license_category,
                "auto_integrable": candidate.auto_integrable,
                "requires_manual_confirmation": not candidate.auto_integrable,
                "note": {
                    "permissive": "Permissive license — safe for automatic adapter generation.",
                    "weak_copyleft": "Weak (file-level) copyleft license (e.g. MPL) — not the same as permissive; "
                                      "a human should review file-level obligations before integration.",
                    "copyleft": "Copyleft license — a human must explicitly approve integration; "
                                "no code will be copied automatically.",
                    "unknown": "License could not be classified — requires manual review before any integration.",
                }[candidate.license_category],
            },
            "recommended_integration": candidate.integration_hint,
            "what_would_be_installed": (
                "Only a thin adapter file under app/integrations/ that calls the project's public "
                "API/SDK/MCP endpoint. No source code from the target repository is copied or executed."
            ),
            "external_apis_used": [candidate.homepage] if candidate.homepage else [],
        }
        return plan
