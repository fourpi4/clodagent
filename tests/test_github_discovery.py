from app.integrations.github_discovery import (
    RepoCandidate,
    _detect_capabilities,
    _integration_hint,
    _license_category,
    _score,
)


def test_license_category_permissive():
    assert _license_category("MIT") == "permissive"
    assert _license_category("Apache-2.0") == "permissive"


def test_license_category_copyleft():
    assert _license_category("GPL-3.0") == "copyleft"
    assert _license_category("AGPL-3.0") == "copyleft"


def test_license_category_unknown():
    assert _license_category(None) == "unknown"
    assert _license_category("NOASSERTION") == "unknown"
    assert _license_category("Some-Weird-License") == "unknown"


def test_detect_capabilities_mcp():
    has_api, has_sdk, has_mcp = _detect_capabilities(
        "This project ships an MCP server for tool access.", ["mcp"], None
    )
    assert has_mcp is True


def test_detect_capabilities_api_and_sdk():
    has_api, has_sdk, has_mcp = _detect_capabilities(
        "Install with `pip install foo`. See our REST API reference for details.",
        [],
        "https://example.com",
    )
    assert has_api is True
    assert has_sdk is True


def _make_candidate(**overrides) -> RepoCandidate:
    base = dict(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        description="",
        stars=1000,
        forks=10,
        open_issues=5,
        language="Python",
        license_spdx="mit",
        license_name="MIT",
        pushed_at="2024-01-01T00:00:00Z",
        created_at="2023-01-01T00:00:00Z",
        contributors_count=20,
        has_wiki=True,
        homepage="https://example.com",
    )
    base.update(overrides)
    candidate = RepoCandidate(**base)
    candidate.license_category = _license_category(candidate.license_spdx)
    return candidate


def test_score_rewards_permissive_license_over_unknown():
    permissive = _make_candidate(license_spdx="mit")
    permissive.license_category = "permissive"
    unknown = _make_candidate(license_spdx=None)
    unknown.license_category = "unknown"

    assert _score(permissive) > _score(unknown)


def test_score_is_not_purely_star_driven():
    """A repo with fewer stars but better docs/recency/contributors can outscore a bigger, stale one."""
    huge_stale = _make_candidate(stars=50_000, contributors_count=1, has_wiki=False, homepage=None, days_since_push=400)
    huge_stale.days_since_push = 400
    huge_stale.license_category = "unknown"

    small_active = _make_candidate(stars=1500, contributors_count=40, has_wiki=True)
    small_active.days_since_push = 5
    small_active.license_category = "permissive"

    assert _score(small_active) > _score(huge_stale)


def test_integration_hint_prefers_mcp():
    candidate = _make_candidate()
    candidate.has_mcp = True
    candidate.has_api = True
    assert "MCP" in _integration_hint(candidate)


def test_integration_hint_falls_back_to_reference_only():
    candidate = _make_candidate()
    candidate.has_mcp = False
    candidate.has_api = False
    candidate.has_sdk = False
    assert "reference only" in _integration_hint(candidate)
