from app.integrations.github_discovery import (
    RepoCandidate,
    _detect_capabilities,
    _integration_hint,
    _lightweight_score,
    _license_category,
    _score,
)


def test_license_category_permissive():
    assert _license_category("MIT") == "permissive"
    assert _license_category("Apache-2.0") == "permissive"


def test_license_category_weak_copyleft():
    """MPL is file-level copyleft — must NOT be classified as plain permissive."""
    assert _license_category("MPL-2.0") == "weak_copyleft"


def test_license_category_copyleft():
    assert _license_category("GPL-3.0") == "copyleft"
    assert _license_category("AGPL-3.0") == "copyleft"


def test_license_category_unknown():
    assert _license_category(None) == "unknown"
    assert _license_category("NOASSERTION") == "unknown"
    assert _license_category("Some-Weird-License") == "unknown"


def test_detect_capabilities_mcp():
    signals = _detect_capabilities("This project ships an MCP server for tool access.", ["mcp"], None)
    assert signals["mcp"].available is True
    assert signals["mcp"].confidence > 0
    assert signals["mcp"].evidence


def test_detect_capabilities_api_and_sdk():
    signals = _detect_capabilities(
        "Install with `pip install foo`. See our REST API reference for details.",
        [],
        "https://example.com",
    )
    assert signals["api"].available is True
    assert signals["sdk"].available is True


def test_detect_capabilities_no_evidence_is_low_confidence():
    signals = _detect_capabilities("Just a plain README with no relevant keywords.", [], None)
    assert signals["api"].available is False
    assert signals["mcp"].available is False
    assert signals["mcp"].confidence == 0.0


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
    candidate.capabilities = _detect_capabilities("", [], candidate.homepage)
    return candidate


def test_score_rewards_permissive_license_over_unknown():
    permissive = _make_candidate(license_spdx="mit")
    permissive.license_category = "permissive"
    unknown = _make_candidate(license_spdx=None)
    unknown.license_category = "unknown"

    assert _score(permissive) > _score(unknown)


def test_score_is_not_purely_star_driven():
    """A repo with fewer stars but better docs/recency/contributors can outscore a bigger, stale one."""
    huge_stale = _make_candidate(stars=50_000, contributors_count=1, has_wiki=False, homepage=None)
    huge_stale.days_since_push = 400
    huge_stale.license_category = "unknown"

    small_active = _make_candidate(stars=1500, contributors_count=40, has_wiki=True)
    small_active.days_since_push = 5
    small_active.license_category = "permissive"

    assert _score(small_active) > _score(huge_stale)


def test_score_penalizes_archived_and_fork():
    normal = _make_candidate()
    normal.days_since_push = 5
    archived = _make_candidate(archived=True)
    archived.days_since_push = 5
    fork = _make_candidate(is_fork=True)
    fork.days_since_push = 5

    assert _score(normal) > _score(archived)
    assert _score(normal) > _score(fork)


def test_integration_hint_prefers_mcp():
    candidate = _make_candidate()
    candidate.capabilities = _detect_capabilities("model context protocol server", ["mcp"], "https://example.com")
    assert "MCP" in _integration_hint(candidate)


def test_integration_hint_falls_back_to_reference_only():
    candidate = _make_candidate()
    candidate.capabilities = _detect_capabilities("nothing relevant here", [], None)
    assert "reference only" in _integration_hint(candidate)


def test_lightweight_score_penalizes_archived():
    active_item = {"stargazers_count": 5000, "pushed_at": "2024-01-01T00:00:00Z", "archived": False}
    archived_item = {"stargazers_count": 5000, "pushed_at": "2024-01-01T00:00:00Z", "archived": True}
    assert _lightweight_score(active_item) > _lightweight_score(archived_item)
