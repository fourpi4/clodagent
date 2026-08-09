"""
SSRF-hardening tests for web_fetch: localhost/private-IP blocking, redirect
re-validation, and bounded response reads. Public-URL cases use
httpx.MockTransport (no real network); localhost blocking is checked via
real (local-only) DNS resolution, which needs no network access.
"""
import httpx

from app.tools.web_tools import WebFetchTool, validate_url, SsrfBlockedError


def test_validate_url_blocks_localhost_hostname():
    try:
        validate_url("http://localhost:8000/admin")
        assert False, "expected SsrfBlockedError"
    except SsrfBlockedError:
        pass


def test_validate_url_blocks_loopback_ip_literal():
    try:
        validate_url("http://127.0.0.1/secret")
        assert False, "expected SsrfBlockedError"
    except SsrfBlockedError:
        pass


def test_validate_url_blocks_link_local_metadata_ip():
    """169.254.169.254 is the common cloud metadata endpoint (AWS/GCP/Azure) — link-local range."""
    try:
        validate_url("http://169.254.169.254/latest/meta-data/")
        assert False, "expected SsrfBlockedError"
    except SsrfBlockedError:
        pass


def test_validate_url_blocks_private_ipv4():
    try:
        validate_url("http://10.0.0.5/")
        assert False, "expected SsrfBlockedError"
    except SsrfBlockedError:
        pass


def test_validate_url_blocks_file_scheme():
    try:
        validate_url("file:///etc/passwd")
        assert False, "expected SsrfBlockedError"
    except SsrfBlockedError:
        pass


def test_validate_url_allows_public_ip_literal():
    validate_url("http://93.184.216.34/")  # example.com's old IP — a real public address, should not raise


async def test_web_fetch_tool_blocks_localhost():
    tool = WebFetchTool()
    result = await tool.execute({"url": "http://localhost/admin"})
    assert result.ok is False
    assert "Blocked" in result.error


async def test_web_fetch_tool_blocks_redirect_to_localhost(mock_httpx):
    def handler(request: httpx.Request) -> httpx.Response:
        if "example.com" in str(request.url):
            return httpx.Response(302, headers={"Location": "http://127.0.0.1:9999/internal"})
        raise AssertionError(f"should never actually request the redirect target: {request.url}")

    mock_httpx["handler"] = handler
    tool = WebFetchTool()
    result = await tool.execute({"url": "http://example.com/redirect-me"})

    assert result.ok is False
    assert "redirect" in result.error.lower()


async def test_web_fetch_tool_enforces_response_size_limit(mock_httpx):
    big_body = b"x" * 1000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=big_body)

    mock_httpx["handler"] = handler
    tool = WebFetchTool(max_response_bytes=100)
    result = await tool.execute({"url": "http://example.com/big"})

    assert result.ok is True
    assert result.metadata["truncated"] is True
    assert len(result.output) <= 100


async def test_web_fetch_tool_rejects_disallowed_method():
    tool = WebFetchTool(allowed_methods=["GET", "HEAD"])
    result = await tool.execute({"url": "http://example.com/", "method": "POST"})
    assert result.ok is False
    assert "not allowed" in result.error.lower()
