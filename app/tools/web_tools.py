"""
Web tools: HTTP fetch, hardened against SSRF.

Blocks requests to loopback/private/link-local/reserved/multicast
addresses (this also covers cloud metadata endpoints like
169.254.169.254, which fall in the link-local range), rejects non-http(s)
schemes (file://, ftp://, ...), resolves DNS up front to catch
IP-literal and simple-hostname SSRF attempts, and re-validates the
destination after every redirect hop instead of blindly following them.
Response bodies are read as a bounded stream so a malicious or oversized
response can never be fully buffered into memory.

Residual risk (documented, not fully closed): there is a small window
between our DNS pre-check and the actual TCP connection where a
DNS-rebinding attacker could swap the resolved address. Closing that
completely requires pinning the HTTP connection to the exact IP we
validated (a custom transport) — tracked as a Phase 2 hardening item.
"""
from __future__ import annotations

import ipaddress
import socket
from typing import Optional
from urllib.parse import urlparse, urljoin

import httpx

from app.tools.base import Tool, ToolResult

MAX_RESPONSE_CHARS = 20_000
DEFAULT_MAX_RESPONSE_BYTES = 2_000_000
MAX_REDIRECTS = 5
ALLOWED_SCHEMES = {"http", "https"}


class SsrfBlockedError(ValueError):
    pass


def _is_blocked_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # unparsable => treat as unsafe
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local  # covers cloud metadata endpoints, e.g. 169.254.169.254
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def validate_url(url: str) -> str:
    """Raises SsrfBlockedError if the URL's scheme/host/resolved IPs are unsafe. Returns the hostname."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise SsrfBlockedError(f"URL scheme '{parsed.scheme}' is not allowed (only http/https)")
    if not parsed.hostname:
        raise SsrfBlockedError("URL has no hostname")

    hostname = parsed.hostname
    if hostname.lower() in {"localhost", "localhost.localdomain"}:
        raise SsrfBlockedError("Requests to localhost are blocked")

    # If the hostname is itself a literal IP, this validates it directly.
    try:
        addrs = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise SsrfBlockedError(f"Could not resolve host '{hostname}': {exc}") from exc

    resolved_ips = {info[4][0] for info in addrs}
    for ip_str in resolved_ips:
        if _is_blocked_ip(ip_str):
            raise SsrfBlockedError(f"Host '{hostname}' resolves to a blocked address ({ip_str})")

    return hostname


class WebFetchTool(Tool):
    name = "web_fetch"
    description = (
        "Fetch the contents of a public URL over HTTP(S) and return the response body as text "
        f"(truncated to ~{MAX_RESPONSE_CHARS} characters). GET and HEAD only. Use this to read "
        "documentation, public APIs, or web pages. Cannot access localhost, private networks, or "
        "cloud metadata endpoints."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Fully-qualified http(s) URL to fetch"},
            "method": {"type": "string", "description": "HTTP method, GET or HEAD (default GET)"},
        },
        "required": ["url"],
    }
    risk_level = "network"
    side_effects = False
    retry_safe = True
    requires_confirmation = False

    def __init__(
        self,
        timeout: int = 20,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        allowed_methods: Optional[list[str]] = None,
    ) -> None:
        self._timeout = timeout
        self._max_response_bytes = max_response_bytes
        self._allowed_methods = {m.upper() for m in (allowed_methods or ["GET", "HEAD"])}

    async def _read_bounded(self, resp: httpx.Response) -> tuple[bytes, bool]:
        chunks: list[bytes] = []
        total = 0
        truncated = False
        async for chunk in resp.aiter_bytes():
            total += len(chunk)
            if total > self._max_response_bytes:
                remaining = self._max_response_bytes - (total - len(chunk))
                if remaining > 0:
                    chunks.append(chunk[:remaining])
                truncated = True
                break
            chunks.append(chunk)
        return b"".join(chunks), truncated

    async def execute(self, arguments: dict) -> ToolResult:
        method = arguments.get("method", "GET").upper()
        if method not in self._allowed_methods:
            return ToolResult(
                ok=False,
                error=(
                    f"Method '{method}' is not allowed for web_fetch (allowed: {sorted(self._allowed_methods)}). "
                    "Non-GET/HEAD requests are side-effecting network actions and require a separate, "
                    "explicitly-confirmed tool."
                ),
            )

        url = arguments["url"]
        try:
            validate_url(url)
        except SsrfBlockedError as exc:
            return ToolResult(ok=False, error=f"Blocked: {exc}")

        current_url = url
        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=False) as client:
                for _ in range(MAX_REDIRECTS + 1):
                    async with client.stream(method, current_url) as resp:
                        if resp.is_redirect:
                            location = resp.headers.get("location")
                            if not location:
                                return ToolResult(ok=False, error="Redirect response missing Location header")
                            next_url = urljoin(current_url, location)
                            try:
                                validate_url(next_url)
                            except SsrfBlockedError as exc:
                                return ToolResult(ok=False, error=f"Blocked redirect target: {exc}")
                            current_url = next_url
                            continue

                        body, truncated = await self._read_bounded(resp)
                        text = body.decode("utf-8", errors="replace")[:MAX_RESPONSE_CHARS]
                        return ToolResult(
                            ok=resp.status_code < 400,
                            output=text,
                            error=None if resp.status_code < 400 else f"HTTP {resp.status_code}",
                            metadata={
                                "status_code": resp.status_code,
                                "url": str(resp.url),
                                "truncated": truncated,
                            },
                        )
                return ToolResult(ok=False, error=f"Too many redirects (max {MAX_REDIRECTS})")
        except httpx.HTTPError as exc:
            return ToolResult(ok=False, error=f"HTTP request failed: {exc}")
