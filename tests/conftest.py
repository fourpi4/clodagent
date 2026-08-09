import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
import pytest  # noqa: E402

from app.providers.base import LLMProvider  # noqa: E402
from app.providers.registry import ProviderEntry, ProviderRegistry  # noqa: E402
from app.providers.router import ProviderRouter  # noqa: E402


def make_router(provider: LLMProvider, name: str = "test", *, default_model: str = "fake-model") -> ProviderRouter:
    """Wraps a single fake/test LLMProvider in a real ProviderRegistry+ProviderRouter,
    matching the production construction shape used by app.main.build_agent()."""
    registry = ProviderRegistry()
    registry.register_provider(
        ProviderEntry(name=name, provider=provider, configured=True, default_model=default_model)
    )
    return ProviderRouter(registry, default_provider=name, fallback_order=[name])


def make_multi_provider_router(providers: dict[str, LLMProvider], default_provider: str) -> tuple[ProviderRouter, ProviderRegistry]:
    """Wraps multiple named fake providers for testing fallback behavior."""
    registry = ProviderRegistry()
    for name, provider in providers.items():
        registry.register_provider(
            ProviderEntry(name=name, provider=provider, configured=True, default_model="fake-model")
        )
    router = ProviderRouter(registry, default_provider=default_provider, fallback_order=list(providers.keys()))
    return router, registry


@pytest.fixture
def mock_httpx(monkeypatch):
    """
    Redirects every httpx.AsyncClient(...) constructed anywhere in the app
    (providers, web_fetch, github_discovery, ...) through a per-test
    MockTransport, so tests never make real network calls. Set
    `mock_httpx["handler"] = fn` where fn(request) -> httpx.Response.
    """
    handler_holder: dict = {"handler": None}
    real_async_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        handler = handler_holder.get("handler")
        if handler is not None:
            kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return handler_holder
