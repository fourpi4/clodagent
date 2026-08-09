"""
ProviderRegistry: tracks every configured LLM provider and exposes safe,
secret-free metadata about each one (used by GET /api/providers and by
ProviderRouter for fallback ordering).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from app.providers.base import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ProviderEntry:
    name: str
    provider: Optional[LLMProvider]
    configured: bool
    default_model: str
    supports_tools: bool = True
    supports_streaming: bool = True
    init_error: Optional[str] = None

    def describe(self, *, available: Optional[bool] = None) -> dict:
        return {
            "name": self.name,
            "configured": self.configured,
            "available": self.configured if available is None else available,
            "default_model": self.default_model,
            "supports_tools": self.supports_tools,
            "supports_streaming": self.supports_streaming,
        }


class ProviderRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, ProviderEntry] = {}

    def register_provider(self, entry: ProviderEntry) -> None:
        self._entries[entry.name] = entry
        if entry.configured:
            logger.info("Registered provider: %s (model=%s)", entry.name, entry.default_model)
        else:
            logger.info("Provider '%s' not configured (%s)", entry.name, entry.init_error or "no API key")

    def get_provider(self, name: str) -> Optional[LLMProvider]:
        entry = self._entries.get(name)
        return entry.provider if entry else None

    def get_entry(self, name: str) -> Optional[ProviderEntry]:
        return self._entries.get(name)

    def list_providers(self) -> list[dict]:
        return [entry.describe() for entry in self._entries.values()]

    async def health_check(self, name: str, *, timeout: float = 8.0) -> bool:
        entry = self._entries.get(name)
        if entry is None or not entry.configured or entry.provider is None:
            return False
        try:
            await asyncio.wait_for(entry.provider.list_models(), timeout=timeout)
            return True
        except Exception as exc:  # noqa: BLE001 - health check must never raise
            logger.warning("Health check failed for provider '%s': %s", name, exc)
            return False

    async def health_check_all(self, *, timeout: float = 8.0) -> dict[str, bool]:
        names = [name for name, entry in self._entries.items() if entry.configured]
        results = await asyncio.gather(*(self.health_check(n, timeout=timeout) for n in names))
        return dict(zip(names, results))

    async def list_providers_with_health(self, *, timeout: float = 8.0) -> list[dict]:
        health = await self.health_check_all(timeout=timeout)
        return [entry.describe(available=health.get(entry.name)) for entry in self._entries.values()]
