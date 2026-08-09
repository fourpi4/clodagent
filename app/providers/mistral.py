"""
Mistral provider (La Plateforme). OpenAI-compatible Chat Completions API:
https://docs.mistral.ai/api/

    Base URL: https://api.mistral.ai/v1
    Auth:     Authorization: Bearer <MISTRAL_API_KEY>
"""
from __future__ import annotations

from app.providers.openai_compatible import OpenAICompatibleProvider


class MistralProvider(OpenAICompatibleProvider):
    def __init__(self, api_key: str, default_model: str, base_url: str = "https://api.mistral.ai/v1", timeout: int = 60) -> None:
        super().__init__(
            name="mistral",
            api_key=api_key,
            default_model=default_model,
            base_url=base_url,
            timeout=timeout,
            env_hint="Set MISTRAL_API_KEY (get a key at https://console.mistral.ai/api-keys).",
        )
