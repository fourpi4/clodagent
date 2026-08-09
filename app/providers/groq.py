"""
Groq provider — recommended default for development (fast inference, generous
free tier). Fully OpenAI-compatible: https://console.groq.com/docs/openai

    Base URL: https://api.groq.com/openai/v1
    Auth:     Authorization: Bearer <GROQ_API_KEY>
"""
from __future__ import annotations

from app.providers.openai_compatible import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    def __init__(self, api_key: str, default_model: str, base_url: str = "https://api.groq.com/openai/v1", timeout: int = 60) -> None:
        super().__init__(
            name="groq",
            api_key=api_key,
            default_model=default_model,
            base_url=base_url,
            timeout=timeout,
            env_hint="Set GROQ_API_KEY (get a free key at https://console.groq.com/keys).",
        )
