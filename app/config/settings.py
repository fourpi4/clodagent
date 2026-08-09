"""
Central application configuration.

Every secret and every tunable limit is read from environment variables
(optionally loaded from a local `.env` file via python-dotenv). Nothing
sensitive is ever hard-coded here.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load .env once, before any Settings are constructed. This is a no-op if
# the file does not exist (e.g. in production containers using real env vars).
load_dotenv(override=False)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_list(name: str, default: List[str]) -> List[str]:
    val = os.getenv(name)
    if not val:
        return default
    return [item.strip() for item in val.split(",") if item.strip()]


class Settings:
    """Runtime configuration, read once and cached via get_settings()."""

    # --- Providers: multi-provider AI with fallback ---------------------------
    DEFAULT_PROVIDER: str = os.getenv("DEFAULT_PROVIDER", "bytez")
    PROVIDER_FALLBACK_ORDER: List[str] = _env_list("PROVIDER_FALLBACK_ORDER", ["bytez"])

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

    # --- Bytez (kept as one of the supported providers) -----------------------
    BYTEZ_API_KEY: str = os.getenv("BYTEZ_API_KEY", "")
    BYTEZ_MODEL: str = os.getenv("BYTEZ_MODEL", "openai-community/gpt2")
    BYTEZ_BASE_URL: str = os.getenv("BYTEZ_BASE_URL", "https://api.bytez.com")
    BYTEZ_TIMEOUT: int = _env_int("BYTEZ_TIMEOUT", 60)
    # Comma-separated fallback model IDs, tried in order if BYTEZ_MODEL is
    # unavailable on the account's catalog. Empty = use the built-in defaults
    # in app/providers/bytez.py (DEFAULT_FALLBACK_MODELS).
    BYTEZ_MODEL_CANDIDATES: List[str] = _env_list("BYTEZ_MODEL_CANDIDATES", [])

    # --- GitHub (discovery module) ---------------------------------------
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    GITHUB_API_URL: str = os.getenv("GITHUB_API_URL", "https://api.github.com")

    # --- Agent execution limits (guard against infinite loops) -----------
    MAX_AGENT_STEPS: int = _env_int("MAX_AGENT_STEPS", 8)
    MAX_TOOL_CALLS: int = _env_int("MAX_TOOL_CALLS", 12)
    TOOL_TIMEOUT: int = _env_int("TOOL_TIMEOUT", 30)
    MAX_RETRIES: int = _env_int("MAX_RETRIES", 2)

    # --- Storage -----------------------------------------------------------
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
    SHORT_TERM_DB: Path = DATA_DIR / "short_term.sqlite3"
    LONG_TERM_DB: Path = DATA_DIR / "long_term.sqlite3"
    AGENT_PROFILES_FILE: Path = DATA_DIR / "agent_profiles.json"
    MCP_SERVERS_FILE: Path = Path(os.getenv("MCP_SERVERS_FILE", str(PROJECT_ROOT / "mcp_servers.json")))

    # --- Filesystem tool sandbox -------------------------------------------
    WORKSPACE_DIR: Path = Path(os.getenv("WORKSPACE_DIR", str(PROJECT_ROOT / "workspace")))
    GENERATED_ADAPTERS_DIR: Path = WORKSPACE_DIR / "generated_adapters"

    # --- Web / API -----------------------------------------------------------
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = _env_int("PORT", 8000)
    # SECURITY: default is intentionally restricted to localhost, not "*".
    # Set CORS_ORIGINS explicitly (comma-separated) to widen this.
    CORS_ORIGINS: List[str] = _env_list(
        "CORS_ORIGINS", ["http://localhost:8000", "http://127.0.0.1:8000"]
    )
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # --- API authentication (MVP bearer token) --------------------------------
    # If set, every /api/* endpoint requires `Authorization: Bearer <key>`.
    # GET /health and GET / (the UI shell) are always exempt.
    AGENT_API_KEY: str = os.getenv("AGENT_API_KEY", "")

    # --- Feature flags -------------------------------------------------------
    ALLOW_GPL_AUTO_INTEGRATION: bool = _env_bool("ALLOW_GPL_AUTO_INTEGRATION", False)

    # --- MCP security ----------------------------------------------------------
    # By default the REST API cannot register a new stdio MCP server with an
    # arbitrary command — stdio servers must come from the trusted
    # mcp_servers.json config file, or from MCP_ALLOWED_COMMANDS.
    ALLOW_DYNAMIC_STDIO_MCP: bool = _env_bool("ALLOW_DYNAMIC_STDIO_MCP", False)
    MCP_ALLOWED_COMMANDS: List[str] = _env_list("MCP_ALLOWED_COMMANDS", [])
    # Environment variable *names* (not values) that may be forwarded to a
    # dynamically-registered stdio MCP server. Empty = none forwarded.
    MCP_ALLOWED_ENV_VARS: List[str] = _env_list("MCP_ALLOWED_ENV_VARS", [])

    # --- Confirmation flow (human-in-the-loop) ---------------------------------
    CONFIRMATION_TTL_SECONDS: int = _env_int("CONFIRMATION_TTL_SECONDS", 600)

    # --- SSRF protection for web_fetch -----------------------------------------
    WEB_FETCH_MAX_RESPONSE_BYTES: int = _env_int("WEB_FETCH_MAX_RESPONSE_BYTES", 2_000_000)
    WEB_FETCH_ALLOWED_METHODS: List[str] = _env_list("WEB_FETCH_ALLOWED_METHODS", ["GET", "HEAD"])

    def ensure_dirs(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        self.GENERATED_ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
