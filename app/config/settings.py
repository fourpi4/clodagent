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

    # --- Bytez (primary LLM provider) -----------------------------------
    BYTEZ_API_KEY: str = os.getenv("BYTEZ_API_KEY", "")
    BYTEZ_MODEL: str = os.getenv("BYTEZ_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    BYTEZ_BASE_URL: str = os.getenv("BYTEZ_BASE_URL", "https://api.bytez.com")
    BYTEZ_TIMEOUT: int = _env_int("BYTEZ_TIMEOUT", 60)

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

    # --- Web / API -----------------------------------------------------------
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = _env_int("PORT", 8000)
    CORS_ORIGINS: List[str] = _env_list("CORS_ORIGINS", ["*"])
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # --- Feature flags -------------------------------------------------------
    ALLOW_GPL_AUTO_INTEGRATION: bool = _env_bool("ALLOW_GPL_AUTO_INTEGRATION", False)

    def ensure_dirs(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
