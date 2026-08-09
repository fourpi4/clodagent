"""
Application entrypoint: wires providers, tools, memory, and the agent core
together, then exposes them via FastAPI (REST API + static Web UI).

Run with:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agent.core import Agent
from app.agent.memory import AgentMemory
from app.agent.profiles import ProfileStore
from app.api.routes import router
from app.config.settings import get_settings
from app.integrations.github_discovery import GitHubClient, GitHubDiscovery
from app.integrations.mcp import McpClientManager
from app.memory.long_term import SqliteLongTermMemory
from app.memory.short_term import SqliteShortTermMemory
from app.providers.bytez import BytezProvider
from app.providers.gemini import GeminiProvider
from app.providers.groq import GroqProvider
from app.providers.mistral import MistralProvider
from app.providers.openrouter import OpenRouterProvider
from app.providers.registry import ProviderEntry, ProviderRegistry
from app.providers.router import ProviderRouter
from app.tools.filesystem_tools import FileListTool, FileReadTool, FileWriteTool, WorkspaceSandbox
from app.tools.github_tools import GitHubGetRepoTool, GitHubSearchTool
from app.tools.memory_tools import ForgetMemoryTool, ListMemoryTool, RememberMemoryTool
from app.tools.registry import ToolRegistry
from app.tools.web_tools import WebFetchTool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent / "ui"

# Endpoints that never require AGENT_API_KEY, even when it's configured.
_AUTH_EXEMPT_PATHS = {"/health", "/"}


def build_provider_registry(settings) -> ProviderRegistry:
    registry = ProviderRegistry()

    def _try_register(name: str, factory, *, default_model: str, supports_tools: bool = True) -> None:
        try:
            provider = factory()
            registry.register_provider(
                ProviderEntry(name=name, provider=provider, configured=True, default_model=default_model, supports_tools=supports_tools)
            )
        except Exception as exc:  # noqa: BLE001 - a missing/invalid key must not crash startup
            registry.register_provider(
                ProviderEntry(
                    name=name, provider=None, configured=False, default_model=default_model,
                    supports_tools=supports_tools, init_error=str(exc),
                )
            )

    _try_register(
        "groq",
        lambda: GroqProvider(api_key=settings.GROQ_API_KEY, default_model=settings.GROQ_MODEL),
        default_model=settings.GROQ_MODEL,
    )
    _try_register(
        "gemini",
        lambda: GeminiProvider(api_key=settings.GEMINI_API_KEY, default_model=settings.GEMINI_MODEL),
        default_model=settings.GEMINI_MODEL,
    )
    _try_register(
        "openrouter",
        lambda: OpenRouterProvider(api_key=settings.OPENROUTER_API_KEY, default_model=settings.OPENROUTER_MODEL),
        default_model=settings.OPENROUTER_MODEL,
        supports_tools=False,  # not every free OpenRouter model supports tool calling; treated as best-effort
    )
    _try_register(
        "bytez",
        lambda: BytezProvider(
            api_key=settings.BYTEZ_API_KEY,
            default_model=settings.BYTEZ_MODEL,
            base_url=settings.BYTEZ_BASE_URL,
            timeout=settings.BYTEZ_TIMEOUT,
            fallback_models=settings.BYTEZ_MODEL_CANDIDATES or None,
        ),
        default_model=settings.BYTEZ_MODEL,
    )
    _try_register(
        "mistral",
        lambda: MistralProvider(api_key=settings.MISTRAL_API_KEY, default_model=settings.MISTRAL_MODEL),
        default_model=settings.MISTRAL_MODEL,
    )

    return registry


def build_agent() -> tuple[Agent, GitHubDiscovery, McpClientManager]:
    settings = get_settings()

    provider_registry = build_provider_registry(settings)
    provider_router = ProviderRouter(
        provider_registry, default_provider=settings.DEFAULT_PROVIDER, fallback_order=settings.PROVIDER_FALLBACK_ORDER
    )

    registry = ToolRegistry()
    sandbox = WorkspaceSandbox(settings.WORKSPACE_DIR)
    github_client = GitHubClient(token=settings.GITHUB_TOKEN, base_url=settings.GITHUB_API_URL)

    memory = AgentMemory(
        short_term=SqliteShortTermMemory(settings.SHORT_TERM_DB),
        long_term=SqliteLongTermMemory(settings.LONG_TERM_DB),
    )

    for tool in (
        WebFetchTool(
            max_response_bytes=settings.WEB_FETCH_MAX_RESPONSE_BYTES,
            allowed_methods=settings.WEB_FETCH_ALLOWED_METHODS,
        ),
        FileReadTool(sandbox),
        FileWriteTool(sandbox),
        FileListTool(sandbox),
        GitHubSearchTool(github_client),
        GitHubGetRepoTool(github_client),
        RememberMemoryTool(memory.long_term),
        ListMemoryTool(memory.long_term),
        ForgetMemoryTool(memory.long_term),
    ):
        registry.register_tool(tool)

    profile_store = ProfileStore(settings.AGENT_PROFILES_FILE)

    agent = Agent(
        provider_router,
        registry,
        memory,
        profile_store,
        max_agent_steps=settings.MAX_AGENT_STEPS,
        max_tool_calls=settings.MAX_TOOL_CALLS,
        tool_timeout=settings.TOOL_TIMEOUT,
        max_retries=settings.MAX_RETRIES,
        confirmation_ttl_seconds=settings.CONFIRMATION_TTL_SECONDS,
        provider_registry=provider_registry,
    )

    discovery = GitHubDiscovery(github_client, allow_gpl_auto=settings.ALLOW_GPL_AUTO_INTEGRATION)
    mcp_manager = McpClientManager(
        settings.MCP_SERVERS_FILE,
        allow_dynamic_stdio=settings.ALLOW_DYNAMIC_STDIO_MCP,
        allowed_commands=settings.MCP_ALLOWED_COMMANDS,
        allowed_env_vars=settings.MCP_ALLOWED_ENV_VARS,
    )

    return agent, discovery, mcp_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        agent, discovery, mcp_manager = build_agent()
    except Exception:
        logger.exception("Failed to initialize agent (check your provider API keys in .env)")
        raise
    app.state.agent = agent
    app.state.github_discovery = discovery
    app.state.mcp_manager = mcp_manager
    configured = [p["name"] for p in agent.provider_registry.list_providers() if p["configured"]]
    logger.info("Universal AI Agent started. Configured providers: %s", configured or ["<none>"])
    logger.info("Registered tools: %s", [t["name"] for t in agent.registry.list_tools()])
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Universal AI Agent",
        description="A self-hosted, provider-agnostic AI agent with multi-provider LLM support and automatic "
        "fallback, a pluggable tool/adapter system with human-in-the-loop confirmation, MCP support, and a "
        "GitHub-integration discovery module.",
        version="0.2.0",
        lifespan=lifespan,
    )

    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if settings.AGENT_API_KEY:
        expected = settings.AGENT_API_KEY

        @app.middleware("http")
        async def require_api_key(request: Request, call_next):
            path = request.url.path
            if path in _AUTH_EXEMPT_PATHS or not path.startswith("/api/"):
                return await call_next(request)
            header = request.headers.get("Authorization", "")
            if header != f"Bearer {expected}":
                # Never echo back the expected key or log the header value.
                return JSONResponse(status_code=401, content={"detail": "Missing or invalid API key"})
            return await call_next(request)

    app.include_router(router)

    if UI_DIR.exists():
        app.mount("/static", StaticFiles(directory=UI_DIR / "static"), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(UI_DIR / "index.html")

    return app


app = create_app()
