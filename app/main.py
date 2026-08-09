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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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
from app.tools.filesystem_tools import FileListTool, FileReadTool, FileWriteTool, WorkspaceSandbox
from app.tools.github_tools import GitHubGetRepoTool, GitHubSearchTool
from app.tools.registry import ToolRegistry
from app.tools.web_tools import WebFetchTool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent / "ui"


def build_agent() -> tuple[Agent, GitHubDiscovery, McpClientManager]:
    settings = get_settings()

    provider = BytezProvider(
        api_key=settings.BYTEZ_API_KEY,
        default_model=settings.BYTEZ_MODEL,
        base_url=settings.BYTEZ_BASE_URL,
        timeout=settings.BYTEZ_TIMEOUT,
        fallback_models=settings.BYTEZ_MODEL_CANDIDATES or None,
    )

    registry = ToolRegistry()
    sandbox = WorkspaceSandbox(settings.WORKSPACE_DIR)
    github_client = GitHubClient(token=settings.GITHUB_TOKEN, base_url=settings.GITHUB_API_URL)

    for tool in (
        WebFetchTool(),
        FileReadTool(sandbox),
        FileWriteTool(sandbox),
        FileListTool(sandbox),
        GitHubSearchTool(github_client),
        GitHubGetRepoTool(github_client),
    ):
        registry.register_tool(tool)

    memory = AgentMemory(
        short_term=SqliteShortTermMemory(settings.SHORT_TERM_DB),
        long_term=SqliteLongTermMemory(settings.LONG_TERM_DB),
    )
    profile_store = ProfileStore(settings.AGENT_PROFILES_FILE)

    agent = Agent(
        provider,
        registry,
        memory,
        profile_store,
        max_agent_steps=settings.MAX_AGENT_STEPS,
        max_tool_calls=settings.MAX_TOOL_CALLS,
        tool_timeout=settings.TOOL_TIMEOUT,
        max_retries=settings.MAX_RETRIES,
    )

    discovery = GitHubDiscovery(github_client, allow_gpl_auto=settings.ALLOW_GPL_AUTO_INTEGRATION)
    mcp_manager = McpClientManager(settings.MCP_SERVERS_FILE)

    return agent, discovery, mcp_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        agent, discovery, mcp_manager = build_agent()
    except Exception:
        logger.exception("Failed to initialize agent (check BYTEZ_API_KEY in your .env)")
        raise
    app.state.agent = agent
    app.state.github_discovery = discovery
    app.state.mcp_manager = mcp_manager
    logger.info("Universal AI Agent started. Registered tools: %s", [t["name"] for t in agent.registry.list_tools()])
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Universal AI Agent",
        description="A self-hosted, provider-agnostic AI agent with Bytez as its primary LLM backend, "
        "a pluggable tool/adapter system, MCP support, and a GitHub-integration discovery module.",
        version="0.1.0",
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

    app.include_router(router)

    if UI_DIR.exists():
        app.mount("/static", StaticFiles(directory=UI_DIR / "static"), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(UI_DIR / "index.html")

    return app


app = create_app()
