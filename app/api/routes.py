"""REST API routes. All shared singletons (agent, registry, discovery, mcp manager) live on `request.app.state`."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from app.agent.profiles import AgentProfile
from app.api.schemas import (
    AgentProfileSchema,
    AgentRunRequest,
    ChatRequest,
    GithubAnalyzeRequest,
    McpServerCreateRequest,
    McpToggleRequest,
    RememberRequest,
)
from app.integrations.github_discovery import DEFAULT_QUERIES
from app.integrations.mcp import McpServerConfig
from app.integrations.mcp_adapter import register_mcp_tools

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# --- Chat / Agent execution -------------------------------------------------

@router.post("/api/chat")
async def chat(req: ChatRequest, request: Request) -> dict:
    agent = request.app.state.agent
    session_id = req.session_id or agent.memory.start_session(req.profile or "default")
    result = await agent.run_turn(session_id, req.message, profile_name=req.profile)
    payload = result.to_dict()
    payload["session_id"] = session_id
    return payload


@router.post("/api/agent/run")
async def agent_run(req: AgentRunRequest, request: Request) -> dict:
    agent = request.app.state.agent
    result = await agent.run_task(req.task, profile_name=req.profile)
    return result.to_dict()


# --- Sessions / short-term memory -------------------------------------------

@router.get("/api/sessions")
async def list_sessions(request: Request) -> list[dict]:
    return request.app.state.agent.memory.short_term.list_sessions()


@router.get("/api/sessions/{session_id}/history")
async def session_history(session_id: str, request: Request) -> list[dict]:
    return request.app.state.agent.memory.short_term.get_history(session_id, limit=200)


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict:
    request.app.state.agent.memory.short_term.clear_session(session_id)
    return {"deleted": session_id}


# --- Long-term memory --------------------------------------------------------

@router.get("/api/memory")
async def list_memory(request: Request, agent_profile: str | None = None) -> list[dict]:
    return request.app.state.agent.memory.long_term.list_all(agent_profile)


@router.post("/api/memory")
async def add_memory(req: RememberRequest, request: Request) -> dict:
    memory_id = request.app.state.agent.memory.remember_fact(req.agent_profile, req.text, req.tags)
    return {"id": memory_id}


@router.delete("/api/memory/{memory_id}")
async def delete_memory(memory_id: str, request: Request) -> dict:
    request.app.state.agent.memory.long_term.forget(memory_id)
    return {"deleted": memory_id}


# --- Agent profiles ------------------------------------------------------------

@router.get("/api/agents")
async def list_agents(request: Request) -> list[dict]:
    return [p.to_dict() for p in request.app.state.agent.profiles.list_profiles()]


@router.post("/api/agents")
async def upsert_agent(req: AgentProfileSchema, request: Request) -> dict:
    profile = AgentProfile(**req.model_dump())
    request.app.state.agent.profiles.upsert_profile(profile)
    return profile.to_dict()


@router.delete("/api/agents/{name}")
async def delete_agent(name: str, request: Request) -> dict:
    request.app.state.agent.profiles.delete_profile(name)
    return {"deleted": name}


# --- Tools ------------------------------------------------------------------

@router.get("/api/tools")
async def list_tools(request: Request) -> list[dict]:
    return request.app.state.agent.registry.list_tools()


# --- Models -------------------------------------------------------------------

@router.get("/api/models")
async def list_models(request: Request) -> list[dict]:
    models = await request.app.state.agent.provider.list_models()
    return [m.__dict__ for m in models]


# --- GitHub Discovery ------------------------------------------------------------

@router.get("/api/github/search")
async def github_search(request: Request, q: str | None = None, limit: int = 25) -> list[dict]:
    discovery = request.app.state.github_discovery
    queries = [q] if q else DEFAULT_QUERIES
    candidates = await discovery.discover(queries, limit=limit)
    return [c.to_dict() for c in candidates]


@router.post("/api/github/analyze")
async def github_analyze(req: GithubAnalyzeRequest, request: Request) -> dict:
    discovery = request.app.state.github_discovery
    try:
        return await discovery.analyze(req.full_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# --- MCP servers ----------------------------------------------------------------

@router.get("/api/mcp")
async def list_mcp_servers(request: Request) -> list[dict]:
    return request.app.state.mcp_manager.list_servers()


@router.post("/api/mcp/servers")
async def create_mcp_server(req: McpServerCreateRequest, request: Request) -> dict:
    manager = request.app.state.mcp_manager
    manager.add_server(McpServerConfig(**req.model_dump()))
    return {"created": req.name}


@router.post("/api/mcp/connect")
async def connect_mcp(request: Request, name: str) -> dict:
    manager = request.app.state.mcp_manager
    try:
        tools = await manager.connect_and_list_tools(name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    registered = await register_mcp_tools(manager, request.app.state.agent.registry)
    return {"server": name, "tools_discovered": [t.name for t in tools], "tools_registered": registered}


@router.post("/api/mcp/{name}/toggle")
async def toggle_mcp(name: str, req: McpToggleRequest, request: Request) -> dict:
    try:
        request.app.state.mcp_manager.set_enabled(name, req.enabled)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"name": name, "enabled": req.enabled}
