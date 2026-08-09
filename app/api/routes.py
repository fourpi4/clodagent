"""REST API routes. All shared singletons (agent, registry, discovery, mcp manager) live on `request.app.state`."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.agent.confirmation import ConfirmationError, ConfirmationExpiredError, ConfirmationNotFoundError
from app.agent.profiles import AgentProfile
from app.api.schemas import (
    AgentProfileSchema,
    AgentRunRequest,
    ChatRequest,
    GithubAnalyzeRequest,
    GithubGenerateAdapterRequest,
    McpServerCreateRequest,
    McpToggleRequest,
    RememberRequest,
)
from app.config.settings import get_settings
from app.integrations.adapter_generator import generate_adapter_source
from app.integrations.github_discovery import DEFAULT_QUERIES
from app.integrations.mcp import McpConnectionError, McpPermissionError, McpServerConfig
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


@router.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    """
    SSE stream of run lifecycle events: planning, plan_ready, provider_selected,
    provider_fallback, tool_requested, confirmation_required, tool_started,
    tool_finished, generating, token, done, error. No hidden chain-of-thought
    is ever emitted — only these structured, user-safe events.

    Note: `token` events chunk the already-completed final answer for a
    typing-effect UI, rather than raw incremental provider tokens — true
    token-level provider streaming isn't wired into the tool-calling loop
    yet (see README: EXPERIMENTAL).
    """
    agent = request.app.state.agent
    session_id = req.session_id or agent.memory.start_session(req.profile or "default")

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()

        async def on_status(event: dict[str, Any]) -> None:
            await queue.put(event)

        async def runner() -> None:
            try:
                result = await agent.run_turn(session_id, req.message, profile_name=req.profile, on_status=on_status)
                if result.final_answer:
                    for word in result.final_answer.split(" "):
                        await queue.put({"type": "token", "text": word + " "})
                payload = result.to_dict()
                payload["session_id"] = session_id
                await queue.put({"type": "done", "result": payload})
            except Exception as exc:  # noqa: BLE001 - must always terminate the stream cleanly
                logger.exception("Streamed run failed")
                await queue.put({"type": "error", "error": str(exc)})
            finally:
                await queue.put(None)

        task = asyncio.create_task(runner())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            await task

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/api/agent/run")
async def agent_run(req: AgentRunRequest, request: Request) -> dict:
    agent = request.app.state.agent
    result = await agent.run_task(req.task, profile_name=req.profile)
    return result.to_dict()


# --- Confirmations (human-in-the-loop) ---------------------------------------

@router.get("/api/confirmations")
async def list_confirmations(request: Request, run_id: Optional[str] = None) -> list[dict]:
    agent = request.app.state.agent
    return [c.to_dict() for c in agent.confirmations.list_pending(run_id)]


def _confirmation_error_status(exc: ConfirmationError) -> int:
    if isinstance(exc, ConfirmationNotFoundError):
        return 404
    if isinstance(exc, ConfirmationExpiredError):
        return 410
    return 409


@router.post("/api/confirmations/{confirmation_id}/approve")
async def approve_confirmation(confirmation_id: str, request: Request) -> dict:
    agent = request.app.state.agent
    try:
        result = await agent.approve_confirmation(confirmation_id)
    except ConfirmationError as exc:
        raise HTTPException(status_code=_confirmation_error_status(exc), detail=str(exc)) from exc
    return result.to_dict()


@router.post("/api/confirmations/{confirmation_id}/deny")
async def deny_confirmation(confirmation_id: str, request: Request) -> dict:
    agent = request.app.state.agent
    try:
        result = await agent.deny_confirmation(confirmation_id)
    except ConfirmationError as exc:
        raise HTTPException(status_code=_confirmation_error_status(exc), detail=str(exc)) from exc
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


# --- Providers ----------------------------------------------------------------

@router.get("/api/providers")
async def list_providers(request: Request, check: bool = True) -> list[dict]:
    """Never returns API keys — only name/configured/available/default_model/capabilities."""
    provider_registry = request.app.state.agent.provider_registry
    if provider_registry is None:
        return []
    if check:
        return await provider_registry.list_providers_with_health()
    return provider_registry.list_providers()


# --- Models (aggregated across all configured providers) -----------------------

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
    """'Generate Integration Plan' — read-only analysis, never installs or executes anything."""
    discovery = request.app.state.github_discovery
    try:
        return await discovery.analyze(req.full_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/api/github/generate-adapter")
async def github_generate_adapter(req: GithubGenerateAdapterRequest, request: Request) -> dict:
    """
    'Generate Adapter' — writes a reviewable adapter STUB to
    workspace/generated_adapters/. Never clones, installs, imports, or
    executes anything; activation is always a manual, human step (see
    app/integrations/adapter_generator.py).
    """
    discovery = request.app.state.github_discovery
    settings = get_settings()
    try:
        plan = await discovery.analyze(req.full_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return generate_adapter_source(plan["repository"], settings.GENERATED_ADAPTERS_DIR)


# --- MCP servers ----------------------------------------------------------------

@router.get("/api/mcp")
async def list_mcp_servers(request: Request) -> list[dict]:
    return request.app.state.mcp_manager.list_servers()


@router.post("/api/mcp/servers")
async def create_mcp_server(req: McpServerCreateRequest, request: Request) -> dict:
    manager = request.app.state.mcp_manager
    try:
        config = manager.add_server(McpServerConfig(**req.model_dump()))
    except McpPermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except McpConnectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "created": req.name,
        "confirmed": config.confirmed,
        "dropped_env_vars": config.dropped_env_vars,
        "note": None if config.confirmed else f"Call POST /api/mcp/{req.name}/confirm before first use.",
    }


@router.post("/api/mcp/{name}/confirm")
async def confirm_mcp_server(name: str, request: Request) -> dict:
    manager = request.app.state.mcp_manager
    try:
        manager.confirm_server(name)
    except McpConnectionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"name": name, "confirmed": True}


@router.post("/api/mcp/connect")
async def connect_mcp(request: Request, name: str) -> dict:
    manager = request.app.state.mcp_manager
    try:
        tools = await manager.connect_and_list_tools(name)
    except McpPermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except McpConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    registered = await register_mcp_tools(manager, request.app.state.agent.registry, only_servers=[name])
    return {"server": name, "tools_discovered": [t.name for t in tools], "tools_registered": registered}


@router.post("/api/mcp/{name}/toggle")
async def toggle_mcp(name: str, req: McpToggleRequest, request: Request) -> dict:
    try:
        request.app.state.mcp_manager.set_enabled(name, req.enabled)
    except McpConnectionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"name": name, "enabled": req.enabled}
