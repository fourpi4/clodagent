"""Pydantic request/response schemas for the REST API."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    profile: Optional[str] = Field(default=None, description="Agent profile name, defaults to 'default'")


class AgentRunRequest(BaseModel):
    task: str = Field(..., min_length=1)
    profile: Optional[str] = None


class AgentRunResponse(BaseModel):
    run_id: str
    session_id: Optional[str] = None
    task: str
    plan: list[str]
    tool_calls: list[dict[str, Any]]
    status: str
    final_answer: str
    steps_used: int
    confirmation: Optional[dict[str, Any]] = None


class AgentProfileSchema(BaseModel):
    name: str
    system_prompt: str = "You are a helpful, precise AI agent. Use tools when they help answer accurately."
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.2
    tools: list[str] = []
    mcp_servers: list[str] = []
    memory_enabled: bool = True
    max_steps: Optional[int] = None


class GithubAnalyzeRequest(BaseModel):
    full_name: str = Field(..., description="'owner/repo'")


class GithubGenerateAdapterRequest(BaseModel):
    full_name: str = Field(..., description="'owner/repo'")


class McpServerCreateRequest(BaseModel):
    name: str
    transport: str = "stdio"
    command: Optional[str] = None
    args: list[str] = []
    env: dict[str, str] = {}
    url: Optional[str] = None
    timeout: float = 30.0
    enabled: bool = True


class McpToggleRequest(BaseModel):
    enabled: bool


class RememberRequest(BaseModel):
    agent_profile: str
    text: str
    tags: list[str] = []
