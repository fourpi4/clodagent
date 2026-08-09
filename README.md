# Universal AI Agent

A self-hosted, provider-agnostic AI agent that belongs to **you** — not to
any single AI vendor or agent framework.

- **Multi-provider LLM routing** with automatic fallback: Groq, Gemini,
  OpenRouter, Bytez, Mistral. Agent Core depends on the `LLMProvider`
  abstraction, never on one specific vendor.
- A **plugin/adapter system** lets you plug in capabilities from other
  open-source AI-agent projects (LangGraph, CrewAI, MCP servers, GitHub
  projects) without touching the core.
- **Human-in-the-loop confirmation**: tools marked risky pause the run and
  wait for an explicit approve/deny before ever executing.
- A **GitHub Discovery** module finds, ranks, and license-gates candidate
  projects to integrate — and only ever generates reviewable adapter
  *stubs*, never auto-installs or auto-executes anything.
- **MCP** (Model Context Protocol) servers plug in as first-class tools,
  gated by a command allowlist and a manual confirmation step for stdio.
- Ships with a **REST API** (+ SSE streaming), **Swagger/OpenAPI docs**, and
  a minimal **Web UI**.

No third-party code is ever executed automatically. Integrations use
official REST APIs, SDKs, or MCP — never blind code copying, never `eval()`.

> ⚠️ **SECURITY WARNING**: this is a development-oriented reference
> implementation. Do **not** expose a running instance directly to the
> public internet without putting it behind your own reverse proxy, TLS,
> and a real authentication layer beyond the MVP bearer-token check
> described below. See [Security notes](#security-notes).

---

## Status: IMPLEMENTED vs EXPERIMENTAL vs PLANNED

Nothing below is claimed as done unless it actually runs and is covered by
a test. Read this section before relying on a feature.

### IMPLEMENTED

- Multi-provider `LLMProvider` abstraction + `ProviderRegistry` + `ProviderRouter`
  with automatic fallback (timeout/429/5xx/model-unavailable), and a hard
  stop (no fallback) on auth errors or malformed requests.
- Providers: **Groq**, **Gemini**, **OpenRouter**, **Bytez**, **Mistral** —
  full `chat`/`stream`/`list_models`, tool-calling response mapping, all
  covered by mocked-HTTP tests (no real API keys needed to run the suite).
- Agent Core loop: context → planner → tool selection → executor →
  observation → reason/replan → final result, bounded by
  `MAX_AGENT_STEPS`/`MAX_TOOL_CALLS`/`TOOL_TIMEOUT`/`MAX_RETRIES`.
- The plan is actually injected into the conversation the model sees (not
  just returned in the API response), with per-step completed/failed
  tracking and a one-shot structured replan when a step fails.
- Tool-call budget correctness: if MAX_TOOL_CALLS runs out mid-batch, every
  requested tool_call still gets a structured response — none are left
  dangling (which would break the conversation for OpenAI-style APIs).
- **Human-in-the-loop confirmation**: a tool with `requires_confirmation`
  pauses the run (`status: "confirmation_required"`), the client approves
  or denies by id only (arguments can never be substituted), and the run
  resumes exactly where it left off. Pending confirmations expire after
  `CONFIRMATION_TTL_SECONDS`.
- Tool safety metadata (`risk_level`, `side_effects`, `retry_safe`,
  `requires_confirmation`) — side-effecting tools are never blindly retried.
- Unknown/hallucinated tool names never crash a run — the model gets a
  clear `"Unknown tool"` error and a chance to self-correct.
- MCP client on the current official SDK (`mcp` 2.x): `stdio` and
  `streamable_http` transports, legacy `sse` kept for backward
  compatibility. stdio commands are gated by `MCP_ALLOWED_COMMANDS` /
  `ALLOW_DYNAMIC_STDIO_MCP`, env vars by `MCP_ALLOWED_ENV_VARS`, and a
  dynamically-added stdio server always needs an explicit confirm before
  its first connection.
- `AgentProfile.mcp_servers` actually restricts which MCP-backed tools a
  profile can see — the safe default (empty list) is *no* MCP tools at all.
- `AgentProfile.provider` / `.model` let a profile pin a specific provider
  and model; if unset, `ProviderRouter` resolves it.
- SSRF-hardened `web_fetch`: blocks loopback/private/link-local/reserved/
  multicast addresses (covers cloud metadata endpoints), rejects non-http(s)
  schemes, resolves DNS up front, re-validates every redirect hop, GET/HEAD
  only by default, and reads responses as a bounded stream
  (`WEB_FETCH_MAX_RESPONSE_BYTES`) instead of buffering unbounded bodies.
- SQLite **FTS5** full-text search for long-term memory (falls back to a
  token-overlap ranking if the platform's SQLite lacks FTS5), plus a
  `memory_remember`/`memory_list`/`memory_forget` toolset with a hard
  policy block on text that looks like a credential/secret.
- GitHub Discovery: two-phase pipeline (cheap search → deep-analyze only
  the top-N candidates), bounded concurrency, a small TTL cache, rate-limit
  awareness, confidence-scored capability signals (`{available, confidence,
  evidence}` — not bare booleans), `archived`/`fork`/`latest_release`
  fields, and license classification split into
  `permissive` / `weak_copyleft` (e.g. MPL) / `copyleft` / `unknown`.
- Safe adapter generation: "Generate Adapter" only writes a reviewable stub
  file to `workspace/generated_adapters/` — it never clones, installs, or
  imports/executes anything; wiring it in is always a manual step.
- MVP bearer-token API auth (`AGENT_API_KEY`) on every `/api/*` route
  except `GET /health` and `GET /`; restrictive `CORS_ORIGINS` default
  (localhost only, not `*`).
- SSE streaming (`POST /api/chat/stream`) with the full lifecycle event
  vocabulary: `planning`, `plan_ready`, `provider_selected`,
  `provider_fallback`, `tool_requested`, `confirmation_required`,
  `tool_started`, `tool_finished`, `generating`, `token`, `done`, `error`.
  No hidden chain-of-thought is ever emitted.
- Web UI: Chat (streaming, confirmation approve/deny cards), Agents,
  Tools, GitHub Explorer, MCP Servers, Memory, **Providers**, Settings
  (local-only API key storage in `localStorage`, never in source).

### EXPERIMENTAL

- **`token` SSE events are not raw incremental provider output.** Tool
  orchestration uses non-streaming `chat()` calls (so tool-call parsing
  stays robust); once a run finishes, its final answer is chunked
  word-by-word into synthetic `token` events for a typing-effect UI. True
  token-level provider streaming through the tool-calling loop is planned
  but not wired in yet.
- **SSRF protection has a residual DNS-rebinding window.** We resolve and
  validate the hostname before connecting, but don't yet pin the actual
  TCP connection to the validated IP — a sufficiently active DNS-rebinding
  attacker could in theory swap the address between our check and the
  request. Direct IP-literal SSRF and simple internal-hostname SSRF are
  fully blocked; full rebinding-proof pinning is a Phase 2 item.
- `langgraph_adapter.py` / `crewai_adapter.py` are thin, tested-at-the-
  interface-level wrappers around an *already-constructed* LangGraph graph
  or CrewAI `Crew` object — they don't manage installing or configuring
  those frameworks themselves.
- Confirmation store and resumable run state are **in-memory, single
  process**. A server restart loses any pending confirmation/paused run.
  Fine for local development; not suitable for multi-instance deployment
  as-is (see Phase 2).

### PLANNED (not implemented)

- Vector-based long-term memory (pgvector/Chroma/Qdrant) — current memory
  is FTS5/keyword-based by design for this phase.
- Multi-agent orchestration (supervisor/worker graphs, agent-to-agent handoff).
- Sandboxed code execution (isolated container/gVisor) — there is currently
  no "run arbitrary code" tool at all, by design.
- Persistent (cross-restart) confirmation/run-state storage.
- Full DNS-rebinding-proof SSRF protection (connection pinning).
- True token-level streaming through the tool-calling loop.
- Per-user/per-tenant auth beyond the single shared `AGENT_API_KEY`.

---

## Architecture

```
USER REQUEST
    │
    ▼
CONTEXT        (agent/context.py)   — system prompt + relevant memory + history
    │
    ▼
PLANNER        (agent/planner.py)   — short structured plan (2-5 steps), can replan
    │
    ▼
PROVIDER SELECTION (providers/router.py) — profile.provider, else DEFAULT_PROVIDER
    │                                      + PROVIDER_FALLBACK_ORDER, with fallback
    ▼
TOOL SELECTION (agent/tool_router.py) — LLM picks tools via function calling
    │
    ▼
CONFIRMATION GATE (agent/confirmation.py) — pauses on risky tools until approved
    │
    ▼
EXECUTOR       (agent/executor.py)  — runs the plan→tool→observe→replan loop
    │
    ▼
OBSERVATION → REASON / REPLAN  (bounded by MAX_AGENT_STEPS / MAX_TOOL_CALLS)
    │
    ▼
FINAL RESULT
```

Only structured data is retained at every stage — task, plan, tool calls,
tool results, status, final answer. The model's raw hidden reasoning is
never stored or shown to the user.

```
app/
  main.py                 FastAPI app factory + provider/tool/agent wiring + auth middleware
  agent/
    core.py                Agent — the single entrypoint (run_turn, run_task, approve/deny)
    planner.py               structured plan generation + replan
    executor.py                plan→tool→observe→replan loop, confirmation gate, budget handling
    confirmation.py              ConfirmationStore (pending/approve/deny, TTL, secret redaction)
    run_state.py                  resumable run state for paused (confirmation_required) runs
    memory.py                       short+long term memory facade
    context.py                        builds the message list sent to the LLM
    tool_router.py                      routes LLM tool_calls to the ToolRegistry
    profiles.py                           Agent Profiles (CRUD, JSON-backed; provider/model/mcp_servers)
  providers/
    base.py                LLMProvider interface (chat / stream / list_models) + ProviderError
    openai_compatible.py     shared OpenAI-Chat-Completions-shaped implementation
    groq.py / openrouter.py / mistral.py   thin OpenAICompatibleProvider subclasses
    bytez.py                  Bytez provider (model-fallback resilience, own auth convention)
    gemini.py                   Gemini provider (translates to/from Gemini's function-calling format)
    registry.py                   ProviderRegistry (configured/available/capabilities, no secrets)
    router.py                       ProviderRouter (fallback, provider_selected/provider_fallback events)
  tools/
    base.py                Tool interface + ToolResult + risk metadata
    registry.py               ToolRegistry (register/remove/list/get/execute; unknown-tool-safe)
    web_tools.py                 SSRF-hardened WebFetchTool
    github_tools.py                GitHubSearchTool, GitHubGetRepoTool
    filesystem_tools.py              sandboxed file_read/file_write/file_list
    memory_tools.py                    memory_remember/memory_list/memory_forget
  integrations/
    github_discovery.py     finds/ranks/license-gates GitHub AI-agent projects (2-phase, bounded)
    adapter_generator.py      writes reviewable adapter stubs, never activates them
    mcp.py                      MCP client manager (stdio / streamable_http / legacy sse)
    mcp_adapter.py                 registers MCP tools into the ToolRegistry, server-filterable
    langgraph_adapter.py             wraps a compiled LangGraph graph as a Tool
    crewai_adapter.py                  wraps a CrewAI Crew as a Tool
    custom_api_adapter.py                wraps any documented REST endpoint as a Tool
  memory/
    short_term.py            SQLite-backed conversation memory
    long_term.py                SQLite FTS5 (or token-ranking fallback) durable facts
  api/
    routes.py                REST endpoints incl. confirmations, providers, SSE stream
    schemas.py                 Pydantic request/response models
  ui/
    index.html, static/      Web UI (Chat w/ streaming+confirmations, Agents, Tools,
                              GitHub Explorer, MCP Servers, Memory, Providers, Settings)
  config/
    settings.py              all configuration, read from environment
tests/                      pytest suite, all offline (mocked HTTP + a real local-only
                             MCP server for the Streamable HTTP test — no external network)
.env.example
requirements.txt
Dockerfile / docker-compose.yml
mcp_servers.json
```

---

## Installation

Requires **Python 3.12+**.

```bash
git clone <this-repo-url> universal-ai-agent
cd universal-ai-agent
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Free Provider Setup

You need **at least one** provider configured. Groq is the fastest way to
get a working agent for local development.

| Provider | Get a key | Free tier | Notes |
|---|---|---|---|
| **Groq** | https://console.groq.com/keys | Yes, generous | Fastest inference; recommended default |
| **Gemini** | https://aistudio.google.com/apikey | Yes | Google's official REST API, no SDK dependency |
| **OpenRouter** | https://openrouter.ai/keys | Yes, rotating `:free` models | Not every free model supports tool calling — router falls back automatically |
| **Bytez** | https://bytez.com/api/key | Yes | Huge catalog (220k+ models); built-in model-fallback resilience |
| **Mistral** | https://console.mistral.ai/api-keys | Limited free tier | OpenAI-compatible |

Put whichever key(s) you have into `.env`:

```bash
DEFAULT_PROVIDER=groq
PROVIDER_FALLBACK_ORDER=groq,gemini,openrouter,bytez
GROQ_API_KEY=your-key-here
GROQ_MODEL=llama-3.3-70b-versatile
```

### How automatic fallback works

1. An `AgentProfile` can pin an exact `provider` (and `model`) — if set,
   that's the only one ever tried for that profile.
2. Otherwise `ProviderRouter` tries `DEFAULT_PROVIDER` first, then walks
   `PROVIDER_FALLBACK_ORDER`.
3. Fallback to the next provider is triggered by: timeout, HTTP 429
   (rate limit — the `Retry-After` header is respected, capped at 5s, before
   moving on), temporary 5xx, or "model unavailable" (404).
4. Fallback is **never** triggered by an invalid API key (401/403) or a
   malformed request/tool schema (400/422) — those abort immediately so a
   misconfiguration is visible instead of silently masked.
5. Every switch is reported via a `provider_fallback` event
   (`{"from": "groq", "to": "gemini", "reason": "rate_limit"}`) — visible on
   the SSE stream and in server logs, with no secrets ever logged.
6. `GET /api/providers` shows, per provider: `configured`, `available`
   (a live cheap health check), `default_model`, `supports_tools`,
   `supports_streaming` — API keys are never included in the response.

## Running

```bash
uvicorn app.main:app --reload
```

- Web UI: http://localhost:8000/
- Swagger / OpenAPI docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Docker

```bash
docker compose up --build
```

Reads `.env` via `env_file`, persists `./data` (SQLite DBs + agent profiles) and
`./workspace` (the sandbox filesystem tools operate in, including
`generated_adapters/`) as volumes.

---

## Creating an Agent Profile

Profiles are stored in `data/agent_profiles.json` and managed via API or the
**Agents** tab in the Web UI.

```bash
curl -X POST http://localhost:8000/api/agents \
  -H "Content-Type: application/json" \
  -d '{
        "name": "Research Agent",
        "system_prompt": "You are a research agent. Cite sources by URL.",
        "provider": "groq",
        "model": null,
        "temperature": 0.3,
        "tools": ["web_fetch", "github_search"],
        "mcp_servers": [],
        "memory_enabled": true,
        "max_steps": 6
      }'
```

- `tools: []` (empty) means "all registered tools are allowed" — restrict
  it to limit what a given profile can do.
- `mcp_servers: []` (empty, the safe default) means "no MCP tools at all"
  for this profile — list the MCP server names you want it to see.
- `provider`/`model` unset means "use `ProviderRouter`'s default + fallback
  order"; set them to pin a profile to a specific provider (e.g. a
  "Coding Agent" pinned to `gemini`).

## Creating a Tool

Implement the `Tool` interface (`app/tools/base.py`), set its risk
metadata honestly, and register it in `app/main.py::build_agent()`:

```python
from app.tools.base import Tool, ToolResult

class WordCountTool(Tool):
    name = "word_count"
    description = "Counts words in a string"
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    risk_level = "read"        # read | network | write | execute | destructive
    side_effects = False
    retry_safe = True
    requires_confirmation = False

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(ok=True, output=len(arguments["text"].split()))
```

```python
# app/main.py, inside build_agent()
registry.register_tool(WordCountTool())
```

No core code changes needed — the LLM automatically sees the new tool's
`name`/`description`/`input_schema` on the next request. If the tool has
side effects, set `requires_confirmation = True` and the Executor will
pause and wait for a human approve/deny before ever running it.

## Connecting an MCP server

Trusted servers go in `mcp_servers.json` (loaded at startup, implicitly
trusted and confirmed):

```json
{
  "mcpServers": {
    "filesystem": {
      "transport": "stdio",
      "enabled": true,
      "timeout": 30,
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "./workspace"],
      "env": {}
    },
    "remote-example": {
      "transport": "streamable_http",
      "enabled": true,
      "timeout": 30,
      "url": "https://example.com/mcp"
    }
  }
}
```

Then connect it (discovers its tools and registers them into the
ToolRegistry):

```bash
curl -X POST "http://localhost:8000/api/mcp/connect?name=filesystem"
```

To add a server dynamically via the API instead of the config file, the
same security rules apply as anywhere else in this project:

- `streamable_http`/`sse` (remote URL) servers are accepted and confirmed
  immediately — they can't execute local commands.
- `stdio` servers are **rejected** unless their command is in
  `MCP_ALLOWED_COMMANDS`, or `ALLOW_DYNAMIC_STDIO_MCP=true` — and even then
  they need one explicit `POST /api/mcp/{name}/confirm` before their first
  connection. Only environment variable *names* listed in
  `MCP_ALLOWED_ENV_VARS` are ever forwarded to them.

Use the **MCP Servers** tab in the Web UI for the same flow with buttons.

## Connecting a GitHub integration

1. **Discover** candidates:
   `GET /api/github/search?q=agent framework` — cheap search first, then
   deep analysis (README, license, contributors, latest release) only for
   the top candidates, with bounded concurrency and rate-limit awareness.
2. **Generate Integration Plan** (read-only, `POST /api/github/analyze
   {"full_name": "owner/repo"}`) — returns a license gate, confidence-scored
   capability signals (`{available, confidence, evidence}`), the
   recommended adapter type, and exactly what *would* be installed.
3. **License gate**:
   - `MIT` / `Apache-2.0` / `BSD-*` / `ISC` → `permissive`, auto-integrable.
   - `MPL-2.0` and similar → `weak_copyleft` — **not** the same as
     permissive; needs a human look at file-level obligations.
   - `GPL` / `AGPL` / `LGPL` → `copyleft` — requires manual confirmation;
     nothing is copied automatically (`ALLOW_GPL_AUTO_INTEGRATION=true`
     only if you've reviewed the license yourself).
   - Unrecognized → `unknown` — always requires manual review.
4. **Generate Adapter** (`POST /api/github/generate-adapter`) writes a
   reviewable stub to `workspace/generated_adapters/` based on whichever
   capability was detected (MCP suggestion / REST adapter scaffold / SDK
   adapter scaffold / reference-only note). **Nothing is activated** —
   wiring a reviewed adapter into `app/main.py` is always a manual step;
   this project never dynamically imports or executes generated code.

The **GitHub Explorer** tab in the Web UI exposes the same flow:
*Generate Integration Plan* and *Generate Adapter (stub only)*.

---

## Security notes

- All secrets (`*_API_KEY`, `GITHUB_TOKEN`, `AGENT_API_KEY`) come from
  environment variables only; `.env` is git-ignored and never logged.
- No `eval()`/`exec()` anywhere in the codebase.
- No third-party GitHub code is ever downloaded and executed automatically
  — discovery only reads public metadata over the GitHub REST API, and
  "Generate Adapter" only writes a stub file, never imports/runs it.
- **API auth**: set `AGENT_API_KEY` and every `/api/*` route requires
  `Authorization: Bearer <key>` (`GET /health` and `GET /` stay open). The
  Web UI stores the key only in the browser's `localStorage`.
- **CORS** defaults to `http://localhost:8000,http://127.0.0.1:8000` —
  widen `CORS_ORIGINS` explicitly, and only if you understand the risk.
- **Human-in-the-loop confirmation**: any tool with `requires_confirmation`
  (e.g. `file_write`) pauses the run; execution only proceeds after an
  explicit approve of that *exact* confirmation id — arguments can never be
  substituted between confirmation and execution. Pending confirmations
  expire after `CONFIRMATION_TTL_SECONDS`.
- **web_fetch is SSRF-hardened**: blocks loopback/private/link-local/
  reserved/multicast addresses (including cloud metadata endpoints),
  non-http(s) schemes, re-validates redirect targets, GET/HEAD only by
  default, bounded response reads. See EXPERIMENTAL notes above for the
  one documented residual gap (DNS rebinding).
- **MCP stdio commands** are allowlisted (`MCP_ALLOWED_COMMANDS` /
  `ALLOW_DYNAMIC_STDIO_MCP`) and require a manual confirm before first use;
  env vars forwarded to them are allowlisted by name
  (`MCP_ALLOWED_ENV_VARS`).
- File tools (`file_read`/`file_write`/`file_list`) are sandboxed to
  `WORKSPACE_DIR`; every path is resolved and validated to stay inside it
  (`app/tools/filesystem_tools.py::WorkspaceSandbox`), blocking `..` escapes.
- Every tool call has an enforced timeout (`TOOL_TIMEOUT`); side-effecting
  tools are never blindly retried (`retry_safe=False`).
- The agent loop is bounded by `MAX_AGENT_STEPS` and `MAX_TOOL_CALLS` — it
  cannot spin forever, and a mid-batch budget cutoff never leaves a tool
  call without a structured response.
- Tool inputs are validated against `input_schema` before execution; an
  unknown/hallucinated tool name returns a clean error instead of crashing.
- `memory_remember` hard-blocks text that looks like a password/API
  key/token/credential — see `app/tools/memory_tools.py`.
- Errors are logged; secrets are never included in log output.

---

## API examples

```bash
# Health (no auth required even if AGENT_API_KEY is set)
curl http://localhost:8000/health

# Chat (creates a session automatically on first call)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What tools do you have access to?"}'

# Streaming chat (Server-Sent Events)
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Search GitHub for MCP servers"}'

# One-off task run (no persistent session)
curl -X POST http://localhost:8000/api/agent/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Search GitHub for popular MCP servers and summarize the top 3", "profile": "GitHub Agent"}'

# Confirmations (approve/deny take only an id — arguments can't be substituted)
curl http://localhost:8000/api/confirmations
curl -X POST http://localhost:8000/api/confirmations/<id>/approve
curl -X POST http://localhost:8000/api/confirmations/<id>/deny

# Providers (never returns API keys)
curl http://localhost:8000/api/providers

# List tools
curl http://localhost:8000/api/tools

# GitHub discovery
curl "http://localhost:8000/api/github/search?q=agentic%20workflow&limit=10"
curl -X POST http://localhost:8000/api/github/analyze -H "Content-Type: application/json" \
  -d '{"full_name": "modelcontextprotocol/servers"}'
curl -X POST http://localhost:8000/api/github/generate-adapter -H "Content-Type: application/json" \
  -d '{"full_name": "modelcontextprotocol/servers"}'

# MCP
curl http://localhost:8000/api/mcp
curl -X POST "http://localhost:8000/api/mcp/connect?name=filesystem"

# Models across all configured providers
curl http://localhost:8000/api/models
```

If `AGENT_API_KEY` is set, add `-H "Authorization: Bearer <key>"` to every
`/api/*` call above.

Full interactive docs (Swagger UI): http://localhost:8000/docs

---

## Running tests

```bash
pytest -q
```

All tests run fully offline: providers are exercised through
`httpx.MockTransport` (no real API keys needed), and the one MCP
Streamable-HTTP test spins up a real server bound to `127.0.0.1` only (no
external network). Test count isn't hard-coded here — run `pytest -q` and
read the summary line.

---

## Phase 2 roadmap

Not implemented in this phase, proposed as next steps:

- **Multi-agent orchestration** — supervisor/worker agent graphs, agent-to-agent handoff.
- **RAG** — document ingestion + retrieval-augmented context building.
- **Vector memory** — swap `SqliteLongTermMemory` for pgvector/Chroma/Qdrant behind the same `LongTermMemory` interface.
- **Sandbox execution** — run agent-written code in an isolated container/gVisor sandbox instead of the current no-code-exec-at-all posture.
- **Browser tools** — headless-browser tool for JS-rendered pages and interactive web tasks.
- **Full DNS-rebinding-proof SSRF protection** — pin the HTTP connection to the exact validated IP.
- **True token-level streaming** through the tool-calling loop, not just post-hoc chunking.
- **Persistent confirmation/run-state storage** — survive process restarts (Redis/Postgres-backed).
- **Stronger auth** — per-user/per-tenant API keys, OAuth, beyond the single shared `AGENT_API_KEY`.
- **PostgreSQL** — swap SQLite for Postgres for multi-instance deployments.
- **Docker isolation per tool** — run risky tools (shell, code exec) in their own throwaway containers.
- **Fine-grained permissions per tool** — per-profile, per-user tool ACLs beyond the current allow-list.
- **Agent marketplace** — publish/install community agent profiles + adapters.
- **Automatic adapter generator** — given an OpenAPI spec or MCP manifest, generate a *working* typed adapter (today's generator only produces a reviewable scaffold).
- **Observability** — structured tracing of every plan/tool-call/step (OpenTelemetry).
- **Evaluation / benchmarks** — regression suite scoring agent runs against golden tasks.
