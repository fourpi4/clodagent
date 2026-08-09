# Universal AI Agent

A self-hosted, provider-agnostic AI agent.

- **Bytez** (https://bytez.com) is the primary LLM provider — one API key, 200k+ models, OpenAI-compatible chat completions.
- A **plugin/adapter system** lets you plug in capabilities from other open-source AI-agent projects on GitHub without touching the core.
- A **GitHub Discovery** module finds, ranks, and license-gates candidate projects to integrate.
- **MCP** (Model Context Protocol) servers plug in as first-class tools.
- Ships with a **REST API**, **Swagger/OpenAPI docs**, and a minimal **Web UI**.

No third-party code is ever executed automatically. Integrations use official REST APIs, SDKs, or MCP — never blind code copying.

---

## Architecture

```
USER REQUEST
    │
    ▼
CONTEXT        (agent/context.py)   — system prompt + relevant memory + history
    │
    ▼
PLANNER        (agent/planner.py)   — short structured plan (2-5 steps)
    │
    ▼
TOOL SELECTION (agent/tool_router.py) — LLM picks tools via function calling
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

Only structured data is retained at every stage — task, plan, tool calls, tool
results, status, final answer. The model's raw hidden reasoning is never
stored or shown to the user.

```
app/
  main.py                 FastAPI app factory + wiring
  agent/
    core.py                Agent — the single entrypoint
    planner.py              structured plan generation
    executor.py              plan→tool→observe→replan loop, step/retry limits
    memory.py                 short+long term memory facade
    context.py                 builds the message list sent to the LLM
    tool_router.py               routes LLM tool_calls to the ToolRegistry
    profiles.py                   Agent Profiles (CRUD, JSON-backed)
  providers/
    base.py                LLMProvider interface (chat / stream / list_models)
    bytez.py                BytezProvider (OpenAI-compatible Bytez API)
  tools/
    base.py                Tool interface + ToolResult
    registry.py              ToolRegistry (register/remove/list/get/execute)
    web_tools.py               WebFetchTool
    github_tools.py             GitHubSearchTool, GitHubGetRepoTool
    filesystem_tools.py           sandboxed file_read/file_write/file_list
  integrations/
    github_discovery.py     finds/ranks/license-gates GitHub AI-agent projects
    mcp.py                    MCP client manager (stdio + sse transports)
    mcp_adapter.py              registers MCP tools into the ToolRegistry
    langgraph_adapter.py        wraps a compiled LangGraph graph as a Tool
    crewai_adapter.py           wraps a CrewAI Crew as a Tool
    custom_api_adapter.py       wraps any documented REST endpoint as a Tool
  memory/
    short_term.py            SQLite-backed conversation memory
    long_term.py                SQLite-backed durable facts
  api/
    routes.py                REST endpoints
    schemas.py                 Pydantic request/response models
  ui/
    index.html, static/      minimal Web UI (Chat, Agents, Tools, GitHub
                              Explorer, MCP Servers, Memory, Settings)
  config/
    settings.py              all configuration, read from environment
tests/                      pytest suite (28 tests, no network required)
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

## Bytez configuration

1. Get an API key at https://bytez.com (Settings → API Keys).
2. Put it in `.env`:

```bash
BYTEZ_API_KEY=your-key-here
BYTEZ_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct   # any Bytez-hosted model id
```

`.env` is git-ignored — **never commit real keys**. Every secret in this
project is read from environment variables only; nothing is hard-coded.

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
`./workspace` (the sandbox filesystem tools operate in) as volumes.

---

## Creating an Agent Profile

Profiles are stored in `data/agent_profiles.json` and managed via API or the
**Agents** tab in the Web UI.

```bash
curl -X POST http://localhost:8000/api/agents \
  -H "Content-Type: application/json" \
  -d '{
        "name": "Docs Writer",
        "system_prompt": "You write clear, concise technical documentation.",
        "model": null,
        "temperature": 0.3,
        "tools": ["web_fetch", "file_read", "file_write"],
        "memory_enabled": true,
        "max_steps": 6
      }'
```

`tools: []` (empty) means "all registered tools are allowed" — restrict it
to limit what a given profile can do.

## Creating a Tool

Implement the `Tool` interface (`app/tools/base.py`) and register it in
`app/main.py::build_agent()`:

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

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(ok=True, output=len(arguments["text"].split()))
```

```python
# app/main.py, inside build_agent()
registry.register_tool(WordCountTool())
```

No core code changes needed — the LLM automatically sees the new tool's
`name`/`description`/`input_schema` on the next request.

## Connecting an MCP server

Add it to `mcp_servers.json`:

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
    }
  }
}
```

Then connect it (discovers its tools and registers them into the
ToolRegistry):

```bash
curl -X POST "http://localhost:8000/api/mcp/connect?name=filesystem"
```

Or use the **MCP Servers** tab in the Web UI. Remote servers use
`"transport": "sse"` with a `"url"` instead of `command`/`args`.

## Connecting a GitHub integration

1. **Discover** candidates:
   `GET /api/github/search?q=agent framework` — ranked by stars, recency,
   contributors, docs, API/SDK/MCP availability, and license (not stars alone).
2. **Analyze** one before touching anything:
   `POST /api/github/analyze {"full_name": "owner/repo"}` — returns a license
   gate, the recommended adapter type (MCP / REST / SDK), and exactly what
   would be installed.
3. **License gate**:
   - `MIT` / `Apache-2.0` / `BSD-*` / `ISC` / `MPL-2.0` / etc. → auto-integrable.
   - `GPL` / `AGPL` / `LGPL` / unknown → **requires manual confirmation**;
     nothing is copied automatically (set `ALLOW_GPL_AUTO_INTEGRATION=true`
     only if you've reviewed the license yourself).
4. **Integrate** using the matching adapter, never by vendoring source:
   - Has an MCP server → `integrations/mcp.py` / `mcp_adapter.py`
   - Has a REST API → `integrations/custom_api_adapter.py`
   - Has a Python SDK → a dedicated adapter (see `langgraph_adapter.py`,
     `crewai_adapter.py` as examples)

The **GitHub Explorer** tab in the Web UI does the same thing with buttons:
*Analyze*, *View Integration Plan*, *Install Adapter* (which only ever shows
you what to add — it never executes downloaded code).

---

## Security notes

- All secrets (`BYTEZ_API_KEY`, `GITHUB_TOKEN`) come from environment
  variables only; `.env` is git-ignored.
- No `eval()`/`exec()` anywhere in the codebase.
- No third-party GitHub code is ever downloaded and executed automatically —
  discovery only reads public metadata (README, license, topics) over the
  GitHub REST API.
- File tools (`file_read`/`file_write`/`file_list`) are sandboxed to
  `WORKSPACE_DIR`; every path is resolved and validated to stay inside it
  (`app/tools/filesystem_tools.py::WorkspaceSandbox`), blocking `..` escapes.
- Every tool call has an enforced timeout (`TOOL_TIMEOUT`) and bounded
  retries (`MAX_RETRIES`).
- The agent loop is bounded by `MAX_AGENT_STEPS` and `MAX_TOOL_CALLS` — it
  cannot spin forever.
- Tool inputs are validated against `input_schema` before execution.
- Tools that perform side effects can set `requires_confirmation = True`
  (e.g. `file_write`) as a signal for UI/clients to prompt the user.
- Errors are logged; secrets are never included in log output.

---

## API examples

```bash
# Health
curl http://localhost:8000/health

# Chat (creates a session automatically on first call)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What tools do you have access to?"}'

# One-off task run (no persistent session)
curl -X POST http://localhost:8000/api/agent/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Search GitHub for popular MCP servers and summarize the top 3", "profile": "GitHub Agent"}'

# List tools
curl http://localhost:8000/api/tools

# GitHub discovery
curl "http://localhost:8000/api/github/search?q=agentic%20workflow&limit=10"
curl -X POST http://localhost:8000/api/github/analyze -H "Content-Type: application/json" \
  -d '{"full_name": "modelcontextprotocol/servers"}'

# MCP
curl http://localhost:8000/api/mcp
curl -X POST "http://localhost:8000/api/mcp/connect?name=filesystem"

# Models available via Bytez
curl http://localhost:8000/api/models
```

Full interactive docs (Swagger UI): http://localhost:8000/docs

---

## Running tests

```bash
pytest -q
```

28 tests, all offline (a scripted fake `LLMProvider` stands in for Bytez;
`GitHubDiscovery`'s scoring/license logic is tested as pure functions).

---

## Phase 2 roadmap

Not implemented in this MVP, proposed as next steps:

- **Multi-agent orchestration** — supervisor/worker agent graphs, agent-to-agent handoff.
- **RAG** — document ingestion + retrieval-augmented context building.
- **Vector memory** — swap `SqliteLongTermMemory` for pgvector/Chroma/Qdrant behind the same `LongTermMemory` interface.
- **Sandbox execution** — run agent-written code in an isolated container/gVisor sandbox instead of the current no-code-exec-at-all posture.
- **Browser tools** — headless-browser tool for JS-rendered pages and interactive web tasks.
- **Authentication** — API keys / OAuth for the REST API and Web UI.
- **PostgreSQL** — swap SQLite for Postgres for multi-instance deployments.
- **Docker isolation per tool** — run risky tools (shell, code exec) in their own throwaway containers.
- **Fine-grained permissions per tool** — per-profile, per-user tool ACLs beyond the current allow-list.
- **Agent marketplace** — publish/install community agent profiles + adapters.
- **Automatic adapter generator** — given an OpenAPI spec or MCP manifest, generate a typed adapter automatically.
- **Observability** — structured tracing of every plan/tool-call/step (OpenTelemetry).
- **Evaluation / benchmarks** — regression suite scoring agent runs against golden tasks.
