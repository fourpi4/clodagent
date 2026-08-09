"""
Safe adapter scaffolding for discovered GitHub projects.

This module NEVER clones a repository, installs a dependency, or executes
any code from (or generated for) a target project. "Generate Adapter" only
writes a reviewable Python *stub* file into workspace/generated_adapters/,
built entirely from string templates — the repository's metadata (name,
description, license) is only ever interpolated as string literals, never
as executable code.

There is deliberately no "activate" endpoint that dynamically imports and
registers the generated file at runtime — that would be a form of
auto-executing fetched/generated code, which this project's security policy
forbids even for its own scaffolds. Wiring a reviewed adapter in is a
manual step: import it in app/main.py and call registry.register_tool(...).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _slugify(full_name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", full_name.lower()).strip("_")


def _class_name(full_name: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", full_name)
    return "".join(p.capitalize() for p in parts if p) + "Adapter"


def build_plan(repository: dict[str, Any]) -> dict[str, Any]:
    """Returns what generation WOULD produce, without writing anything (used by 'Generate Integration Plan')."""
    capabilities = repository.get("capabilities", {})
    tools_exposed: list[str] = []
    network_destinations: list[str] = []
    dependencies: list[str] = []
    permissions: list[str] = ["outbound network access to the target project's API/MCP endpoint"]

    if capabilities.get("mcp", {}).get("available"):
        tools_exposed.append("(dynamic) whatever tools the MCP server advertises via list_tools()")
        dependencies.append("none beyond the already-installed mcp SDK")
    elif capabilities.get("api", {}).get("available"):
        tools_exposed.append(f"{_slugify(repository['name'])}_call (generic REST adapter stub, needs endpoint filled in)")
        dependencies.append("none beyond the already-installed httpx")
        if repository.get("repository"):
            network_destinations.append(repository["repository"])
    elif capabilities.get("sdk", {}).get("available"):
        tools_exposed.append(f"{_slugify(repository['name'])}_call (SDK adapter stub, needs the project's SDK installed manually)")
        dependencies.append("the target project's own SDK (not installed automatically — review before adding)")
    else:
        tools_exposed.append("(none) — no API/SDK/MCP detected; reference-only, no adapter recommended")

    return {
        "repository": repository.get("repository"),
        "license": repository.get("license"),
        "license_category": repository.get("license_category"),
        "dependencies": dependencies,
        "permissions": permissions,
        "network_destinations": network_destinations,
        "tools_exposed": tools_exposed,
    }


_MCP_TEMPLATE = '''\
"""
Suggested mcp_servers.json entry for {full_name} — this project exposes an
MCP server, so no custom Python adapter is needed. Add this (reviewed and
adjusted) entry manually, then use the MCP Servers page to confirm and
connect it. Nothing here is executed automatically.

{{
  "mcpServers": {{
    "{slug}": {{
      "transport": "streamable_http",
      "enabled": true,
      "timeout": 30,
      "url": "REPLACE_WITH_THE_PROJECT_S_MCP_ENDPOINT_URL"
    }}
  }}
}}
"""
'''

_API_TEMPLATE = '''\
"""
Generated adapter STUB for {full_name} ({repository}).

License: {license} ({license_category})
This file is a scaffold only — it is NOT imported or registered
automatically anywhere. Review it, fill in the real endpoint details, then
manually wire it into app/main.py:

    from workspace.generated_adapters.{module_name} import build_adapter
    registry.register_tool(build_adapter())

Never register a generated adapter without reading it first.
"""
from __future__ import annotations

from app.integrations.custom_api_adapter import CustomApiAdapter


def build_adapter() -> CustomApiAdapter:
    return CustomApiAdapter(
        name="{tool_name}",
        description="TODO: describe what this call does ({full_name}).",
        input_schema={{
            "type": "object",
            "properties": {{
                # TODO: fill in the real parameters for this endpoint.
            }},
            "required": [],
        }},
        base_url="{repository}",  # TODO: replace with the project's actual API base URL
        path_template="/TODO",
        method="GET",
        requires_confirmation=True,  # keep True until you've reviewed the endpoint's side effects
    )
'''

_REFERENCE_TEMPLATE = '''\
"""
No adapter generated for {full_name} ({repository}).

No REST API, SDK, or MCP server was detected for this project — it is
reference-only. Do not vendor its source code directly; if you want to use
it, do so through its own official install instructions, reviewed manually.
"""
'''


def generate_adapter_source(repository: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """
    Writes a reviewable adapter stub (or a reference-only note) into
    `output_dir`. Returns the same plan as build_plan() plus the list of
    files actually written, and an explicit reminder that nothing was
    activated.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    full_name = repository["repository"].removeprefix("https://github.com/")
    slug = _slugify(full_name)
    capabilities = repository.get("capabilities", {})

    if capabilities.get("mcp", {}).get("available"):
        content = _MCP_TEMPLATE.format(full_name=full_name, slug=slug)
        filename = f"{slug}_mcp_suggestion.py"
    elif capabilities.get("api", {}).get("available"):
        content = _API_TEMPLATE.format(
            full_name=full_name,
            repository=repository["repository"],
            license=repository.get("license", "unknown"),
            license_category=repository.get("license_category", "unknown"),
            module_name=f"{slug}_adapter",
            tool_name=f"{slug}_call",
        )
        filename = f"{slug}_adapter.py"
    elif capabilities.get("sdk", {}).get("available"):
        content = _API_TEMPLATE.format(
            full_name=full_name,
            repository=repository["repository"],
            license=repository.get("license", "unknown"),
            license_category=repository.get("license_category", "unknown"),
            module_name=f"{slug}_adapter",
            tool_name=f"{slug}_call",
        )
        filename = f"{slug}_adapter.py"
    else:
        content = _REFERENCE_TEMPLATE.format(full_name=full_name, repository=repository["repository"])
        filename = f"{slug}_reference_only.py"

    file_path = output_dir / filename
    file_path.write_text(content, encoding="utf-8")

    plan = build_plan(repository)
    plan["generated_files"] = [str(file_path)]
    plan["activation_required"] = True
    plan["activation_note"] = (
        "Nothing was activated. Review the generated file, then manually import it in app/main.py and call "
        "registry.register_tool(...) — this project never auto-imports or executes generated/fetched code."
    )
    return plan
