"""
Filesystem tools, sandboxed to a single workspace directory.

Every path argument is resolved against WORKSPACE_DIR and validated to stay
inside it, so a tool call can never read or write anywhere else on disk
(no `..` escapes, no absolute paths, no symlink escapes).
"""
from __future__ import annotations

from pathlib import Path

from app.tools.base import Tool, ToolResult

MAX_READ_CHARS = 50_000
MAX_WRITE_CHARS = 200_000


class WorkspaceSandbox:
    """Resolves and validates paths against a single sandbox root."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(
                f"Path '{relative_path}' escapes the workspace sandbox ({self.root})"
            ) from exc
        return candidate


class FileReadTool(Tool):
    name = "file_read"
    description = "Read a text file from the agent's sandboxed workspace directory."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path relative to the workspace root"}},
        "required": ["path"],
    }
    risk_level = "read"
    side_effects = False
    retry_safe = True
    requires_confirmation = False

    def __init__(self, sandbox: WorkspaceSandbox) -> None:
        self._sandbox = sandbox

    async def execute(self, arguments: dict) -> ToolResult:
        try:
            path = self._sandbox.resolve(arguments["path"])
        except PermissionError as exc:
            return ToolResult(ok=False, error=str(exc))
        if not path.exists() or not path.is_file():
            return ToolResult(ok=False, error=f"File not found: {arguments['path']}")
        content = path.read_text(encoding="utf-8", errors="replace")[:MAX_READ_CHARS]
        return ToolResult(ok=True, output=content)


class FileWriteTool(Tool):
    name = "file_write"
    description = "Write text content to a file inside the agent's sandboxed workspace directory."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace root"},
            "content": {"type": "string", "description": "Text content to write"},
        },
        "required": ["path", "content"],
    }
    risk_level = "write"
    side_effects = True
    retry_safe = False  # writing twice on a spurious failure could double-append/clobber unexpectedly
    requires_confirmation = True

    def __init__(self, sandbox: WorkspaceSandbox) -> None:
        self._sandbox = sandbox

    async def execute(self, arguments: dict) -> ToolResult:
        content = arguments["content"]
        if len(content) > MAX_WRITE_CHARS:
            return ToolResult(ok=False, error=f"content exceeds max size of {MAX_WRITE_CHARS} characters")
        try:
            path = self._sandbox.resolve(arguments["path"])
        except PermissionError as exc:
            return ToolResult(ok=False, error=str(exc))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult(ok=True, output=f"Wrote {len(content)} characters to {arguments['path']}")


class FileListTool(Tool):
    name = "file_list"
    description = "List files and directories inside the agent's sandboxed workspace directory."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Directory relative to the workspace root, default '.'"}},
        "required": [],
    }
    risk_level = "read"
    side_effects = False
    retry_safe = True
    requires_confirmation = False

    def __init__(self, sandbox: WorkspaceSandbox) -> None:
        self._sandbox = sandbox

    async def execute(self, arguments: dict) -> ToolResult:
        rel = arguments.get("path", ".")
        try:
            path = self._sandbox.resolve(rel)
        except PermissionError as exc:
            return ToolResult(ok=False, error=str(exc))
        if not path.exists() or not path.is_dir():
            return ToolResult(ok=False, error=f"Directory not found: {rel}")
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
        return ToolResult(ok=True, output=entries)
