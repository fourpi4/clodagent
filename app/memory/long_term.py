"""
Long-term memory.

Stores durable facts/notes the agent chooses to remember across sessions,
keyed by a simple text tag so they can be recalled later. This is a plain
keyword-search MVP; the abstraction is designed so a vector-search backend
(Postgres+pgvector, Chroma, etc.) can be dropped in later behind the same
interface (see README Phase 2).
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class LongTermMemory(ABC):
    @abstractmethod
    def remember(self, agent_profile: str, text: str, tags: list[str] | None = None) -> str: ...

    @abstractmethod
    def recall(self, agent_profile: str, query: str, limit: int = 5) -> list[dict[str, Any]]: ...

    @abstractmethod
    def forget(self, memory_id: str) -> None: ...

    @abstractmethod
    def list_all(self, agent_profile: str | None = None) -> list[dict[str, Any]]: ...


class SqliteLongTermMemory(LongTermMemory):
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    agent_profile TEXT NOT NULL,
                    text TEXT NOT NULL,
                    tags TEXT,
                    created_at REAL NOT NULL
                )
                """
            )

    def remember(self, agent_profile: str, text: str, tags: list[str] | None = None) -> str:
        memory_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memories (id, agent_profile, text, tags, created_at) VALUES (?, ?, ?, ?, ?)",
                (memory_id, agent_profile, text, ",".join(tags or []), time.time()),
            )
        return memory_id

    def recall(self, agent_profile: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        like = f"%{query}%"
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, text, tags, created_at FROM memories "
                "WHERE agent_profile = ? AND (text LIKE ? OR tags LIKE ?) "
                "ORDER BY created_at DESC LIMIT ?",
                (agent_profile, like, like, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def forget(self, memory_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    def list_all(self, agent_profile: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if agent_profile:
                rows = conn.execute(
                    "SELECT id, agent_profile, text, tags, created_at FROM memories "
                    "WHERE agent_profile = ? ORDER BY created_at DESC",
                    (agent_profile,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, agent_profile, text, tags, created_at FROM memories ORDER BY created_at DESC"
                ).fetchall()
        return [dict(r) for r in rows]
