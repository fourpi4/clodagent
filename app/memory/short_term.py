"""
Short-term (conversation) memory.

Stores the rolling message history for a single chat session. Backed by
SQLite for the MVP; swap the implementation behind `ShortTermMemory` for
Postgres/Redis later without touching Agent Core.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class ShortTermMemory(ABC):
    @abstractmethod
    def create_session(self, agent_profile: str = "default") -> str: ...

    @abstractmethod
    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[dict] = None) -> None: ...

    @abstractmethod
    def get_history(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]: ...

    @abstractmethod
    def list_sessions(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def clear_session(self, session_id: str) -> None: ...


class SqliteShortTermMemory(ShortTermMemory):
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
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    agent_profile TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")

    def create_session(self, agent_profile: str = "default") -> str:
        session_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, agent_profile, created_at) VALUES (?, ?, ?)",
                (session_id, agent_profile, time.time()),
            )
        return session_id

    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[dict] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, json.dumps(metadata or {}), time.time()),
            )

    def get_history(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, metadata, created_at FROM messages "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [
            {
                "role": r["role"],
                "content": r["content"],
                "metadata": json.loads(r["metadata"] or "{}"),
                "created_at": r["created_at"],
            }
            for r in reversed(rows)
        ]

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, agent_profile, created_at FROM sessions ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def clear_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
