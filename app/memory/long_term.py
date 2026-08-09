"""
Long-term memory.

Stores durable facts/notes the agent chooses to remember across sessions.
Search is backed by SQLite FTS5 (full-text ranking via bm25) when available,
with an automatic fallback to a simple token-overlap ranking over LIKE
queries on platforms whose SQLite build lacks the FTS5 extension. Either
way this is a plain keyword-search MVP; the abstraction is designed so a
vector-search backend (Postgres+pgvector, Chroma, etc.) can be dropped in
later behind the same interface (see README Phase 2 notes).
"""
from __future__ import annotations

import logging
import re
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class LongTermMemory(ABC):
    @abstractmethod
    def remember(self, agent_profile: str, text: str, tags: list[str] | None = None) -> str: ...

    @abstractmethod
    def recall(self, agent_profile: str, query: str, limit: int = 5) -> list[dict[str, Any]]: ...

    @abstractmethod
    def forget(self, memory_id: str) -> None: ...

    @abstractmethod
    def list_all(self, agent_profile: str | None = None) -> list[dict[str, Any]]: ...


def _fts_query(text: str) -> str:
    """Turns free text into a forgiving FTS5 MATCH expression (OR of tokens)."""
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens[:16])


class SqliteLongTermMemory(LongTermMemory):
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fts_available = False
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
            try:
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(id UNINDEXED, text, tags)"
                )
                self._fts_available = True
            except sqlite3.OperationalError:
                logger.warning("SQLite build lacks FTS5 — falling back to token-overlap ranking for memory recall")
                self._fts_available = False

    def remember(self, agent_profile: str, text: str, tags: list[str] | None = None) -> str:
        memory_id = str(uuid.uuid4())
        tags_str = ",".join(tags or [])
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memories (id, agent_profile, text, tags, created_at) VALUES (?, ?, ?, ?, ?)",
                (memory_id, agent_profile, text, tags_str, time.time()),
            )
            if self._fts_available:
                conn.execute(
                    "INSERT INTO memories_fts (id, text, tags) VALUES (?, ?, ?)", (memory_id, text, tags_str)
                )
        return memory_id

    def _recall_fts(self, agent_profile: str, query: str, limit: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT m.id, m.text, m.tags, m.created_at, bm25(memories_fts) AS rank
                FROM memories_fts
                JOIN memories m ON m.id = memories_fts.id
                WHERE memories_fts MATCH ? AND m.agent_profile = ?
                ORDER BY rank
                LIMIT ?
                """,
                (_fts_query(query), agent_profile, limit),
            ).fetchall()
        return [{"id": r["id"], "text": r["text"], "tags": r["tags"], "created_at": r["created_at"]} for r in rows]

    def _recall_fallback(self, agent_profile: str, query: str, limit: int) -> list[dict[str, Any]]:
        """Token-overlap ranking: counts how many query tokens appear in each memory's text/tags."""
        tokens = [t.lower() for t in _TOKEN_RE.findall(query)]
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, text, tags, created_at FROM memories WHERE agent_profile = ?",
                (agent_profile,),
            ).fetchall()

        def score(row: sqlite3.Row) -> int:
            haystack = f"{row['text']} {row['tags'] or ''}".lower()
            return sum(1 for t in tokens if t in haystack)

        scored = [(score(r), r) for r in rows]
        scored = [(s, r) for s, r in scored if s > 0]
        scored.sort(key=lambda pair: (-pair[0], -pair[1]["created_at"]))
        return [
            {"id": r["id"], "text": r["text"], "tags": r["tags"], "created_at": r["created_at"]}
            for _, r in scored[:limit]
        ]

    def recall(self, agent_profile: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if self._fts_available:
            try:
                return self._recall_fts(agent_profile, query, limit)
            except sqlite3.OperationalError:
                logger.exception("FTS5 query failed, falling back to token-overlap ranking")
        return self._recall_fallback(agent_profile, query, limit)

    def forget(self, memory_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            if self._fts_available:
                conn.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))

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
