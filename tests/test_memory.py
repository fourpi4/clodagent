from app.memory.long_term import SqliteLongTermMemory
from app.memory.short_term import SqliteShortTermMemory


def test_short_term_session_roundtrip(tmp_path):
    memory = SqliteShortTermMemory(tmp_path / "short.sqlite3")
    session_id = memory.create_session("default")
    memory.add_message(session_id, "user", "hello")
    memory.add_message(session_id, "assistant", "hi there")

    history = memory.get_history(session_id)
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == "hello"

    sessions = memory.list_sessions()
    assert any(s["id"] == session_id for s in sessions)


def test_short_term_clear_session(tmp_path):
    memory = SqliteShortTermMemory(tmp_path / "short.sqlite3")
    session_id = memory.create_session("default")
    memory.add_message(session_id, "user", "hello")
    memory.clear_session(session_id)
    assert memory.get_history(session_id) == []


def test_long_term_remember_and_recall(tmp_path):
    memory = SqliteLongTermMemory(tmp_path / "long.sqlite3")
    memory.remember("default", "The user prefers concise answers", tags=["preference"])
    memory.remember("default", "The project uses Bytez as the LLM provider", tags=["project"])

    results = memory.recall("default", "concise")
    assert len(results) == 1
    assert "concise" in results[0]["text"]


def test_long_term_forget(tmp_path):
    memory = SqliteLongTermMemory(tmp_path / "long.sqlite3")
    memory_id = memory.remember("default", "temporary fact")
    memory.forget(memory_id)
    assert memory.list_all("default") == []
