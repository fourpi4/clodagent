import pytest

from app.tools.filesystem_tools import FileListTool, FileReadTool, FileWriteTool, WorkspaceSandbox


@pytest.fixture
def sandbox(tmp_path):
    return WorkspaceSandbox(tmp_path / "workspace")


async def test_write_then_read(sandbox):
    write = FileWriteTool(sandbox)
    read = FileReadTool(sandbox)

    write_result = await write.execute({"path": "notes.txt", "content": "hello world"})
    assert write_result.ok is True

    read_result = await read.execute({"path": "notes.txt"})
    assert read_result.ok is True
    assert read_result.output == "hello world"


async def test_read_missing_file(sandbox):
    read = FileReadTool(sandbox)
    result = await read.execute({"path": "nope.txt"})
    assert result.ok is False


async def test_path_escape_is_blocked(sandbox):
    read = FileReadTool(sandbox)
    result = await read.execute({"path": "../../etc/passwd"})
    assert result.ok is False
    assert "escapes" in result.error


async def test_list_directory(sandbox):
    write = FileWriteTool(sandbox)
    await write.execute({"path": "a.txt", "content": "1"})
    await write.execute({"path": "b.txt", "content": "2"})
    listing = FileListTool(sandbox)
    result = await listing.execute({})
    assert result.ok is True
    assert set(result.output) == {"a.txt", "b.txt"}
