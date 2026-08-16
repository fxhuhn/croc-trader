"""Unit and security tests for the read-only SQLite MCP Server."""

import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

from scripts.tools.sqlite_mcp_server import (
    _resolve_database_path,
    _sanitize_identifier,
    sqlite_count_rows,
    sqlite_describe_table,
    sqlite_list_tables,
    sqlite_query,
)


@pytest.fixture
def mock_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Path, None, None]:
    """Creates a temporary SQLite test database and patches the DATABASE_MAP."""
    db_file = tmp_path / "mock_test.db"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE test_items (id INTEGER PRIMARY KEY, name TEXT, value REAL);"
    )
    cursor.execute(
        "INSERT INTO test_items (name, value) VALUES ('Item A', 10.5), ('Item B', 20.0), ('Item C', 35.2);"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        "scripts.tools.sqlite_mcp_server.DATABASE_MAP",
        {"test_db": db_file},
    )
    yield db_file


def test_sanitize_identifier_valid() -> None:
    """Valid table/column identifiers are accepted."""
    assert _sanitize_identifier("trades") == "trades"
    assert _sanitize_identifier("market_prices_2026") == "market_prices_2026"
    assert _sanitize_identifier("UserTable1") == "UserTable1"


def test_sanitize_identifier_invalid() -> None:
    """Invalid table/column identifiers with spaces or injection syntax raise ValueError."""
    with pytest.raises(ValueError, match="Invalid SQL identifier"):
        _sanitize_identifier("trades; DROP TABLE trades;")

    with pytest.raises(ValueError, match="Invalid SQL identifier"):
        _sanitize_identifier("table name with spaces")

    with pytest.raises(ValueError, match="Invalid SQL identifier"):
        _sanitize_identifier("test--comment")


def test_resolve_database_path_valid(mock_db: Path) -> None:
    """Canonical alias maps to existing path."""
    resolved = _resolve_database_path("test_db")
    assert resolved == mock_db


def test_resolve_database_path_nonexistent() -> None:
    """Nonexistent database throws FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        _resolve_database_path("non_existent_database_name.db")


def test_sqlite_list_tables(mock_db: Path) -> None:
    """Lists user tables correctly."""
    tables = sqlite_list_tables("test_db")
    assert "test_items" in tables
    assert len(tables) == 1


def test_sqlite_describe_table(mock_db: Path) -> None:
    """Returns correct column metadata."""
    columns = sqlite_describe_table("test_db", "test_items")
    col_names = [col["name"] for col in columns]
    assert col_names == ["id", "name", "value"]

    id_col = next(col for col in columns if col["name"] == "id")
    assert id_col["pk"] is True
    assert id_col["type"].upper() == "INTEGER"


def test_sqlite_query_select(mock_db: Path) -> None:
    """Executes valid parameterized SELECT queries."""
    rows = sqlite_query(
        "test_db",
        "SELECT name, value FROM test_items WHERE value > ? ORDER BY id",
        [15.0],
    )
    assert len(rows) == 2
    assert rows[0]["name"] == "Item B"
    assert rows[0]["value"] == 20.0
    assert rows[1]["name"] == "Item C"


def test_sqlite_query_blocks_mutation(mock_db: Path) -> None:
    """Mutating statements are strictly blocked by security filter."""
    with pytest.raises(PermissionError, match="Only read-only SELECT"):
        sqlite_query(
            "test_db", "INSERT INTO test_items (name, value) VALUES ('Hack', 99.9)"
        )

    with pytest.raises(PermissionError, match="Only read-only SELECT"):
        sqlite_query("test_db", "UPDATE test_items SET value = 0")

    with pytest.raises(PermissionError, match="Only read-only SELECT"):
        sqlite_query("test_db", "DROP TABLE test_items")

    with pytest.raises(PermissionError, match="Only read-only SELECT"):
        sqlite_query("test_db", "DELETE FROM test_items")


def test_sqlite_query_blocks_multi_statements(mock_db: Path) -> None:
    """Multiple statements separated by semicolon are blocked."""
    with pytest.raises(ValueError, match="Multi-statement queries are forbidden"):
        sqlite_query("test_db", "SELECT 1; SELECT 2")


def test_sqlite_count_rows(mock_db: Path) -> None:
    """Counts matching rows correctly."""
    assert sqlite_count_rows("test_db", "test_items") == 3
    assert sqlite_count_rows("test_db", "test_items", "value >= 20.0") == 2
    assert sqlite_count_rows("test_db", "test_items", "name = 'Unknown'") == 0


def test_sqlite_count_rows_unsafe_where(mock_db: Path) -> None:
    """Unsafe WHERE clauses are rejected."""
    with pytest.raises(PermissionError, match="Unsafe WHERE clause rejected"):
        sqlite_count_rows("test_db", "test_items", "1=1; DROP TABLE test_items")
