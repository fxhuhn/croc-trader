"""Unit tests for DatabaseSession contract and context manager usage."""

import sqlite3

import pytest

from app.database.session import DatabaseSession


def test_database_session_does_not_support_context_manager_protocol(tmp_path):
    """Verify that DatabaseSession itself is not a context manager.

    Using 'with DatabaseSession(...)' must raise TypeError/AttributeError
    because context management is explicitly scoped to session.connect().
    """
    db_file = tmp_path / "test.db"
    session = DatabaseSession(str(db_file))

    with pytest.raises((TypeError, AttributeError)):
        with session:  # type: ignore[attr-defined]
            pass


def test_database_session_connect_context_manager(tmp_path):
    """Verify that session.connect() provides a valid SQLite connection context manager."""
    db_file = tmp_path / "test.db"
    session = DatabaseSession(str(db_file))

    # Create table via connect()
    with session.connect() as conn:
        assert isinstance(conn, sqlite3.Connection)
        conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO test_table (value) VALUES ('sample')")

    # Read back data in a new connect() block
    with session.connect() as conn:
        cursor = conn.execute("SELECT value FROM test_table WHERE id = 1")
        row = cursor.fetchone()
        assert row is not None
        assert row["value"] == "sample"


def test_database_session_rollback_on_exception(tmp_path):
    """Verify that session.connect() automatically rolls back on unhandled exceptions."""
    db_file = tmp_path / "test.db"
    session = DatabaseSession(str(db_file))

    with session.connect() as conn:
        conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT)")

    with pytest.raises(RuntimeError):
        with session.connect() as conn:
            conn.execute("INSERT INTO test_table (value) VALUES ('rollback_me')")
            raise RuntimeError("Simulated failure during transaction")

    with session.connect() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM test_table")
        count = cursor.fetchone()[0]
        assert count == 0
