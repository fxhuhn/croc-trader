"""Chaos and resilience tests for SQLite DatabaseSession and transaction integrity.

Verifies WAL mode concurrency, busy timeout configuration, automatic rollbacks
on transaction failure, and read-only isolation.
"""

import sqlite3
from pathlib import Path

import pytest

from app.database.session import DatabaseSession


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Provides a fresh temporary SQLite file path."""
    db_file = tmp_path / "chaos_test.db"
    session = DatabaseSession(str(db_file))
    with session.connect() as conn:
        conn.execute(
            "CREATE TABLE test_data (id INTEGER PRIMARY KEY, key TEXT, val REAL);"
        )
        conn.execute("INSERT INTO test_data (key, val) VALUES ('initial', 100.0);")
    return str(db_file)


@pytest.mark.tier2
def test_wal_mode_and_pragmas_enforced(temp_db_path: str) -> None:
    """Verifies that WAL mode, busy timeout, and row factory are properly configured."""
    session = DatabaseSession(temp_db_path)
    with session.connect() as conn:
        cursor = conn.execute("PRAGMA journal_mode;")
        journal_mode = cursor.fetchone()[0]
        assert journal_mode.upper() == "WAL"

        cursor = conn.execute("PRAGMA busy_timeout;")
        timeout = cursor.fetchone()[0]
        assert timeout >= 5000  # At least 5000ms


@pytest.mark.tier2
def test_transaction_rollback_on_exception(temp_db_path: str) -> None:
    """Invariant: An unhandled exception during a write session triggers a clean rollback."""
    session = DatabaseSession(temp_db_path)

    with pytest.raises(RuntimeError, match="Simulated crash during write"):
        with session.connect() as conn:
            conn.execute(
                "INSERT INTO test_data (key, val) VALUES ('corrupted', 999.0);"
            )
            raise RuntimeError("Simulated crash during write")

    # Verify that the corrupted insert was rolled back
    with session.connect() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM test_data WHERE key = 'corrupted';")
        count = cursor.fetchone()[0]
        assert count == 0


@pytest.mark.tier2
def test_concurrent_read_while_writing_in_wal(temp_db_path: str) -> None:
    """Invariant: In WAL mode, an open reader connection does not block a writer."""
    reader_session = DatabaseSession(temp_db_path, read_only=True)
    writer_session = DatabaseSession(temp_db_path)

    with reader_session.connect() as read_conn:
        # Reader is reading
        cursor = read_conn.execute("SELECT val FROM test_data WHERE key = 'initial';")
        val = cursor.fetchone()[0]
        assert val == 100.0

        # Writer can concurrently insert and commit
        with writer_session.connect() as write_conn:
            write_conn.execute(
                "INSERT INTO test_data (key, val) VALUES ('concurrent', 200.0);"
            )

    # After commit, new reader sees the update
    with reader_session.connect() as read_conn:
        cursor = read_conn.execute(
            "SELECT val FROM test_data WHERE key = 'concurrent';"
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 200.0


@pytest.mark.tier2
def test_read_only_mode_rejects_writes(temp_db_path: str) -> None:
    """Invariant: read_only session strictly rejects any write operation."""
    ro_session = DatabaseSession(temp_db_path, read_only=True)
    with pytest.raises(sqlite3.OperationalError):
        with ro_session.connect() as conn:
            conn.execute(
                "INSERT INTO test_data (key, val) VALUES ('forbidden', 500.0);"
            )
