"""Base repository providing common database access patterns.

Provides reusable fetch/execute methods for all concrete repositories,
ensuring consistent connection handling via DatabaseSession.
"""

import logging
import sqlite3

from ..session import DatabaseSession

logger = logging.getLogger(__name__)


class BaseRepository:
    """Abstract base providing query execution and row fetching helpers."""

    def __init__(self, session: DatabaseSession) -> None:
        self.session = session

    def execute(
        self,
        sql: str,
        params: tuple = (),
        conn: sqlite3.Connection | None = None,
    ) -> sqlite3.Cursor:
        """Executes a SQL statement.

        Args:
            sql: The SQL query to execute.
            params: Query parameters for parameterized queries.
            conn: Optional existing connection for transactional use.
        """

        if conn:
            return conn.execute(sql, params)

        with self.session.connect() as new_conn:
            return new_conn.execute(sql, params)

    def fetch_one(
        self,
        sql: str,
        params: tuple = (),
        conn: sqlite3.Connection | None = None,
    ) -> sqlite3.Row | None:
        """Fetches a single row from the database."""

        if conn:
            return conn.execute(sql, params).fetchone()

        with self.session.connect() as new_conn:
            return new_conn.execute(sql, params).fetchone()

    def fetch_all(
        self,
        sql: str,
        params: tuple = (),
        conn: sqlite3.Connection | None = None,
    ) -> list[sqlite3.Row]:
        """Fetches all matching rows from the database."""

        if conn:
            return conn.execute(sql, params).fetchall()

        with self.session.connect() as new_conn:
            return new_conn.execute(sql, params).fetchall()

    def fetch_val(
        self,
        sql: str,
        params: tuple = (),
        conn: sqlite3.Connection | None = None,
    ) -> int | float | str | None:
        """Fetches a single scalar value (e.g., COUNT(*))."""

        row = self.fetch_one(sql, params, conn)
        return row[0] if row else None
