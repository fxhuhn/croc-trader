import logging
import sqlite3
from typing import Any

from ..session import DatabaseSession

logger = logging.getLogger(__name__)


class BaseRepository:
    """Abstract base providing query execution and row fetching helpers."""

    def __init__(self, session: DatabaseSession) -> None:
        self.session = session

    def execute(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
        connection: sqlite3.Connection | None = None,
    ) -> sqlite3.Cursor:
        """Executes a SQL statement.

        Args:
            sql: The SQL query to execute.
            params: Query parameters for parameterized queries.
            connection: Optional existing connection for transactional use.
        """

        if connection:
            return connection.execute(sql, params)

        with self.session.connect() as new_connection:
            return new_connection.execute(sql, params)

    def fetch_one(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
        connection: sqlite3.Connection | None = None,
    ) -> sqlite3.Row | None:
        """Fetches a single row from the database."""
        try:
            if connection:
                row: sqlite3.Row | None = connection.execute(sql, params).fetchone()
                return row

            with self.session.connect() as new_connection:
                row_new: sqlite3.Row | None = new_connection.execute(
                    sql, params
                ).fetchone()
                return row_new
        except sqlite3.OperationalError as e:
            logger.error("Database fetch_one failed for query %s: %s", sql, e)
            return None

    def fetch_all(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
        connection: sqlite3.Connection | None = None,
    ) -> list[sqlite3.Row]:
        """Fetches all matching rows from the database."""
        try:
            if connection:
                rows: list[sqlite3.Row] = connection.execute(sql, params).fetchall()
                return rows

            with self.session.connect() as new_connection:
                rows_new: list[sqlite3.Row] = new_connection.execute(
                    sql, params
                ).fetchall()
                return rows_new
        except sqlite3.OperationalError as e:
            logger.error("Database fetch_all failed for query %s: %s", sql, e)
            return []

    def fetch_value(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
        connection: sqlite3.Connection | None = None,
    ) -> int | float | str | None:
        """Fetches a single scalar value (e.g., COUNT(*))."""

        row = self.fetch_one(sql, params, connection)
        if row is None:
            return None
        value = row[0]
        if isinstance(value, int | float | str):
            return value
        return str(value)
