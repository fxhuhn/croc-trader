import logging
import sqlite3
from typing import Any

from ..session import DatabaseSession

logger = logging.getLogger(__name__)


class BaseRepository:
    def __init__(self, session: DatabaseSession):
        self.session = session

    def execute(
        self, sql: str, params: tuple = (), conn: sqlite3.Connection = None
    ) -> sqlite3.Cursor:
        """
        Führt ein SQL-Statement aus.
        :param conn: Optionale Connection für Transaktionen. Falls None, wird eine neue Session geöffnet.
        """

        if conn:
            return conn.execute(sql, params)

        with self.session.connect() as new_conn:
            return new_conn.execute(sql, params)

    def fetch_one(
        self, sql: str, params: tuple = (), conn: sqlite3.Connection = None
    ) -> sqlite3.Row | None:
        """Lädt einen einzelnen Eintrag."""

        if conn:
            return conn.execute(sql, params).fetchone()

        with self.session.connect() as new_conn:
            return new_conn.execute(sql, params).fetchone()

    def fetch_all(
        self, sql: str, params: tuple = (), conn: sqlite3.Connection = None
    ) -> list[sqlite3.Row]:
        """Lädt mehrere Einträge."""

        if conn:
            return conn.execute(sql, params).fetchall()

        with self.session.connect() as new_conn:
            return new_conn.execute(sql, params).fetchall()

    def fetch_val(
        self, sql: str, params: tuple = (), conn: sqlite3.Connection = None
    ) -> Any:
        """Lädt einen einzelnen Wert (Skalar), z.B. COUNT(*)."""

        row = self.fetch_one(sql, params, conn)
        return row[0] if row else None
