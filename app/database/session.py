"""Database session management with SQLite WAL mode.

Provides a context-managed connection factory ensuring consistent
PRAGMA configuration, row_factory setup, and transaction handling.
"""

import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Pre-built PRAGMA statements from class constants to avoid f-strings in SQL
_PRAGMA_BUSY_TIMEOUT = "PRAGMA busy_timeout = 30000;"
_PRAGMA_JOURNAL_MODE = "PRAGMA journal_mode = WAL;"
_PRAGMA_SYNCHRONOUS = "PRAGMA synchronous = NORMAL;"


class DatabaseSession:
    """Context-managed SQLite connection factory with WAL mode.

    All connections share the same PRAGMA configuration for consistency.
    Uses WAL journal mode for improved read concurrency.
    """

    def __init__(self, db_path: str, read_only: bool = False) -> None:
        self.db_path = db_path
        self.read_only = read_only

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Yields a configured SQLite connection with automatic commit/rollback."""
        if self.read_only:
            from pathlib import Path

            abs_path = Path(self.db_path).resolve().as_posix()
            connection = sqlite3.connect(f"file:{abs_path}?mode=ro", uri=True)
        else:
            connection = sqlite3.connect(self.db_path)

        # WAL Mode & Timeout & Performance optimizations
        connection.execute(_PRAGMA_BUSY_TIMEOUT)
        if not self.read_only:
            try:
                connection.execute(_PRAGMA_JOURNAL_MODE)
                connection.execute(_PRAGMA_SYNCHRONOUS)
            except sqlite3.OperationalError as e:
                logger.warning(
                    "Could not configure WAL mode for %s (possibly read-only): %s",
                    self.db_path,
                    e,
                )

        connection.row_factory = sqlite3.Row
        try:
            yield connection
            if not self.read_only:
                connection.commit()
        except Exception:
            if not self.read_only:
                connection.rollback()
            raise
        finally:
            connection.close()
