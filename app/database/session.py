import sqlite3
from collections.abc import Generator
from contextlib import contextmanager


class DatabaseSession:
    BUSY_TIMEOUT = 30000  # Increased to 30s for high concurrency
    JOURNAL_MODE = "WAL"

    def __init__(self, db_path: str):
        self.db_path = db_path

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)

        # WAL Mode & Timeout & Performance optimizations
        conn.execute(f"PRAGMA busy_timeout = {self.BUSY_TIMEOUT};")
        conn.execute(f"PRAGMA journal_mode = {self.JOURNAL_MODE};")
        conn.execute("PRAGMA synchronous = NORMAL;")  # Faster WAL writes

        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
