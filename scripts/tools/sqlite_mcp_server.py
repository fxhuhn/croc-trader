"""Read-only SQLite Model Context Protocol (MCP) Server for Croc-Trader.

Provides safe, read-only inspection tools for stocks.db, signals.db, and
trading.db without allowing any data mutations or schema alterations.
"""

import re
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

# Base workspace directory (repository root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Canonical database aliases mapped to relative repository paths
DATABASE_MAP: dict[str, Path] = {
    "stocks": PROJECT_ROOT / "data" / "stocks.db",
    "signals": PROJECT_ROOT / "data" / "signals.db",
    "trading": PROJECT_ROOT / "data" / "trading.db",
}

# Forbidden SQL keywords to prevent mutations even if query_only pragma is bypassed
FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|VACUUM|REINDEX|TRUNCATE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)

server = MCPServer(
    name="croc-sqlite-readonly",
    instructions="Read-only SQLite database inspection server for Croc-Trader.",
)


def _resolve_database_path(database: str) -> Path:
    """Resolves a database identifier or relative path safely within data/."""
    canonical_key = database.strip().lower()
    if canonical_key in DATABASE_MAP:
        resolved_path = DATABASE_MAP[canonical_key]
    else:
        # Fallback to direct path resolution within data directory
        candidate = (PROJECT_ROOT / "data" / database).resolve()
        if not candidate.is_relative_to(PROJECT_ROOT / "data"):
            raise ValueError(
                f"Access denied: database path '{database}' outside data directory"
            )
        resolved_path = candidate

    if not resolved_path.exists():
        raise FileNotFoundError(f"Database file does not exist: {resolved_path.name}")

    return resolved_path


@contextmanager
def _get_readonly_connection(
    database: str,
) -> Generator[sqlite3.Connection, None, None]:
    """Yields a strictly read-only SQLite connection configured with WAL support."""
    db_path = _resolve_database_path(database)
    # URI connection with mode=ro enforces filesystem read-only access
    uri = f"file:{db_path.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=10.0)
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.cursor()
        cursor.execute("PRAGMA query_only = ON;")
        yield connection
    finally:
        connection.close()


def _sanitize_identifier(identifier: str) -> str:
    """Validates SQL table/column identifiers against injection patterns."""
    clean_identifier = identifier.strip()
    if not re.match(r"^[A-Za-z0-9_]+$", clean_identifier):
        raise ValueError(f"Invalid SQL identifier: '{identifier}'")
    return clean_identifier


def _validate_table_name(connection: sqlite3.Connection, table_name: str) -> str:
    """Validates that a table exists in sqlite_master and conforms to identifier syntax."""
    safe_table = _sanitize_identifier(table_name)
    cursor = connection.cursor()
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?;",
        (safe_table,),
    )
    if not cursor.fetchone():
        raise ValueError(f"Table '{table_name}' does not exist in database.")
    return safe_table


@server.tool()
def sqlite_list_tables(database: str) -> list[str]:
    """Lists all user tables in the specified database ('stocks', 'signals', or 'trading').

    Args:
        database: Database alias ('stocks', 'signals', 'trading') or DB filename.

    Returns:
        List of table names.
    """
    with _get_readonly_connection(database) as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        )
        return [row[0] for row in cursor.fetchall()]


@server.tool()
def sqlite_describe_table(database: str, table_name: str) -> list[dict[str, Any]]:
    """Returns schema metadata for a specific table in the database.

    Args:
        database: Database alias ('stocks', 'signals', 'trading').
        table_name: Name of the table to describe.

    Returns:
        List of column metadata dictionaries (cid, name, type, notnull, dflt_value, pk).
    """
    with _get_readonly_connection(database) as connection:
        safe_table = _validate_table_name(connection, table_name)
        cursor = connection.cursor()
        cursor.execute(f"PRAGMA table_info({safe_table});")  # nosec B608: table name validated against sqlite_master
        columns = cursor.fetchall()
        return [
            {
                "cid": row[0],
                "name": row[1],
                "type": row[2],
                "notnull": bool(row[3]),
                "default_value": row[4],
                "pk": bool(row[5]),
            }
            for row in columns
        ]


@server.tool()
def sqlite_query(
    database: str,
    query: str,
    params: list[Any] | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Executes a parameterized read-only SELECT query against the specified database.

    Args:
        database: Database alias ('stocks', 'signals', 'trading').
        query: Parameterized SELECT SQL query.
        params: List of positional parameter values for '?' placeholders.
        limit: Maximum number of rows to return (capped at 500).

    Returns:
        List of row dictionaries mapping column names to values.
    """
    clean_query = query.strip()
    if FORBIDDEN_SQL_PATTERN.search(clean_query):
        raise PermissionError(
            "Query rejected: Only read-only SELECT/PRAGMA operations are permitted."
        )

    # Guard against multiple statement injection via semicolon
    if ";" in clean_query.rstrip(";"):
        raise ValueError("Multi-statement queries are forbidden.")

    effective_limit = max(1, min(limit, 500))
    query_params = params or []

    with _get_readonly_connection(database) as connection:
        cursor = connection.cursor()
        cursor.execute(clean_query, query_params)
        rows = cursor.fetchmany(effective_limit)
        return [{key: row[key] for key in row.keys()} for row in rows]


@server.tool()
def sqlite_count_rows(
    database: str,
    table_name: str,
    where_clause: str | None = None,
) -> int:
    """Counts records in a table with an optional WHERE filter condition.

    Args:
        database: Database alias ('stocks', 'signals', 'trading').
        table_name: Name of the table.
        where_clause: Optional WHERE filter expression (e.g. "status = 'ACTIVE'").

    Returns:
        Total number of matching rows.
    """
    with _get_readonly_connection(database) as connection:
        safe_table = _validate_table_name(connection, table_name)
        sql = f"SELECT COUNT(*) FROM {safe_table}"  # nosec B608: table name validated against sqlite_master

        if where_clause:
            clean_where = where_clause.strip()
            if FORBIDDEN_SQL_PATTERN.search(clean_where) or ";" in clean_where:
                raise PermissionError("Unsafe WHERE clause rejected.")
            sql += f" WHERE {clean_where}"

        cursor = connection.cursor()
        cursor.execute(sql)
        row = cursor.fetchone()
        return int(row[0]) if row else 0


def main() -> None:
    """Runs the MCP server over standard I/O."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
