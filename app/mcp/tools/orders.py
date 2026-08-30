"""Broker orders, executions, and CSV export inspection tools for Croc-Trader MCP."""

import datetime
import logging
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from ...database.repositories.broker import BrokerRepository
from ...database.session import DatabaseSession
from ...routes.views.dependencies import _get_database_path

logger = logging.getLogger(__name__)


def _get_broker_repository() -> BrokerRepository | None:
    """Instantiates a read-only BrokerRepository if trading.db exists."""
    try:
        trading_db_path = _get_database_path("trading")
        if trading_db_path.exists():
            session = DatabaseSession(str(trading_db_path), read_only=True)
            return BrokerRepository(session)
    except Exception as error:
        logger.warning("Could not instantiate BrokerRepository: %s", error)
    return None


def get_broker_active_orders(
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Returns list of broker orders.

    Args:
        status: Optional order status filter (e.g. 'Submitted', 'PreSubmitted', 'Created').
    """
    try:
        repo = _get_broker_repository()
        if not repo:
            return []
        if status:
            orders = repo.get_orders_by_status(status)
        else:
            orders = repo.get_all_orders()
        return [dict(o) for o in orders]
    except Exception as error:
        logger.error("Error fetching broker orders: %s", error)
        return []


def get_broker_settlements(limit: int = 50) -> list[dict[str, Any]]:
    """Returns list of completed trade settlement records.

    Args:
        limit: Maximum records to return (default 50).
    """
    try:
        effective_limit = max(1, min(limit, 200))
        repo = _get_broker_repository()
        if not repo:
            return []
        settlements = repo.get_settlements()
        return [dict(s) for s in settlements[:effective_limit]]
    except Exception as error:
        logger.error("Error fetching broker settlements: %s", error)
        return []


def list_order_csv_files() -> list[dict[str, Any]]:
    """Returns list of order export CSV files."""
    try:
        orders_dir = Path("data/orders").resolve()
        if not orders_dir.exists():
            return []

        files = []
        for file_path in sorted(orders_dir.glob("*.csv"), reverse=True):
            stat = file_path.stat()
            mod_time = datetime.datetime.fromtimestamp(
                stat.st_mtime, tz=datetime.UTC
            ).strftime("%Y-%m-%d %H:%M:%S UTC")
            files.append(
                {
                    "filename": file_path.name,
                    "size_bytes": stat.st_size,
                    "last_modified": mod_time,
                }
            )
        return files
    except Exception as error:
        logger.error("Error listing order CSV files: %s", error)
        return []


def register(server: MCPServer) -> None:
    """Registers broker and order inspection tools on the MCP server."""
    server.tool(
        name="get_broker_active_orders",
        description=(
            "Fetches active broker orders (ENTRY, STOP LOSS, TAKE PROFIT) from trading.db."
        ),
    )(get_broker_active_orders)

    server.tool(
        name="get_broker_settlements",
        description=(
            "Fetches settled trade groups from trading.db with execution VWAP, "
            "slippage against target entry, total commissions, and net PnL."
        ),
    )(get_broker_settlements)

    server.tool(
        name="list_order_csv_files",
        description=(
            "Lists all generated CSV order export files in data/orders/ with date and size."
        ),
    )(list_order_csv_files)
