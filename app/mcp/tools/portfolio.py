"""Portfolio and position inspection tools for Croc-Trader MCP."""

import json
import logging
from typing import Any

from mcp.server import MCPServer

from ...const import TradeStatus
from ...database.repositories.trade import TradeRepository
from ...database.session import DatabaseSession
from ...routes.views.dependencies import (
    _get_database_path,
    _get_trade_view_service,
)

logger = logging.getLogger(__name__)


def _extract_float(val: object, default: float = 0.0) -> float:
    """Safely extracts a float from any object or returns default."""
    if val is None:
        return default
    try:
        return float(str(val))
    except (ValueError, TypeError):
        return default


def _sanitize_trade_view(trade_view: dict[str, Any]) -> dict[str, Any]:
    """Sanitizes trade view dict by removing HTML sparklines and raw Plotly strings."""
    clean = {k: v for k, v in trade_view.items() if k != "sparkline"}
    raw_context = clean.get("context") or clean.get("signal_context")
    if isinstance(raw_context, str):
        try:
            clean["context"] = json.loads(raw_context)
        except Exception:
            clean["context"] = {}
    return clean


def get_active_positions(
    strategy: str | None = None,
) -> list[dict[str, Any]]:
    """Returns list of active trade positions.

    Args:
        strategy: Optional strategy filter (e.g. 'dip_buyer', 'tgim', 'ndx_momentum').
    """
    try:
        trade_view_service = _get_trade_view_service()
        trades = trade_view_service.get_trades(
            strategies=strategy,
            status=TradeStatus.ACTIVE,
        )
        return [_sanitize_trade_view(dict(t)) for t in trades]
    except Exception as error:
        logger.error("Error fetching active positions: %s", error)
        return []


def get_portfolio_summary() -> dict[str, Any]:
    """Returns summary statistics for the trading portfolio."""
    try:
        trade_view_service = _get_trade_view_service()
        active_trades = trade_view_service.get_trades(status=TradeStatus.ACTIVE)

        total_invested = sum(
            _extract_float(t.get("current_size")) * _extract_float(t.get("entry_price"))
            for t in active_trades
        )
        total_unrealized_pnl = sum(
            _extract_float(t.get("unrealized_pnl")) for t in active_trades
        )

        by_strategy: dict[str, dict[str, Any]] = {}
        for t in active_trades:
            strat = str(t.get("strategy", "unknown"))
            if strat not in by_strategy:
                by_strategy[strat] = {
                    "count": 0,
                    "invested": 0.0,
                    "unrealized_pnl": 0.0,
                }
            size = _extract_float(t.get("current_size"))
            entry = _extract_float(t.get("entry_price"))
            pnl = _extract_float(t.get("unrealized_pnl"))
            by_strategy[strat]["count"] += 1
            by_strategy[strat]["invested"] += size * entry
            by_strategy[strat]["unrealized_pnl"] += pnl

        return {
            "active_positions_count": len(active_trades),
            "total_invested": round(total_invested, 2),
            "total_unrealized_pnl": round(total_unrealized_pnl, 2),
            "strategy_breakdown": by_strategy,
        }
    except Exception as error:
        logger.error("Error calculating portfolio summary: %s", error)
        return {"error": str(error)}


def get_trade_history(
    strategy: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Returns list of closed trades.

    Args:
        strategy: Optional strategy filter.
        limit: Maximum records to return (default 50, max 200).
    """
    try:
        effective_limit = max(1, min(limit, 200))
        trade_view_service = _get_trade_view_service()
        trades = trade_view_service.get_trades(
            strategies=strategy,
            status=TradeStatus.CLOSED,
        )
        sanitized = [_sanitize_trade_view(dict(t)) for t in trades]
        return sanitized[:effective_limit]
    except Exception as error:
        logger.error("Error fetching trade history: %s", error)
        return []


def get_trade_detail(trade_id: int) -> dict[str, Any]:
    """Returns full trade record with audit trail logs.

    Args:
        trade_id: Unique local identifier of the trade.
    """
    try:
        signals_db = str(_get_database_path("signals"))
        session = DatabaseSession(signals_db, read_only=True)
        repo = TradeRepository(session)
        trade = repo.get_trade(trade_id)
        if not trade:
            return {"error": f"Trade with ID {trade_id} not found"}

        clean_trade = dict(trade)
        raw_context = clean_trade.get("signal_context")
        if isinstance(raw_context, str):
            try:
                clean_trade["signal_context"] = json.loads(raw_context)
            except (json.JSONDecodeError, TypeError):
                logger.debug("Signal context for trade %d is not valid JSON.", trade_id)

        logs_rows = repo.fetch_all(
            "SELECT * FROM trade_logs WHERE trade_id = ? ORDER BY timestamp ASC",
            (trade_id,),
        )
        clean_trade["audit_logs"] = [dict(log) for log in logs_rows]
        return clean_trade
    except Exception as error:
        logger.error("Error fetching trade detail for %d: %s", trade_id, error)
        return {"error": str(error)}


def register(server: MCPServer) -> None:
    """Registers portfolio and trade inspection tools on the MCP server."""
    server.tool(
        name="get_active_positions",
        description=(
            "Fetches all currently active portfolio positions with real-time "
            "unrealized PnL, current price, stop loss, and target profit levels."
        ),
    )(get_active_positions)

    server.tool(
        name="get_portfolio_summary",
        description=(
            "Returns aggregated portfolio metrics including strategy budget allocations, "
            "invested capital, available cash, and open PnL."
        ),
    )(get_portfolio_summary)

    server.tool(
        name="get_trade_history",
        description=(
            "Fetches historical closed trades with holding period, exit reason, "
            "and realized profit/loss."
        ),
    )(get_trade_history)

    server.tool(
        name="get_trade_detail",
        description=(
            "Retrieves full details for a single trade record including its "
            "complete lifecycle audit log events."
        ),
    )(get_trade_detail)
