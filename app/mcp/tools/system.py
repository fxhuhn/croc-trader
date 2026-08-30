"""System status and strategy configuration inspection tools for Croc-Trader MCP."""

import logging
from typing import Any

from flask import current_app
from mcp.server import MCPServer

from ...const import Strategies
from ...database.repositories.market import MarketRepository
from ...database.repositories.trade import TradeRepository
from ...database.session import DatabaseSession
from ...routes.views.dependencies import _get_database_path

logger = logging.getLogger(__name__)


def get_system_health() -> dict[str, Any]:
    """Returns diagnostic health status for the Croc-Trader system."""
    try:
        signals_path = _get_database_path("signals")
        stocks_path = _get_database_path("stocks")
        trading_path = _get_database_path("trading")

        latest_price_update = None
        if stocks_path.exists():
            stocks_session = DatabaseSession(str(stocks_path), read_only=True)
            market_repo = MarketRepository(stocks_session)
            latest_price_update = market_repo.get_latest_updated_at()

        latest_trade_update = None
        if signals_path.exists():
            signals_session = DatabaseSession(str(signals_path), read_only=True)
            trade_repo = TradeRepository(signals_session)
            latest_trade_update = trade_repo.get_latest_updated_at()

        return {
            "status": "healthy",
            "databases": {
                "signals_db": {
                    "exists": signals_path.exists(),
                    "size_bytes": signals_path.stat().st_size
                    if signals_path.exists()
                    else 0,
                    "latest_update": latest_trade_update,
                },
                "stocks_db": {
                    "exists": stocks_path.exists(),
                    "size_bytes": stocks_path.stat().st_size
                    if stocks_path.exists()
                    else 0,
                    "latest_quote_update": latest_price_update,
                },
                "trading_db": {
                    "exists": trading_path.exists(),
                    "size_bytes": trading_path.stat().st_size
                    if trading_path.exists()
                    else 0,
                },
            },
            "scheduler_running": "scheduler" in current_app.extensions,
        }
    except Exception as error:
        logger.error("Error retrieving system health: %s", error)
        return {"status": "error", "error": str(error)}


def get_strategy_list() -> list[dict[str, Any]]:
    """Returns list of strategies with their allocation settings."""
    try:
        config = current_app.config.get("APP_CONFIG")
        portfolio_config = config.app.portfolio if config else None

        strategies_info = []
        for strategy in Strategies:
            budget = (
                portfolio_config.get_budget(strategy.value) if portfolio_config else 0.0
            )
            risk = (
                portfolio_config.get_risk_amount(strategy.value)
                if portfolio_config
                else 0.0
            )
            strategies_info.append(
                {
                    "enum_name": strategy.name,
                    "canonical_identifier": strategy.value,
                    "budget": budget,
                    "risk_amount": risk,
                }
            )
        return strategies_info
    except Exception as error:
        logger.error("Error retrieving strategy list: %s", error)
        return []


def register(server: MCPServer) -> None:
    """Registers system health and strategy registry tools on the MCP server."""
    server.tool(
        name="get_system_health",
        description=(
            "Checks status of SQLite databases (stocks.db, signals.db, trading.db), "
            "data freshness, latest price quote timestamp, and scheduler configuration."
        ),
    )(get_system_health)

    server.tool(
        name="get_strategy_list",
        description=(
            "Returns all canonical trading strategies registered in the system "
            "with their configured allocation budgets and risk amounts."
        ),
    )(get_strategy_list)
