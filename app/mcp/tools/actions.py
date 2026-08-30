"""Control and execution tools for triggering Croc-Trader workflows via MCP."""

import logging
from typing import Any, cast

from flask import current_app
from mcp.server import MCPServer

from ...const import STRATEGY_ALIASES, Strategies
from ...database.session import DatabaseSession
from ...services.backfill_engine import run_strategy_backfill
from ...tasks import run_daily_eod_pipeline

logger = logging.getLogger(__name__)


def _resolve_strategy_enum(strategy_name: str) -> Strategies | None:
    """Resolves a raw strategy string to its canonical Strategies Enum."""
    raw_name = strategy_name.lower().strip()
    canonical_name = raw_name.replace("-", "_")
    resolved = STRATEGY_ALIASES.get(raw_name) or STRATEGY_ALIASES.get(canonical_name)
    if resolved:
        return resolved
    for item in Strategies:
        if (
            item.value.lower().replace("-", "_") == canonical_name
            or item.name.lower() == canonical_name
        ):
            return item
    return None


def trigger_screener(
    strategy_name: str | None = None,
    lookback_days: int = 0,
) -> dict[str, Any]:
    """Runs the screener engine.

    Args:
        strategy_name: Optional strategy to run exclusively (e.g. 'dip_buyer', 'two_percent').
        lookback_days: Lookback offset in trading days (0 for latest market data).
    """
    try:
        screener_engine = current_app.extensions.get("screener_engine")
        if not screener_engine:
            return {"status": "error", "message": "ScreenerEngine not initialized"}

        stats = screener_engine.run_all(
            days=lookback_days,
            strategy_filter=strategy_name,
        )
        return {"status": "success", "hits_per_strategy": stats}
    except Exception as error:
        logger.exception("Error executing screener via MCP: %s", error)
        return {"status": "error", "message": str(error)}


def trigger_single_symbol_debug(
    strategy_name: str,
    symbol: str,
) -> dict[str, Any]:
    """Runs single-symbol debug analysis.

    Args:
        strategy_name: Name of strategy (e.g. 'dip_buyer', 'turnover_timing').
        symbol: Stock ticker symbol (e.g. 'AAPL').
    """
    try:
        strategy_enum = _resolve_strategy_enum(strategy_name)
        if not strategy_enum:
            return {
                "status": "error",
                "message": f"Strategy '{strategy_name}' not found",
            }

        screener_engine = current_app.extensions.get("screener_engine")
        if not screener_engine:
            return {"status": "error", "message": "ScreenerEngine not initialized"}

        strategy = screener_engine.get_strategy(strategy_enum)
        if not strategy:
            return {
                "status": "error",
                "message": f"Strategy '{strategy_enum.name}' not found in engine",
            }

        if not hasattr(strategy, "analyze_single_symbol"):
            return {
                "status": "error",
                "message": f"Strategy '{strategy_name}' does not implement single-symbol analysis",
            }

        result = strategy.analyze_single_symbol(symbol.upper().strip())
        return {"status": "success", "analysis": result}
    except Exception as error:
        logger.exception("Error debugging symbol %s: %s", symbol, error)
        return {"status": "error", "message": str(error)}


def trigger_order_generation() -> dict[str, Any]:
    """Triggers the TradeManager daily order generation."""
    try:
        trade_manager = current_app.extensions.get("trade_manager")
        if not trade_manager:
            return {"status": "error", "message": "TradeManager not initialized"}

        order_file_path = trade_manager.generate_daily_orders()
        if order_file_path:
            return {
                "status": "success",
                "orders_generated": True,
                "order_file": str(order_file_path),
            }
        return {
            "status": "success",
            "orders_generated": False,
            "message": "No new orders generated (no actions required).",
        }
    except Exception as error:
        logger.exception("Error generating orders: %s", error)
        return {"status": "error", "message": str(error)}


def trigger_eod_pipeline() -> dict[str, Any]:
    """Runs the entire synchronous EOD batch pipeline."""
    try:
        app_instance = cast(Any, current_app)._get_current_object()
        summary = run_daily_eod_pipeline(app_instance)
        return {"status": "success", "pipeline_summary": summary}
    except Exception as error:
        logger.exception("Error running EOD pipeline: %s", error)
        return {"status": "error", "message": str(error)}


def trigger_strategy_backfill(
    strategy_name: str,
    start_date: str = "2025-01-01",
    end_date: str | None = None,
    budget: float = 10000.0,
    clear_existing: bool = True,
) -> dict[str, Any]:
    """Executes a strategy historical backfill run.

    Args:
        strategy_name: Strategy identifier (e.g. 'tgim', 'bridge_scout', 'bounce_bandit').
        start_date: Simulation start date (YYYY-MM-DD).
        end_date: Optional simulation end date.
        budget: Strategy allocation budget amount.
        clear_existing: Whether to clear previous backtest trades for this strategy.
    """
    try:
        config = current_app.config.get("APP_CONFIG")
        if not config:
            return {"status": "error", "message": "Configuration missing"}

        canonical_strategy = strategy_name.lower().replace("-", "_")
        stocks_session = DatabaseSession(str(config.get_path("stocks")))
        signals_session = DatabaseSession(str(config.get_path("signals")))

        result = run_strategy_backfill(
            stocks_session=stocks_session,
            signals_session=signals_session,
            strategy_name=canonical_strategy,
            start_date=start_date,
            end_date=end_date,
            budget=budget,
            clear_existing=clear_existing,
        )
        return {"status": "success", "result": result}
    except Exception as error:
        logger.exception("Error running strategy backfill: %s", error)
        return {"status": "error", "message": str(error)}


def register(server: MCPServer) -> None:
    """Registers execution and workflow trigger tools on the MCP server."""
    server.tool(
        name="trigger_screener",
        description=(
            "Executes screening for all active strategies or a specific named strategy. "
            "Saves any discovered setup candidates to signals.db."
        ),
    )(trigger_screener)

    server.tool(
        name="trigger_single_symbol_debug",
        description=(
            "Runs a detailed rule evaluation and debug inspection for a single symbol "
            "against a specific screener strategy (e.g. DipBuyer, TurnoverTiming)."
        ),
    )(trigger_single_symbol_debug)

    server.tool(
        name="trigger_order_generation",
        description=(
            "Generates daily bracket orders for all active positions and new screener setups, "
            "and exports the CSV order file to data/orders/."
        ),
    )(trigger_order_generation)

    server.tool(
        name="trigger_eod_pipeline",
        description=(
            "Runs the complete daily End-of-Day (EOD) pipeline sequentially: "
            "TradeManager -> Screener Engine -> Order Generation -> Cache Pre-warming."
        ),
    )(trigger_eod_pipeline)

    server.tool(
        name="trigger_strategy_backfill",
        description=(
            "Runs a historical simulation backfill for a given strategy and saves "
            "simulated trades to signals.db."
        ),
    )(trigger_strategy_backfill)
