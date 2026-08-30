"""Market price and historical quote tools for Croc-Trader MCP."""

import logging
from typing import Any

from mcp.server import MCPServer

from ...database.repositories.market import MarketRepository
from ...database.session import DatabaseSession
from ...routes.views.dependencies import _get_database_path

logger = logging.getLogger(__name__)


def _get_market_repository() -> MarketRepository:
    """Instantiates a read-only MarketRepository."""
    stocks_db = str(_get_database_path("stocks"))
    session = DatabaseSession(stocks_db, read_only=True)
    return MarketRepository(session)


def get_price_history(
    symbol: str,
    days: int = 60,
) -> list[dict[str, Any]]:
    """Returns historical OHLCV price bars.

    Args:
        symbol: Ticker symbol (e.g. 'AAPL', 'QQQ').
        days: Lookback days limit (default 60, max 500).
    """
    try:
        effective_days = max(1, min(days, 500))
        repo = _get_market_repository()
        sql = """
            SELECT symbol, date, open, high, low, close, volume, provider
            FROM market_prices
            WHERE symbol = ? AND timeframe = '1D'
            ORDER BY date DESC
            LIMIT ?
        """
        rows = repo.fetch_all(sql, (symbol.upper().strip(), effective_days))
        return [dict(r) for r in reversed(rows)]
    except Exception as error:
        logger.error("Error fetching price history for %s: %s", symbol, error)
        return []


def get_latest_prices(symbols: list[str]) -> dict[str, float | None]:
    """Returns mapping of ticker symbols to their latest closing price.

    Args:
        symbols: List of equity ticker symbols.
    """
    try:
        repo = _get_market_repository()
        results: dict[str, float | None] = {}
        for sym in symbols:
            clean_sym = sym.strip().upper()
            price = repo.get_latest_price(clean_sym)
            results[clean_sym] = round(price, 4) if price is not None else None
        return results
    except Exception as error:
        logger.error("Error fetching latest prices: %s", error)
        return {}


def get_symbol_universe() -> dict[str, Any]:
    """Returns tracked and ignored symbols."""
    try:
        repo = _get_market_repository()
        known = repo.get_all_known_symbols()
        ignored = repo.get_ignored_symbols()
        return {
            "total_tracked_symbols": len(known),
            "tracked_symbols_sample": known[:50],
            "ignored_symbols_count": len(ignored),
            "ignored_symbols": sorted(ignored),
        }
    except Exception as error:
        logger.error("Error fetching symbol universe: %s", error)
        return {"error": str(error)}


def register(server: MCPServer) -> None:
    """Registers market data inspection tools on the MCP server."""
    server.tool(
        name="get_price_history",
        description=(
            "Fetches historical daily OHLCV candlestick quotes for a given symbol."
        ),
    )(get_price_history)

    server.tool(
        name="get_latest_prices",
        description=(
            "Fetches the most recent daily closing price for one or more symbols."
        ),
    )(get_latest_prices)

    server.tool(
        name="get_symbol_universe",
        description=(
            "Returns the universe of tracked stock symbols and the blacklist of ignored symbols."
        ),
    )(get_symbol_universe)
