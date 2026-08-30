"""Unit tests for MCP market data tools."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from flask import Flask

from app.mcp.tools.market import (
    get_latest_prices,
    get_price_history,
    get_symbol_universe,
)


@pytest.fixture
def market_app(tmp_path: Path) -> Flask:
    """Configures test Flask app fixture with market mock data."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    stocks_db = str(tmp_path / "stocks.db")

    with sqlite3.connect(stocks_db) as conn:
        conn.execute(
            """
            CREATE TABLE market_prices (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                provider TEXT NOT NULL DEFAULT 'yahoo',
                timeframe TEXT NOT NULL DEFAULT '1D',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date, timeframe, provider)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE ignored_symbols (
                symbol TEXT PRIMARY KEY,
                reason TEXT,
                ignored_since TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Insert historical bars
        conn.execute(
            """
            INSERT INTO market_prices (symbol, date, open, high, low, close, volume)
            VALUES ('AAPL', '2026-08-27', 150.0, 155.0, 149.0, 154.0, 1000000),
                   ('AAPL', '2026-08-28', 154.0, 158.0, 153.0, 157.0, 1200000),
                   ('MSFT', '2026-08-28', 300.0, 305.0, 298.0, 304.0, 800000)
            """
        )
        conn.execute(
            "INSERT INTO ignored_symbols (symbol, reason) VALUES ('BADSYM', 'Delisted')"
        )

    mock_config = MagicMock()
    mock_config.get_db_path.side_effect = lambda name: stocks_db
    app.config["APP_CONFIG"] = mock_config

    return app


def test_get_price_history(market_app: Flask) -> None:
    """Verifies get_price_history returns sorted historical price bars."""
    with market_app.app_context():
        bars = get_price_history("AAPL", days=10)
        assert len(bars) == 2
        assert bars[0]["date"] == "2026-08-27"
        assert bars[1]["date"] == "2026-08-28"
        assert bars[1]["close"] == 157.0


def test_get_latest_prices(market_app: Flask) -> None:
    """Verifies get_latest_prices returns mapping of symbols to prices."""
    with market_app.app_context():
        prices = get_latest_prices(["AAPL", "MSFT", "UNKNOWN"])
        assert prices["AAPL"] == 157.0
        assert prices["MSFT"] == 304.0
        assert prices["UNKNOWN"] is None


def test_get_symbol_universe(market_app: Flask) -> None:
    """Verifies get_symbol_universe lists tracked and ignored symbols."""
    with market_app.app_context():
        universe = get_symbol_universe()
        assert universe["total_tracked_symbols"] == 2
        assert universe["ignored_symbols_count"] == 1
        assert "BADSYM" in universe["ignored_symbols"]
