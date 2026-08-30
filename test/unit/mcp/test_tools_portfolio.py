"""Unit tests for MCP portfolio inspection tools."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from flask import Flask

from app.mcp.tools.portfolio import (
    get_active_positions,
    get_portfolio_summary,
    get_trade_detail,
    get_trade_history,
)


@pytest.fixture
def mock_app(tmp_path: Path) -> Flask:
    """Configures test Flask application context with temporary SQLite database."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    signals_db = str(tmp_path / "signals.db")
    stocks_db = str(tmp_path / "stocks.db")
    trading_db = str(tmp_path / "trading.db")

    # Initialize test schema in signals.db
    with sqlite3.connect(signals_db) as conn:
        conn.execute(
            """
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                status TEXT DEFAULT 'CREATED',
                initial_size REAL DEFAULT 0,
                current_size REAL DEFAULT 0,
                entry_price REAL,
                entry_date TIMESTAMP,
                current_price REAL,
                current_stop_loss REAL,
                current_target REAL,
                avg_exit_price REAL,
                realized_pnl REAL DEFAULT 0,
                exit_price REAL,
                exit_date TIMESTAMP,
                exit_reason TEXT,
                signal_context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE trade_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER,
                event_type TEXT,
                old_value TEXT,
                new_value TEXT,
                reason TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Insert sample active and closed trades
        conn.execute(
            """
            INSERT INTO trades (id, symbol, strategy, status, current_size, entry_price, current_price, current_stop_loss, current_target, realized_pnl)
            VALUES (1, 'AAPL', 'dip_buyer', 'ACTIVE', 10, 150.0, 155.0, 145.0, 160.0, 0.0),
                   (2, 'MSFT', 'tgim', 'CLOSED', 5, 300.0, 310.0, 290.0, 320.0, 50.0)
            """
        )
        conn.execute(
            """
            INSERT INTO trade_logs (trade_id, event_type, old_value, new_value, reason)
            VALUES (1, 'ENTRY', '0', '10', 'DipBuyer signal filled')
            """
        )

    # Initialize stocks.db and trading.db dummy tables
    with sqlite3.connect(stocks_db) as conn:
        conn.execute(
            "CREATE TABLE market_prices (symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER, provider TEXT, timeframe TEXT, updated_at TIMESTAMP, PRIMARY KEY (symbol, date, timeframe, provider))"
        )
    with sqlite3.connect(trading_db) as conn:
        conn.execute(
            "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, trade_group_id TEXT, account_id TEXT, symbol TEXT, action TEXT, quantity REAL, order_type TEXT, target_price REAL, strategy_name TEXT, status TEXT)"
        )
        conn.execute(
            "CREATE TABLE trades_settlement (account_id TEXT, trade_group_id TEXT, avg_entry_price REAL, avg_exit_price REAL, price_diff_slippage REAL, total_commissions REAL, net_pnl REAL, PRIMARY KEY (account_id, trade_group_id))"
        )

    mock_config = MagicMock()
    mock_config.get_db_path.side_effect = lambda name: {
        "signals": signals_db,
        "stocks": stocks_db,
        "trading": trading_db,
    }.get(name, f"/mock/{name}.db")
    app.config["APP_CONFIG"] = mock_config

    return app


def test_get_active_positions(mock_app: Flask) -> None:
    """Verifies get_active_positions returns active trade records."""
    with mock_app.app_context():
        positions = get_active_positions()
        assert len(positions) >= 1
        aapl = next(p for p in positions if p["symbol"] == "AAPL")
        assert aapl["status"] == "ACTIVE"
        assert aapl["strategy"] == "dip_buyer"
        assert aapl["current_size"] == 10


def test_get_portfolio_summary(mock_app: Flask) -> None:
    """Verifies get_portfolio_summary calculates invested amounts and active counts."""
    with mock_app.app_context():
        summary = get_portfolio_summary()
        assert "active_positions_count" in summary
        assert summary["active_positions_count"] >= 1
        assert "strategy_breakdown" in summary


def test_get_trade_history(mock_app: Flask) -> None:
    """Verifies get_trade_history returns closed trades."""
    with mock_app.app_context():
        history = get_trade_history()
        assert len(history) >= 1
        msft = next(p for p in history if p["symbol"] == "MSFT")
        assert msft["status"] == "CLOSED"
        assert msft["realized_pnl"] == 50.0


def test_get_trade_detail_with_audit_logs(mock_app: Flask) -> None:
    """Verifies get_trade_detail fetches trade details along with trade logs."""
    with mock_app.app_context():
        detail = get_trade_detail(trade_id=1)
        assert detail["id"] == 1
        assert detail["symbol"] == "AAPL"
        assert "audit_logs" in detail
        assert len(detail["audit_logs"]) == 1
        assert detail["audit_logs"][0]["event_type"] == "ENTRY"


def test_get_trade_detail_not_found(mock_app: Flask) -> None:
    """Verifies get_trade_detail returns error when trade does not exist."""
    with mock_app.app_context():
        detail = get_trade_detail(trade_id=9999)
        assert "error" in detail
