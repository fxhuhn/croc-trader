"""Unit tests for MCP broker and order tools."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from flask import Flask

from app.mcp.tools.orders import (
    get_broker_active_orders,
    get_broker_settlements,
    list_order_csv_files,
)


@pytest.fixture
def orders_app(tmp_path: Path) -> Flask:
    """Configures test Flask app fixture with broker trading mock data."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    trading_db = str(tmp_path / "trading.db")

    with sqlite3.connect(trading_db) as conn:
        conn.execute(
            """
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                perm_id INTEGER,
                parent_id INTEGER,
                trade_group_id TEXT NOT NULL,
                account_id TEXT NOT NULL,
                bracket_role TEXT,
                symbol TEXT NOT NULL,
                sec_type TEXT,
                exchange TEXT,
                action TEXT,
                quantity INTEGER,
                order_type TEXT,
                target_price REAL,
                tif TEXT,
                strategy_name TEXT,
                status TEXT,
                retry_count INTEGER DEFAULT 0,
                transmitted_at TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE trades_settlement (
                account_id TEXT,
                trade_group_id TEXT,
                avg_entry_price REAL,
                avg_exit_price REAL,
                price_diff_slippage REAL,
                total_commissions REAL,
                net_pnl REAL,
                settled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (account_id, trade_group_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO orders (order_id, trade_group_id, account_id, symbol, action, quantity, order_type, target_price, strategy_name, status)
            VALUES (101, '101_DipBuyer_AAPL', 'DU12345', 'AAPL', 'BUY', 10, 'LMT', 150.0, 'dip_buyer', 'Submitted')
            """
        )
        conn.execute(
            """
            INSERT INTO trades_settlement (account_id, trade_group_id, avg_entry_price, avg_exit_price, price_diff_slippage, total_commissions, net_pnl)
            VALUES ('DU12345', '100_TGIM_MSFT', 300.0, 310.0, 0.05, 2.0, 48.0)
            """
        )

    mock_config = MagicMock()
    mock_config.get_db_path.side_effect = lambda name: trading_db
    app.config["APP_CONFIG"] = mock_config

    return app


def test_get_broker_active_orders(orders_app: Flask) -> None:
    """Verifies get_broker_active_orders returns orders from trading.db."""
    with orders_app.app_context():
        orders = get_broker_active_orders()
        assert len(orders) == 1
        assert orders[0]["symbol"] == "AAPL"
        assert orders[0]["status"] == "Submitted"


def test_get_broker_settlements(orders_app: Flask) -> None:
    """Verifies get_broker_settlements returns settlement records."""
    with orders_app.app_context():
        settlements = get_broker_settlements()
        assert len(settlements) == 1
        assert settlements[0]["trade_group_id"] == "100_TGIM_MSFT"
        assert settlements[0]["net_pnl"] == 48.0


def test_list_order_csv_files(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies list_order_csv_files lists files in data/orders."""
    orders_dir = Path("data/orders")
    orders_dir.mkdir(parents=True, exist_ok=True)
    test_file = orders_dir / "orders_2026_08_28.csv"
    test_file.write_text(
        "trade_group_id,symbol\n1_DipBuyer_AAPL,AAPL\n", encoding="utf-8"
    )

    try:
        files = list_order_csv_files()
        file_names = [f["filename"] for f in files]
        assert "orders_2026_08_28.csv" in file_names
    finally:
        if test_file.exists():
            test_file.unlink()
