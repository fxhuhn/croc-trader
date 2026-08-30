"""Unit tests for MCP screener and webhook inspection tools."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from flask import Flask

from app.mcp.tools.screener import (
    get_screener_candidates,
    get_turnover_candidates,
    get_webhook_signals,
)


@pytest.fixture
def screener_app(tmp_path: Path) -> Flask:
    """Configures test Flask app fixture with screener mock data."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    signals_db = str(tmp_path / "signals.db")

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
            CREATE TABLE croc (
                symbol TEXT NOT NULL,
                timeframe TEXT,
                signal TEXT,
                timestamp TEXT,
                exchange TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, signal, timestamp)
            )
            """
        )
        # Insert sample candidate in trades
        conn.execute(
            """
            INSERT INTO trades (id, symbol, strategy, status, entry_price, signal_context)
            VALUES (10, 'NVDA', 'dip_buyer', 'CREATED', 120.0, '{"date": "2026-08-28", "setup_score": 85.0}'),
                   (11, 'AMD', 'turnover_timing', 'CREATED', 140.0, '{"date": "2026-08-28", "setup_turnover_sma": 1000000.0}')
            """
        )
        # Insert sample webhook in croc
        conn.execute(
            """
            INSERT INTO croc (symbol, timeframe, signal, timestamp, exchange, data)
            VALUES ('TSLA', '1D', 'BUY', '2026-08-28T10:00:00', 'NASDAQ', '{"rsi": 28.5}')
            """
        )

    mock_config = MagicMock()
    mock_config.get_db_path.side_effect = lambda name: signals_db
    app.config["APP_CONFIG"] = mock_config

    return app


def test_get_screener_candidates(screener_app: Flask) -> None:
    """Verifies get_screener_candidates returns candidate setups."""
    with screener_app.app_context():
        candidates = get_screener_candidates(strategy="dip_buyer")
        assert len(candidates) >= 1
        nvda = next(c for c in candidates if c["symbol"] == "NVDA")
        assert nvda["strategy"] == "dip_buyer"


def test_get_turnover_candidates(screener_app: Flask) -> None:
    """Verifies get_turnover_candidates aggregates and returns candidates."""
    with screener_app.app_context():
        candidates = get_turnover_candidates()
        assert len(candidates) >= 1
        amd = next(c for c in candidates if c["symbol"] == "AMD")
        assert amd["dollar_volume"] == 1000000.0


def test_get_webhook_signals(screener_app: Flask) -> None:
    """Verifies get_webhook_signals returns records from croc table."""
    with screener_app.app_context():
        signals = get_webhook_signals()
        assert len(signals) >= 1
        tsla = next(s for s in signals if s["symbol"] == "TSLA")
        assert tsla["signal"] == "BUY"
        assert isinstance(tsla["data"], dict)
        assert tsla["data"]["rsi"] == 28.5
