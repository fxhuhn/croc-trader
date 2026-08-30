"""Unit tests for MCP system health and strategy registry tools."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from flask import Flask

from app.mcp.tools.system import get_strategy_list, get_system_health


@pytest.fixture
def system_app(tmp_path: Path) -> Flask:
    """Configures test Flask app fixture for system health testing."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    signals_db = str(tmp_path / "signals.db")
    stocks_db = str(tmp_path / "stocks.db")
    trading_db = str(tmp_path / "trading.db")

    for db_path in (signals_db, stocks_db, trading_db):
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE dummy (id INT)")

    mock_config = MagicMock()
    mock_config.get_db_path.side_effect = lambda name: {
        "signals": signals_db,
        "stocks": stocks_db,
        "trading": trading_db,
    }.get(name, f"/mock/{name}.db")
    mock_config.app.portfolio.get_budget.return_value = 6000.0
    mock_config.app.portfolio.get_risk_amount.return_value = 100.0
    app.config["APP_CONFIG"] = mock_config

    return app


def test_get_system_health(system_app: Flask) -> None:
    """Verifies get_system_health reports database existence and scheduler state."""
    with system_app.app_context():
        health = get_system_health()
        assert health["status"] == "healthy"
        assert health["databases"]["signals_db"]["exists"] is True
        assert health["databases"]["stocks_db"]["exists"] is True


def test_get_strategy_list(system_app: Flask) -> None:
    """Verifies get_strategy_list returns all strategies with budget info."""
    with system_app.app_context():
        strategies = get_strategy_list()
        assert len(strategies) > 0
        names = [s["canonical_identifier"] for s in strategies]
        assert "dip_buyer" in names
        assert "tgim" in names
