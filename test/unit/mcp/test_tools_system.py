"""Unit tests for MCP system health and strategy registry tools."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from flask import Flask

from app.mcp.tools.system import get_strategy_list, get_system_health, get_system_logs


@pytest.fixture
def system_app(tmp_path: Path) -> Flask:
    """Configures test Flask app fixture for system health testing."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    signals_db = str(tmp_path / "signals.db")
    stocks_db = str(tmp_path / "stocks.db")
    trading_db = str(tmp_path / "trading.db")
    log_file = str(tmp_path / "test_app.log")

    for db_path in (signals_db, stocks_db, trading_db):
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE dummy (id INT)")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("21:29:16 [INFO] root: Croc-Trader App initialized.\n")
        f.write("21:30:01 [WARNING] app.routes.mcp: Unknown method requested.\n")
        f.write("21:30:05 [ERROR] app.services.market: Failed fetching quote.\n")
        f.write("21:30:10 [DEBUG] app.routes.mcp: MCP tool executed successfully.\n")

    mock_config = MagicMock()
    mock_config.get_db_path.side_effect = lambda name: {
        "signals": signals_db,
        "stocks": stocks_db,
        "trading": trading_db,
    }.get(name, f"/mock/{name}.db")
    mock_config.get_log_path.return_value = log_file
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


def test_get_system_logs_success_and_filtering(system_app: Flask) -> None:
    """Verifies get_system_logs retrieves and filters lines by level and module."""
    with system_app.app_context():
        # 1. All logs
        res_all = get_system_logs(lines=10)
        assert res_all["status"] == "ok"
        assert res_all["exists"] is True
        assert res_all["returned_lines"] == 4

        # 2. Filter by level ERROR
        res_err = get_system_logs(lines=10, level="ERROR")
        assert res_err["returned_lines"] == 1
        assert "Failed fetching quote" in res_err["lines"][0]

        # 3. Filter by module app.routes.mcp
        res_mod = get_system_logs(lines=10, module="app.routes.mcp")
        assert res_mod["returned_lines"] == 2


def test_get_system_logs_file_not_found(system_app: Flask) -> None:
    """Verifies get_system_logs handles non-existent log file gracefully."""
    with system_app.app_context():
        mock_config = system_app.config["APP_CONFIG"]
        mock_config.get_log_path.return_value = "/non/existent/path/app.log"

        res = get_system_logs(lines=10)
        assert res["status"] == "ok"
        assert res["exists"] is False
        assert res["lines"] == []
