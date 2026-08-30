"""Unit tests for MCP operational trigger and action tools."""

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from app.mcp.tools.actions import (
    trigger_eod_pipeline,
    trigger_order_generation,
    trigger_screener,
    trigger_single_symbol_debug,
    trigger_strategy_backfill,
)


@pytest.fixture
def actions_app() -> Flask:
    """Configures test Flask app fixture with mock services."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    mock_config = MagicMock()
    mock_config.get_path.side_effect = lambda name: f"/mock/path/{name}.db"
    mock_config.get_db_path.side_effect = lambda name: f"/mock/path/{name}.db"
    app.config["APP_CONFIG"] = mock_config

    return app


def test_trigger_screener_success(actions_app: Flask) -> None:
    """Verifies trigger_screener invokes screener_engine.run_all."""
    mock_engine = MagicMock()
    mock_engine.run_all.return_value = {"dip_buyer": 3}
    actions_app.extensions["screener_engine"] = mock_engine

    with actions_app.app_context():
        res = trigger_screener(strategy_name="dip_buyer", lookback_days=0)
        assert res["status"] == "success"
        assert res["hits_per_strategy"] == {"dip_buyer": 3}
        mock_engine.run_all.assert_called_once_with(days=0, strategy_filter="dip_buyer")


def test_trigger_single_symbol_debug(actions_app: Flask) -> None:
    """Verifies trigger_single_symbol_debug calls analyze_single_symbol on the strategy."""
    mock_strategy = MagicMock()
    mock_strategy.analyze_single_symbol.return_value = {
        "symbol": "AAPL",
        "passed": True,
    }
    mock_engine = MagicMock()
    mock_engine.get_strategy.return_value = mock_strategy
    actions_app.extensions["screener_engine"] = mock_engine

    with actions_app.app_context():
        res = trigger_single_symbol_debug(strategy_name="dip_buyer", symbol="AAPL")
        assert res["status"] == "success"
        assert res["analysis"]["passed"] is True


def test_trigger_order_generation(actions_app: Flask) -> None:
    """Verifies trigger_order_generation invokes trade_manager.generate_daily_orders."""
    mock_tm = MagicMock()
    mock_tm.generate_daily_orders.return_value = "/data/orders/orders_2026_08_28.csv"
    actions_app.extensions["trade_manager"] = mock_tm

    with actions_app.app_context():
        res = trigger_order_generation()
        assert res["status"] == "success"
        assert res["orders_generated"] is True
        assert "orders_2026_08_28.csv" in res["order_file"]


@patch("app.mcp.tools.actions.run_daily_eod_pipeline")
def test_trigger_eod_pipeline(mock_run_pipeline: MagicMock, actions_app: Flask) -> None:
    """Verifies trigger_eod_pipeline invokes the EOD orchestrator."""
    mock_run_pipeline.return_value = {"status": "completed", "orders": 2}

    with actions_app.app_context():
        res = trigger_eod_pipeline()
        assert res["status"] == "success"
        assert res["pipeline_summary"]["status"] == "completed"


@patch("app.mcp.tools.actions.run_strategy_backfill")
def test_trigger_strategy_backfill(
    mock_backfill: MagicMock, actions_app: Flask
) -> None:
    """Verifies trigger_strategy_backfill executes strategy backfill engine."""
    mock_backfill.return_value = {"total_trades": 15, "win_rate": 0.65}

    with actions_app.app_context():
        res = trigger_strategy_backfill(
            strategy_name="tgim",
            start_date="2025-01-01",
            budget=10000.0,
        )
        assert res["status"] == "success"
        assert res["result"]["win_rate"] == 0.65
