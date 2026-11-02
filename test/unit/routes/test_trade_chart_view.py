"""Unit tests for the trade chart popout route and view service context."""

from collections.abc import Generator
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from flask import Flask
from flask.testing import FlaskClient

from app.services.trade_manager.view_service import (
    TradeViewService,
    _calculate_chart_date_range,
    _snap_to_trading_day,
)


@pytest.fixture
def test_application() -> Generator[Flask]:
    """Provides a configured Flask application instance for testing the chart route."""
    from app import create_app

    application_instance = create_app()
    application_instance.config["TESTING"] = True
    application_instance.config["APP_CONFIG"] = MagicMock()
    application_instance.config["APP_CONFIG"].app.security.whitelist = ["127.0.0.1"]
    application_instance.config["APP_CONFIG"].app.security.mode = "block"
    application_instance.config["APP_CONFIG"].get_db_path.return_value = ":memory:"

    yield application_instance


@pytest.fixture
def test_client(test_application: Flask) -> FlaskClient:
    """Provides a test client for the configured Flask application."""
    return test_application.test_client()


def test_snap_to_trading_day() -> None:
    """Verifies date snapping to the nearest trading bar."""
    available_dates = ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"]

    # Exact match
    assert _snap_to_trading_day("2026-10-13", available_dates) == "2026-10-13"

    # Weekend / off-market date snaps to next available trading day
    assert _snap_to_trading_day("2026-10-11", available_dates) == "2026-10-12"

    # After last available date snaps to the latest available day
    assert _snap_to_trading_day("2026-10-20", available_dates) == "2026-10-15"

    # Empty list
    assert _snap_to_trading_day("2026-10-13", []) is None


def test_calculate_chart_date_range() -> None:
    """Verifies lookback start and lookahead end date calculations."""
    start_date, end_date = _calculate_chart_date_range("2026-10-10", "2026-10-20")
    assert start_date < "2026-10-10"
    assert end_date > "2026-10-20"

    # Fallbacks on missing or invalid dates
    fallback_start, fallback_end = _calculate_chart_date_range(None, None)
    assert fallback_start == "2020-01-01"
    assert fallback_end > "2020-01-01"


def test_get_trade_chart_context_not_found() -> None:
    """Verifies that non-existent trade returns None."""
    trade_repository = MagicMock()
    trade_repository.get_trade.return_value = None

    service = TradeViewService(
        trade_repository=trade_repository,
        market_repository=MagicMock(),
        broker_repository=MagicMock(),
    )
    context = service.get_trade_chart_context(999)
    assert context is None


def test_get_trade_chart_context_happy_path() -> None:
    """Verifies that valid trade with price candles produces complete chart context."""
    trade_repository = MagicMock()
    trade_repository.get_trade.return_value = {
        "id": 42,
        "symbol": "AAPL",
        "strategy": "dip_buyer",
        "status": "CLOSED",
        "entry_date": "2026-10-05 09:30:00",
        "exit_date": "2026-10-15 16:00:00",
        "entry_price": 220.0,
        "exit_price": 230.0,
        "current_price": 230.0,
        "initial_size": 100,
        "current_size": 0,
        "current_target": 235.0,
        "realized_pnl": 1000.0,
        "exit_reason": "TARGET_HIT",
    }

    market_repository = MagicMock()
    sample_df = pd.DataFrame(
        [
            {
                "date": datetime(2026, 9, 20),
                "open": 215.0,
                "high": 218.0,
                "low": 214.0,
                "close": 217.0,
                "volume": 1000000,
            },
            {
                "date": datetime(2026, 10, 5),
                "open": 219.0,
                "high": 222.0,
                "low": 218.0,
                "close": 221.0,
                "volume": 1200000,
            },
            {
                "date": datetime(2026, 10, 15),
                "open": 228.0,
                "high": 231.0,
                "low": 227.0,
                "close": 230.0,
                "volume": 1500000,
            },
        ]
    )
    market_repository.get_symbol_history_raw.return_value = sample_df

    service = TradeViewService(
        trade_repository=trade_repository,
        market_repository=market_repository,
        broker_repository=MagicMock(),
    )

    context = service.get_trade_chart_context(42)
    assert context is not None
    assert context["is_empty"] is False
    assert context["entry_date"] == "2026-10-05"
    assert context["exit_date"] == "2026-10-15"
    assert context["target_price"] == 235.0
    assert "AAPL" in context["trade"]["symbol"]
    assert '"open": 219.0' in context["candles_json"]


def test_get_trade_chart_context_empty_market_data() -> None:
    """Verifies graceful handling when market prices are unavailable."""
    trade_repository = MagicMock()
    trade_repository.get_trade.return_value = {
        "id": 43,
        "symbol": "UNKNOWN",
        "strategy": "dip_buyer",
        "status": "CLOSED",
        "entry_date": "2026-10-05",
        "exit_date": "2026-10-15",
        "entry_price": 50.0,
        "exit_price": 55.0,
    }

    market_repository = MagicMock()
    market_repository.get_symbol_history_raw.return_value = pd.DataFrame()

    service = TradeViewService(
        trade_repository=trade_repository,
        market_repository=market_repository,
        broker_repository=MagicMock(),
    )

    context = service.get_trade_chart_context(43)
    assert context is not None
    assert context["is_empty"] is True
    assert context["candles_json"] == "[]"


def test_view_trade_chart_route_404(test_client: FlaskClient) -> None:
    """Verifies that accessing a non-existent trade returns 404."""
    with patch(
        "app.routes.views.trades._get_trade_view_service"
    ) as mock_service_getter:
        mock_service = MagicMock()
        mock_service.get_trade_chart_context.return_value = None
        mock_service_getter.return_value = mock_service

        response = test_client.get("/trades/99999/chart")
        assert response.status_code == 404


def test_view_trade_chart_route_200(test_client: FlaskClient) -> None:
    """Verifies that accessing an existing trade returns 200 with chart HTML."""
    with patch(
        "app.routes.views.trades._get_trade_view_service"
    ) as mock_service_getter:
        mock_service = MagicMock()
        mock_service.get_trade_chart_context.return_value = {
            "trade": {
                "id": "123",
                "symbol": "NVDA",
                "strategy": "dip_buyer",
                "entry_price": 120.50,
                "exit_price": 135.00,
                "days_held": 10,
                "realized_pnl": 1500.0,
                "pnl_percentage": 12.03,
                "exit_reason": "TARGET_HIT",
            },
            "chart_payload": {
                "candles": [
                    {
                        "time": "2026-10-01",
                        "open": 118.0,
                        "high": 122.0,
                        "low": 117.0,
                        "close": 121.0,
                        "volume": 1000,
                    }
                ],
                "entry_date": "2026-10-01",
                "exit_date": "2026-10-10",
                "entry_price": 120.50,
                "exit_price": 135.00,
                "target_price": 135.0,
                "exit_reason": "TARGET_HIT",
                "is_profitable": True,
            },
            "candles_json": '[{"time":"2026-10-01","open":118.0,"high":122.0,"low":117.0,"close":121.0,"volume":1000}]',
            "entry_date": "2026-10-01",
            "exit_date": "2026-10-10",
            "target_price": 135.0,
            "is_empty": False,
        }
        mock_service_getter.return_value = mock_service

        response = test_client.get("/trades/123/chart")
        assert response.status_code == 200
        content = response.data.decode("utf-8")
        assert "NVDA" in content
        assert "Trade #123" in content
        assert "TradingView Lightweight Charts" in content
