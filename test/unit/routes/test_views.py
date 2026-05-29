# filename: test_views.py
"""Unit and integration test suite for the views sub-package.

This module targets all view routes, ensuring they render correctly under happy
paths, gracefully manage empty datasets, and strictly validate calculation
metrics without touching the actual database or disk.
"""

from typing import Generator
from unittest.mock import MagicMock, patch
import pytest
from flask import Flask
from flask.testing import FlaskClient

from app.const import Strategies


@pytest.fixture
def test_application() -> Generator[Flask, None, None]:
    """Provides a configured Flask application instance for testing views."""
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


def test_view_screener_overview_returns_correct_response(
    test_client: FlaskClient,
) -> None:
    """Verifies that the screener overview page renders all strategy cards."""
    # Arrange
    with (
        patch("app.routes.views.screener._get_signal_repository") as mock_signal_repo,
        patch(
            "app.routes.views.screener._get_screener_view_service"
        ) as mock_screener_service,
    ):
        mock_signal_repo.return_value.get_trade_candidates.return_value = []
        mock_screener_service.return_value.get_candidates.return_value = []

        # Act
        response = test_client.get("/screener")

        # Assert
        assert response.status_code == 200
        assert b"Croc Setup" in response.data
        assert b"Dip Buyer" in response.data
        assert b"Turnover Timing" in response.data


@pytest.mark.parametrize(
    "strategy_route, expected_title",
    [
        ("/screener/croc", b"Croc Setup"),
        ("/screener/dip-buyer", b"Dip Buyer"),
        ("/screener/turnover", b"Turnover Signale"),
        ("/screener/twopercent", b"Two Percent"),
        ("/screener/ndx-momentum", b"NDX Momentum"),
    ],
)
def test_view_screener_strategy_specific_routes(
    test_client: FlaskClient, strategy_route: str, expected_title: bytes
) -> None:
    """Verifies that each strategy-specific screener renders correctly."""
    # Arrange
    with patch(
        "app.routes.views.screener._get_screener_view_service"
    ) as mock_screener_service:
        mock_service_instance = mock_screener_service.return_value
        mock_service_instance.get_candidates.return_value = []
        mock_service_instance.get_turnover_candidates.return_value = []

        # Act
        response = test_client.get(strategy_route)

        # Assert
        assert response.status_code == 200
        assert expected_title in response.data


def test_view_trades_overview_returns_correct_response(
    test_client: FlaskClient,
) -> None:
    """Verifies that trades overview dashboard renders summary and donut chart."""
    # Arrange
    with patch("app.routes.views.trades._get_trade_view_service") as mock_trade_service:
        mock_service_instance = mock_trade_service.return_value
        mock_service_instance.get_trades.return_value = []
        mock_service_instance.get_portfolio_summary.return_value = {
            "invested": 0.0,
            "open_pnl": 0.0,
            "win_rate": 0.0,
        }
        mock_service_instance.generate_donut_chart.return_value = (
            "<div>Donut Chart Mock</div>"
        )

        # Act
        response = test_client.get("/trades")

        # Assert
        assert response.status_code == 200
        assert b"Strategies" in response.data


@pytest.mark.parametrize(
    "trades_route, expected_title",
    [
        ("/trades/croc", b"Croc Setup"),
        ("/trades/dip-buyer", b"Dip Buyer"),
        ("/trades/turnover", b"Turnover"),
        ("/trades/ndx-momentum", b"NDX Momentum"),
        ("/trades/twopercent", b"Two Percent"),
    ],
)
def test_view_trades_strategy_specific_routes(
    test_client: FlaskClient, trades_route: str, expected_title: bytes
) -> None:
    """Verifies that each strategy-specific trades dashboard renders correctly."""
    # Arrange
    with patch("app.routes.views.trades._get_trade_view_service") as mock_trade_service:
        mock_service_instance = mock_trade_service.return_value
        mock_service_instance.get_trades.return_value = []
        mock_service_instance.get_portfolio_summary.return_value = {
            "invested": 0.0,
            "open_pnl": 0.0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
        }
        mock_service_instance.get_closed_summary.return_value = {
            "count": 0,
            "average_pnl": 0.0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
        }
        mock_service_instance.get_index_stats.return_value = {}
        mock_service_instance.group_trades_by_symbol.return_value = {}
        mock_service_instance.group_trades_history.return_value = {}

        # Act
        response = test_client.get(trades_route)

        # Assert
        assert response.status_code == 200
        assert expected_title in response.data


def test_view_analytics_dashboard_handles_empty_data_gracefully(
    test_client: FlaskClient,
) -> None:
    """Verifies that the analytics dashboard handles empty trade list gracefully."""
    # Arrange
    with patch(
        "app.routes.views.analytics._get_trade_view_service"
    ) as mock_trade_service:
        mock_service_instance = mock_trade_service.return_value
        mock_service_instance.get_trades.return_value = []

        # Act
        response = test_client.get("/analytics")

        # Assert
        assert b"Total Trades" in response.data


def test_view_analytics_dashboard_calculates_correct_vectorized_metrics(
    test_client: FlaskClient,
) -> None:
    """Verifies analytics dashboard calculates correct performance metrics."""
    # Arrange
    mock_trades = [
        {
            "exit_date": "2026-01-10",
            "realized_pnl": 500.0,
            "strategy": Strategies.CrocSetup,
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "initial_size": 100.0,
        },
        {
            "exit_date": "2026-01-15",
            "realized_pnl": -200.0,
            "strategy": Strategies.CrocSetup,
            "entry_price": 50.0,
            "stop_loss": 48.0,
            "initial_size": 100.0,
        },
    ]

    with patch(
        "app.routes.views.analytics._get_trade_view_service"
    ) as mock_trade_service:
        mock_service_instance = mock_trade_service.return_value
        # Closed and Active
        mock_service_instance.get_trades.side_effect = [mock_trades, []]
        mock_service_instance.resolve_strategy.return_value = Strategies.CrocSetup

        # Act
        response = test_client.get("/analytics")

        # Assert
        assert response.status_code == 200
        # Win rate: 1 win out of 2 trades = 50.0%
        assert b"50.0%" in response.data
        # PnL: 500 - 200 = 300
        assert b"300.0" in response.data or b"300" in response.data


def test_view_analytics_dashboard_handles_corrupted_data_gracefully(
    test_client: FlaskClient,
) -> None:
    """Verifies vectorized metrics handle missing and malformed numeric fields."""
    # Arrange
    mock_trades = [
        {
            "exit_date": "2026-01-10",
            "realized_pnl": 100.0,
            "strategy": Strategies.CrocSetup,
            "entry_price": None,  # Corrupted entry
            "stop_loss": 90.0,
            "initial_size": 10.0,
        },
        {
            "exit_date": "2026-01-12",
            "realized_pnl": -50.0,
            "strategy": Strategies.CrocSetup,
            "entry_price": 10.0,
            "stop_loss": 9.0,
            "initial_size": "NaN",  # Corrupted size
        },
    ]

    with patch(
        "app.routes.views.analytics._get_trade_view_service"
    ) as mock_trade_service:
        mock_service_instance = mock_trade_service.return_value
        mock_service_instance.get_trades.side_effect = [mock_trades, []]
        mock_service_instance.resolve_strategy.return_value = Strategies.CrocSetup

        # Act
        response = test_client.get("/analytics")

        # Assert
        assert response.status_code == 200
        # The view should render correctly without raising TypeError/ValueError
        assert b"Analytics" in response.data


def test_view_backtest_dashboard_handles_missing_run_id_gracefully(
    test_client: FlaskClient,
) -> None:
    """Verifies backtest route handles missing run identifier gracefully."""
    # Arrange
    with (
        patch("app.routes.views.backtest._get_database_path"),
        patch("app.routes.views.backtest._get_backtest_database_path"),
        patch("app.routes.views.backtest.ResultsPersistence") as mock_persistence,
    ):
        mock_persistence.return_value.get_latest_run_id.return_value = None

        # Act
        response = test_client.get("/backtest")

        # Assert
        assert response.status_code == 200
        assert b"No backtest results found." in response.data
