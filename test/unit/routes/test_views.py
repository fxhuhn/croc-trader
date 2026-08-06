# filename: test_views.py
"""Unit and integration test suite for the views sub-package.

This module targets all view routes, ensuring they render correctly under happy
paths, gracefully manage empty datasets, and strictly validate calculation
metrics without touching the actual database or disk.
"""

from collections.abc import Generator
from typing import Any
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
        ("/trades/tgim", b"TGIM"),
        ("/trades/bridge-scout", b"Bridge Scout"),
        ("/trades/bounce-bandit", b"Bounce Bandit"),
    ],
)
def test_view_trades_strategy_specific_routes(
    test_client: FlaskClient, trades_route: str, expected_title: bytes
) -> None:
    """Verifies that each strategy-specific trades dashboard renders correctly with active and history trades."""
    mock_active_trade = {
        "id": "trade-1",
        "symbol": "AAPL",
        "entry_date": "2026-06-01",
        "display_entry": "2026-06-01",
        "exit_date": None,
        "days_held": 5,
        "initial_size": 100,
        "current_size": 100,
        "entry_price": 150.0,
        "current_price": 155.0,
        "current_stop_loss": 145.0,
        "current_target": 160.0,
        "unrealized_pnl": 500.0,
        "realized_pnl": 0.0,
        "pnl_percentage": 3.33,
        "strategy": "NDXMomentum",
        "version": "1.0",
        "variant": "1.0",
        "context": {},
        "executions": [],
        "exit_reason": None,
    }
    mock_history_group = {
        "symbol": "MSFT",
        "trades": [
            {
                "id": "trade-2",
                "symbol": "MSFT",
                "entry_date": "2026-05-01",
                "exit_date": "2026-05-11",
                "display_entry": "2026-05-01",
                "days_held": 10,
                "initial_size": 50,
                "current_size": 0,
                "entry_price": 400.0,
                "exit_price": 420.0,
                "realized_pnl": 1000.0,
                "unrealized_pnl": 0.0,
                "pnl_percentage": 5.0,
                "exit_reason": "PROFIT_TARGET",
                "strategy": "NDXMomentum",
                "version": "1.0",
                "variant": "1.0",
                "context": {},
                "executions": [],
            }
        ],
    }
    mock_active_group = {
        "symbol": "AAPL",
        "total_pnl": 500.0,
        "total_invested": 15000.0,
        "total_pnl_percentage": 3.33,
        "variants": [mock_active_trade],
    }

    # Arrange
    with patch("app.routes.views.trades._get_trade_view_service") as mock_trade_service:
        mock_service_instance = mock_trade_service.return_value
        mock_service_instance.get_trades.return_value = [mock_active_trade]
        mock_service_instance.get_portfolio_summary.return_value = {
            "invested": 15000.0,
            "open_pnl": 500.0,
            "win_rate": 100.0,
            "total_pnl": 1000.0,
        }
        mock_service_instance.get_closed_summary.return_value = {
            "count": 1,
            "average_pnl": 1000.0,
            "total_pnl": 1000.0,
            "win_rate": 100.0,
        }
        mock_service_instance.get_index_stats.return_value = {
            "NASDAQ": {
                "name": "NASDAQ 100",
                "count": 1,
                "win": 1,
                "loss": 0,
                "pnl": 1000.0,
                "average_pnl": 1000.0,
            }
        }
        mock_service_instance.get_weekday_stats.return_value = {}
        mock_service_instance.group_trades_by_symbol.return_value = [mock_active_group]
        mock_service_instance.group_trades_history.return_value = [mock_history_group]

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


def test_view_analytics_dashboard_calculates_correct_95_percentile_utilization(
    test_client: FlaskClient,
) -> None:
    """Verifies that 95th percentile of active trade concurrency is calculated as whole number."""
    # Arrange
    mock_trades = [
        {
            "entry_date": "2026-01-10 10:00:00",
            "exit_date": "2026-01-12 16:00:00",
            "realized_pnl": 100.0,
            "strategy": Strategies.CrocSetup,
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "initial_size": 100.0,
        },
        {
            "entry_date": "2026-01-11 11:00:00",
            "exit_date": "2026-01-13 15:00:00",
            "realized_pnl": 100.0,
            "strategy": Strategies.CrocSetup,
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "initial_size": 100.0,
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
        # Check that the merged "Open (100% / 95%)" layout exists in the rendered HTML
        assert b"Open (100% / 95%)" in response.data
        # With 2 concurrent trades active, the peak is 2, and the 95th percentile should also round to 2.
        # Format should be '2 / 2'
        assert b"2 / 2" in response.data


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


def test_prepare_active_orders_hierarchical_sorting_and_child_marking() -> None:
    """Verifies that active orders are sorted ascending by order_id, grouped by trade_group_id,
    and child orders are properly flagged depending on whether their parent is open.
    """
    from app.routes.views.trades import _prepare_active_orders

    # Test Case 1: Both Parent (731) and Child (732) are open
    # Test Case 2: Standalone Order (735)
    # Test Case 3: Order (742) whose parent (739) is Filled/missing (Active Position)
    orders: list[dict[str, Any]] = [
        {
            "order_id": 742,
            "parent_id": 739,
            "trade_group_id": "TG_NVDA",
            "symbol": "NVDA",
        },
        {
            "order_id": 735,
            "parent_id": None,
            "trade_group_id": "TG_AAPL",
            "symbol": "AAPL",
        },
        {
            "order_id": 732,
            "parent_id": 731,
            "trade_group_id": "TG_GILD",
            "symbol": "GILD",
        },
        {
            "order_id": 731,
            "parent_id": None,
            "trade_group_id": "TG_GILD",
            "symbol": "GILD",
        },
    ]

    result = _prepare_active_orders(orders)

    # Assert correct order of elements (731, 732, 735, 742)
    assert [o["order_id"] for o in result] == [731, 732, 735, 742]

    # Assert is_child flags:
    # 731 is parent -> is_child=False
    # 732 has parent 731 in open orders -> is_child=True
    # 735 is standalone -> is_child=False
    # 742 parent 739 is NOT in open orders (active position) -> is_child=False
    assert result[0]["is_child"] is False
    assert result[1]["is_child"] is True
    assert result[2]["is_child"] is False
    assert result[3]["is_child"] is False


def test_view_broker_dashboard_renders_dom_elements_and_headers(
    test_client: FlaskClient,
) -> None:
    """Verifies that the broker dashboard renders table wrappers, section headers, and column headers."""
    mock_order = {
        "order_id": 101,
        "symbol": "AAPL",
        "action": "BUY",
        "quantity": 10,
        "status": "Submitted",
        "strategy_name": "DipBuyer",
        "strategy_filter": "DipBuyer",
        "trade_group_id": "TG_1",
        "parent_id": None,
        "is_child": False,
        "target_price": 150.0,
        "order_type": "LMT",
    }

    with patch(
        "app.routes.views.trades._get_trade_view_service"
    ) as mock_broker_service:
        mock_service_instance = mock_broker_service.return_value
        default_metric = {
            "pnl": 0.0,
            "pnlText": "0",
            "winrate": "0.0%",
            "slippage": "0.00",
            "fees": 0.0,
            "win_count": 0,
            "total_count": 0,
            "slippage_sum": 0.0,
        }
        mock_service_instance.get_broker_summary.return_value = {
            strat: default_metric.copy()
            for strat in [
                "all",
                "DipBuyer",
                "TurnoverTiming",
                "TwoPercent",
                "NDXMomentum",
            ]
        }
        mock_service_instance.get_broker_active_trades.return_value = []
        mock_service_instance.get_broker_settlements.return_value = []
        mock_service_instance.broker_repository.get_orders_by_status.return_value = [
            mock_order
        ]

        response = test_client.get("/broker")

        assert response.status_code == 200
        # Section Headers
        assert b"Orders" in response.data
        assert b"Fehler" in response.data
        assert b"Positions" in response.data
        assert b"History" in response.data
        # Desktop Table Wrapper & Desktop Column Headers
        assert b"hidden md:block bg-white rounded-3xl" in response.data
        assert (
            b"Quantity &amp; Direction" in response.data
            or b"Quantity & Direction" in response.data
        )
        assert (
            b"Exit &amp; Target" in response.data or b"Exit & Target" in response.data
        )


def test_view_trades_dip_buyer_history_omits_signal_label(
    test_client: FlaskClient,
) -> None:
    """Verifies that single-strategy dip buyer history omits redundant signal labels."""
    mock_history_trade = {
        "id": "trade-dip-1",
        "symbol": "AMD",
        "entry_date": "2026-06-01",
        "exit_date": "2026-06-05",
        "display_entry": "2026-06-01",
        "days_held": 4,
        "initial_size": 20,
        "current_size": 0,
        "entry_price": 160.0,
        "exit_price": 170.0,
        "realized_pnl": 200.0,
        "unrealized_pnl": 0.0,
        "pnl_percentage": 6.25,
        "exit_reason": "PROFIT_TARGET",
        "strategy": "DipBuyer",
        "version": "1.0",
        "variant": "1.0",
        "context": {},
        "executions": [],
    }

    with patch("app.routes.views.trades._get_trade_view_service") as mock_trade_service:
        mock_service = mock_trade_service.return_value
        mock_service.get_trades.return_value = []
        mock_service.get_portfolio_summary.return_value = {
            "invested": 0,
            "open_pnl": 0,
            "win_rate": 0,
            "total_pnl": 0,
        }
        mock_service.get_closed_summary.return_value = {
            "count": 1,
            "average_pnl": 200.0,
            "total_pnl": 200.0,
            "win_rate": 100.0,
        }
        mock_service.get_index_stats.return_value = {}
        mock_service.get_weekday_stats.return_value = {}
        mock_service.group_trades_by_symbol.return_value = []
        mock_service.group_trades_history.return_value = [
            {"symbol": "AMD", "trades": [mock_history_trade]}
        ]

        response = test_client.get("/trades/dip-buyer")

        assert response.status_code == 200
        # Symbol and date should be present
        assert b"AMD" in response.data
        assert b"2026-06-01" in response.data
        # Signal name should NOT be rendered in history subline when show_signal=false
        # Verify no 'DipBuyer &bull;' or 'DipBuyer •' in history row
        assert b"DipBuyer &bull;" not in response.data
        assert b"DipBuyer \xe2\x80\xa2" not in response.data


def test_view_analytics_monthly_matrix_returns_correct_response(
    test_client: FlaskClient,
) -> None:
    """Verifies that the monthly matrix analytics view renders correctly."""
    with patch(
        "app.routes.views.analytics._get_trade_view_service"
    ) as mock_trade_service:
        mock_service = mock_trade_service.return_value
        mock_service.get_trades.return_value = [
            {
                "exit_date": "2026-03-15",
                "realized_pnl": 500.0,
                "strategy": "dip_buyer",
                "entry_price": 100.0,
                "initial_size": 10,
            }
        ]
        import pandas as pd

        mock_service.market_repository.get_symbol_history_raw.return_value = (
            pd.DataFrame()
        )

        response = test_client.get("/analytics/monthly-matrix?year=2026")

        assert response.status_code == 200
        assert b"Monthlymatrix" in response.data
        assert b"Dip Buyer" in response.data
        assert b"Gesamt Portfolio" in response.data
        assert b"SPY (S&amp;P 500)" in response.data
        assert b"QQQ (Nasdaq 100)" in response.data


def test_view_analytics_monthly_matrix_compounded_return(
    test_client: FlaskClient,
) -> None:
    """Verifies that monthly matrix calculates unweighted average trade returns per month and compounded gesamt return."""
    with patch(
        "app.routes.views.analytics._get_trade_view_service"
    ) as mock_trade_service:
        mock_service = mock_trade_service.return_value
        # Month 1: Trade 1 (+10%), Trade 2 (+30%) -> Unweighted Month 1 = +20.0%
        # Month 2: Trade 3 (-10%) -> Unweighted Month 2 = -10.0%
        # Compounded Gesamt = (1.20 * 0.90 - 1) = +8.0%
        mock_service.get_trades.return_value = [
            {
                "exit_date": "2026-01-10",
                "realized_pnl": 100.0,
                "strategy": "dip_buyer",
                "entry_price": 100.0,
                "initial_size": 10,  # 1000 invested -> +10.0%
            },
            {
                "exit_date": "2026-01-20",
                "realized_pnl": 30.0,
                "strategy": "dip_buyer",
                "entry_price": 10.0,
                "initial_size": 10,  # 100 invested -> +30.0%
            },
            {
                "exit_date": "2026-02-15",
                "realized_pnl": -1000.0,
                "strategy": "dip_buyer",
                "entry_price": 100.0,
                "initial_size": 100,  # 10000 invested -> -10.0%
            },
        ]
        import pandas as pd

        mock_service.market_repository.get_symbol_history_raw.return_value = (
            pd.DataFrame()
        )

        response = test_client.get("/analytics/monthly-matrix?year=2026")
        assert response.status_code == 200
        # Dip Buyer Month 1 cell shows +20.0%
        assert b"+20.0%" in response.data
        # Dip Buyer Compounded Gesamt shows +8.0%
        assert b"+8.0%" in response.data
        # Portfolio Month 1 average across 8 strategies (+20.0% / 8) shows +2.5%
        assert b"+2.5%" in response.data


def test_view_analytics_monthly_matrix_badge_styles(
    test_client: FlaskClient,
) -> None:
    """Verifies that monthly matrix badge styles match the required return threshold tiers."""
    with patch(
        "app.routes.views.analytics._get_trade_view_service"
    ) as mock_trade_service:
        mock_service = mock_trade_service.return_value
        # Month 1: +2.0% (<= 3%)
        # Month 2: +15.0% (3% to 20%)
        # Month 3: +25.0% (> 20%)
        # Month 4: -15.0% (-20% to -3%)
        # Month 5: -25.0% (< -20%)
        mock_service.get_trades.return_value = [
            {
                "exit_date": "2026-01-10",
                "realized_pnl": 20.0,
                "strategy": "dip_buyer",
                "entry_price": 100.0,
                "initial_size": 10,  # +2.0%
            },
            {
                "exit_date": "2026-02-10",
                "realized_pnl": 150.0,
                "strategy": "dip_buyer",
                "entry_price": 100.0,
                "initial_size": 10,  # +15.0%
            },
            {
                "exit_date": "2026-03-10",
                "realized_pnl": 250.0,
                "strategy": "dip_buyer",
                "entry_price": 100.0,
                "initial_size": 10,  # +25.0%
            },
            {
                "exit_date": "2026-04-10",
                "realized_pnl": -150.0,
                "strategy": "dip_buyer",
                "entry_price": 100.0,
                "initial_size": 10,  # -15.0%
            },
            {
                "exit_date": "2026-05-10",
                "realized_pnl": -250.0,
                "strategy": "dip_buyer",
                "entry_price": 100.0,
                "initial_size": 10,  # -25.0%
            },
        ]
        import pandas as pd

        mock_service.market_repository.get_symbol_history_raw.return_value = (
            pd.DataFrame()
        )

        response = test_client.get("/analytics/monthly-matrix?year=2026")
        assert response.status_code == 200

        # Tier 1 (<= 3%): font-semibold
        assert (
            b"bg-emerald-50 text-emerald-700 font-semibold border border-emerald-100"
            in response.data
        )

        # Tier 2 (3% to 20%): font-bold (light bg)
        assert (
            b"bg-emerald-50 text-emerald-700 font-bold border border-emerald-100"
            in response.data
        )
        assert (
            b"bg-rose-50 text-rose-700 font-bold border border-rose-100"
            in response.data
        )

        # Tier 3 (> 20%): dark area (bg-emerald-500 / bg-rose-500)
        assert b"bg-emerald-500 text-white font-bold shadow-sm" in response.data
        assert b"bg-rose-500 text-white font-bold shadow-sm" in response.data
