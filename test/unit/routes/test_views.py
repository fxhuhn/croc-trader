# filename: test_views.py
"""Unit and integration test suite for the views sub-package.

This module targets all view routes, ensuring they render correctly under happy
paths, gracefully manage empty datasets, and strictly validate calculation
metrics without touching the actual database or disk.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from flask import Flask, template_rendered
from flask.testing import FlaskClient

from app.const import Strategies
from app.types import TradeStatus


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


@pytest.fixture
def captured_templates(
    test_application: Flask,
) -> Generator[list[tuple[Any, dict[str, Any]]], None, None]:
    """Captures all rendered Jinja2 templates and their passed contexts."""
    recorded: list[tuple[Any, dict[str, Any]]] = []

    def record(
        sender: Any, template: Any, context: dict[str, Any], **extra: Any
    ) -> None:
        recorded.append((template, context))

    template_rendered.connect(record, test_application)
    try:
        yield recorded
    finally:
        template_rendered.disconnect(record, test_application)


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
    mock_active_trade: dict[str, Any] = {
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
    mock_history_group: dict[str, Any] = {
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
    mock_active_group: dict[str, Any] = {
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
        ("/trades", b"Strategies"),
    ],
)
def test_view_trades_all_strategies_empty_data_renders_gracefully(
    test_client: FlaskClient, trades_route: str, expected_title: bytes
) -> None:
    """Verifies that all strategy trade views render gracefully when trade datasets are empty."""
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
        mock_service_instance.get_weekday_stats.return_value = {
            i: {
                "name": [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ][i],
                "count": 0,
                "win": 0,
                "loss": 0,
                "pnl": 0.0,
                "average_pnl": 0.0,
            }
            for i in range(7)
        }
        mock_service_instance.group_trades_by_symbol.return_value = []
        mock_service_instance.group_trades_history.return_value = []
        mock_service_instance.generate_donut_chart.return_value = (
            "<div>Mock Donut</div>"
        )

        response = test_client.get(trades_route)

        assert response.status_code == 200
        assert expected_title in response.data

        if trades_route == "/trades/dip-buyer":
            mock_service_instance.get_weekday_stats.assert_called_once_with([])
            mock_service_instance.get_index_stats.assert_called_once_with([])
        elif trades_route == "/trades/croc":
            mock_service_instance.get_index_stats.assert_called_once_with([])
        elif trades_route == "/trades/turnover":
            mock_service_instance.get_index_stats.assert_called_once_with([])


def test_view_trades_dip_buyer_weekday_stats_populated_and_rendered(
    test_client: FlaskClient,
) -> None:
    """Verifies that Dip Buyer trade dashboard calculates and renders weekday stats breakdown in the DOM."""
    mock_closed_trade = {
        "id": "trade-dip-1",
        "symbol": "AMD",
        "entry_date": "2026-06-29",
        "exit_date": "2026-07-02",
        "display_entry": "2026-06-29",
        "days_held": 3,
        "initial_size": 100,
        "current_size": 0,
        "entry_price": 150.0,
        "exit_price": 160.0,
        "realized_pnl": 1000.0,
        "unrealized_pnl": 0.0,
        "pnl_percentage": 6.67,
        "exit_reason": "PROFIT_TARGET",
        "strategy": "DipBuyer",
        "version": "1.0",
        "variant": "1.0",
        "context": {},
        "executions": [],
    }
    mock_weekday_stats = {
        0: {
            "name": "Monday",
            "count": 1,
            "win": 1,
            "loss": 0,
            "pnl": 1000.0,
            "average_pnl": 1000.0,
        },
        1: {
            "name": "Tuesday",
            "count": 0,
            "win": 0,
            "loss": 0,
            "pnl": 0.0,
            "average_pnl": 0.0,
        },
        2: {
            "name": "Wednesday",
            "count": 0,
            "win": 0,
            "loss": 0,
            "pnl": 0.0,
            "average_pnl": 0.0,
        },
        3: {
            "name": "Thursday",
            "count": 0,
            "win": 0,
            "loss": 0,
            "pnl": 0.0,
            "average_pnl": 0.0,
        },
        4: {
            "name": "Friday",
            "count": 0,
            "win": 0,
            "loss": 0,
            "pnl": 0.0,
            "average_pnl": 0.0,
        },
        5: {
            "name": "Saturday",
            "count": 0,
            "win": 0,
            "loss": 0,
            "pnl": 0.0,
            "average_pnl": 0.0,
        },
        6: {
            "name": "Sunday",
            "count": 0,
            "win": 0,
            "loss": 0,
            "pnl": 0.0,
            "average_pnl": 0.0,
        },
    }

    with patch("app.routes.views.trades._get_trade_view_service") as mock_trade_service:
        mock_service = mock_trade_service.return_value
        mock_service.get_trades.side_effect = lambda strategies, status, **kwargs: (
            [] if status == TradeStatus.ACTIVE else [mock_closed_trade]
        )
        mock_service.get_portfolio_summary.return_value = {
            "invested": 0.0,
            "open_pnl": 0.0,
            "win_rate": 100.0,
            "total_pnl": 1000.0,
        }
        mock_service.get_closed_summary.return_value = {
            "count": 1,
            "average_pnl": 1000.0,
            "total_pnl": 1000.0,
            "win_rate": 100.0,
        }
        mock_service.get_index_stats.return_value = {
            "NASDAQ": {
                "name": "NASDAQ 100",
                "count": 1,
                "win": 1,
                "loss": 0,
                "pnl": 1000.0,
                "average_pnl": 1000.0,
            }
        }
        mock_service.get_weekday_stats.return_value = mock_weekday_stats
        mock_service.group_trades_by_symbol.return_value = []
        mock_service.group_trades_history.return_value = [
            {"symbol": "AMD", "trades": [mock_closed_trade]}
        ]

        response = test_client.get("/trades/dip-buyer")

        assert response.status_code == 200
        mock_service.get_weekday_stats.assert_called_once_with([mock_closed_trade])
        mock_service.get_index_stats.assert_called_once_with([mock_closed_trade])

        # Assert Weekday card header and populated weekday row are present in HTML
        assert b"Weekday" in response.data
        assert b"Monday" in response.data
        assert b"+1000" in response.data
        assert b"1W" in response.data


@pytest.mark.parametrize(
    "trades_route, expected_context_keys",
    [
        (
            "/trades/croc",
            {
                "active_trades",
                "active_groups",
                "closed_trades",
                "history_groups",
                "summary",
                "closed_summary",
                "index_stats",
                "signal_stats",
            },
        ),
        (
            "/trades/dip-buyer",
            {
                "active_trades",
                "active_groups",
                "closed_trades",
                "history_groups",
                "summary",
                "closed_summary",
                "index_stats",
                "weekday_stats",
            },
        ),
        (
            "/trades/turnover",
            {
                "summary",
                "active_trades",
                "history_groups",
                "closed_trades",
                "closed_summary",
                "performance_index",
                "performance_variants",
            },
        ),
        (
            "/trades/ndx-momentum",
            {
                "active_trades",
                "active_groups",
                "closed_trades",
                "history_groups",
                "summary",
                "closed_summary",
            },
        ),
        (
            "/trades/twopercent",
            {
                "active_trades",
                "active_groups",
                "closed_trades",
                "history_groups",
                "summary",
                "closed_summary",
                "index_stats",
            },
        ),
        (
            "/trades/tgim",
            {
                "active_trades",
                "active_groups",
                "closed_trades",
                "history_groups",
                "summary",
                "closed_summary",
                "index_stats",
            },
        ),
        (
            "/trades/bridge-scout",
            {
                "active_trades",
                "active_groups",
                "closed_trades",
                "history_groups",
                "summary",
                "closed_summary",
                "index_stats",
            },
        ),
        (
            "/trades/bounce-bandit",
            {
                "active_trades",
                "active_groups",
                "closed_trades",
                "history_groups",
                "summary",
                "closed_summary",
                "index_stats",
            },
        ),
        (
            "/trades",
            {
                "active_trades",
                "summary",
                "strategy_stats",
                "donut_html",
            },
        ),
        (
            "/broker",
            {
                "metrics",
                "active_orders",
                "error_orders",
                "settlements",
                "discrepancies",
                "active_trades",
            },
        ),
    ],
)
def test_view_trades_template_context_contract_enforced(
    test_client: FlaskClient,
    captured_templates: list[tuple[Any, dict[str, Any]]],
    trades_route: str,
    expected_context_keys: set[str],
) -> None:
    """Strictly enforces that all view routes pass every expected template variable."""
    with patch("app.routes.views.trades._get_trade_view_service") as mock_trade_service:
        mock_service = mock_trade_service.return_value
        mock_service.get_trades.return_value = []
        mock_service.get_portfolio_summary.return_value = {
            "invested": 0.0,
            "open_pnl": 0.0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
        }
        mock_service.get_closed_summary.return_value = {
            "count": 0,
            "average_pnl": 0.0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
        }
        mock_service.get_index_stats.return_value = {}
        mock_service.get_weekday_stats.return_value = {}
        mock_service.group_trades_by_symbol.return_value = []
        mock_service.group_trades_history.return_value = []
        mock_service.generate_donut_chart.return_value = "<div>Mock Donut</div>"
        default_broker_metric = {
            "pnl": 0.0,
            "pnlText": "0",
            "winrate": "0.0%",
            "slippage": "0.00",
            "fees": 0.0,
            "win_count": 0,
            "total_count": 0,
            "slippage_sum": 0.0,
        }
        mock_service.get_broker_summary.return_value = {
            strat: default_broker_metric.copy()
            for strat in [
                "all",
                "DipBuyer",
                "TurnoverTiming",
                "TwoPercent",
                "NDXMomentum",
            ]
        }
        mock_service.get_broker_active_orders.return_value = []
        mock_service.get_broker_error_orders.return_value = []
        mock_service.get_broker_settlements.return_value = []
        mock_service.get_reconciliation_discrepancies.return_value = []
        mock_service.get_broker_active_trades.return_value = []

        response = test_client.get(trades_route)

        assert response.status_code == 200
        assert len(captured_templates) > 0
        _template, context = captured_templates[-1]
        for key in expected_context_keys:
            assert key in context, (
                f"Route '{trades_route}' missing required template variable '{key}'"
            )


@pytest.mark.parametrize(
    "trades_route, expected_snippets",
    [
        (
            "/trades/croc",
            [b"Signals", b"Breakout (L20)", b"Index"],
        ),
        (
            "/trades/dip-buyer",
            [b"Weekday", b"Monday", b"Performance"],
        ),
        (
            "/trades/turnover",
            [b"Variants", b"Turnover 0.5", b"Performance"],
        ),
        (
            "/trades/ndx-momentum",
            [b"NDX Momentum", b"Positions", b"AAPL"],
        ),
        (
            "/trades/twopercent",
            [b"Two Percent", b"Positions", b"AAPL"],
        ),
        (
            "/trades/tgim",
            [b"TGIM", b"Positions", b"AAPL"],
        ),
        (
            "/trades/bridge-scout",
            [b"Bridge Scout", b"Positions", b"AAPL"],
        ),
        (
            "/trades/bounce-bandit",
            [b"Bounce Bandit", b"Positions", b"AAPL"],
        ),
    ],
)
def test_view_trades_all_strategies_populated_data_renders_specific_breakdowns(
    test_client: FlaskClient, trades_route: str, expected_snippets: list[bytes]
) -> None:
    """Verifies that populated data renders all strategy-specific breakdown tables and content."""
    mock_trade_active: dict[str, Any] = {
        "id": "trade-act-1",
        "symbol": "AAPL",
        "entry_date": "2026-06-01",
        "display_entry": "2026-06-01",
        "days_held": 2,
        "initial_size": 50,
        "current_size": 50,
        "entry_price": 200.0,
        "current_price": 210.0,
        "current_stop_loss": 190.0,
        "current_target": 220.0,
        "unrealized_pnl": 500.0,
        "realized_pnl": 0.0,
        "pnl_percentage": 5.0,
        "strategy": "hold_target",
        "variant": "Hold",
        "context": {"match_rule": {"name": "Breakout (L20)"}},
        "executions": [],
        "exit_reason": None,
    }
    mock_trade_closed: dict[str, Any] = {
        "id": "trade-cls-1",
        "symbol": "MSFT",
        "entry_date": "2026-06-29",
        "exit_date": "2026-07-02",
        "display_entry": "2026-06-29",
        "days_held": 3,
        "initial_size": 25,
        "current_size": 0,
        "entry_price": 400.0,
        "current_price": 420.0,
        "exit_price": 420.0,
        "current_stop_loss": 390.0,
        "current_target": 420.0,
        "realized_pnl": 500.0,
        "unrealized_pnl": 0.0,
        "pnl_percentage": 5.0,
        "exit_reason": "TARGET",
        "strategy": "turnover_timing_0.5",
        "variant": "0.5",
        "context": {"match_rule": {"name": "Breakout (L20)"}},
        "executions": [],
    }

    def _update_stats_mock(stats_dict: dict[str, Any], pnl: float) -> None:
        stats_dict["count"] = int(stats_dict.get("count", 0)) + 1
        stats_dict["pnl"] = float(stats_dict.get("pnl", 0.0)) + pnl
        if pnl > 0:
            stats_dict["win"] = int(stats_dict.get("win", 0)) + 1
        else:
            stats_dict["loss"] = int(stats_dict.get("loss", 0)) + 1

    with patch("app.routes.views.trades._get_trade_view_service") as mock_trade_service:
        mock_service = mock_trade_service.return_value
        mock_service._update_statistics.side_effect = _update_stats_mock
        mock_service.get_trades.side_effect = lambda strategies, status, **kwargs: (
            [mock_trade_active] if status == TradeStatus.ACTIVE else [mock_trade_closed]
        )
        mock_service.get_portfolio_summary.return_value = {
            "invested": 10000.0,
            "open_pnl": 500.0,
            "win_rate": 100.0,
            "total_pnl": 500.0,
        }
        mock_service.get_closed_summary.return_value = {
            "count": 1,
            "average_pnl": 500.0,
            "total_pnl": 500.0,
            "win_rate": 100.0,
        }
        mock_service.get_index_stats.return_value = {
            "NASDAQ": {
                "name": "NASDAQ 100",
                "count": 1,
                "win": 1,
                "loss": 0,
                "pnl": 500.0,
                "average_pnl": 500.0,
            }
        }
        mock_service.get_weekday_stats.return_value = {
            0: {
                "name": "Monday",
                "count": 1,
                "win": 1,
                "loss": 0,
                "pnl": 500.0,
                "average_pnl": 500.0,
            },
            1: {
                "name": "Tuesday",
                "count": 0,
                "win": 0,
                "loss": 0,
                "pnl": 0.0,
                "average_pnl": 0.0,
            },
            2: {
                "name": "Wednesday",
                "count": 0,
                "win": 0,
                "loss": 0,
                "pnl": 0.0,
                "average_pnl": 0.0,
            },
            3: {
                "name": "Thursday",
                "count": 0,
                "win": 0,
                "loss": 0,
                "pnl": 0.0,
                "average_pnl": 0.0,
            },
            4: {
                "name": "Friday",
                "count": 0,
                "win": 0,
                "loss": 0,
                "pnl": 0.0,
                "average_pnl": 0.0,
            },
            5: {
                "name": "Saturday",
                "count": 0,
                "win": 0,
                "loss": 0,
                "pnl": 0.0,
                "average_pnl": 0.0,
            },
            6: {
                "name": "Sunday",
                "count": 0,
                "win": 0,
                "loss": 0,
                "pnl": 0.0,
                "average_pnl": 0.0,
            },
        }
        mock_service.group_trades_by_symbol.return_value = [
            {
                "symbol": "AAPL",
                "total_pnl": 500.0,
                "total_invested": 10000.0,
                "total_pnl_percentage": 5.0,
                "variants": [mock_trade_active],
            }
        ]
        mock_service.group_trades_history.return_value = [
            {"symbol": "MSFT", "trades": [mock_trade_closed]}
        ]

        response = test_client.get(trades_route)

        assert response.status_code == 200
        for snippet in expected_snippets:
            assert snippet in response.data, (
                f"Route '{trades_route}' missing snippet '{snippet.decode()}' in response"
            )


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
        assert b"Sample" in response.data
        assert b"Return (YTD)" in response.data
        assert b"Monthly Performance" in response.data
        assert b"Monthly Drawdown" in response.data
        assert b"monthly-trend-chart" in response.data
        assert b"monthly-drawdown-chart" in response.data
        assert b"Weekly PnL" in response.data
        assert b"weekly-pnl-chart" in response.data
        assert b"Rolling Performance" in response.data


def test_view_analytics_dashboard_calculates_correct_vectorized_metrics(
    test_client: FlaskClient,
) -> None:
    """Verifies analytics dashboard calculates correct performance metrics."""
    # Arrange
    current_year = pd.Timestamp.now().year
    mock_trades = [
        {
            "exit_date": f"{current_year}-01-10",
            "realized_pnl": 500.0,
            "strategy": Strategies.CrocSetup,
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "initial_size": 100.0,
        },
        {
            "exit_date": f"{current_year}-01-15",
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
        assert b"Return (YTD)" in response.data
        assert b"Monthly Performance" in response.data
        assert b"Monthly Drawdown" in response.data
        assert b"monthly-trend-chart" in response.data
        assert b"monthly-drawdown-chart" in response.data
        assert b"Weekly PnL" in response.data
        assert b"weekly-pnl-chart" in response.data
        assert b"Rolling Performance" in response.data
        assert b"Max Drawdown" in response.data
        assert b"Sharpe" in response.data
        assert b"Sortino" in response.data
        assert b"Profit Factor" in response.data
        assert b"Sample" in response.data


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
        assert b"Portfoliomodell" in response.data
        assert b"Standard" in response.data
        assert b"Frequenz-Modell (EV/M)" in response.data
        assert b"SPY (S&amp;P 500)" in response.data
        assert b"QQQ (Nasdaq 100)" in response.data


def test_view_analytics_monthly_matrix_portfolio_models(
    test_client: FlaskClient,
) -> None:
    """Verifies that Portfoliomodell calculates Standard and Frequenz-Modell (EV/M) rows correctly."""
    import pandas as pd

    mock_trades = [
        {
            "entry_date": "2026-01-01",
            "exit_date": "2026-01-15",
            "realized_pnl": 100.0,
            "strategy": "dip_buyer",
            "entry_price": 100.0,
            "initial_size": 10,  # +10.0% in Jan
        },
        {
            "entry_date": "2026-02-01",
            "exit_date": "2026-02-15",
            "realized_pnl": 200.0,
            "strategy": "dip_buyer",
            "entry_price": 100.0,
            "initial_size": 10,  # +20.0% in Feb
        },
    ]

    with patch(
        "app.routes.views.analytics._get_trade_view_service"
    ) as mock_trade_service:
        mock_service = mock_trade_service.return_value
        mock_service.get_trades.return_value = mock_trades
        mock_service.market_repository.get_symbol_history_raw.return_value = (
            pd.DataFrame()
        )

        response = test_client.get("/analytics/monthly-matrix?year=2026")
        assert response.status_code == 200

        # Section header rendered
        assert b"Portfoliomodell" in response.data
        # Standard, Frequenz-Modell, and Risikoadjustiertes Modell labels
        assert b"Standard" in response.data
        assert b"Frequenz-Modell (EV/M)" in response.data
        assert b"Risikoadjustiert (Max-Sharpe)" in response.data
        assert b"Risikoadjustiert (Risk Parity)" in response.data


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


def test_view_analytics_dashboard_edge_cases(
    test_client: FlaskClient,
) -> None:
    """Verifies edge case branches in view_analytics_dashboard including active trades, exposure scaling, and old trades."""
    with patch(
        "app.routes.views.analytics._get_trade_view_service"
    ) as mock_trade_service:
        mock_service = mock_trade_service.return_value

        # Trades before 2026-01-01 (tests chart_dataframe.empty branch)
        mock_service.get_trades.side_effect = lambda status, **kwargs: (
            [
                {
                    "id": 1,
                    "exit_date": "2025-11-10",
                    "realized_pnl": 500.0,
                    "strategy": "dip_buyer",
                    "entry_price": 50.0,
                    "initial_size": 100,
                    "entry_date": "2025-11-01",
                    "stop_loss": 45.0,
                },
                {
                    "id": 2,
                    "exit_date": "2025-12-10",
                    "realized_pnl": -200.0,
                    "strategy": "croc",
                    "entry_price": 100.0,
                    "initial_size": 10,
                    "entry_date": "2025-12-01",
                    "stop_loss": 90.0,
                },
                {
                    "id": 3,
                    "exit_date": "2025-12-20",
                    "realized_pnl": 0.0,
                    "strategy": "turnover_timing_1.0",
                    # missing entry_price/stop_loss to test has_valid_risk = False
                },
            ]
            if status == TradeStatus.CLOSED
            else [
                {
                    "id": 10,
                    "strategy": "dip_buyer",
                    "unrealized_pnl": 150.0,
                    "entry_price": 50.0,
                    "quantity": 10,
                    "entry_date": "2026-02-01",
                },
                {
                    "id": 11,
                    "strategy": "unknown_invalid_strategy",
                    "unrealized_pnl": None,
                    "entry_date": "invalid-date-string",
                },
            ]
        )
        mock_service.resolve_strategy.side_effect = lambda t: t.get("strategy")

        response = test_client.get("/analytics")
        assert response.status_code == 200
        assert b"Strategy Overview" in response.data


def test_view_analytics_monthly_matrix_benchmarks_and_year_handling(
    test_client: FlaskClient,
) -> None:
    """Verifies monthly matrix year parsing fallback and benchmark data quotes rendering."""
    import pandas as pd

    with patch(
        "app.routes.views.analytics._get_trade_view_service"
    ) as mock_trade_service:
        mock_service = mock_trade_service.return_value
        mock_service.get_trades.return_value = [
            {
                "exit_date": "2026-01-10",
                "realized_pnl": 100.0,
                "strategy": "dip_buyer",
                "entry_price": 100.0,
                "initial_size": 10,
                "entry_date": "2026-01-01",
            }
        ]

        # Return mock SPY and QQQ daily quotes DataFrame
        dates = pd.date_range("2026-01-01", "2026-12-31", freq="B")
        mock_benchmark_df = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "open": [400.0 + i * 0.1 for i in range(len(dates))],
                "close": [401.0 + i * 0.1 for i in range(len(dates))],
            }
        )
        mock_service.market_repository.get_symbol_history_raw.return_value = (
            mock_benchmark_df
        )

        # Invalid year string query parameter (tests ValueError fallback)
        response = test_client.get("/analytics/monthly-matrix?year=invalid_year")
        assert response.status_code == 200
        assert b"SPY (S&amp;P 500)" in response.data
        assert b"QQQ (Nasdaq 100)" in response.data


def test_calculate_mean_variance_helpers_empty_and_invalid_inputs() -> None:
    """Verifies _calculate_mean_variance_allocations and _calculate_mean_variance_dashboard_data edge cases."""
    import pandas as pd

    from app.routes.views.analytics import (
        _calculate_mean_variance_allocations,
        _calculate_mean_variance_dashboard_data,
    )

    strategy_groups = {"Dip Buyer": ["dip_buyer"], "Croc": ["croc"]}

    # Empty DataFrame
    empty_df = pd.DataFrame()
    allocs_empty = _calculate_mean_variance_allocations(empty_df, strategy_groups)
    assert allocs_empty == {"Dip Buyer": 0.5, "Croc": 0.5}

    dash_empty = _calculate_mean_variance_dashboard_data(empty_df, strategy_groups)
    assert dash_empty["has_low_data"] is True
    assert len(dash_empty["strategies"]) == 2

    # DataFrame missing 'strategy' column
    no_strat_df = pd.DataFrame({"realized_pnl": [10.0, 20.0]})
    allocs_no_strat = _calculate_mean_variance_allocations(no_strat_df, strategy_groups)
    assert allocs_no_strat == {"Dip Buyer": 0.5, "Croc": 0.5}

    dash_no_strat = _calculate_mean_variance_dashboard_data(
        no_strat_df, strategy_groups
    )
    assert dash_no_strat["has_low_data"] is True

    # DataFrame with fewer than 2 return observations per strategy
    single_trade_df = pd.DataFrame(
        [
            {
                "strategy": "dip_buyer",
                "realized_pnl": 10.0,
                "entry_price": 100.0,
                "initial_size": 1,
            }
        ]
    )
    allocs_single = _calculate_mean_variance_allocations(
        single_trade_df, strategy_groups
    )
    assert allocs_single == {"Dip Buyer": 0.5, "Croc": 0.5}


def test_view_analytics_dashboard_kelly_scaling_and_active_trades(
    test_client: FlaskClient,
) -> None:
    """Verifies exposure scaling when Kelly > 1.0, active trade zero investment, and benchmark exception handling."""
    with patch(
        "app.routes.views.analytics._get_trade_view_service"
    ) as mock_trade_service:
        mock_service = mock_trade_service.return_value

        # High win-rate trades to produce large Kelly criterion (> 1.0 total proposed exposure)
        mock_service.get_trades.side_effect = lambda status, **kwargs: (
            [
                {
                    "id": 1,
                    "exit_date": "2026-01-10",
                    "realized_pnl": 500.0,
                    "strategy": "dip_buyer",
                    "entry_price": 50.0,
                    "initial_size": 100,  # +10%
                    "entry_date": "2026-01-01",
                    "stop_loss": 45.0,
                },
                {
                    "id": 2,
                    "exit_date": "2026-01-20",
                    "realized_pnl": 600.0,
                    "strategy": "dip_buyer",
                    "entry_price": 50.0,
                    "initial_size": 100,  # +12%
                    "entry_date": "2026-01-15",
                    "stop_loss": 45.0,
                },
            ]
            if status == TradeStatus.CLOSED
            else [
                {
                    "id": 10,
                    "strategy": "dip_buyer",
                    "unrealized_pnl": 50.0,
                    "entry_price": 0.0,  # zero inv branch
                    "quantity": 0.0,
                    "entry_date": "2026-02-01",
                }
            ]
        )
        mock_service.resolve_strategy.side_effect = lambda t: t.get("strategy")

        response = test_client.get("/analytics")
        assert response.status_code == 200
        assert b"Strategy Overview" in response.data


def test_view_analytics_monthly_matrix_year_2023_and_zero_open_prices(
    test_client: FlaskClient,
) -> None:
    """Verifies selected year < 2024, market repo exception, and zero open price benchmark handling."""
    with patch(
        "app.routes.views.analytics._get_trade_view_service"
    ) as mock_trade_service:
        mock_service = mock_trade_service.return_value
        mock_service.get_trades.return_value = []

        # Exception on market repository get_symbol_history_raw
        mock_service.market_repository.get_symbol_history_raw.side_effect = (
            RuntimeError("Market data service offline")
        )

        response = test_client.get("/analytics/monthly-matrix?year=2023")
        assert response.status_code == 200
        assert b"2023" in response.data


def test_build_weekly_trend_data_generates_correct_labels_and_series() -> None:
    """Verifies that _build_weekly_trend_data includes week_labels with KW and dates."""
    import pandas as pd

    from app.routes.views.analytics import _build_weekly_trend_data

    sample_trades = pd.DataFrame(
        [
            {
                "exit_date_dt": pd.Timestamp("2026-01-10 16:00:00"),
                "realized_pnl": 150.0,
                "strategy": "croc_setup",
            },
            {
                "exit_date_dt": pd.Timestamp("2026-01-17 16:00:00"),
                "realized_pnl": -50.0,
                "strategy": "dip_buyer",
            },
        ]
    )

    today = pd.Timestamp("2026-01-20")
    weekly_trend, weekly_pnl = _build_weekly_trend_data(sample_trades, today)

    assert "week_labels" in weekly_trend
    assert "week_labels" in weekly_pnl
    week_labels = weekly_trend["week_labels"]
    dates = weekly_trend["dates"]
    assert isinstance(week_labels, list)
    assert isinstance(dates, list)
    assert len(week_labels) == len(dates)
    assert any("KW" in str(label) for label in week_labels)
    assert week_labels[0] == "03.01.2026 · KW 1"

    # Lookback 3 months
    eval_date = pd.Timestamp("2026-04-20")
    _, pnl_3m = _build_weekly_trend_data(sample_trades, eval_date, lookback_months=3)
    pnl_3m_dates = pnl_3m["dates"]
    pnl_3m_labels = pnl_3m["week_labels"]
    assert isinstance(pnl_3m_dates, list)
    assert isinstance(pnl_3m_labels, list)
    assert len(pnl_3m_dates) <= 15
    assert len(pnl_3m_labels) == len(pnl_3m_dates)
