from unittest.mock import MagicMock

import pandas as pd

from app.services.trade_manager.view_service import TradeViewData, TradeViewService


def test_get_portfolio_summary_empty() -> None:
    """Verifies portfolio summary calculations when no active trades exist."""
    service = TradeViewService(
        trade_repository=MagicMock(),
        market_repository=MagicMock(),
    )
    summary = service.get_portfolio_summary([])
    assert summary["invested"] == 0.0
    assert summary["open_pnl"] == 0.0
    assert summary["open_pnl_5d_change"] == 0.0
    assert summary["count"] == 0


def test_get_portfolio_summary_5d_change() -> None:
    """Verifies 5-day Open PnL change calculation for older and newly opened active trades."""
    reference_date = pd.Timestamp("2026-07-21")

    # Mock market repository history dataframe
    history_data = pd.DataFrame(
        [
            # AAPL history: 6 trading days
            {"symbol": "AAPL", "date": "2026-07-10", "close": 100.0},
            {"symbol": "AAPL", "date": "2026-07-13", "close": 102.0},
            {"symbol": "AAPL", "date": "2026-07-14", "close": 104.0},
            {"symbol": "AAPL", "date": "2026-07-15", "close": 106.0},
            {"symbol": "AAPL", "date": "2026-07-16", "close": 108.0},
            {"symbol": "AAPL", "date": "2026-07-17", "close": 110.0},
            {"symbol": "AAPL", "date": "2026-07-20", "close": 115.0},
            # MSFT history
            {"symbol": "MSFT", "date": "2026-07-10", "close": 200.0},
            {"symbol": "MSFT", "date": "2026-07-20", "close": 210.0},
        ]
    )

    market_repo_mock = MagicMock()
    market_repo_mock.get_batch_history_raw.return_value = history_data

    service = TradeViewService(
        trade_repository=MagicMock(),
        market_repository=market_repo_mock,
    )

    trade_aapl: TradeViewData = {
        "id": 1,
        "symbol": "AAPL",
        "strategy": "CrocSetup",
        "version": None,
        "status": "ACTIVE",
        "entry_date": "2026-07-01",
        "exit_date": None,
        "entry_price": 95.0,
        "exit_price": 0.0,
        "current_price": 115.0,
        "initial_size": 10.0,
        "current_size": 10.0,
        "current_stop_loss": 90.0,
        "current_target": 120.0,
        "budget": 1000.0,
        "signal_context": None,
        "exit_reason": None,
        "stop_loss": 90.0,
        "take_profit": 120.0,
        "display_entry": "2026-07-01",
        "display_exit": "-",
        "days_held": 15,
        "unrealized_pnl": 200.0,  # (115 - 95) * 10
        "realized_pnl": 0.0,
        "pnl_percentage": 21.05,
        "is_critical": False,
        "progress": 83.3,
        "display_size": 10.0,
        "sparkline": "",
        "max_days": None,
        "context": {"direction": "long"},
        "tws_status": None,
        "tws_orders": [],
    }

    # MSFT entered after 5 days ago (e.g. 2026-07-18 > 2026-07-10)
    trade_msft: TradeViewData = {
        "id": 2,
        "symbol": "MSFT",
        "strategy": "DipBuyer",
        "version": None,
        "status": "ACTIVE",
        "entry_date": "2026-07-18",
        "exit_date": None,
        "entry_price": 205.0,
        "exit_price": 0.0,
        "current_price": 210.0,
        "initial_size": 5.0,
        "current_size": 5.0,
        "current_stop_loss": 200.0,
        "current_target": 220.0,
        "budget": 1000.0,
        "signal_context": None,
        "exit_reason": None,
        "stop_loss": 200.0,
        "take_profit": 220.0,
        "display_entry": "2026-07-18",
        "display_exit": "-",
        "days_held": 2,
        "unrealized_pnl": 25.0,  # (210 - 205) * 5
        "realized_pnl": 0.0,
        "pnl_percentage": 2.44,
        "is_critical": False,
        "progress": 50.0,
        "display_size": 5.0,
        "sparkline": "",
        "max_days": None,
        "context": {"direction": "long"},
        "tws_status": None,
        "tws_orders": [],
    }

    summary = service.get_portfolio_summary(
        [trade_aapl, trade_msft], reference_date=reference_date
    )

    # AAPL 1D change (previous trading day 2026-07-17 close 110.0 to current 115.0): (115 - 110) * 10 = +50.0
    # MSFT entered 2026-07-18 > past date 2026-07-10 -> 1D change is unrealized_pnl = +25.0
    # Expected total_open_pnl_5d_change = 50.0 + 25.0 = 75.0
    assert summary["open_pnl"] == 225.0
    assert summary["open_pnl_5d_change"] == 75.0
    assert summary["count"] == 2
