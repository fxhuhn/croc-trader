"""Negative Testing & Boundary Value Analysis (BVA) for Strategy Execution.

Verifies that Screener and Trade Manager strategies safely handle extreme market conditions,
zero volume, zero volatility, gap jumps, and simultaneous Stop/Target events without unhandled crashes.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.trade_manager.strategies.dip_buyer import DipBuyerStrategy
from app.services.trade_manager.strategies.hold_target import HoldTargetStrategy
from app.services.trade_manager.strategies.two_percent_strategy import (
    TwoPercentStrategy,
)
from app.types import ExitReason, TradeStatus


@pytest.fixture
def mock_trade_repo() -> MagicMock:
    return MagicMock()


@pytest.mark.tier1
def test_dip_buyer_entry_boundary_zero_range_candle(mock_trade_repo: MagicMock) -> None:
    """BVA: When candle has High == Low == Open == Close (flat line / halt), entry evaluates safely."""
    strategy = DipBuyerStrategy()
    trade = {
        "id": 1,
        "symbol": "FLAT",
        "entry_price": 100.0,
        "signal_context": '{"date": "2026-02-17"}',
        "current_size": 10,
    }
    candle = pd.Series(
        {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "date": pd.Timestamp("2026-02-18"),
        },
        name=pd.Timestamp("2026-02-18"),
    )
    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        result = strategy.check_entry(
            trade, candle, pd.DataFrame([candle]), mock_trade_repo
        )
        assert result is not None
        assert "FILLED" in result


@pytest.mark.tier1
def test_dip_buyer_entry_boundary_huge_gap_down(mock_trade_repo: MagicMock) -> None:
    """BVA: Huge gap down below limit price fills at open price, not at the higher limit price."""
    strategy = DipBuyerStrategy()
    trade = {
        "id": 2,
        "symbol": "GAP",
        "entry_price": 100.0,
        "signal_context": '{"date": "2026-02-17"}',
        "current_size": 10,
    }
    # Limit is 100, but open gaps down directly to 80.0
    candle = pd.Series(
        {
            "open": 80.0,
            "high": 85.0,
            "low": 78.0,
            "close": 82.0,
            "date": pd.Timestamp("2026-02-18"),
        },
        name=pd.Timestamp("2026-02-18"),
    )
    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        result = strategy.check_entry(
            trade, candle, pd.DataFrame([candle]), mock_trade_repo
        )
        assert result is not None
        assert "FILLED" in result
        args, _ = mock_trade_repo.update_trade.call_args
        assert args[1]["entry_price"] == 80.0  # Filled at actual Open price


@pytest.mark.tier1
def test_hold_target_simultaneous_stop_and_target_collision(
    mock_trade_repo: MagicMock,
) -> None:
    """BVA: On an extreme volatility bar touching both Stop-Loss and Target, worst-case exit is prioritized."""
    strategy = HoldTargetStrategy()
    trade = {
        "id": 3,
        "symbol": "VOLA",
        "entry_price": 100.0,
        "current_stop_loss": 90.0,
        "current_target": 120.0,
        "current_size": 10,
        "status": TradeStatus.ACTIVE,
        "entry_date": "2026-02-17",
    }
    # Bar touches High 125 (Target hit) AND Low 85 (Stop hit)
    candle = pd.Series(
        {
            "open": 100.0,
            "high": 125.0,
            "low": 85.0,
            "close": 110.0,
            "date": "2026-02-18",
        },
        name=pd.Timestamp("2026-02-18"),
    )
    df_history = pd.DataFrame([candle])
    result = strategy.manage_active_trade(trade, df_history)
    assert result is not None
    assert result.updates["status"] == TradeStatus.CLOSED
    assert result.updates["exit_reason"] == ExitReason.STOP_LOSS  # Worst-case priority


@pytest.mark.tier1
def test_two_percent_entry_with_zero_size(mock_trade_repo: MagicMock) -> None:
    """BVA: Trade setup with 0 size evaluates safely without crashing."""
    strategy = TwoPercentStrategy()
    trade = {
        "id": 4,
        "symbol": "ZEROSIZE",
        "entry_price": 50.0,
        "current_size": 0,
        "signal_context": '{"date": "2026-02-17"}',
    }
    candle = pd.Series(
        {
            "open": 50.0,
            "high": 52.0,
            "low": 48.0,
            "close": 51.0,
            "date": pd.Timestamp("2026-02-18"),
        },
        name=pd.Timestamp("2026-02-18"),
    )
    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        result = strategy.check_entry(
            trade, candle, pd.DataFrame([candle]), mock_trade_repo
        )
        assert result is not None
