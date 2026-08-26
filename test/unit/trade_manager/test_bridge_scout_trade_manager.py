"""Unit tests for BridgeScoutTradeStrategy."""

from decimal import Decimal

import pandas as pd
import pytest

from app.const import Strategies
from app.services.trade_manager.strategies.bridge_scout import (
    BridgeScoutTradeStrategy,
)
from app.types import ExitReason, TradeStatus


@pytest.fixture
def strategy() -> BridgeScoutTradeStrategy:
    return BridgeScoutTradeStrategy()


@pytest.fixture
def sample_trade() -> dict:
    return {
        "id": 101,
        "symbol": "AAPL",
        "strategy": Strategies.BridgeScout,
        "status": "CREATED",
        "entry_price": 150.0,
        "current_size": 10.0,
        "entry_date": "2026-01-15T00:00:00",
        "signal_context": '{"date": "2026-01-15"}',
    }


@pytest.fixture
def sample_history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-01-15",
                "open": 149.0,
                "high": 152.0,
                "low": 148.0,
                "close": 150.0,
                "volume": 1000,
            },
            {
                "date": "2026-02-01",
                "open": 151.0,
                "high": 155.0,
                "low": 150.0,
                "close": 154.0,
                "volume": 1200,
            },
        ]
    )


def test_get_current_parameters(
    strategy: BridgeScoutTradeStrategy, sample_trade: dict
) -> None:
    params = strategy.get_current_parameters(sample_trade)
    assert params is not None
    assert params.stop_loss == 0.0
    assert params.take_profit_1 == 0.0
    assert params.extras["entry_price"] == 150.0
    assert params.extras["current_size"] == 10.0


def test_generate_entry_order(
    strategy: BridgeScoutTradeStrategy,
    sample_trade: dict,
    sample_history: pd.DataFrame,
) -> None:
    # Valid budget
    order = strategy._generate_entry_order(sample_trade, sample_history, budget=1500.0)
    assert order is not None
    assert order.symbol == "AAPL"
    assert order.quantity == 10
    assert order.entry is not None
    assert order.entry.type == "MKT"

    # Budget too small
    assert (
        strategy._generate_entry_order(sample_trade, sample_history, budget=50.0)
        is None
    )

    # Invalid entry price
    invalid_trade = dict(sample_trade, entry_price=0.0)
    assert (
        strategy._generate_entry_order(invalid_trade, sample_history, budget=1500.0)
        is None
    )


def test_generate_exit_order(
    strategy: BridgeScoutTradeStrategy,
    sample_trade: dict,
    sample_history: pd.DataFrame,
) -> None:
    order = strategy._generate_exit_order(sample_trade, sample_history, budget=1000.0)
    assert order is not None
    assert order.symbol == "AAPL"
    assert order.quantity == 10
    assert len(order.exits) > 0
    assert order.exits[0].price == Decimal("154.0")

    # Empty size
    no_size = dict(sample_trade, current_size=0)
    assert strategy._generate_exit_order(no_size, sample_history, budget=1000.0) is None

    # Empty dataframe
    assert (
        strategy._generate_exit_order(sample_trade, pd.DataFrame(), budget=1000.0)
        is None
    )


def test_check_entry(
    strategy: BridgeScoutTradeStrategy,
    sample_trade: dict,
    sample_history: pd.DataFrame,
) -> None:
    candle = sample_history.iloc[
        0
    ]  # close is 150.0, date is 2026-01-15, threshold is 150.0
    transition = strategy.check_entry(sample_trade, candle, sample_history)
    assert transition is not None
    assert transition.updates.get("status") == "ACTIVE"

    invalid_trade = dict(sample_trade, entry_price=0.0)
    assert strategy.check_entry(invalid_trade, candle, sample_history) is None


def test_check_entry_threshold_invalidation(
    strategy: BridgeScoutTradeStrategy,
    sample_trade: dict,
    sample_history: pd.DataFrame,
) -> None:
    """Tests that check_entry invalidates trade when close > threshold."""
    candle = sample_history.iloc[0]  # close is 150.0
    # Threshold is lower than actual close (e.g. 145.0)
    trade_with_low_threshold = dict(
        sample_trade,
        entry_price=145.0,
        signal_context='{"setup_date": "2026-01-15", "req_close_rsi40": 145.0}',
    )
    transition = strategy.check_entry(trade_with_low_threshold, candle, sample_history)
    assert transition is not None
    assert transition.updates.get("status") == TradeStatus.INVALID
    assert "Bridge Scout condition failed" in str(transition.reason)


def test_check_entry_missed_window_invalidation(
    strategy: BridgeScoutTradeStrategy,
    sample_trade: dict,
    sample_history: pd.DataFrame,
) -> None:
    """Tests that check_entry invalidates trade when candle date is after setup date."""
    candle_later = sample_history.iloc[1]  # date is 2026-02-01, setup was 2026-01-15
    transition = strategy.check_entry(sample_trade, candle_later, sample_history)
    assert transition is not None
    assert transition.updates.get("status") == TradeStatus.INVALID
    assert "Missed Entry Window" in str(transition.reason)


def test_check_entry_before_setup_date_returns_none(
    strategy: BridgeScoutTradeStrategy,
    sample_trade: dict,
) -> None:
    """Tests that check_entry returns None when candle date is before setup date."""
    future_trade = dict(
        sample_trade,
        signal_context='{"setup_date": "2026-01-20"}',
    )
    candle_earlier = pd.Series({"date": "2026-01-15", "close": 140.0})
    assert strategy.check_entry(future_trade, candle_earlier, pd.DataFrame()) is None


def test_do_manage_active_trade(
    strategy: BridgeScoutTradeStrategy,
    sample_trade: dict,
    sample_history: pd.DataFrame,
) -> None:
    active_trade = dict(sample_trade, status="ACTIVE", entry_date="2026-01-15")

    # Same month: no exit
    jan_candle = sample_history.iloc[0]
    assert (
        strategy._do_manage_active_trade(
            active_trade, jan_candle, "2026-01-15", sample_history
        )
        is None
    )

    # Next month: exit triggered
    feb_candle = sample_history.iloc[1]
    transition = strategy._do_manage_active_trade(
        active_trade, feb_candle, "2026-02-01", sample_history
    )
    assert transition is not None
    assert transition.updates.get("status") == "CLOSED"
    assert transition.updates.get("exit_reason") == ExitReason.TIME_STOP.value


def test_do_manage_active_trade_missing_entry_date_or_empty_history(
    strategy: BridgeScoutTradeStrategy,
) -> None:
    """Tests _do_manage_active_trade returns None when entry_date missing or history empty."""
    trade_no_date = {"status": "ACTIVE"}
    candle = pd.Series({"date": "2026-02-01", "close": 150.0})
    assert (
        strategy._do_manage_active_trade(
            trade_no_date, candle, "2026-02-01", pd.DataFrame([candle])
        )
        is None
    )
    trade_with_date = {"status": "ACTIVE", "entry_date": "2026-01-15"}
    assert (
        strategy._do_manage_active_trade(
            trade_with_date, candle, "2026-02-01", pd.DataFrame()
        )
        is None
    )
