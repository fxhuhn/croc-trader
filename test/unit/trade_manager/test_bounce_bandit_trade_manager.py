"""Unit tests for the Bounce Bandit trade manager execution strategy."""

import json

import pandas as pd
import pytest

from app.services.trade_manager.strategies.bounce_bandit import (
    BounceBanditTradeStrategy,
)
from app.types import TradeStatus


@pytest.fixture
def trade_strategy() -> BounceBanditTradeStrategy:
    """Fixture providing a BounceBanditTradeStrategy instance."""
    return BounceBanditTradeStrategy()


def test_bounce_bandit_check_entry_prevents_same_day_entry(
    trade_strategy: BounceBanditTradeStrategy,
) -> None:
    """Tests that check_entry returns None on setup day t."""
    trade = {
        "id": 1,
        "symbol": "QQQ",
        "strategy": "bounce_bandit",
        "status": TradeStatus.CREATED.value,
        "entry_price": 500.0,
        "signal_context": json.dumps({"date": "2026-07-20"}),
    }
    setup_day_candle = pd.Series(
        {
            "date": "2026-07-20",
            "open": 502.0,
            "high": 505.0,
            "low": 498.0,
            "close": 500.0,
        }
    )
    df_history = pd.DataFrame([setup_day_candle])

    transition = trade_strategy.check_entry(trade, setup_day_candle, df_history)
    assert transition is None


def test_bounce_bandit_check_entry_activates_on_next_day_open(
    trade_strategy: BounceBanditTradeStrategy,
) -> None:
    """Tests that check_entry executes MOO entry at next day's open (bar t+1)."""
    trade = {
        "id": 1,
        "symbol": "QQQ",
        "strategy": "bounce_bandit",
        "status": TradeStatus.CREATED.value,
        "entry_price": 500.0,
        "budget": 10000.0,
        "signal_context": json.dumps({"date": "2026-07-20"}),
    }

    next_day_candle = pd.Series(
        {
            "date": "2026-07-21",
            "open": 503.50,
            "high": 508.0,
            "low": 501.0,
            "close": 506.0,
        }
    )
    df_history = pd.DataFrame([next_day_candle])

    transition = trade_strategy.check_entry(trade, next_day_candle, df_history)

    assert transition is not None
    assert transition.updates["status"] == TradeStatus.ACTIVE.value
    assert transition.updates["entry_price"] == 503.50


def test_bounce_bandit_exit_triggers_on_sma_8_cross(
    trade_strategy: BounceBanditTradeStrategy,
) -> None:
    """Tests that active trade exits MOC when Close > SMA(8)."""
    trade = {
        "id": 1,
        "symbol": "QQQ",
        "strategy": "bounce_bandit",
        "status": TradeStatus.ACTIVE.value,
        "entry_price": 500.0,
        "entry_date": "2026-07-21",
    }
    # Create 10 days of prices where the last close rises above 8-day SMA
    prices = [500.0, 498.0, 495.0, 492.0, 490.0, 491.0, 493.0, 494.0, 495.0, 520.0]
    dates = pd.date_range("2026-07-10", periods=10, freq="B")

    records = [
        {"date": d, "open": p, "high": p + 2, "low": p - 2, "close": p}
        for d, p in zip(dates, prices, strict=True)
    ]
    df_history = pd.DataFrame(records)

    transition = trade_strategy.manage_active_trade(trade, df_history)

    assert transition is not None
    assert transition.updates["status"] == TradeStatus.CLOSED.value
    assert transition.updates["exit_reason"] in ("SMA", "RSI", "RSI / SMA")
    assert transition.updates["exit_price"] == 520.0


def test_bounce_bandit_exit_triggers_on_rsi_2_overbought(
    trade_strategy: BounceBanditTradeStrategy,
) -> None:
    """Tests that active trade exits MOC when RSI(2) > 75."""
    trade = {
        "id": 1,
        "symbol": "QQQ",
        "strategy": "bounce_bandit",
        "status": TradeStatus.ACTIVE.value,
        "entry_price": 500.0,
        "entry_date": "2026-07-21",
    }
    # Sharp up move causes RSI(2) > 75
    prices = [500.0, 490.0, 485.0, 480.0, 475.0, 470.0, 465.0, 460.0, 470.0, 490.0]
    dates = pd.date_range("2026-07-10", periods=10, freq="B")

    records = [
        {"date": d, "open": p, "high": p + 2, "low": p - 2, "close": p}
        for d, p in zip(dates, prices, strict=True)
    ]
    df_history = pd.DataFrame(records)

    transition = trade_strategy.manage_active_trade(trade, df_history)

    assert transition is not None
    assert transition.updates["status"] == TradeStatus.CLOSED.value
    assert transition.updates["exit_reason"] in ("SMA", "RSI", "RSI / SMA")


def test_bounce_bandit_holds_when_no_exit_condition_met(
    trade_strategy: BounceBanditTradeStrategy,
) -> None:
    """Tests that trade remains ACTIVE when Close <= SMA(8) and RSI(2) <= 75."""
    trade = {
        "id": 1,
        "symbol": "QQQ",
        "strategy": "bounce_bandit",
        "status": TradeStatus.ACTIVE.value,
        "entry_price": 500.0,
        "entry_date": "2026-07-21",
    }
    # Flat / downward drift where SMA(8) is above close and RSI(2) is low
    prices = [520.0, 518.0, 515.0, 512.0, 510.0, 508.0, 506.0, 504.0, 502.0, 500.0]
    dates = pd.date_range("2026-07-10", periods=10, freq="B")

    records = [
        {"date": d, "open": p, "high": p + 2, "low": p - 2, "close": p}
        for d, p in zip(dates, prices, strict=True)
    ]
    df_history = pd.DataFrame(records)

    transition = trade_strategy.manage_active_trade(trade, df_history)

    assert transition is None


def test_bounce_bandit_get_daily_updates_returns_sma_8_and_rsi_2(
    trade_strategy: BounceBanditTradeStrategy,
) -> None:
    """Tests that get_daily_updates calculates and returns sma_8 and rsi_2."""
    trade = {
        "id": 1,
        "symbol": "QQQ",
        "strategy": "bounce_bandit",
        "status": TradeStatus.ACTIVE.value,
        "entry_price": 500.0,
    }
    prices = [520.0, 518.0, 515.0, 512.0, 510.0, 508.0, 506.0, 504.0, 502.0, 500.0]
    dates = pd.date_range("2026-07-10", periods=10, freq="B")

    records = [
        {"date": d, "open": p, "high": p + 2, "low": p - 2, "close": p}
        for d, p in zip(dates, prices, strict=True)
    ]
    df_history = pd.DataFrame(records)

    updates = trade_strategy.get_daily_updates(trade, df_history)

    assert "sma_8" in updates
    assert "rsi_2" in updates
    assert "target" in updates
    assert "target_price" in updates
    assert "required_sma_exit" in updates
    assert "required_rsi_exit" in updates
    assert isinstance(updates["sma_8"], float)
    assert isinstance(updates["rsi_2"], float)
    assert isinstance(updates["target"], float)
    assert updates["target"] <= max(
        updates["required_sma_exit"], updates["required_rsi_exit"]
    )


def test_bounce_bandit_raises_error_on_insufficient_history(
    trade_strategy: BounceBanditTradeStrategy,
) -> None:
    """Tests that BounceBanditTradeStrategy raises ValueError when history is less than 8 candles."""
    trade = {
        "id": 1136,
        "symbol": "QQQ",
        "strategy": "bounce_bandit",
        "status": TradeStatus.ACTIVE.value,
        "entry_price": 690.25,
    }
    # Short history (only 5 candles < 8)
    prices = [680.0, 682.0, 675.0, 661.0, 683.0]
    dates = pd.date_range("2026-07-24", periods=5, freq="B")
    df_short = pd.DataFrame(
        [
            {"date": d, "open": p, "high": p + 2, "low": p - 2, "close": p}
            for d, p in zip(dates, prices, strict=True)
        ]
    )

    with pytest.raises(
        ValueError,
        match="Insufficient price history for Bounce Bandit trade ID 1136",
    ):
        trade_strategy.manage_active_trade(trade, df_short)


def test_resolve_history_start_date_includes_lookback_buffer() -> None:
    """Tests that _resolve_history_start_date subtracts a 60-day lookback buffer from the trade anchor date."""
    from app.services.trade_manager.manager import _resolve_history_start_date

    trade = {
        "id": 1136,
        "symbol": "QQQ",
        "strategy": "bounce_bandit",
        "entry_date": "2026-07-24",
    }
    start_date = _resolve_history_start_date(trade, lookback_days=60)
    assert start_date == "2026-05-25"
