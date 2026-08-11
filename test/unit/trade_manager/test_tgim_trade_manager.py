"""Unit tests for the TGIM trade manager execution strategy."""

from decimal import Decimal

import pandas as pd
import pytest

from app.services.trade_manager.strategies.tgim import (
    TGIMTradeStrategy,
    calculate_tgim_position_quantity,
    evaluate_tgim_exit,
)
from app.types import ExitReason, TradeStatus

# =====================================================================
# Functional Core Unit Tests
# =====================================================================


def test_evaluate_tgim_exit_bar1_take_profit() -> None:
    """Tests evaluate_tgim_exit triggers TAKE_PROFIT on Bar 1 if current > previous."""
    reason = evaluate_tgim_exit(
        bars_held=1,
        current_close=Decimal("504.0"),
        previous_close=Decimal("500.0"),
    )
    assert reason == ExitReason.TAKE_PROFIT


def test_evaluate_tgim_exit_bar1_hold() -> None:
    """Tests evaluate_tgim_exit returns None on Bar 1 if current <= previous."""
    reason = evaluate_tgim_exit(
        bars_held=1,
        current_close=Decimal("497.0"),
        previous_close=Decimal("500.0"),
    )
    assert reason is None


def test_evaluate_tgim_exit_bar2_take_profit() -> None:
    """Tests evaluate_tgim_exit triggers TAKE_PROFIT on Bar 2 if current > previous."""
    reason = evaluate_tgim_exit(
        bars_held=2,
        current_close=Decimal("499.0"),
        previous_close=Decimal("497.0"),
    )
    assert reason == ExitReason.TAKE_PROFIT


def test_evaluate_tgim_exit_bar2_time_stop() -> None:
    """Tests evaluate_tgim_exit triggers TIME_STOP on Bar 2 if current <= previous."""
    reason = evaluate_tgim_exit(
        bars_held=2,
        current_close=Decimal("495.0"),
        previous_close=Decimal("497.0"),
    )
    assert reason == ExitReason.TIME_STOP


def test_calculate_tgim_position_quantity_valid() -> None:
    """Tests calculate_tgim_position_quantity floors share calculation cleanly."""
    qty = calculate_tgim_position_quantity(
        allocated_budget=Decimal("10000.0"),
        entry_price=Decimal("495.0"),
    )
    assert qty == 20  # 10000 / 495 = 20.2020 -> 20 shares


def test_calculate_tgim_position_quantity_invalid() -> None:
    """Tests calculate_tgim_position_quantity handles zero or negative values."""
    assert calculate_tgim_position_quantity(Decimal("0.0"), Decimal("495.0")) == 0
    assert calculate_tgim_position_quantity(Decimal("10000.0"), Decimal("0.0")) == 0


# =====================================================================
# Imperative Shell Integration Tests
# =====================================================================


@pytest.fixture
def trade_strategy() -> TGIMTradeStrategy:
    """Fixture providing a TGIMTradeStrategy instance."""
    return TGIMTradeStrategy()


def test_tgim_check_entry_activates_on_monday_close(
    trade_strategy: TGIMTradeStrategy,
) -> None:
    """Tests that check_entry activates a CREATED trade at Monday MOC."""
    trade = {
        "id": 1,
        "symbol": "SPY",
        "strategy": "tgim",
        "status": TradeStatus.CREATED.value,
        "entry_price": 500.0,
        "entry_date": "2026-07-20",
        "budget": 10000.0,
    }
    candle = pd.Series(
        {
            "date": "2026-07-20",
            "open": 502.0,
            "high": 505.0,
            "low": 498.0,
            "close": 500.0,
        }
    )
    df_history = pd.DataFrame([candle])

    transition = trade_strategy.check_entry(trade, candle, df_history)

    assert transition is not None
    assert transition.updates["status"] == TradeStatus.ACTIVE.value
    assert transition.updates["entry_price"] == 500.0


def test_tgim_check_entry_invalidates_when_close_above_threshold(
    trade_strategy: TGIMTradeStrategy,
) -> None:
    """Tests check_entry invalidates setup when Monday Close > threshold price."""
    trade = {
        "id": 1,
        "symbol": "SPY",
        "strategy": "tgim",
        "status": TradeStatus.CREATED.value,
        "entry_price": 500.0,
        "entry_date": "2026-07-20",
        "budget": 10000.0,
    }
    candle = pd.Series(
        {
            "date": "2026-07-20",
            "open": 502.0,
            "high": 508.0,
            "low": 501.0,
            "close": 505.0,  # > 500.0 threshold
        }
    )
    df_history = pd.DataFrame([candle])

    transition = trade_strategy.check_entry(trade, candle, df_history)

    assert transition is not None
    assert transition.updates["status"] == TradeStatus.INVALID.value
    assert transition.updates["exit_reason"] == ExitReason.INVALIDATED.value
    assert "Missed Entry Window" in transition.reason


def test_tgim_c1exit_on_tuesday_close(
    trade_strategy: TGIMTradeStrategy,
) -> None:
    """Tests c1exit on Tuesday when Tuesday Close > Monday Close."""
    trade = {
        "id": 1,
        "symbol": "SPY",
        "strategy": "tgim",
        "status": TradeStatus.ACTIVE.value,
        "entry_price": 500.0,
        "entry_date": "2026-07-20",
        "budget": 10000.0,
    }
    monday_candle = {"date": pd.Timestamp("2026-07-20"), "close": 500.0}
    tuesday_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-07-21"),
            "open": 501.0,
            "high": 506.0,
            "low": 500.0,
            "close": 504.0,
        }
    )

    df_history = pd.DataFrame([monday_candle, tuesday_candle])

    transition = trade_strategy.manage_active_trade(trade, df_history)

    assert transition is not None
    assert transition.updates["status"] == TradeStatus.CLOSED.value
    assert transition.updates["exit_reason"] == ExitReason.TAKE_PROFIT.value
    assert transition.updates["exit_price"] == 504.0


def test_tgim_holds_on_tuesday_when_close_lower(
    trade_strategy: TGIMTradeStrategy,
) -> None:
    """Tests trade is held on Tuesday when Tuesday Close <= Monday Close."""
    trade = {
        "id": 1,
        "symbol": "SPY",
        "strategy": "tgim",
        "status": TradeStatus.ACTIVE.value,
        "entry_price": 500.0,
        "entry_date": "2026-07-20",
        "budget": 10000.0,
    }
    monday_candle = {"date": pd.Timestamp("2026-07-20"), "close": 500.0}
    tuesday_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-07-21"),
            "open": 499.0,
            "high": 500.0,
            "low": 495.0,
            "close": 497.0,
        }
    )

    df_history = pd.DataFrame([monday_candle, tuesday_candle])

    transition = trade_strategy.manage_active_trade(trade, df_history)

    assert transition is None


def test_tgim_time_exit_on_wednesday_close_if_c1exit_false(
    trade_strategy: TGIMTradeStrategy,
) -> None:
    """Tests TE (Time Exit) on Wednesday when Wednesday Close <= Tuesday Close."""
    trade = {
        "id": 1,
        "symbol": "SPY",
        "strategy": "tgim",
        "status": TradeStatus.ACTIVE.value,
        "entry_price": 500.0,
        "entry_date": "2026-07-20",
        "budget": 10000.0,
    }
    monday_candle = {"date": pd.Timestamp("2026-07-20"), "close": 500.0}
    tuesday_candle = {"date": pd.Timestamp("2026-07-21"), "close": 497.0}
    wednesday_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-07-22"),
            "open": 497.0,
            "high": 498.0,
            "low": 494.0,
            "close": 495.0,
        }
    )

    df_history = pd.DataFrame([monday_candle, tuesday_candle, wednesday_candle])

    transition = trade_strategy.manage_active_trade(trade, df_history)

    assert transition is not None
    assert transition.updates["status"] == TradeStatus.CLOSED.value
    assert transition.updates["exit_reason"] == ExitReason.TIME_STOP.value
    assert transition.updates["exit_price"] == 495.0
