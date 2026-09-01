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


def test_tgim_get_current_parameters(
    trade_strategy: TGIMTradeStrategy,
) -> None:
    """Tests get_current_parameters returns TradeParams with correct extras for Bar 0/1."""
    trade = {"entry_price": 500.0, "current_size": 20}
    params = trade_strategy.get_current_parameters(trade)
    assert params is not None
    assert params.take_profit_1 == 500.0
    assert params.stop_loss == 0.0
    assert params.extras["entry_price"] == 500.0
    assert params.extras["current_size"] == 20.0
    assert params.extras["exit_rule_label"] == "LOC: > 500.00"
    assert params.extras["max_holding_bars"] == 2


def test_tgim_get_current_parameters_bar2_time_exit(
    trade_strategy: TGIMTradeStrategy,
) -> None:
    """Tests get_current_parameters returns Time Exit for Bar 2 (Wednesday)."""
    trade = {
        "entry_price": 500.0,
        "current_size": 20,
        "entry_date": "2026-07-20",
    }
    monday_candle = {"date": pd.Timestamp("2026-07-20"), "close": 500.0}
    tuesday_candle = {"date": pd.Timestamp("2026-07-21"), "close": 497.0}
    wednesday_candle = {"date": pd.Timestamp("2026-07-22"), "close": 495.0}
    df_history = pd.DataFrame([monday_candle, tuesday_candle, wednesday_candle])

    params = trade_strategy.get_current_parameters(trade, df_history)
    assert params is not None
    assert params.take_profit_1 == 0.0
    assert params.stop_loss == 0.0
    assert params.extras["exit_rule_label"] == "Time Exit (Wed MOC)"
    assert params.extras["max_holding_bars"] == 2


def test_tgim_generate_entry_order(
    trade_strategy: TGIMTradeStrategy,
) -> None:
    """Tests _generate_entry_order creates valid MKT entry order and handles invalid inputs."""
    trade = {"symbol": "SPY", "entry_price": 500.0, "budget": 10000.0}
    df_history = pd.DataFrame()
    order = trade_strategy._generate_entry_order(trade, df_history, budget=10000.0)
    assert order is not None
    assert order.symbol == "SPY"
    assert order.quantity == 20
    assert order.entry is not None
    assert order.entry.type == "MKT"

    # Invalid entry_price or budget
    assert (
        trade_strategy._generate_entry_order(
            {"symbol": "SPY", "entry_price": 0.0}, df_history, budget=10000.0
        )
        is None
    )

    # Quantity < 1
    assert (
        trade_strategy._generate_entry_order(
            {"symbol": "SPY", "entry_price": 500.0, "budget": 10.0},
            df_history,
            budget=10.0,
        )
        is None
    )


def test_tgim_generate_exit_order(
    trade_strategy: TGIMTradeStrategy,
) -> None:
    """Tests _generate_exit_order creates valid MOC exit order and handles invalid inputs."""
    trade = {"symbol": "SPY", "current_size": 20}
    df_history = pd.DataFrame([{"close": 505.0}])
    order = trade_strategy._generate_exit_order(trade, df_history, budget=10000.0)
    assert order is not None
    assert order.symbol == "SPY"
    assert order.quantity == 20
    assert len(order.exits) > 0
    assert order.exits[0].type == "MKT"
    assert order.exits[0].time_in_force == "DAY"

    # Quantity <= 0
    assert (
        trade_strategy._generate_exit_order(
            {"symbol": "SPY", "current_size": 0}, df_history, budget=10000.0
        )
        is None
    )

    # Empty history
    assert (
        trade_strategy._generate_exit_order(trade, pd.DataFrame(), budget=10000.0)
        is None
    )


def test_tgim_check_entry_zero_threshold_returns_none(
    trade_strategy: TGIMTradeStrategy,
) -> None:
    """Tests check_entry returns None when threshold price <= 0."""
    trade = {"symbol": "SPY", "entry_price": 0.0}
    candle = pd.Series({"date": "2026-07-20", "close": 500.0})
    assert trade_strategy.check_entry(trade, candle, pd.DataFrame([candle])) is None


def test_tgim_manage_active_trade_missing_entry_date_or_empty_history(
    trade_strategy: TGIMTradeStrategy,
) -> None:
    """Tests manage_active_trade returns None when entry_date missing or history empty."""
    trade_no_date = {"symbol": "SPY", "status": TradeStatus.ACTIVE.value}
    assert trade_strategy.manage_active_trade(trade_no_date, pd.DataFrame()) is None

    trade_with_date = {
        "symbol": "SPY",
        "status": TradeStatus.ACTIVE.value,
        "entry_date": "2026-07-20",
    }
    assert trade_strategy.manage_active_trade(trade_with_date, pd.DataFrame()) is None


def test_evaluate_tgim_exit_zero_bars() -> None:
    """Tests pure function evaluate_tgim_exit returns None when bars_held < 1."""
    from decimal import Decimal

    from app.services.trade_manager.strategies.tgim import evaluate_tgim_exit

    assert (
        evaluate_tgim_exit(
            bars_held=0,
            current_close=Decimal("500.0"),
            previous_close=Decimal("490.0"),
        )
        is None
    )


def test_tgim_manage_active_trade_bars_held_zero(
    trade_strategy: TGIMTradeStrategy,
) -> None:
    """Tests manage_active_trade returns None on entry day (bars_held == 0)."""
    trade = {
        "symbol": "SPY",
        "status": TradeStatus.ACTIVE.value,
        "entry_date": "2026-07-20",
    }
    monday_candle = pd.Series({"date": "2026-07-20", "close": 500.0})
    df_history = pd.DataFrame([monday_candle])
    assert trade_strategy.manage_active_trade(trade, df_history) is None


def test_tgim_check_entry_ignores_premarket_friday_candle(
    trade_strategy: TGIMTradeStrategy,
) -> None:
    """Tests that check_entry returns None when given a pre-market Friday candle before Monday setup."""
    trade = {
        "id": 1,
        "symbol": "SPY",
        "strategy": "tgim",
        "status": TradeStatus.CREATED.value,
        "entry_price": 500.0,
        "budget": 10000.0,
        "signal_context": '{"setup_date": "2026-07-20", "threshold_price": 500.0}',
    }
    friday_candle = pd.Series(
        {
            "date": "2026-07-17",
            "open": 498.0,
            "high": 502.0,
            "low": 495.0,
            "close": 500.0,
        }
    )
    df_history = pd.DataFrame([friday_candle])

    transition = trade_strategy.check_entry(trade, friday_candle, df_history)
    assert transition is None


def test_tgim_check_entry_activates_with_exact_monday_close(
    trade_strategy: TGIMTradeStrategy,
) -> None:
    """Tests check_entry activates on Monday close using exact Monday close as fill price."""
    trade = {
        "id": 1,
        "symbol": "SPY",
        "strategy": "tgim",
        "status": TradeStatus.CREATED.value,
        "entry_price": 500.0,  # threshold price
        "budget": 10000.0,
        "signal_context": '{"setup_date": "2026-07-20", "threshold_price": 500.0}',
    }
    monday_candle = pd.Series(
        {
            "date": "2026-07-20",
            "open": 496.0,
            "high": 498.0,
            "low": 490.0,
            "close": 492.50,  # lower than 500.0 threshold
        }
    )
    df_history = pd.DataFrame([monday_candle])

    transition = trade_strategy.check_entry(trade, monday_candle, df_history)
    assert transition is not None
    assert transition.updates["status"] == TradeStatus.ACTIVE.value
    assert transition.updates["entry_price"] == 492.50
    assert transition.updates["entry_date"] == "2026-07-20"


def test_tgim_check_entry_rejects_past_monday_candle(
    trade_strategy: TGIMTradeStrategy,
) -> None:
    """Tests check_entry invalidates trade when evaluated on Tuesday after missing Monday."""
    trade = {
        "id": 1,
        "symbol": "SPY",
        "strategy": "tgim",
        "status": TradeStatus.CREATED.value,
        "entry_price": 500.0,
        "budget": 10000.0,
        "signal_context": '{"setup_date": "2026-07-20", "threshold_price": 500.0}',
    }
    tuesday_candle = pd.Series(
        {
            "date": "2026-07-21",
            "open": 490.0,
            "high": 495.0,
            "low": 488.0,
            "close": 492.0,
        }
    )
    df_history = pd.DataFrame([tuesday_candle])

    transition = trade_strategy.check_entry(trade, tuesday_candle, df_history)
    assert transition is not None
    assert transition.updates["status"] == TradeStatus.INVALID.value
    assert transition.updates["exit_reason"] == ExitReason.INVALIDATED.value
