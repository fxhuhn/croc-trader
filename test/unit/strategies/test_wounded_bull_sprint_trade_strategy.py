"""Unit test suite for the Wounded Bull Sprint Trade Manager strategy."""

import datetime
from decimal import Decimal

import pandas as pd

from app.const import ExitReason, Strategies
from app.services.trade_manager.strategies.wounded_bull_sprint import (
    WoundedBullSprintTradeStrategy,
    calculate_wbs_bars_held,
    calculate_wbs_position_quantity,
    evaluate_wbs_exit,
)
from app.types import TradeData, TradeStatus


def test_calculate_wbs_bars_held_scenario_a() -> None:
    """Verifies Scenario A: day of entry EOD is 1 bar held, day 6 EOD is 6 bars held."""
    entry_date = datetime.date(2026, 7, 20)
    dates = pd.date_range(start="2026-07-20", periods=6, freq="B")
    dates_series = pd.Series(dates.date)

    # Day 1 EOD (only 1 bar on or after entry date)
    assert calculate_wbs_bars_held(entry_date, dates_series.iloc[:1]) == 1

    # Day 3 EOD
    assert calculate_wbs_bars_held(entry_date, dates_series.iloc[:3]) == 3

    # Day 6 EOD
    assert calculate_wbs_bars_held(entry_date, dates_series) == 6

    # Empty or None date
    assert calculate_wbs_bars_held(None, dates_series) == 0
    assert calculate_wbs_bars_held(entry_date, pd.Series([], dtype=object)) == 0


def test_evaluate_wbs_exit_clexit() -> None:
    """Verifies clexit triggers when Close > High[1] and not is_setup_today."""
    # current_close (105.0) > prev_high (104.0) and not setup -> clexit
    should_exit, reason = evaluate_wbs_exit(
        bars_held=2, current_close=105.0, previous_high=104.0, is_setup_today=False
    )
    assert should_exit is True
    assert reason == "clexit"

    # Close <= prev_high -> no exit
    should_exit, reason = evaluate_wbs_exit(
        bars_held=2, current_close=103.0, previous_high=104.0, is_setup_today=False
    )
    assert should_exit is False
    assert reason is None

    # Close > prev_high BUT fresh setup today -> no exit (keep position)
    should_exit, reason = evaluate_wbs_exit(
        bars_held=2, current_close=105.0, previous_high=104.0, is_setup_today=True
    )
    assert should_exit is False
    assert reason is None


def test_evaluate_wbs_exit_time_exit() -> None:
    """Verifies TE triggers when bars_held >= 6 even if Close <= prev_high."""
    should_exit, reason = evaluate_wbs_exit(
        bars_held=6, current_close=95.0, previous_high=100.0, is_setup_today=False
    )
    assert should_exit is True
    assert reason == "TE"

    should_exit, reason = evaluate_wbs_exit(
        bars_held=5, current_close=95.0, previous_high=100.0, is_setup_today=False
    )
    assert should_exit is False
    assert reason is None


def test_calculate_wbs_position_quantity() -> None:
    """Verifies floor rounding of share calculation using Decimal precision."""
    # 10,000 / 150 = 66.666... -> 66
    qty = calculate_wbs_position_quantity(Decimal("10000.0"), Decimal("150.0"))
    assert qty == 66

    # Zero budget or price
    assert calculate_wbs_position_quantity(Decimal("0.0"), Decimal("150.0")) == 0
    assert calculate_wbs_position_quantity(Decimal("10000.0"), Decimal("0.0")) == 0


def test_wbs_check_entry() -> None:
    """Verifies NextOpen entry execution on session following setup."""
    strategy = WoundedBullSprintTradeStrategy()
    trade: TradeData = {
        "id": "WBS_1",
        "symbol": "AAPL",
        "strategy": Strategies.WoundedBullSprint.value,
        "entry_price": 150.0,
        "signal_context": '{"date": "2026-07-20"}',
    }

    # Same day as setup: no activation
    same_day_candle = pd.Series(
        {
            "date": "2026-07-20",
            "open": 149.0,
            "high": 152.0,
            "low": 148.0,
            "close": 150.0,
        }
    )
    transition = strategy.check_entry(
        trade, same_day_candle, pd.DataFrame([same_day_candle])
    )
    assert transition is None

    # Next trading day: activated at open price
    next_day_candle = pd.Series(
        {
            "date": "2026-07-21",
            "open": 151.0,
            "high": 153.0,
            "low": 150.0,
            "close": 152.0,
        }
    )
    transition = strategy.check_entry(
        trade, next_day_candle, pd.DataFrame([same_day_candle, next_day_candle])
    )
    assert transition is not None
    assert transition.updates["status"] == TradeStatus.ACTIVE
    assert transition.updates["entry_price"] == 151.0


def test_wbs_generate_entry_order() -> None:
    """Verifies MKT / OPG entry order generation with budget sizing."""
    strategy = WoundedBullSprintTradeStrategy()
    trade: TradeData = {
        "id": "WBS_2",
        "symbol": "MSFT",
        "strategy": Strategies.WoundedBullSprint.value,
        "entry_price": 200.0,
        "signal_context": '{"date": "2026-07-20"}',
    }

    history = pd.DataFrame(
        [
            {
                "date": "2026-07-20",
                "open": 198.0,
                "high": 202.0,
                "low": 197.0,
                "close": 200.0,
            }
        ]
    )

    order = strategy.generate_orders(
        trade=trade,
        dataframe_history=history,
        budget=10000.0,
    )

    assert order is not None
    assert order.symbol == "MSFT"
    assert order.quantity == 50  # 10,000 / 200
    assert order.entry is not None
    assert order.entry.action == "BUY"
    assert order.entry.type == "MKT"
    assert order.entry.time_in_force == "OPG"


def test_wbs_generate_exit_order_on_clexit() -> None:
    """Verifies MKT / OPG exit order generation when clexit is met at EOD."""
    strategy = WoundedBullSprintTradeStrategy()
    trade: TradeData = {
        "id": "WBS_3",
        "symbol": "NVDA",
        "strategy": Strategies.WoundedBullSprint.value,
        "status": TradeStatus.ACTIVE.value,
        "current_size": 40,
        "entry_date": "2026-07-21",
        "signal_context": '{"date": "2026-07-20"}',
    }

    history = pd.DataFrame(
        [
            {
                "date": "2026-07-21",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 102.0,
            },
            {
                "date": "2026-07-22",
                "open": 102.0,
                "high": 108.0,
                "low": 101.0,
                "close": 107.0,
            },
        ]
    )

    order = strategy.generate_orders(
        trade=trade,
        dataframe_history=history,
        budget=10000.0,
    )

    assert order is not None
    assert order.symbol == "NVDA"
    assert order.quantity == 40
    assert order.exits is not None
    assert len(order.exits) == 1
    assert order.exits[0].action == "SELL"
    assert order.exits[0].type == "MKT"
    assert order.exits[0].time_in_force == "OPG"


def test_wbs_do_manage_active_trade_exit_fill_at_open() -> None:
    """Verifies active trade is closed at current session Open when prior bar triggered exit."""
    strategy = WoundedBullSprintTradeStrategy()
    trade: TradeData = {
        "id": "WBS_4",
        "symbol": "GOOGL",
        "strategy": Strategies.WoundedBullSprint.value,
        "status": TradeStatus.ACTIVE.value,
        "current_size": 50,
        "entry_date": "2026-07-20",
        "signal_context": '{"date": "2026-07-17"}',
    }

    # Day 1: close 100, high 102
    # Day 2: close 105 (> prev high 102) -> triggers clexit for Day 3 Open!
    # Day 3: current_candle open 106.0
    history = pd.DataFrame(
        [
            {
                "date": "2026-07-20",
                "open": 98.0,
                "high": 102.0,
                "low": 97.0,
                "close": 100.0,
            },
            {
                "date": "2026-07-21",
                "open": 101.0,
                "high": 106.0,
                "low": 100.0,
                "close": 105.0,
            },
            {
                "date": "2026-07-22",
                "open": 106.0,
                "high": 109.0,
                "low": 105.0,
                "close": 108.0,
            },
        ]
    )

    transition = strategy.manage_active_trade(
        trade=trade,
        dataframe_history=history,
    )

    assert transition is not None
    assert transition.updates["status"] == TradeStatus.CLOSED
    assert transition.updates["exit_price"] == 106.0
    assert transition.updates["exit_reason"] == ExitReason.TAKE_PROFIT


def test_wbs_manage_active_trade_exit_suppressed_when_setup_active() -> None:
    """Verifies clexit is suppressed when Close > High[1] but symbol is in latest_leaders."""
    strategy = WoundedBullSprintTradeStrategy()
    trade: TradeData = {
        "id": "WBS_5",
        "symbol": "CAT",
        "strategy": Strategies.WoundedBullSprint.value,
        "status": TradeStatus.ACTIVE.value,
        "current_size": 14,
        "entry_date": "2026-07-20",
        "signal_context": '{"date": "2026-07-17"}',
    }

    history = pd.DataFrame(
        [
            {
                "date": "2026-07-20",
                "open": 98.0,
                "high": 102.0,
                "low": 97.0,
                "close": 100.0,
            },
            {
                "date": "2026-07-21",
                "open": 101.0,
                "high": 106.0,
                "low": 100.0,
                "close": 105.0,  # Close 105 > prev High 102
            },
            {
                "date": "2026-07-22",
                "open": 106.0,
                "high": 109.0,
                "low": 105.0,
                "close": 108.0,
            },
        ]
    )

    # When CAT is in latest_leaders (qualifying as setup today): exit MUST be suppressed!
    transition = strategy.manage_active_trade(
        trade=trade,
        dataframe_history=history,
        latest_leaders={"CAT"},
    )
    assert transition is None

    # When CAT is NOT in latest_leaders and short history: clexit triggers normally
    transition_exit = strategy.manage_active_trade(
        trade=trade,
        dataframe_history=history,
        latest_leaders={"MSFT"},
    )
    assert transition_exit is not None
    assert transition_exit.updates["status"] == TradeStatus.CLOSED
    assert transition_exit.updates["exit_reason"] == ExitReason.TAKE_PROFIT


def test_is_wbs_setup_active_validation() -> None:
    """Verifies is_wbs_setup_active contracts across technical and calendar filters."""
    from app.services.trade_manager.strategies.wounded_bull_sprint import (
        is_wbs_setup_active,
    )

    # Short history (< 200 bars) -> False
    dates_short = pd.date_range("2026-07-01", periods=100, freq="B")
    df_short = pd.DataFrame(
        {
            "date": dates_short,
            "open": [100.0] * 100,
            "high": [105.0] * 100,
            "low": [98.0] * 100,
            "close": [102.0] * 100,
            "volume": [2_000_000.0] * 100,
        }
    )
    assert is_wbs_setup_active(df_short) is False

    # 226 bars in second half of month (2026-07-24 is TDOM 18, td_left >= 3)
    dates_long = pd.date_range("2025-08-01", "2026-07-24", freq="B")
    n_bars = len(dates_long)
    assert n_bars >= 200

    # Base price path in structural uptrend with ROC(10) < 0 and ATR(3)/Close > 2.5%
    closes = [100.0 + i * 0.5 for i in range(n_bars)]
    # Dip last 10 bars so ROC(10) < 0
    closes[-1] = closes[-11] - 5.0
    highs = [c + 4.0 for c in closes]
    lows = [c - 2.0 for c in closes]
    volumes = [2_000_000.0] * n_bars

    df_valid = pd.DataFrame(
        {
            "date": dates_long,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )

    assert is_wbs_setup_active(df_valid) is True

    # If ROC(10) >= 0 (no pullback) -> False
    df_no_pullback = df_valid.copy()
    df_no_pullback.loc[df_no_pullback.index[-1], "close"] = (
        df_no_pullback.loc[df_no_pullback.index[-11], "close"] + 5.0
    )
    assert is_wbs_setup_active(df_no_pullback) is False
