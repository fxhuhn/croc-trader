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


def test_evaluate_wbs_exit_blocks_zero_bars_held() -> None:
    """Verifies bars_held < 1 never triggers clexit or TE."""
    should_exit, reason = evaluate_wbs_exit(
        bars_held=0, current_close=150.0, previous_high=100.0, is_setup_today=False
    )
    assert should_exit is False
    assert reason is None


def test_wbs_check_entry_guards_capacity_duplicate_share_class_and_expiry() -> None:
    """Verifies check_entry enforces 1-bar expiry, share-class deduplication, and 5-slot cap."""
    from app.tools.symbol_filter import SymbolFilter

    strategy = WoundedBullSprintTradeStrategy()
    symbol_filter = SymbolFilter()
    original_mapping = dict(symbol_filter._mapping)
    symbol_filter._mapping = {"GOOGL": ["GOOG"]}
    try:
        setup_bar = {
            "date": "2026-07-20",
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
        }
        next_bar = {
            "date": "2026-07-21",
            "open": 102.0,
            "high": 104.0,
            "low": 101.0,
            "close": 103.0,
        }
        day2_bar = {
            "date": "2026-07-22",
            "open": 104.0,
            "high": 106.0,
            "low": 103.0,
            "close": 105.0,
        }
        history_next = pd.DataFrame([setup_bar, next_bar])
        history_late = pd.DataFrame([setup_bar, next_bar, day2_bar])

        # 1. Share-class duplicate: GOOGL active blocks GOOG
        goog_trade: TradeData = {
            "id": "WBS_SC",
            "symbol": "GOOG",
            "strategy": Strategies.WoundedBullSprint.value,
            "entry_price": 101.0,
            "signal_context": '{"date": "2026-07-20"}',
        }
        res_dup = strategy.check_entry(
            goog_trade,
            pd.Series(next_bar),
            history_next,
            active_symbols={"GOOGL"},
        )
        assert res_dup is not None
        assert res_dup.updates["status"] == TradeStatus.INVALIDATED
        assert res_dup.updates["exit_reason"] == ExitReason.INVALIDATED

        # 2. GOOGL counts as 1 position (not 2): 4 active trades still allow 5th slot
        fifth_trade: TradeData = {
            "id": "WBS_5TH",
            "symbol": "UNP",
            "strategy": Strategies.WoundedBullSprint.value,
            "entry_price": 101.0,
            "signal_context": '{"date": "2026-07-20"}',
        }
        res_fifth = strategy.check_entry(
            fifth_trade,
            pd.Series(next_bar),
            history_next,
            active_symbols={"GOOGL", "AAPL", "MSFT", "UPS"},
        )
        assert res_fifth is not None
        assert res_fifth.updates["status"] == TradeStatus.ACTIVE

        # 3. 5 active trades block 6th candidate
        res_full = strategy.check_entry(
            fifth_trade,
            pd.Series(next_bar),
            history_next,
            active_symbols={"GOOGL", "AAPL", "MSFT", "UPS", "AMT"},
        )
        assert res_full is not None
        assert res_full.updates["status"] == TradeStatus.INVALIDATED

        # 4. Missed NextOpen window (> 1 trading day post signal) expires
        res_expired = strategy.check_entry(
            fifth_trade,
            pd.Series(day2_bar),
            history_late,
            active_symbols=set(),
        )
        assert res_expired is not None
        assert res_expired.updates["status"] == TradeStatus.INVALIDATED
        assert res_expired.updates["exit_reason"] == ExitReason.INVALIDATED
    finally:
        symbol_filter._mapping = original_mapping


def test_wbs_holiday_and_weekend_rerun_do_not_prematurely_close() -> None:
    """Regression: Thanksgiving / Monday 06:30 re-run on same candle must not close pending_exit."""
    strategy = WoundedBullSprintTradeStrategy()

    # Case A: Day 1 entry (entry_date == candle_date) with pending_exit already written
    day1_trade: TradeData = {
        "id": "1867",
        "symbol": "UPS",
        "strategy": Strategies.WoundedBullSprint.value,
        "status": TradeStatus.ACTIVE.value,
        "entry_date": "2026-11-25",
        "entry_price": 96.0,
        "current_size": 52,
        "signal_context": (
            '{"date": "2026-11-24", "bars_held": 1, '
            '"last_processed_date": "2026-11-25", "pending_exit": "clexit"}'
        ),
    }
    wed_history = pd.DataFrame(
        [
            {
                "date": "2026-11-24",
                "open": 95.0,
                "high": 97.0,
                "low": 94.0,
                "close": 96.0,
            },
            {
                "date": "2026-11-25",
                "open": 96.0,
                "high": 98.0,
                "low": 95.5,
                "close": 97.5,
            },
        ]
    )
    assert strategy.manage_active_trade(day1_trade, wed_history) is None

    # Case B: Multi-day trade re-run on same last_processed_date (Saturday -> Monday 06:30)
    mon_rerun_trade: TradeData = {
        "id": "1870",
        "symbol": "AMT",
        "strategy": Strategies.WoundedBullSprint.value,
        "status": TradeStatus.ACTIVE.value,
        "entry_date": "2026-11-23",
        "entry_price": 175.0,
        "current_size": 28,
        "signal_context": (
            '{"date": "2026-11-20", "bars_held": 3, '
            '"last_processed_date": "2026-11-25", "pending_exit": "clexit"}'
        ),
    }
    # Prior bar (2026-11-24) did NOT trigger clexit; 2026-11-25 triggered pending_exit
    multi_history = pd.DataFrame(
        [
            {
                "date": "2026-11-23",
                "open": 175.0,
                "high": 180.0,
                "low": 174.0,
                "close": 176.0,
            },
            {
                "date": "2026-11-24",
                "open": 176.0,
                "high": 179.0,
                "low": 175.0,
                "close": 177.0,
            },
            {
                "date": "2026-11-25",
                "open": 177.0,
                "high": 182.0,
                "low": 176.5,
                "close": 181.0,
            },
        ]
    )
    assert strategy.manage_active_trade(mon_rerun_trade, multi_history) is None

    # Case C: Next real trading day arrives (2026-11-27) -> closes at Friday Open
    fri_history = pd.concat(
        [
            multi_history,
            pd.DataFrame(
                [
                    {
                        "date": "2026-11-27",
                        "open": 183.5,
                        "high": 185.0,
                        "low": 182.0,
                        "close": 184.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    transition = strategy.manage_active_trade(mon_rerun_trade, fri_history)
    assert transition is not None
    assert transition.updates["status"] == TradeStatus.CLOSED
    assert transition.updates["exit_price"] == 183.5
    assert transition.updates["exit_reason"] == ExitReason.TAKE_PROFIT


def test_wbs_get_daily_updates_and_exit_order_setup_parity() -> None:
    """Verifies get_daily_updates and _generate_exit_order suppress clexit when in qualifying_setups."""
    strategy = WoundedBullSprintTradeStrategy()
    trade: TradeData = {
        "id": "WBS_PARITY",
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

    # Active setup suppresses both pending_exit and exit order
    updates_suppressed = strategy.get_daily_updates(
        trade, history, qualifying_setups={"NVDA"}
    )
    assert updates_suppressed["bars_held"] == 2
    assert updates_suppressed["last_processed_date"] == "2026-07-22"
    assert updates_suppressed["pending_exit"] is None
    assert (
        strategy.generate_orders(
            trade, history, budget=10000.0, created_symbols={"NVDA"}
        )
        is None
    )

    # Not in qualifying setups triggers both pending_exit and SELL MKT/OPG order
    updates_exit = strategy.get_daily_updates(trade, history, qualifying_setups=set())
    assert updates_exit["pending_exit"] == "clexit"
    exit_order = strategy.generate_orders(
        trade, history, budget=10000.0, created_symbols=set()
    )
    assert exit_order is not None
    assert exit_order.exits[0].action == "SELL"
