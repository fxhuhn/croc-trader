"""TGIM (Thank God It's Monday) Strategy Hardening Suite (Tier 1 & Tier 2).

Provides exhaustive Boundary Value Analysis (BVA), Hypothesis Property-Based Fuzzing,
and Zero Lookahead-Bias validation for the TGIM Screener and Trade Manager strategies.
"""

from decimal import Decimal
from unittest.mock import MagicMock

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.const import Strategies, TradeStatus
from app.services.screener.strategies.tgim import (
    TGIMStrategy,
    evaluate_tgim_setup,
)
from app.services.trade_manager.strategies.tgim import (
    TGIMTradeStrategy,
    calculate_tgim_position_quantity,
    evaluate_tgim_exit,
)
from app.types import ExitReason


# ==============================================================================
# 1. Boundary Value Analysis (BVA) — Screener & Calendar Logic
# ==============================================================================
@pytest.mark.tier1
def test_bva_tgim_target_date_resolution() -> None:
    """BVA: Target date resolution for all days of the week.

    Friday, Saturday, Sunday must roll forward to Monday.
    Monday must stay Monday.
    Tuesday, Wednesday, Thursday must resolve to themselves and be rejected by run().
    """
    trade_repo = MagicMock()
    data_provider = MagicMock()
    strategy = TGIMStrategy(trade_repository=trade_repo, data_provider=data_provider)

    # Friday 2026-07-24 -> rolls forward to Monday 2026-07-27
    assert (
        strategy._resolve_target_date(0, "2026-07-24").strftime("%Y-%m-%d")
        == "2026-07-27"
    )
    # Saturday 2026-07-25 -> rolls forward to Monday 2026-07-27
    assert (
        strategy._resolve_target_date(0, "2026-07-25").strftime("%Y-%m-%d")
        == "2026-07-27"
    )
    # Sunday 2026-07-26 -> rolls forward to Monday 2026-07-27
    assert (
        strategy._resolve_target_date(0, "2026-07-26").strftime("%Y-%m-%d")
        == "2026-07-27"
    )
    # Monday 2026-07-27 -> stays Monday 2026-07-27
    assert (
        strategy._resolve_target_date(0, "2026-07-27").strftime("%Y-%m-%d")
        == "2026-07-27"
    )

    # Tuesday 2026-07-28 -> run() returns 0 (not Monday)
    assert strategy.run(analysis_date="2026-07-28") == 0
    # Wednesday 2026-07-29 -> run() returns 0
    assert strategy.run(analysis_date="2026-07-29") == 0
    # Thursday 2026-07-30 -> run() returns 0
    assert strategy.run(analysis_date="2026-07-30") == 0


@pytest.mark.tier1
def test_bva_tgim_setup_strict_inequality() -> None:
    """BVA: Monday Close < min(Friday Close, Thursday Close) exact threshold boundary.

    If Monday Close == min(Friday, Thursday) -> False (No signal)
    If Monday Close == min(Friday, Thursday) - 0.01 -> True (Signal)
    """
    thursday = Decimal("550.00")
    friday = Decimal("540.00")  # min is 540.00

    # Case 1: Monday Close == 540.00 (Equal to min) -> False
    res_equal = evaluate_tgim_setup(
        current_close=Decimal("540.00"),
        friday_close=friday,
        thursday_close=thursday,
    )
    assert not res_equal.is_signal
    assert res_equal.threshold_price == Decimal("540.00")

    # Case 2: Monday Close == 539.99 (0.01 below min) -> True
    res_below = evaluate_tgim_setup(
        current_close=Decimal("539.99"),
        friday_close=friday,
        thursday_close=thursday,
    )
    assert res_below.is_signal
    assert res_below.threshold_price == Decimal("540.00")

    # Case 3: Thursday is lower than Friday (Thursday=530, Friday=540) -> min is 530.00
    res_thursday_min = evaluate_tgim_setup(
        current_close=Decimal("529.99"),
        friday_close=Decimal("540.00"),
        thursday_close=Decimal("530.00"),
    )
    assert res_thursday_min.is_signal
    assert res_thursday_min.threshold_price == Decimal("530.00")


# ==============================================================================
# 2. Boundary Value Analysis (BVA) — Trade Manager Entry & Exit Lifecycle
# ==============================================================================
@pytest.mark.tier1
def test_bva_tgim_check_entry_threshold_and_timing() -> None:
    """BVA: check_entry threshold comparison and timing boundaries."""
    strategy = TGIMTradeStrategy()
    trade = {
        "id": 1,
        "symbol": "SPY",
        "strategy": Strategies.TGIM.value,
        "status": TradeStatus.CREATED.value,
        "entry_price": 540.00,
        "signal_context": '{"setup_date": "2026-07-27"}',
    }

    # Case 1: Candle before setup date (e.g. Friday 2026-07-24) -> None
    candle_early = pd.Series({"date": "2026-07-24", "close": 530.00})
    assert (
        strategy.check_entry(trade, candle_early, pd.DataFrame([candle_early])) is None
    )

    # Case 2: Candle on setup date, Close == 540.00 (<= threshold 540.00) -> ACTIVE
    candle_exact = pd.Series({"date": "2026-07-27", "close": 540.00})
    trans_active = strategy.check_entry(
        trade, candle_exact, pd.DataFrame([candle_exact])
    )
    assert trans_active is not None
    assert trans_active.updates["status"] == TradeStatus.ACTIVE.value
    assert trans_active.updates["entry_price"] == 540.00

    # Case 3: Candle on setup date, Close == 540.01 (> threshold 540.00) -> INVALID
    candle_high = pd.Series({"date": "2026-07-27", "close": 540.01})
    trans_invalid = strategy.check_entry(
        trade, candle_high, pd.DataFrame([candle_high])
    )
    assert trans_invalid is not None
    assert trans_invalid.updates["status"] == TradeStatus.INVALID.value

    # Case 4: Candle after setup date (Missed Entry Window) -> INVALID
    candle_late = pd.Series({"date": "2026-07-28", "close": 530.00})
    trans_missed = strategy.check_entry(trade, candle_late, pd.DataFrame([candle_late]))
    assert trans_missed is not None
    assert trans_missed.updates["status"] == TradeStatus.INVALID.value


@pytest.mark.tier1
def test_bva_tgim_exit_bars_held_boundaries() -> None:
    """BVA: evaluate_tgim_exit boundaries for Bar 0, Bar 1, and Bar 2."""
    # Bar 0 (Monday holding): Never exits
    assert (
        evaluate_tgim_exit(
            bars_held=0, current_close=Decimal("550"), previous_close=Decimal("540")
        )
        is None
    )

    # Bar 1 (Tuesday):
    # Case 1.1: Tuesday Close > Monday Close -> TAKE_PROFIT
    assert (
        evaluate_tgim_exit(
            bars_held=1,
            current_close=Decimal("540.01"),
            previous_close=Decimal("540.00"),
        )
        == ExitReason.TAKE_PROFIT
    )
    # Case 1.2: Tuesday Close == Monday Close -> None (Hold)
    assert (
        evaluate_tgim_exit(
            bars_held=1,
            current_close=Decimal("540.00"),
            previous_close=Decimal("540.00"),
        )
        is None
    )
    # Case 1.3: Tuesday Close < Monday Close -> None (Hold)
    assert (
        evaluate_tgim_exit(
            bars_held=1,
            current_close=Decimal("539.99"),
            previous_close=Decimal("540.00"),
        )
        is None
    )

    # Bar 2 (Wednesday):
    # Case 2.1: Wednesday Close > Tuesday Close -> TAKE_PROFIT
    assert (
        evaluate_tgim_exit(
            bars_held=2,
            current_close=Decimal("541.00"),
            previous_close=Decimal("540.00"),
        )
        == ExitReason.TAKE_PROFIT
    )
    # Case 2.2: Wednesday Close == Tuesday Close -> TIME_STOP
    assert (
        evaluate_tgim_exit(
            bars_held=2,
            current_close=Decimal("540.00"),
            previous_close=Decimal("540.00"),
        )
        == ExitReason.TIME_STOP
    )
    # Case 2.3: Wednesday Close < Tuesday Close -> TIME_STOP
    assert (
        evaluate_tgim_exit(
            bars_held=2,
            current_close=Decimal("539.00"),
            previous_close=Decimal("540.00"),
        )
        == ExitReason.TIME_STOP
    )


# ==============================================================================
# 3. Property-Based Fuzzing (Hypothesis)
# ==============================================================================
@pytest.mark.tier2
@given(
    monday=st.decimals(
        min_value=Decimal("1.00"), max_value=Decimal("2000.00"), places=2
    ),
    friday=st.decimals(
        min_value=Decimal("1.00"), max_value=Decimal("2000.00"), places=2
    ),
    thursday=st.decimals(
        min_value=Decimal("1.00"), max_value=Decimal("2000.00"), places=2
    ),
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_tgim_setup_invariants(
    monday: Decimal,
    friday: Decimal,
    thursday: Decimal,
) -> None:
    """Invariant: evaluate_tgim_setup is_signal == (monday < min(friday, thursday))."""
    res = evaluate_tgim_setup(monday, friday, thursday)
    expected_threshold = min(friday, thursday)
    assert res.threshold_price == expected_threshold
    assert res.is_signal == (monday < expected_threshold)
    assert res.setup_close == monday


@pytest.mark.tier2
@given(
    bars_held=st.integers(min_value=0, max_value=10),
    current_close=st.decimals(
        min_value=Decimal("1.00"), max_value=Decimal("2000.00"), places=2
    ),
    previous_close=st.decimals(
        min_value=Decimal("1.00"), max_value=Decimal("2000.00"), places=2
    ),
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_tgim_exit_invariants(
    bars_held: int,
    current_close: Decimal,
    previous_close: Decimal,
) -> None:
    """Invariant: TGIM exit is deterministic and non-None for all bars_held >= 2."""
    exit_reason = evaluate_tgim_exit(bars_held, current_close, previous_close)

    if bars_held < 1:
        assert exit_reason is None
    elif bars_held == 1:
        if current_close > previous_close:
            assert exit_reason == ExitReason.TAKE_PROFIT
        else:
            assert exit_reason is None
    else:  # bars_held >= 2
        assert exit_reason in (ExitReason.TAKE_PROFIT, ExitReason.TIME_STOP)
        if current_close > previous_close:
            assert exit_reason == ExitReason.TAKE_PROFIT
        else:
            assert exit_reason == ExitReason.TIME_STOP


@pytest.mark.tier2
@given(
    budget=st.decimals(
        min_value=Decimal("100.00"), max_value=Decimal("1000000.00"), places=2
    ),
    price=st.decimals(
        min_value=Decimal("1.00"), max_value=Decimal("5000.00"), places=2
    ),
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_tgim_position_quantity_invariants(
    budget: Decimal,
    price: Decimal,
) -> None:
    """Invariant: calculate_tgim_position_quantity is non-negative and <= budget / price."""
    qty = calculate_tgim_position_quantity(budget, price)
    assert isinstance(qty, int)
    assert qty >= 0
    assert Decimal(qty) * price <= budget


# ==============================================================================
# 4. Zero Lookahead-Bias Guard
# ==============================================================================
@pytest.mark.tier2
def test_tgim_screener_zero_lookahead_bias() -> None:
    """Time-Shift Invariance: Monday screening signal must be identical with or without future bars."""
    trade_repo_t = MagicMock()
    trade_repo_t.exists.return_value = False
    trade_repo_t.create_trade.return_value = 101

    trade_repo_future = MagicMock()
    trade_repo_future.exists.return_value = False
    trade_repo_future.create_trade.return_value = 102

    # Setup days: Thursday 2026-07-23 (550), Friday 2026-07-24 (540), Monday 2026-07-27 (530)
    # Future days: Tuesday 2026-07-28 (535), Wednesday 2026-07-29 (545)
    records_t = [
        {
            "date": "2026-07-23",
            "open": 555.0,
            "high": 556.0,
            "low": 548.0,
            "close": 550.0,
        },
        {
            "date": "2026-07-24",
            "open": 548.0,
            "high": 550.0,
            "low": 538.0,
            "close": 540.0,
        },
        {
            "date": "2026-07-27",
            "open": 538.0,
            "high": 540.0,
            "low": 528.0,
            "close": 530.0,
        },
    ]
    df_t = pd.DataFrame(records_t)

    records_future = list(records_t) + [
        {
            "date": "2026-07-28",
            "open": 532.0,
            "high": 538.0,
            "low": 530.0,
            "close": 535.0,
        },
        {
            "date": "2026-07-29",
            "open": 536.0,
            "high": 548.0,
            "low": 535.0,
            "close": 545.0,
        },
    ]
    df_future = pd.DataFrame(records_future)

    target_monday_str = "2026-07-27"

    # Run 1: History up to Monday close
    provider_t = MagicMock()
    provider_t.get_batch_history.return_value = {"SPY": df_t}
    strategy_t = TGIMStrategy(trade_repository=trade_repo_t, data_provider=provider_t)
    result_t = strategy_t.run(analysis_date=target_monday_str)

    # Run 2: History with future dataset filtered by end_date
    provider_future = MagicMock()
    provider_future.get_batch_history.side_effect = lambda symbols, days, end_date: (
        {"SPY": df_future[df_future["date"] <= end_date]}
        if end_date
        else {"SPY": df_future}
    )
    strategy_future = TGIMStrategy(
        trade_repository=trade_repo_future, data_provider=provider_future
    )
    result_future = strategy_future.run(analysis_date=target_monday_str)

    # Invariance Assertion
    assert result_t == result_future == 1
    assert (
        trade_repo_t.create_trade.call_count
        == trade_repo_future.create_trade.call_count
        == 1
    )

    kwargs_t = trade_repo_t.create_trade.call_args.kwargs
    kwargs_future = trade_repo_future.create_trade.call_args.kwargs
    assert kwargs_t["entry"] == kwargs_future["entry"] == 540.0
    assert (
        kwargs_t["context"]["threshold_price"]
        == kwargs_future["context"]["threshold_price"]
        == 540.0
    )
    assert (
        kwargs_t["context"]["setup_close"]
        == kwargs_future["context"]["setup_close"]
        == 530.0
    )
