"""Two Percent Strategy Hardening Suite (Tier 1 & Tier 2).

Provides exhaustive Boundary Value Analysis (BVA), Hypothesis Property-Based Fuzzing,
and Zero Lookahead-Bias validation for the Two Percent Screener and Trade Manager strategies.
"""

from decimal import Decimal
from unittest.mock import MagicMock

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.const import Strategies, TradeStatus
from app.services.screener.strategies.two_percent_strategy import TwoPercentStrategy
from app.services.trade_manager.strategies.two_percent_strategy import (
    TwoPercentStrategy as TwoPercentTradeStrategy,
)
from app.types import ExitReason


# ==============================================================================
# 1. Boundary Value Analysis (BVA) — Screener & Limits
# ==============================================================================
@pytest.mark.tier1
def test_bva_two_percent_entry_limit_calculation() -> None:
    """BVA: Entry limit discount is exact round(setup_close * 0.99, 2)."""
    strategy = TwoPercentStrategy(
        trade_repository=MagicMock(), data_provider=MagicMock()
    )

    # Exact cent calculations
    assert strategy._calculate_entry_price(100.00) == 99.00
    assert strategy._calculate_entry_price(100.50) == 99.50
    assert strategy._calculate_entry_price(53.25) == 52.72


@pytest.mark.tier1
def test_bva_two_percent_screener_friday_and_thursday_holiday_detection() -> None:
    """BVA: Friday is intrinsic end-of-week; Thursday triggers only if Friday is holiday/missing."""
    trade_repo = MagicMock()
    trade_repo.exists.return_value = False
    data_provider = MagicMock()
    strategy = TwoPercentStrategy(
        trade_repository=trade_repo, data_provider=data_provider
    )

    # Mock real today far in the future so historical fallbacks are treated as past
    strategy._get_real_today = MagicMock(return_value=pd.Timestamp("2026-12-31").date())  # type: ignore[method-assign]

    # 1. Regular Friday (2026-07-24)
    dates_fri = pd.date_range(
        "2026-07-01", periods=18, freq="B"
    )  # ends on Friday 2026-07-24
    df_fri = pd.DataFrame(
        [
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
            }
            for d in dates_fri
        ]
    )
    data_provider.get_symbol_history.return_value = df_fri
    assert strategy.run(analysis_date="2026-07-24") == 1
    assert trade_repo.create_trade.call_count == 1

    # 2. Thursday before a Friday holiday (df does not contain Friday 2026-07-24, ends on Thu 2026-07-23)
    trade_repo.create_trade.reset_mock()
    df_thu = df_fri.iloc[:-1].copy()
    data_provider.get_symbol_history.return_value = df_thu
    assert strategy.run(analysis_date="2026-07-23") == 1
    assert trade_repo.create_trade.call_count == 1

    # 3. Wednesday on a normal week (Thursday and Friday exist in DB) -> returns 0
    trade_repo.create_trade.reset_mock()
    data_provider.get_symbol_history.return_value = df_fri
    assert strategy.run(analysis_date="2026-07-22") == 0
    trade_repo.create_trade.assert_not_called()


# ==============================================================================
# 2. Boundary Value Analysis (BVA) — Trade Manager Entry & Exit Lifecycle
# ==============================================================================
@pytest.mark.tier1
def test_bva_two_percent_entry_fill_boundaries() -> None:
    """BVA: Day 1 entry fill conditions: Gap Down vs Limit Hit vs Missed Limit."""
    strategy = TwoPercentTradeStrategy()
    trade = {
        "id": 1,
        "symbol": "SXRV.DE",
        "strategy": Strategies.TwoPercent.value,
        "status": TradeStatus.CREATED.value,
        "entry_price": 99.00,  # Limit is 99.00
        "signal_context": '{"date": "2026-07-24"}',  # Friday setup
    }

    # History containing Friday setup + Monday candidate
    # Case 1: Gap Down (Open = 98.50 < Limit 99.00) -> Entry at Open (98.50), Target = 98.50 * 1.02 = 100.47
    candle_gap = pd.Series(
        {
            "date": "2026-07-27",
            "open": 98.50,
            "high": 99.50,
            "low": 98.00,
            "close": 99.00,
        }
    )
    df_gap = pd.DataFrame([{"date": "2026-07-24", "close": 100.0}, candle_gap])
    trans_gap = strategy.check_entry(trade, candle_gap, df_gap)
    assert trans_gap is not None
    assert trans_gap.updates["status"] == TradeStatus.ACTIVE.value
    assert trans_gap.updates["entry_price"] == 98.50
    assert trans_gap.updates["current_target"] == 100.47

    # Case 2: Limit Hit (Open = 100.00, Low = 99.00 == Limit) -> Entry at Limit (99.00), Target = 99.00 * 1.02 = 100.98
    candle_limit = pd.Series(
        {
            "date": "2026-07-27",
            "open": 100.00,
            "high": 100.50,
            "low": 99.00,
            "close": 99.50,
        }
    )
    df_limit = pd.DataFrame([{"date": "2026-07-24", "close": 100.0}, candle_limit])
    trans_limit = strategy.check_entry(trade, candle_limit, df_limit)
    assert trans_limit is not None
    assert trans_limit.updates["status"] == TradeStatus.ACTIVE.value
    assert trans_limit.updates["entry_price"] == 99.00
    assert trans_limit.updates["current_target"] == 100.98

    # Case 3: Missed Limit (Open = 100.00, Low = 99.01 > Limit 99.00) -> INVALID
    candle_miss = pd.Series(
        {
            "date": "2026-07-27",
            "open": 100.00,
            "high": 100.50,
            "low": 99.01,
            "close": 99.50,
        }
    )
    df_miss = pd.DataFrame([{"date": "2026-07-24", "close": 100.0}, candle_miss])
    trans_miss = strategy.check_entry(trade, candle_miss, df_miss)
    assert trans_miss is not None
    assert trans_miss.updates["status"] == TradeStatus.INVALID.value


@pytest.mark.tier1
def test_bva_two_percent_holiday_monday_entry_fallback() -> None:
    """BVA: Monday holiday allows Day 2 (Tuesday) entry fallback; normal Monday rejects Day 2."""
    holiday_checker = MagicMock()
    strategy = TwoPercentTradeStrategy(holiday_checker=holiday_checker)

    trade = {
        "id": 1,
        "symbol": "SXRV.DE",
        "strategy": Strategies.TwoPercent.value,
        "status": TradeStatus.CREATED.value,
        "entry_price": 99.00,
        "signal_context": '{"date": "2026-07-24"}',  # Friday setup
    }

    # Case 1: Monday 2026-07-27 IS a holiday -> Tuesday 2026-07-28 is accepted
    holiday_checker.is_holiday.side_effect = lambda d: (
        d == pd.Timestamp("2026-07-27").date()
    )

    candle_tue = pd.Series(
        {
            "date": "2026-07-28",
            "open": 100.0,
            "high": 100.5,
            "low": 98.50,
            "close": 99.00,
        }
    )
    # df has Friday + Monday holiday + Tuesday
    df_history_tue = pd.DataFrame(
        [
            {"date": "2026-07-24", "close": 100.0},
            {"date": "2026-07-27", "close": 100.0},
            candle_tue,
        ]
    )
    trans_tue = strategy.check_entry(trade, candle_tue, df_history_tue)
    assert trans_tue is not None
    assert trans_tue.updates["status"] == TradeStatus.ACTIVE.value

    # Case 2: Monday 2026-07-27 was NOT a holiday -> Tuesday 2026-07-28 is rejected as Stale
    holiday_checker.is_holiday.side_effect = None
    holiday_checker.is_holiday.return_value = False
    trans_stale = strategy.check_entry(trade, candle_tue, df_history_tue)
    assert trans_stale is not None
    assert trans_stale.updates["status"] == TradeStatus.INVALID.value


@pytest.mark.tier1
def test_bva_two_percent_target_exit_and_gap_up_benefit() -> None:
    """BVA: High >= Target triggers target exit; Open > Target benefits from gap-up."""
    strategy = TwoPercentTradeStrategy()
    trade = {
        "id": 1,
        "symbol": "SXRV.DE",
        "strategy": Strategies.TwoPercent.value,
        "status": TradeStatus.ACTIVE.value,
        "entry_price": 99.00,
        "current_target": 100.98,  # Target is 100.98
        "entry_date": "2026-07-27",  # Monday entry
    }

    # Case 1: Same-day candle (Monday 2026-07-27) does NOT trigger exit
    candle_same_day = pd.Series(
        {"date": "2026-07-27", "open": 99.0, "high": 102.0, "low": 98.0, "close": 101.5}
    )
    df_same_day = pd.DataFrame([candle_same_day])
    assert strategy.manage_active_trade(trade, df_same_day) is None

    # Case 2: Tuesday candle with High == Target (100.98) -> Exit at 100.98
    candle_exact = pd.Series(
        {
            "date": "2026-07-28",
            "open": 99.5,
            "high": 100.98,
            "low": 99.0,
            "close": 100.5,
        }
    )
    df_exact = pd.DataFrame([candle_same_day, candle_exact])
    trans_exact = strategy.manage_active_trade(trade, df_exact)
    assert trans_exact is not None
    assert trans_exact.updates["status"] == TradeStatus.CLOSED.value
    assert trans_exact.updates["exit_price"] == 100.98
    assert trans_exact.updates["exit_reason"] == ExitReason.TARGET_HIT

    # Case 3: Tuesday candle gap-up (Open = 101.50 > Target 100.98) -> Exit at Open (101.50)
    candle_gap_up = pd.Series(
        {
            "date": "2026-07-28",
            "open": 101.50,
            "high": 102.00,
            "low": 101.0,
            "close": 101.8,
        }
    )
    df_gap_up = pd.DataFrame([candle_same_day, candle_gap_up])
    trans_gap_up = strategy.manage_active_trade(trade, df_gap_up)
    assert trans_gap_up is not None
    assert trans_gap_up.updates["exit_price"] == 101.50
    assert trans_gap_up.updates["exit_reason"] == ExitReason.TARGET_HIT


# ==============================================================================
# 3. Property-Based Fuzzing (Hypothesis)
# ==============================================================================
@pytest.mark.tier2
@given(
    entry_price=st.decimals(
        min_value=Decimal("0.50"), max_value=Decimal("10000.00"), places=2
    )
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_two_percent_target_price_invariants(entry_price: Decimal) -> None:
    """Invariant: _calculate_target_price is strictly greater than entry_price and has 2 decimals."""
    strategy = TwoPercentTradeStrategy()
    target = strategy._calculate_target_price(entry_price)
    assert target > entry_price
    assert target == (entry_price * Decimal("1.02")).quantize(Decimal("0.01"))


@pytest.mark.tier2
@given(
    close_price=st.floats(
        min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False
    )
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_two_percent_entry_discount_invariants(close_price: float) -> None:
    """Invariant: _calculate_entry_price is strictly less than close_price."""
    strategy = TwoPercentStrategy(
        trade_repository=MagicMock(), data_provider=MagicMock()
    )
    entry = strategy._calculate_entry_price(close_price)
    assert entry < close_price
    assert entry == round(close_price * 0.99, 2)


@pytest.mark.tier2
@given(
    budget=st.floats(
        min_value=100.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False
    ),
    entry_price=st.floats(
        min_value=1.0, max_value=5000.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_two_percent_order_sizing(budget: float, entry_price: float) -> None:
    """Invariant: Generated entry order quantity is non-negative integer = int(budget / price)."""
    strategy = TwoPercentTradeStrategy()
    trade = {
        "id": 1,
        "symbol": "SXRV.DE",
        "entry_price": entry_price,
        "budget": budget,
    }

    order = strategy._generate_entry_order(trade, pd.DataFrame(), budget=budget)
    expected_qty = int(budget / entry_price)

    if expected_qty >= 1:
        assert order is not None
        assert order.quantity == expected_qty
        assert isinstance(order.quantity, int)
    else:
        assert order is None


# ==============================================================================
# 4. Zero Lookahead-Bias Guard
# ==============================================================================
@pytest.mark.tier2
def test_two_percent_screener_zero_lookahead_bias() -> None:
    """Time-Shift Invariance: Friday screening signal on date T must not be affected by future bars."""
    trade_repo_t = MagicMock()
    trade_repo_t.exists.return_value = False
    trade_repo_t.create_trade.return_value = 101

    trade_repo_future = MagicMock()
    trade_repo_future.exists.return_value = False
    trade_repo_future.create_trade.return_value = 102

    # Friday 2026-07-24
    dates_t = pd.date_range("2026-07-01", periods=18, freq="B")
    records_t = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
        }
        for d in dates_t
    ]
    df_t = pd.DataFrame(records_t)

    # Future week (Mon 2026-07-27 .. Fri 2026-07-31)
    dates_future = pd.date_range("2026-07-01", periods=23, freq="B")
    records_future = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
        }
        for d in dates_future
    ]
    df_future = pd.DataFrame(records_future)

    target_friday_str = "2026-07-24"

    # Run 1: Data available only up to Friday
    provider_t = MagicMock()
    provider_t.get_symbol_history.return_value = df_t
    strategy_t = TwoPercentStrategy(
        trade_repository=trade_repo_t, data_provider=provider_t
    )
    strategy_t._get_real_today = MagicMock(
        return_value=pd.Timestamp("2026-12-31").date()
    )  # type: ignore[method-assign]
    result_t = strategy_t.run(analysis_date=target_friday_str)

    # Run 2: Provider returns dataset with future bars, filtered up to target_friday_str
    provider_future = MagicMock()
    # When _get_last_valid_candle filters history <= analysis_timestamp, it must produce identical results
    provider_future.get_symbol_history.return_value = df_future
    strategy_future = TwoPercentStrategy(
        trade_repository=trade_repo_future, data_provider=provider_future
    )
    strategy_future._get_real_today = MagicMock(
        return_value=pd.Timestamp("2026-12-31").date()
    )  # type: ignore[method-assign]
    result_future = strategy_future.run(analysis_date=target_friday_str)

    # Invariance Assertion
    assert result_t == result_future == 1
    assert (
        trade_repo_t.create_trade.call_count
        == trade_repo_future.create_trade.call_count
        == 1
    )

    kwargs_t = trade_repo_t.create_trade.call_args.kwargs
    kwargs_future = trade_repo_future.create_trade.call_args.kwargs
    assert kwargs_t["entry"] == kwargs_future["entry"] == 99.0
    assert (
        kwargs_t["context"]["limit_entry"]
        == kwargs_future["context"]["limit_entry"]
        == 99.0
    )
    assert (
        kwargs_t["context"]["setup_close"]
        == kwargs_future["context"]["setup_close"]
        == 100.0
    )
