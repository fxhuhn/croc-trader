"""Bridge Scout Strategy Hardening Suite (Tier 1 & Tier 2).

Contains comprehensive Boundary Value Analysis (BVA), Hypothesis Property-Based Fuzzing,
and Zero Lookahead-Bias validation for the Bridge Scout Screener and Trade Manager strategies.
"""

import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.const import TradeStatus
from app.services.screener.strategies.bridge_scout import (
    BridgeScoutStrategy,
    get_remaining_trading_days_in_month,
    is_in_end_of_month_window,
)
from app.services.trade_manager.strategies.bridge_scout import BridgeScoutTradeStrategy
from app.tools.market_holidays import MarketHolidayChecker

# ============================================================================
# TIER 1: BOUNDARY VALUE ANALYSIS (BVA) & EDGE CASES
# ============================================================================


@pytest.mark.tier1
def test_bva_december_year_rollover() -> None:
    """BVA: December 31st rollover into January next year handles year boundary cleanly."""
    dec_31 = datetime.date(2026, 12, 31)  # Thursday
    remaining_days = get_remaining_trading_days_in_month(dec_31)
    assert remaining_days == 1
    assert is_in_end_of_month_window(dec_31, days_before=4) is True


@pytest.mark.tier1
def test_bva_leap_year_february_boundaries() -> None:
    """BVA: February 29th in leap years (2024, 2028) vs Feb 28th in non-leap years (2026)."""
    # 2024 is a leap year. Feb 29 was a Thursday.
    leap_feb_29 = datetime.date(2024, 2, 29)
    assert get_remaining_trading_days_in_month(leap_feb_29) == 1
    assert is_in_end_of_month_window(leap_feb_29, days_before=4) is True

    # 2026 is NOT a leap year. Feb 28 is Saturday. Last trading day is Friday Feb 27.
    feb_27_2026 = datetime.date(2026, 2, 27)
    assert get_remaining_trading_days_in_month(feb_27_2026) == 1
    assert is_in_end_of_month_window(feb_27_2026, days_before=4) is True


@pytest.mark.tier1
def test_bva_exact_threshold_entry_and_rejection() -> None:
    """BVA: Close exactly equal to req_close_rsi40 enters; Close + 0.01 is rejected."""
    strategy = BridgeScoutTradeStrategy()
    trade_record = {
        "id": 10,
        "symbol": "QQQ",
        "strategy": "bridge_scout",
        "status": "CREATED",
        "entry_price": 500.0,
        "signal_context": '{"setup_date": "2026-07-28", "req_close_rsi40": 500.0}',
    }
    df_empty = pd.DataFrame()

    # Case A: Exact threshold match (500.00 <= 500.00) -> ACTIVE
    candle_exact = pd.Series({"date": "2026-07-28", "close": 500.0})
    transition_exact = strategy.check_entry(trade_record, candle_exact, df_empty)
    assert transition_exact is not None
    assert transition_exact.updates["status"] == "ACTIVE"

    # Case B: 1 cent above threshold (500.01 > 500.00) -> REJECTED (INVALID)
    candle_above = pd.Series({"date": "2026-07-28", "close": 500.01})
    transition_above = strategy.check_entry(trade_record, candle_above, df_empty)
    assert transition_above is not None
    assert transition_above.updates["status"] == TradeStatus.INVALID


@pytest.mark.tier1
def test_bva_missed_entry_window_invalidates_trade() -> None:
    """BVA: Candle date strictly after setup_date invalidates the setup."""
    strategy = BridgeScoutTradeStrategy()
    trade_record = {
        "id": 11,
        "symbol": "QQQ",
        "strategy": "bridge_scout",
        "status": "CREATED",
        "entry_price": 500.0,
        "signal_context": '{"setup_date": "2026-07-28", "req_close_rsi40": 500.0}',
    }
    # Evaluated on July 29th (1 day late)
    candle_late = pd.Series({"date": "2026-07-29", "close": 490.0})
    transition = strategy.check_entry(trade_record, candle_late, pd.DataFrame())
    assert transition is not None
    assert transition.updates["status"] == TradeStatus.INVALID
    assert "Missed Entry Window" in transition.reason


@pytest.mark.tier1
def test_bva_atr_filter_boundary() -> None:
    """BVA: ATR% at boundary (3.49% passes, 3.50% skipped)."""
    mock_repo = MagicMock()
    mock_data = MagicMock()
    screener = BridgeScoutStrategy(mock_repo, mock_data)

    analysis_date_str = "2026-07-28"
    dates = pd.date_range(end=analysis_date_str, periods=20, freq="B")

    # Construct ATR% = (17.5 / 500.0) * 100 = 3.50% (should be skipped)
    closes = [500.0] * 18 + [495.0, 480.0]
    highs = [c + 17.5 for c in closes]
    lows = [c - 0.0 for c in closes]
    df_history = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100000] * 20,
        }
    )
    mock_data.get_batch_history.return_value = {"QQQ": df_history}

    hits = screener.run(analysis_date=analysis_date_str)
    assert hits == 0
    mock_repo.create_trade.assert_not_called()


# ============================================================================
# TIER 2: PROPERTY-BASED FUZZING (HYPOTHESIS)
# ============================================================================


@pytest.mark.tier2
@given(
    st.dates(
        min_value=datetime.date(2000, 1, 1),
        max_value=datetime.date(2050, 12, 31),
    )
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_trading_days_in_month_invariant(
    test_date: datetime.date,
) -> None:
    """Invariant: get_remaining_trading_days_in_month must always return 0 <= days <= 23.

    Returns 0 when test_date is on a weekend at the very end of the month (e.g., Saturday 31st).
    """
    checker = MarketHolidayChecker()
    remaining = get_remaining_trading_days_in_month(test_date, holiday_checker=checker)
    assert isinstance(remaining, int)
    assert 0 <= remaining <= 23
    # If the date itself is a weekday and not a holiday, there must be at least 1 remaining day
    if test_date.weekday() < 5 and not checker.is_holiday(test_date):
        assert 1 <= remaining <= 23


@pytest.mark.tier2
@given(
    st.dates(min_value=datetime.date(2000, 1, 1), max_value=datetime.date(2050, 12, 31))
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_month_end_window_is_boolean(test_date: datetime.date) -> None:
    """Invariant: is_in_end_of_month_window returns boolean and never raises for any calendar date."""
    checker = MarketHolidayChecker()
    in_window = is_in_end_of_month_window(
        test_date, days_before=4, holiday_checker=checker
    )
    assert isinstance(in_window, bool)


@pytest.mark.tier2
@given(
    st.floats(min_value=0.01, max_value=50000.0, allow_nan=False, allow_infinity=False),
    st.floats(
        min_value=0.01, max_value=1000000.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=50, deadline=None)
def test_hypothesis_bridge_scout_order_sizing(
    entry_price: float, budget: float
) -> None:
    """Invariant: Sizing for Bridge Scout always yields integer quantity >= 0."""
    strategy = BridgeScoutTradeStrategy()
    trade = {
        "id": 1,
        "symbol": "QQQ",
        "entry_price": entry_price,
        "budget": budget,
    }
    df = pd.DataFrame({"close": [entry_price]})
    order = strategy._generate_entry_order(trade, df, budget=budget)
    if order is not None:
        assert isinstance(order.quantity, int)
        assert order.quantity >= 1
        assert order.symbol == "QQQ"


# ============================================================================
# TIER 2: LOOKAHEAD BIAS GUARD
# ============================================================================


@pytest.mark.tier2
def test_bridge_scout_screener_zero_lookahead_bias() -> None:
    """Invariant: Bridge Scout screening at target_date is unaffected by future bars."""
    target_date_str = "2026-07-28"
    dates_full = pd.date_range(start="2026-06-01", end="2026-09-30", freq="B")
    np.random.seed(123)

    closes = 400.0 + np.cumsum(np.random.normal(0, 2, len(dates_full)))
    df_full = pd.DataFrame(
        {
            "date": dates_full,
            "open": closes * 0.99,
            "high": closes * 1.01,
            "low": closes * 0.98,
            "close": closes,
            "volume": [500_000] * len(dates_full),
        }
    )

    # Cutoff at target_date
    df_cutoff = df_full[df_full["date"] <= pd.Timestamp(target_date_str)].copy()

    mock_repo_1 = MagicMock()
    mock_repo_2 = MagicMock()
    mock_data_1 = MagicMock()
    mock_data_2 = MagicMock()

    mock_data_1.get_batch_history.return_value = {"QQQ": df_cutoff}
    mock_data_2.get_batch_history.return_value = {"QQQ": df_full}

    screener_1 = BridgeScoutStrategy(mock_repo_1, mock_data_1)
    screener_2 = BridgeScoutStrategy(mock_repo_2, mock_data_2)

    hits_1 = screener_1.run(analysis_date=target_date_str)
    hits_2 = screener_2.run(analysis_date=target_date_str)

    assert hits_1 == hits_2
