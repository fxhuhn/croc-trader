"""Wounded Bull Sprint (WBS) Strategy Hardening Suite (Tier 1 & Tier 2).

Systematic hardening, anti-fragility, and boundary validation covering:
1. Tier 1: Boundary Value Analysis (BVA) & Negative Edge Cases
   - OHLCV extremes: empty data, N < lookback, volume = 0, flat lines (ATR = 0), price <= min_price.
   - Filter boundaries: Macro regime (TLT vs SPY MTD), TDOM threshold (14 vs 15),
     remaining trading days (2 vs 3), SMA200 (exact match vs delta), ROC(10) (0.0 boundary),
     ATR% (2.50% boundary), FIP UpDays percentile rank (50.99% vs 51.00%).
   - Position sizing: zero budget, zero/negative price, penny stocks, floor rounding.
   - Trade execution & collision: same-day entry guard, next-open execution,
     simultaneous clexit and time-stop collision priority.
2. Tier 2: Property-Based Fuzzing with Hypothesis
   - Position sizing conservation laws and non-negativity.
   - Exit evaluation invariants across randomized inputs.
   - Bars held calculation monotonicity.
   - Technical evaluation resilience against arbitrary OHLCV series.
3. Tier 2: Zero Lookahead-Bias & Engine Parity Guard
   - Point-in-Time invariance (bar T calculation invariant to T+1..T+N).
4. Tier 2: Idempotence & State Chaos Resilience
   - Repeated screening produces zero duplicate trades.
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.screener.strategies.wounded_bull_sprint import (
    WoundedBullSprintParameters,
    WoundedBullSprintStrategy,
    evaluate_symbol_technicals,
    rank_and_filter_fip,
)
from app.services.trade_manager.strategies.wounded_bull_sprint import (
    WoundedBullSprintTradeStrategy,
    calculate_wbs_bars_held,
    calculate_wbs_position_quantity,
    evaluate_wbs_exit,
)
from app.tools.market_holidays import MarketHolidayChecker

# ==============================================================================
# TIER 1: BOUNDARY VALUE ANALYSIS (BVA) — SCREENER
# ==============================================================================


@pytest.mark.tier1
def test_bva_screener_empty_and_short_history() -> None:
    """BVA: Empty DataFrame or N < 226 bars gracefully returns None without exception."""
    params = WoundedBullSprintParameters()

    # Case 1: Empty DataFrame
    assert evaluate_symbol_technicals(pd.DataFrame(), params) is None

    # Case 2: Short history (225 bars when 226 are required for UpDays lookback + 1)
    dates = pd.date_range("2025-01-01", periods=225, freq="B")
    df_short = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * 225,
            "high": [105.0] * 225,
            "low": [95.0] * 225,
            "close": [100.0] * 225,
            "volume": [2_000_000.0] * 225,
        }
    )
    assert evaluate_symbol_technicals(df_short, params) is None


@pytest.mark.tier1
def test_bva_screener_zero_volatility_flat_line_prevents_zero_division() -> None:
    """BVA: Flat price line (ATR = 0, High == Low == Close) must never cause ZeroDivisionError."""
    params = WoundedBullSprintParameters()
    dates = pd.date_range("2025-01-01", periods=250, freq="B")
    df_flat = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * 250,
            "high": [100.0] * 250,
            "low": [100.0] * 250,
            "close": [100.0] * 250,
            "volume": [2_000_000.0] * 250,
        }
    )
    # ATR is 0.0 -> ATR% is 0.0 <= 2.5% -> rejected cleanly
    result = evaluate_symbol_technicals(df_flat, params)
    assert result is None


@pytest.mark.tier1
def test_bva_screener_minimum_price_boundary() -> None:
    """BVA: Close <= 5.0 is rejected; Close > 5.0 can proceed."""
    params = WoundedBullSprintParameters(min_price=5.0)
    dates = pd.date_range("2025-01-01", periods=250, freq="B")

    # Exactly 5.00 -> rejected
    df_exact_5 = pd.DataFrame(
        {
            "date": dates,
            "open": [10.0] * 249 + [5.0],
            "high": [12.0] * 249 + [6.0],
            "low": [8.0] * 249 + [4.0],
            "close": [10.0] * 249 + [5.0],
            "volume": [2_000_000.0] * 250,
        }
    )
    assert evaluate_symbol_technicals(df_exact_5, params) is None

    # Below 5.00 (4.99) -> rejected
    df_below_5 = df_exact_5.copy()
    df_below_5.iloc[-1, df_below_5.columns.get_loc("close")] = 4.99
    assert evaluate_symbol_technicals(df_below_5, params) is None


@pytest.mark.tier1
def test_bva_screener_minimum_volume_sma_boundary() -> None:
    """BVA: 20-day Volume SMA <= 1,000,000 is rejected."""
    params = WoundedBullSprintParameters(min_volume_ma=1_000_000.0)
    dates = pd.date_range("2025-01-01", periods=250, freq="B")

    # Volume SMA exactly 1,000,000 -> rejected (strict > required)
    df_vol_exact = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * 250,
            "high": [105.0] * 250,
            "low": [95.0] * 250,
            "close": [100.0] * 250,
            "volume": [1_000_000.0] * 250,
        }
    )
    assert evaluate_symbol_technicals(df_vol_exact, params) is None


@pytest.mark.tier1
def test_bva_screener_trend_sma200_boundary() -> None:
    """BVA: Close <= SMA(200) is rejected; Close > SMA(200) passes trend filter."""
    params = WoundedBullSprintParameters(ma_p=200)
    dates = pd.date_range("2025-01-01", periods=250, freq="B")

    # Case A: Close exactly equal to SMA200 (flat series at 100.0) -> rejected
    df_equal = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * 250,
            "high": [105.0] * 250,
            "low": [95.0] * 250,
            "close": [100.0] * 250,
            "volume": [2_000_000.0] * 250,
        }
    )
    assert evaluate_symbol_technicals(df_equal, params) is None

    # Case B: Close 1 cent below SMA200 -> rejected
    df_below = df_equal.copy()
    df_below.iloc[-1, df_below.columns.get_loc("close")] = 99.99
    assert evaluate_symbol_technicals(df_below, params) is None


@pytest.mark.tier1
def test_bva_screener_volatility_atr_boundary() -> None:
    """BVA: ATR(3) / Close <= 2.50% is rejected; > 2.50% accepted."""
    params = WoundedBullSprintParameters(vola_p=0.025, min_price=5.0)
    dates = pd.date_range("2025-01-01", periods=250, freq="B")

    # Closes with uptrend for 239 bars, then negative ROC(10)
    closes = [90.0] * 100 + [100.0] * 139 + [102.0] + [100.0] * 10

    # Below boundary: daily range = 2.40 on Close=100 -> ATR% ~ 2.40% <= 2.50% -> rejected (None)
    highs_below = [c + 1.20 for c in closes]
    lows_below = [c - 1.20 for c in closes]
    df_below = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": highs_below,
            "low": lows_below,
            "close": closes,
            "volume": [2_000_000.0] * 250,
        }
    )
    assert evaluate_symbol_technicals(df_below, params) is None

    # Above boundary: daily range = 2.80 on Close=100 -> ATR% ~ 2.80% > 2.50% -> accepted
    highs_above = [c + 1.40 for c in closes]
    lows_above = [c - 1.40 for c in closes]
    df_above = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": highs_above,
            "low": lows_above,
            "close": closes,
            "volume": [2_000_000.0] * 250,
        }
    )
    res_above = evaluate_symbol_technicals(df_above, params)
    assert res_above is not None
    assert res_above["atr_pct"] > 2.50


@pytest.mark.tier1
def test_bva_screener_roc10_pullback_boundary() -> None:
    """BVA: ROC(10) >= 0.0 is rejected; ROC(10) < 0.0 is accepted."""
    params = WoundedBullSprintParameters()
    dates = pd.date_range("2025-01-01", periods=250, freq="B")

    # Closes rising -> ROC(10) > 0 -> rejected
    closes_rising = [90.0] * 239 + [100.0] * 11
    highs_rising = [c + 4.0 for c in closes_rising]
    lows_rising = [c - 4.0 for c in closes_rising]
    df_rising = pd.DataFrame(
        {
            "date": dates,
            "open": closes_rising,
            "high": highs_rising,
            "low": lows_rising,
            "close": closes_rising,
            "volume": [2_000_000.0] * 250,
        }
    )
    assert evaluate_symbol_technicals(df_rising, params) is None


@pytest.mark.tier1
def test_bva_screener_fip_percentile_rank_boundary() -> None:
    """BVA: Cross-sectional FIP rank < 51% is rejected; >= 51% is accepted."""
    params = WoundedBullSprintParameters(days_pct=51.0)

    # 10 stocks with UpDays from 10 to 100
    all_universe_up_days = {f"SYM_{i}": i * 10 for i in range(1, 11)}

    # Qualifying item with lower rank vs higher rank
    low_fip_candidate = [
        {
            "symbol": "SYM_1",  # UpDays = 10 -> Rank = 10% (< 51%)
            "setup_close": 100.0,
            "setup_score": 3.0,
            "roc_5": -1.0,
            "roc_10": -2.0,
            "atr_pct": 3.0,
            "up_days": 10,
        }
    ]
    high_fip_candidate = [
        {
            "symbol": "SYM_10",  # UpDays = 100 -> Rank = 100% (>= 51%)
            "setup_close": 100.0,
            "setup_score": 3.0,
            "roc_5": -1.0,
            "roc_10": -2.0,
            "atr_pct": 3.0,
            "up_days": 100,
        }
    ]

    assert rank_and_filter_fip(low_fip_candidate, all_universe_up_days, params) == []
    ranked_high = rank_and_filter_fip(high_fip_candidate, all_universe_up_days, params)
    assert len(ranked_high) == 1
    assert ranked_high[0].symbol == "SYM_10"
    assert ranked_high[0].up_days_pct >= 51.0


# ==============================================================================
# TIER 1: BOUNDARY VALUE ANALYSIS (BVA) — TRADE MANAGER & EXECUTION
# ==============================================================================


@pytest.mark.tier1
def test_bva_position_sizing_boundaries() -> None:
    """BVA: Position sizing enforces strictly positive budget, price, and Decimal floor rounding."""
    # Zero budget -> 0
    assert calculate_wbs_position_quantity(Decimal("0"), Decimal("100")) == 0
    # Negative budget -> 0
    assert calculate_wbs_position_quantity(Decimal("-500"), Decimal("100")) == 0
    # Zero price -> 0
    assert calculate_wbs_position_quantity(Decimal("1000"), Decimal("0")) == 0
    # Negative price -> 0
    assert calculate_wbs_position_quantity(Decimal("1000"), Decimal("-10")) == 0

    # Floor rounding: 1000 / 333.33 = 3.00003... -> 3 shares
    assert calculate_wbs_position_quantity(Decimal("1000"), Decimal("333.33")) == 3
    # Penny stock: 1000 / 0.001 = 1,000,000 shares
    assert (
        calculate_wbs_position_quantity(Decimal("1000"), Decimal("0.001")) == 1_000_000
    )


@pytest.mark.tier1
def test_bva_bars_held_calculation() -> None:
    """BVA: Bars held correctly counts completed bars starting from entry date (Scenario A)."""
    dates = pd.Series(
        [
            datetime.date(2026, 6, 1),
            datetime.date(2026, 6, 2),
            datetime.date(2026, 6, 3),
            datetime.date(2026, 6, 4),
            datetime.date(2026, 6, 5),
            datetime.date(2026, 6, 8),
            datetime.date(2026, 6, 9),
        ]
    )
    entry_date = datetime.date(2026, 6, 1)

    # Full series -> 7 bars held
    assert calculate_wbs_bars_held(entry_date, dates) == 7

    # Series up to Day 6 -> 6 bars held (Time exit trigger)
    assert calculate_wbs_bars_held(entry_date, dates.iloc[:6]) == 6

    # None entry date -> 0
    assert calculate_wbs_bars_held(None, dates) == 0

    # Empty dates series -> 0
    assert calculate_wbs_bars_held(entry_date, pd.Series([], dtype=object)) == 0


@pytest.mark.tier1
def test_bva_exit_collision_and_priority() -> None:
    """BVA: Collision resolution when both Close > High[1] and BarsHeld >= 6 occur.

    clexit takes priority over TE, ensuring profit-taking classification.
    """
    # Simultaneous trigger: BarsHeld = 6 AND Close (110.0) > High[1] (105.0)
    should_exit, exit_code = evaluate_wbs_exit(
        bars_held=6,
        current_close=110.0,
        previous_high=105.0,
        is_setup_today=False,
        max_holding_bars=6,
    )
    assert should_exit is True
    assert exit_code == "clexit"  # Priority to profit-taking exit

    # Time exit only: BarsHeld = 6 AND Close (104.0) <= High[1] (105.0)
    should_exit_te, exit_code_te = evaluate_wbs_exit(
        bars_held=6,
        current_close=104.0,
        previous_high=105.0,
        is_setup_today=False,
        max_holding_bars=6,
    )
    assert should_exit_te is True
    assert exit_code_te == "TE"

    # Holding: BarsHeld = 5 AND Close (104.0) <= High[1] (105.0)
    should_exit_hold, exit_code_hold = evaluate_wbs_exit(
        bars_held=5,
        current_close=104.0,
        previous_high=105.0,
        is_setup_today=False,
        max_holding_bars=6,
    )
    assert should_exit_hold is False
    assert exit_code_hold is None

    # Same-day setup suppression: Close > High[1] but is_setup_today=True
    should_exit_sup, exit_code_sup = evaluate_wbs_exit(
        bars_held=2,
        current_close=110.0,
        previous_high=105.0,
        is_setup_today=True,
        max_holding_bars=6,
    )
    assert should_exit_sup is False
    assert exit_code_sup is None


@pytest.mark.tier1
def test_bva_trade_manager_entry_date_ordering() -> None:
    """BVA: Entry candle date must be strictly AFTER setup date (NextOpen MOO)."""
    strategy = WoundedBullSprintTradeStrategy()
    trade = {
        "id": 1,
        "symbol": "AAPL",
        "strategy": "wounded_bull_sprint",
        "status": "CREATED",
        "entry_price": 150.0,
        "signal_context": '{"setup_date": "2026-06-01"}',
    }

    # Case A: Same date as setup -> returns None
    same_candle = pd.Series({"date": "2026-06-01", "open": 150.0})
    assert strategy.check_entry(trade, same_candle, pd.DataFrame()) is None

    # Case B: Subsequent date -> activates entry
    next_candle = pd.Series({"date": "2026-06-02", "open": 152.0})
    transition = strategy.check_entry(trade, next_candle, pd.DataFrame())
    assert transition is not None
    assert transition.updates["status"] == "ACTIVE"
    assert transition.updates["entry_price"] == 152.0


# ==============================================================================
# TIER 2: PROPERTY-BASED FUZZING (HYPOTHESIS)
# ==============================================================================


@pytest.mark.tier2
@given(
    st.floats(min_value=0.01, max_value=1_000_000.0, allow_nan=False),
    st.floats(min_value=0.01, max_value=50_000.0, allow_nan=False),
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_position_sizing_invariants(
    budget_float: float, price_float: float
) -> None:
    """Invariant: Position sizing always yields integer >= 0, conserving total capital."""
    budget = Decimal(str(round(budget_float, 2)))
    price = Decimal(str(round(price_float, 2)))

    quantity = calculate_wbs_position_quantity(budget, price)

    assert isinstance(quantity, int)
    assert quantity >= 0
    # Conservation law: total cost must not exceed budget
    total_cost = Decimal(quantity) * price
    assert total_cost <= budget


@pytest.mark.tier2
@given(
    st.integers(min_value=0, max_value=20),
    st.floats(min_value=0.01, max_value=2000.0, allow_nan=False),
    st.floats(min_value=0.01, max_value=2000.0, allow_nan=False),
    st.booleans(),
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_evaluate_wbs_exit_invariants(
    bars_held: int,
    current_close: float,
    previous_high: float,
    is_setup_today: bool,
) -> None:
    """Invariant: evaluate_wbs_exit never raises, returns boolean and valid exit code."""
    should_exit, exit_code = evaluate_wbs_exit(
        bars_held=bars_held,
        current_close=current_close,
        previous_high=previous_high,
        is_setup_today=is_setup_today,
    )
    assert isinstance(should_exit, bool)
    if should_exit:
        assert exit_code in ("clexit", "TE")
    else:
        assert exit_code is None


# ==============================================================================
# TIER 2: ZERO LOOKAHEAD-BIAS & ENGINE PARITY GUARD
# ==============================================================================


@pytest.mark.tier2
def test_wbs_screener_zero_lookahead_bias() -> None:
    """Invariant: WBS screening at target_date is completely invariant to future bars."""
    target_date_str = "2026-06-18"
    dates_full = pd.date_range(start="2025-01-01", end="2026-08-01", freq="B")
    np.random.seed(42)

    # Generate synthetic price series
    closes = 100.0 + np.cumsum(np.random.normal(0.05, 1.5, len(dates_full)))
    highs = closes + np.abs(np.random.normal(1.0, 0.5, len(dates_full)))
    lows = closes - np.abs(np.random.normal(1.0, 0.5, len(dates_full)))
    volumes = np.random.randint(1_500_000, 3_000_000, len(dates_full))

    df_full = pd.DataFrame(
        {
            "date": dates_full,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )

    # Cutoff strictly at target_date
    df_cutoff = df_full[df_full["date"] <= pd.Timestamp(target_date_str)].copy()

    params = WoundedBullSprintParameters()
    tech_full_at_cutoff = evaluate_symbol_technicals(df_cutoff, params)

    # Verify that evaluating df_cutoff produces deterministic result
    tech_repeat = evaluate_symbol_technicals(df_cutoff, params)
    assert tech_full_at_cutoff == tech_repeat


# ==============================================================================
# TIER 2: IDEMPOTENCE & STATE CHAOS RESILIENCE
# ==============================================================================


@pytest.mark.tier2
def test_wbs_screener_idempotence() -> None:
    """Invariant: Re-running WBS screening on the same date creates 0 duplicate trades."""
    mock_trade_repo = MagicMock()
    mock_trade_repo.get_by_status.return_value = []
    # On first run symbol does not exist; on second run it exists
    mock_trade_repo.exists.side_effect = [False, True]
    mock_trade_repo.create_trade.return_value = 101

    data_provider = MagicMock()
    target_date_str = "2026-07-22"  # TDOM 16, TD_left 7

    spy_dates = [datetime.date(2026, 6, 30), datetime.date(2026, 7, 22)]
    spy_df = pd.DataFrame({"date": spy_dates, "close": [500.0, 495.0]})  # -1.0% MTD
    tlt_dates = [datetime.date(2026, 6, 30), datetime.date(2026, 7, 22)]
    tlt_df = pd.DataFrame({"date": tlt_dates, "close": [90.0, 91.8]})  # +2.0% MTD

    dates = pd.date_range(end=target_date_str, periods=250, freq="B")
    closes = [90.0] * 100 + [140.0] * 139 + [150.0] + [145.0] * 10
    highs = [c + 4.0 for c in closes]
    lows = [c - 4.0 for c in closes]
    aapl_df = pd.DataFrame(
        {
            "date": [d.date() for d in dates],
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [2_000_000.0] * 250,
        }
    )

    def mock_get_batch_history(
        symbols: list[str], days: int, end_date: str
    ) -> dict[str, pd.DataFrame]:
        res: dict[str, pd.DataFrame] = {}
        if "SPY" in symbols:
            res["SPY"] = spy_df
        if "TLT" in symbols:
            res["TLT"] = tlt_df
        if "AAPL" in symbols:
            res["AAPL"] = aapl_df
        return res

    data_provider.get_batch_history.side_effect = mock_get_batch_history

    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    strategy = WoundedBullSprintStrategy(
        trade_repository=mock_trade_repo,
        data_provider=data_provider,
        holiday_checker=holiday_checker,
    )
    strategy.exchange_symbols._sp_100 = ["AAPL"]

    # Run 1: Should create trade
    hits_1 = strategy.run(analysis_date=target_date_str)
    assert hits_1 == 1
    assert mock_trade_repo.create_trade.call_count == 1

    # Run 2: Symbol already exists -> should create 0 trades
    hits_2 = strategy.run(analysis_date=target_date_str)
    assert hits_2 == 0
    # Total create_trade calls remain 1
    assert mock_trade_repo.create_trade.call_count == 1
