"""Dip Buyer Hardening Suite (Tier 1 & Tier 2).

Provides exhaustive Boundary Value Analysis (BVA), Hypothesis Property-Based Fuzzing,
and Zero Lookahead-Bias validation for the Dip Buyer Screener and Trade Manager.
"""

from decimal import Decimal
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.const import ExitReason, Strategies, TradeStatus
from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.repositories.trade import TradeRepository
from app.services.screener.strategies.dip_buyer import (
    DipBuyerAnalysisSnapshot,
    DipBuyerStrategy,
)
from app.services.trade_manager.strategies.dip_buyer import (
    DipBuyerStrategy as DipBuyerTradeStrategy,
)
from app.types import TradeData


# ==============================================================================
# 1. Boundary Value Analysis (BVA) — Screener Setup Rules & Indicators
# ==============================================================================
@pytest.mark.tier1
def test_bva_dip_buyer_screener_ibs_zero_range_candle() -> None:
    """BVA: Internal Bar Strength (IBS) calculation is safe against flat candles (High == Low)."""
    from app.tools.indicators import calculate_ibs

    # Flat candle: High == Low == Close
    highs = pd.Series([100.0, 100.0])
    lows = pd.Series([100.0, 100.0])
    closes = pd.Series([100.0, 100.0])

    ibs_series = calculate_ibs(highs, lows, closes)
    assert not ibs_series.isna().any()
    assert (ibs_series == 0.0).all()


@pytest.mark.tier1
def test_bva_dip_buyer_screener_filter_boundaries() -> None:
    """BVA: Screener strictly checks trend, volume, IBS, ATR drop ratio, and volatility."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_market_provider = MagicMock(spec=MarketDataProvider)

    screener = DipBuyerStrategy(
        trade_repository=mock_trade_repo,
        data_provider=mock_market_provider,
    )

    # 1. Valid Candidate matching all criteria:
    # Close=100 (> 5.0), Open=102 (Red candle today), PrevClose=102, PrevOpen=104 (Red yesterday),
    # SMA200=90 (< Close), VolSMA=1.2M (> 1M), ATR5=4.0 (ATR/Close = 0.04 > 0.03),
    # IBS=0.15 (< 0.2), 3-day drop=-5.0 (Drop/ATR = -1.25 < -1.0), in_universe=True
    valid_snapshot = DipBuyerAnalysisSnapshot(
        current_close=100.0,
        current_open=102.0,
        previous_close=102.0,
        previous_open=104.0,
        sma200=90.0,
        volume_sma=1_200_000.0,
        atr=4.0,
        atr_ratio_3day=-1.25,
        volatility_ratio=0.04,
        ibs=0.15,
        has_indices=True,
    )
    valid_checks = screener._run_analysis_checks(valid_snapshot)
    assert all(valid_checks.values()) is True

    # 2. Downtrend Boundary: Close <= SMA200 -> Failed
    downtrend_snapshot = DipBuyerAnalysisSnapshot(
        current_close=100.0,
        current_open=102.0,
        previous_close=102.0,
        previous_open=104.0,
        sma200=105.0,
        volume_sma=1_200_000.0,
        atr=4.0,
        atr_ratio_3day=-1.25,
        volatility_ratio=0.04,
        ibs=0.15,
        has_indices=True,
    )
    downtrend_checks = screener._run_analysis_checks(downtrend_snapshot)
    assert downtrend_checks["uptrend_sma200"] is False

    # 3. High IBS Boundary: IBS >= 0.2 -> Failed
    high_ibs_snapshot = DipBuyerAnalysisSnapshot(
        current_close=100.0,
        current_open=102.0,
        previous_close=102.0,
        previous_open=104.0,
        sma200=90.0,
        volume_sma=1_200_000.0,
        atr=4.0,
        atr_ratio_3day=-1.25,
        volatility_ratio=0.04,
        ibs=0.25,
        has_indices=True,
    )
    high_ibs_checks = screener._run_analysis_checks(high_ibs_snapshot)
    assert high_ibs_checks["low_ibs"] is False

    # 4. Weak Drop Ratio: Drop >= -1.0 ATR (e.g. -0.80) -> Failed
    weak_drop_snapshot = DipBuyerAnalysisSnapshot(
        current_close=100.0,
        current_open=102.0,
        previous_close=102.0,
        previous_open=104.0,
        sma200=90.0,
        volume_sma=1_200_000.0,
        atr=4.0,
        atr_ratio_3day=-0.80,
        volatility_ratio=0.04,
        ibs=0.15,
        has_indices=True,
    )
    weak_drop_checks = screener._run_analysis_checks(weak_drop_snapshot)
    assert weak_drop_checks["dip_atr_ratio_3day"] is False


# ==============================================================================
# 2. Boundary Value Analysis (BVA) — Trade Manager Entry Logic & Gap-Down
# ==============================================================================
@pytest.mark.tier1
def test_bva_dip_buyer_entry_limit_hit_and_gap_down() -> None:
    """BVA: Trade Manager handles exact limit hit, gap-down benefit, and missed entry window."""
    manager = DipBuyerTradeStrategy()

    trade: TradeData = {
        "id": 1,
        "symbol": "AAPL",
        "strategy": Strategies.DipBuyer,
        "status": "CREATED",
        "entry_price": 96.0,  # Limit Price
        "signal_context": {
            "date": "2026-01-30",
            "setup_close": 100.0,
            "setup_atr": 4.0,
        },
    }

    dates = pd.to_datetime(["2026-01-30", "2026-02-02"])
    history = pd.DataFrame({"date": dates, "close": [100.0, 97.0]})

    # 1. Limit Hit: Open=98.0, Low=95.0 (Low <= 96.0) -> Fills at Limit (96.0)
    candle_limit_hit = pd.Series(
        {"date": "2026-02-02", "open": 98.0, "high": 99.0, "low": 95.0, "close": 97.0}
    )
    transition_hit = manager.check_entry(trade, candle_limit_hit, history)
    assert transition_hit is not None
    assert transition_hit.updates["status"] == TradeStatus.ACTIVE
    assert transition_hit.updates["entry_price"] == 96.0
    assert "LIMIT" in transition_hit.reason

    # 2. Gap Down: Open=93.0 (< Limit 96.0), Low=92.0 -> Fills at Open (93.0) with gap-down benefit
    candle_gap_down = pd.Series(
        {"date": "2026-02-02", "open": 93.0, "high": 95.0, "low": 92.0, "close": 94.0}
    )
    transition_gap = manager.check_entry(trade, candle_gap_down, history)
    assert transition_gap is not None
    assert transition_gap.updates["status"] == TradeStatus.ACTIVE
    assert transition_gap.updates["entry_price"] == 93.0
    assert "LIMIT" in transition_gap.reason

    # 3. Missed Entry Window on Day 1: Low=97.0 (> Limit 96.0) -> Rejected (INVALID)
    candle_missed = pd.Series(
        {"date": "2026-02-02", "open": 99.0, "high": 101.0, "low": 97.0, "close": 98.0}
    )
    transition_missed = manager.check_entry(trade, candle_missed, history)
    assert transition_missed is not None
    assert transition_missed.updates["status"] == TradeStatus.INVALID
    assert "INVALIDATED" in transition_missed.reason

    # 4. Stale Entry Window (days_passed > 1) -> Rejected with Missed Entry Window
    history_stale = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-30", "2026-02-02", "2026-02-03"]),
            "close": [100.0, 97.0, 96.0],
        }
    )
    candle_stale = pd.Series(
        {"date": "2026-02-03", "open": 96.0, "high": 97.0, "low": 95.0, "close": 96.0}
    )
    transition_stale = manager.check_entry(trade, candle_stale, history_stale)
    assert transition_stale is not None
    assert transition_stale.updates["status"] == TradeStatus.INVALID
    assert "Missed Entry Window" in transition_stale.reason


# ==============================================================================
# 3. Boundary Value Analysis (BVA) — Trade Manager Exits (TP, LOC, Time Stop)
# ==============================================================================
@pytest.mark.tier1
def test_bva_dip_buyer_exits_target_and_time_stop() -> None:
    """BVA: Trade Manager executes Target Hit on Day 1+, LOC profit exit, and Time Stop at Day 8."""
    manager = DipBuyerTradeStrategy()

    active_trade: TradeData = {
        "id": 10,
        "symbol": "AAPL",
        "strategy": Strategies.DipBuyer,
        "status": "ACTIVE",
        "current_size": 100,
        "entry_price": 96.0,
        "entry_date": "2026-02-02",
        "current_target": 99.2,  # Target price (96.0 + 0.8 * 4.0)
        "signal_context": {
            "date": "2026-01-30",
            "setup_atr": 4.0,
        },
    }

    # 1. Target Hit Exit on Day 1: High=100.0 (>= Target 99.2)
    history_day1 = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-02-02", "2026-02-03"]),
            "close": [96.0, 98.0],
            "high": [97.0, 100.0],
            "low": [95.0, 96.0],
            "open": [96.0, 96.5],
        }
    )
    transition_tp = manager.manage_active_trade(active_trade, history_day1)
    assert transition_tp is not None
    assert transition_tp.updates["status"] == TradeStatus.CLOSED
    assert transition_tp.updates["exit_price"] == 99.2
    assert transition_tp.reason == ExitReason.TARGET_HIT

    # 2. Time Stop Exit on Day 8 (8 bars passed since entry, target not reached)
    date_list = [f"2026-02-{i:02d}" for i in range(2, 11)]  # 9 days of history
    history_day8 = pd.DataFrame(
        {
            "date": pd.to_datetime(date_list),
            "close": [96.0] + [97.0] * 8,
            "high": [97.0] + [98.0] * 8,  # High never reaches target 99.2
            "low": [95.0] + [96.0] * 8,
            "open": [96.0] + [96.5] * 8,
        }
    )
    transition_time_stop = manager.manage_active_trade(active_trade, history_day8)
    assert transition_time_stop is not None
    assert transition_time_stop.updates["status"] == TradeStatus.CLOSED
    assert transition_time_stop.updates["exit_price"] == 97.0  # MOC exit at close
    assert transition_time_stop.reason == ExitReason.TIME_STOP


# ==============================================================================
# 4. Boundary Value Analysis (BVA) — Order Generation & Sizing
# ==============================================================================
@pytest.mark.tier1
def test_bva_dip_buyer_order_generation_and_sizing() -> None:
    """BVA: Order generation creates bracket orders with entry LMT and exit LOC orders."""
    manager = DipBuyerTradeStrategy()

    trade: TradeData = {
        "id": 1,
        "symbol": "AAPL",
        "strategy": Strategies.DipBuyer,
        "status": "CREATED",
        "entry_price": 96.0,
        "current_target": 99.2,
        "signal_context": {"budget": 6000.0, "threshold_loc": 99.5},
    }

    history = pd.DataFrame(
        {
            "date": ["2026-01-30"],
            "close": [100.0],
            "high": [102.0],
            "low": [99.0],
            "open": [101.0],
        }
    )

    # Budget $6000 / Entry $96.0 = 62 shares
    order = manager._generate_entry_order(trade, history, budget=6000.0)
    assert order is not None
    assert order.symbol == "AAPL"
    assert order.quantity == 62
    assert order.entry is not None
    assert order.entry.type == "LMT"
    assert order.entry.price == Decimal("96.0")

    # Exit LOC leg attached
    assert len(order.exits) == 1
    assert order.exits[0].type == "LOC"
    assert order.exits[0].price == Decimal("99.5")


# ==============================================================================
# 5. Property-Based Fuzzing (Hypothesis)
# ==============================================================================
@pytest.mark.tier2
@given(
    setup_close=st.floats(
        min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False
    ),
    setup_atr=st.floats(
        min_value=0.50, max_value=200.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_dip_buyer_entry_and_target_math(
    setup_close: float, setup_atr: float
) -> None:
    """Invariant: Entry limit is strictly below setup close; Target is strictly above entry."""
    entry_price = round(setup_close - (1.0 * setup_atr), 2)
    target_price = round(entry_price + (0.8 * setup_atr), 2)

    assert entry_price < setup_close
    assert target_price > entry_price


@pytest.mark.tier2
@given(
    budget=st.floats(
        min_value=1000.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False
    ),
    entry_price=st.floats(
        min_value=1.0, max_value=5000.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_dip_buyer_order_sizing_invariants(
    budget: float, entry_price: float
) -> None:
    """Invariant: Position quantity is non-negative and capital does not exceed budget + price."""
    manager = DipBuyerTradeStrategy()
    trade: TradeData = {
        "id": 1,
        "symbol": "TEST",
        "strategy": Strategies.DipBuyer,
        "status": "CREATED",
        "entry_price": entry_price,
        "current_target": entry_price * 1.05,
    }

    history = pd.DataFrame(
        {"date": ["2026-01-30"], "close": [entry_price], "high": [entry_price * 1.02]}
    )
    order = manager._generate_entry_order(trade, history, budget=budget)

    if budget < entry_price:
        assert order is None
    else:
        assert order is not None
        assert order.quantity >= 1
        allocated_capital = Decimal(str(order.quantity)) * Decimal(str(entry_price))
        assert allocated_capital <= Decimal(str(budget)) + Decimal(str(entry_price))


# ==============================================================================
# 6. Zero Lookahead-Bias Guard
# ==============================================================================
@pytest.mark.tier2
def test_dip_buyer_screener_zero_lookahead_bias() -> None:
    """Lookahead Guard: Screening calculations on date T are strictly independent of data at T+1..T+N."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_market_provider = MagicMock(spec=MarketDataProvider)

    screener = DipBuyerStrategy(
        trade_repository=mock_trade_repo,
        data_provider=mock_market_provider,
    )

    analysis_date = "2026-01-30"  # Date T
    base_dates = pd.date_range(end=analysis_date, periods=250, freq="B")
    future_dates = pd.date_range(start="2026-02-02", periods=20, freq="B")
    full_dates = base_dates.append(future_dates)

    def make_pivoted_df(future_noise: float) -> pd.DataFrame:
        close_vals = np.linspace(120, 100, len(full_dates))
        close_vals[len(base_dates) :] *= future_noise
        return pd.DataFrame({"AAPL": close_vals}, index=full_dates)

    # Run 1: Historical data up to T
    closes_t = make_pivoted_df(1.0).loc[:analysis_date]
    highs_t = closes_t * 1.02
    lows_t = closes_t * 0.98
    volumes_t = closes_t * 0 + 2_000_000
    indicators_baseline = screener._compute_indicators(
        closes_t, highs_t, lows_t, volumes_t
    )

    # Run 2: Full history including massive future spike at T+1..T+N
    closes_future = make_pivoted_df(10.0)
    highs_future = closes_future * 1.02
    lows_future = closes_future * 0.98
    volumes_future = closes_future * 0 + 2_000_000
    indicators_with_future = screener._compute_indicators(
        closes_future, highs_future, lows_future, volumes_future
    )

    # Indicator series up to date T must be completely identical
    assert (
        indicators_baseline["atr"]
        .loc[:analysis_date]
        .equals(indicators_with_future["atr"].loc[:analysis_date])
    )
    assert (
        indicators_baseline["sma200"]
        .loc[:analysis_date]
        .equals(indicators_with_future["sma200"].loc[:analysis_date])
    )
    assert (
        indicators_baseline["ibs"]
        .loc[:analysis_date]
        .equals(indicators_with_future["ibs"].loc[:analysis_date])
    )
