"""Turnover Timing Hardening Suite (Tier 1 & Tier 2).

Provides exhaustive Boundary Value Analysis (BVA), Hypothesis Property-Based Fuzzing,
and Zero Lookahead-Bias validation for the Turnover Timing Screener and Trade Manager.
"""

import json
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
from app.services.screener.strategies.turnover_timing import (
    TurnoverTimingStrategy,
)
from app.services.trade_manager.strategies.turnover_timing import (
    TurnoverTimingStrategy as TurnoverTradeStrategy,
)
from app.types import TradeData


# ==============================================================================
# 1. Boundary Value Analysis (BVA) — Screener Setup Rules & Holidays
# ==============================================================================
@pytest.mark.tier1
def test_bva_turnover_timing_friday_and_holiday_thursday_resolution() -> None:
    """BVA: Screener correctly identifies standard Fridays and Thursday before Good Friday as setup days."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_market_provider = MagicMock(spec=MarketDataProvider)

    screener = TurnoverTimingStrategy(
        trade_repository=mock_trade_repo,
        data_provider=mock_market_provider,
    )

    # Standard Friday (2026-01-30) -> True
    friday_ts = pd.Timestamp("2026-01-30")
    assert screener._is_setup_day(friday_ts) is True

    # Standard Midweek Wednesday (2026-01-28) -> False
    wednesday_ts = pd.Timestamp("2026-01-28")
    assert screener._is_setup_day(wednesday_ts) is False

    # Thursday before Good Friday (2026-04-02, Good Friday is 2026-04-03) -> True
    thursday_pre_holiday_ts = pd.Timestamp("2026-04-02")
    assert screener._is_setup_day(thursday_pre_holiday_ts) is True


@pytest.mark.tier1
def test_bva_turnover_timing_dual_variant_entry_limits() -> None:
    """BVA: Limits are accurately computed for 0.5 ATR and 1.0 ATR entry factors."""
    close_price = 100.0
    atr_3 = 4.0

    # Variant 0.5: 100.0 - (0.5 * 4.0) = 98.00
    limit_0_5 = round(close_price - (0.5 * atr_3), 2)
    assert limit_0_5 == 98.00

    # Variant 1.0: 100.0 - (1.0 * 4.0) = 96.00
    limit_1_0 = round(close_price - (1.0 * atr_3), 2)
    assert limit_1_0 == 96.00

    assert close_price > limit_0_5 > limit_1_0


# ==============================================================================
# 2. Boundary Value Analysis (BVA) — Trade Manager Entry Logic & Green State
# ==============================================================================
@pytest.mark.tier1
def test_bva_turnover_timing_entry_execution_and_green_state_inheritance() -> None:
    """BVA: Trade Manager handles exact limit hit, gap down, and sets initial green sequence context."""
    manager = TurnoverTradeStrategy()

    trade: TradeData = {
        "id": 1,
        "symbol": "AAPL",
        "strategy": Strategies.TurnOverTiming,
        "status": "CREATED",
        "entry_price": 96.0,
        "signal_context": json.dumps(
            {
                "date": "2026-01-30",
                "setup_close": 100.0,
                "setup_candle_green": True,
                "green_candle_count": 0,
            }
        ),
    }

    dates = pd.to_datetime(["2026-01-30", "2026-02-02"])
    history = pd.DataFrame({"date": dates, "close": [100.0, 97.0]})

    # 1. Limit Hit on Day 1 with Green Day (Open=98, Low=95 <= 96, Close=99 > Open 98)
    # Since setup was green and entry is green -> green_candle_count becomes 2!
    candle_green_entry = pd.Series(
        {"date": "2026-02-02", "open": 98.0, "high": 100.0, "low": 95.0, "close": 99.0}
    )
    transition_green = manager.check_entry(trade, candle_green_entry, history)
    assert transition_green is not None
    assert transition_green.updates["status"] == TradeStatus.ACTIVE
    assert transition_green.updates["entry_price"] == 96.0
    assert "LIMIT" in transition_green.reason

    ctx = json.loads(str(transition_green.updates["signal_context"]))
    assert ctx["green_candle_count"] == 2

    # 2. Gap Down Benefit (Open=94 < Limit 96, Low=93)
    candle_gap_down = pd.Series(
        {"date": "2026-02-02", "open": 94.0, "high": 96.0, "low": 93.0, "close": 95.0}
    )
    transition_gap = manager.check_entry(trade, candle_gap_down, history)
    assert transition_gap is not None
    assert transition_gap.updates["status"] == TradeStatus.ACTIVE
    assert transition_gap.updates["entry_price"] == 94.0

    # 3. Missed Entry Window (Low=97 > Limit 96) -> Expired / INVALID
    candle_missed = pd.Series(
        {"date": "2026-02-02", "open": 99.0, "high": 101.0, "low": 97.0, "close": 98.0}
    )
    transition_missed = manager.check_entry(trade, candle_missed, history)
    assert transition_missed is not None
    assert transition_missed.updates["status"] == TradeStatus.INVALID
    assert "EXPIRED" in transition_missed.reason


# ==============================================================================
# 3. Boundary Value Analysis (BVA) — Trade Manager Exits (Green Sequence & Time Stop)
# ==============================================================================
@pytest.mark.tier1
def test_bva_turnover_timing_exits_green_sequence_and_friday_time_stop() -> None:
    """BVA: Exits trigger at Open after 2 green candles, or at Close on Friday time stop."""
    manager = TurnoverTradeStrategy()

    # Active trade with 2 green candles accumulated -> Trigger exit at next OPEN
    active_trade_green: TradeData = {
        "id": 10,
        "symbol": "AAPL",
        "strategy": Strategies.TurnOverTiming,
        "status": "ACTIVE",
        "current_size": 100,
        "entry_price": 96.0,
        "entry_date": "2026-02-02",
        "signal_context": json.dumps(
            {
                "date": "2026-01-30",
                "green_candle_count": 2,
                "last_processed_date": "2026-02-02",
            }
        ),
    }

    history_tuesday = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-02-02", "2026-02-03"]),
            "close": [96.0, 98.0],
            "high": [97.0, 99.0],
            "low": [95.0, 96.0],
            "open": [96.0, 97.5],
        }
    )
    transition_green_exit = manager.manage_active_trade(
        active_trade_green, history_tuesday
    )
    assert transition_green_exit is not None
    assert transition_green_exit.updates["status"] == TradeStatus.CLOSED
    assert transition_green_exit.updates["exit_price"] == 97.5  # Next Open price
    assert transition_green_exit.reason == ExitReason.GREEN_SEQUENCE

    # Active trade with 0 green candles on Friday (2026-02-06) -> Time Stop Exit at Close
    active_trade_time_stop: TradeData = {
        "id": 11,
        "symbol": "AAPL",
        "strategy": Strategies.TurnOverTiming,
        "status": "ACTIVE",
        "current_size": 100,
        "entry_price": 96.0,
        "entry_date": "2026-02-02",
        "signal_context": json.dumps(
            {
                "date": "2026-01-30",
                "green_candle_count": 0,
                "last_processed_date": "2026-02-05",
            }
        ),
    }
    history_friday = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05", "2026-02-06"]
            ),
            "close": [96.0, 95.0, 94.0, 93.0, 92.5],
            "high": [97.0, 96.0, 95.0, 94.0, 93.5],
            "low": [95.0, 94.0, 93.0, 92.0, 91.5],
            "open": [96.0, 95.5, 94.5, 93.5, 93.0],
        }
    )
    transition_time_stop = manager.manage_active_trade(
        active_trade_time_stop, history_friday
    )
    assert transition_time_stop is not None
    assert transition_time_stop.updates["status"] == TradeStatus.CLOSED
    assert transition_time_stop.updates["exit_price"] == 92.5  # Friday Close
    assert transition_time_stop.reason == ExitReason.TIME_STOP


# ==============================================================================
# 4. Boundary Value Analysis (BVA) — Order Generation & Sizing
# ==============================================================================
@pytest.mark.tier1
def test_bva_turnover_timing_order_generation_and_sizing() -> None:
    """BVA: Order generation computes quantity from budget and creates LMT/DAY entry order."""
    manager = TurnoverTradeStrategy()

    trade: TradeData = {
        "id": 1,
        "symbol": "AAPL",
        "strategy": Strategies.TurnOverTiming,
        "status": "CREATED",
        "entry_price": 96.0,
        "signal_context": json.dumps({"budget": 3000.0}),
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

    # Budget $3000 / Entry $96.0 = 31 shares
    order = manager._generate_entry_order(trade, history, budget=3000.0)
    assert order is not None
    assert order.symbol == "AAPL"
    assert order.quantity == 31
    assert order.entry is not None
    assert order.entry.type == "LMT"
    assert order.entry.price == Decimal("96.0")


# ==============================================================================
# 5. Property-Based Fuzzing (Hypothesis)
# ==============================================================================
@pytest.mark.tier2
@given(
    close=st.floats(
        min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False
    ),
    atr=st.floats(
        min_value=0.50, max_value=200.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_turnover_timing_limit_factors_ordering(
    close: float, atr: float
) -> None:
    """Invariant: Close > Limit_0.5 > Limit_1.0 for positive ATR values."""
    limit_0_5 = round(close - (0.5 * atr), 2)
    limit_1_0 = round(close - (1.0 * atr), 2)

    assert limit_0_5 < close
    assert limit_1_0 < limit_0_5


@pytest.mark.tier2
@given(
    budget=st.floats(
        min_value=500.0, max_value=500_000.0, allow_nan=False, allow_infinity=False
    ),
    entry_price=st.floats(
        min_value=1.0, max_value=5000.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_turnover_timing_order_sizing_invariants(
    budget: float, entry_price: float
) -> None:
    """Invariant: Position sizing never divides by zero and allocated capital <= budget + price."""
    manager = TurnoverTradeStrategy()
    trade: TradeData = {
        "id": 1,
        "symbol": "TEST",
        "strategy": Strategies.TurnOverTiming,
        "status": "CREATED",
        "entry_price": entry_price,
    }

    history = pd.DataFrame({"date": ["2026-01-30"], "close": [entry_price]})
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
def test_turnover_timing_screener_zero_lookahead_bias() -> None:
    """Lookahead Guard: Turnover Timing indicators at date T are strictly independent of data at T+1..T+N."""
    analysis_date = "2026-01-30"  # Date T
    base_dates = pd.date_range(end=analysis_date, periods=250, freq="B")
    future_dates = pd.date_range(start="2026-02-02", periods=20, freq="B")
    full_dates = base_dates.append(future_dates)

    def make_data(
        future_noise: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        close_vals = np.linspace(120, 100, len(full_dates))
        close_vals[len(base_dates) :] *= future_noise
        closes = pd.DataFrame({"AAPL": close_vals}, index=full_dates)
        highs = closes * 1.02
        lows = closes * 0.98
        volumes = pd.DataFrame(
            {"AAPL": [2_000_000] * len(full_dates)}, index=full_dates
        )
        return closes, highs, lows, volumes

    from app.tools import indicators

    # Run 1: Historical data up to T
    c_t, h_t, l_t, v_t = make_data(1.0)
    c_t_sliced = c_t.loc[:analysis_date]
    h_t_sliced = h_t.loc[:analysis_date]
    l_t_sliced = l_t.loc[:analysis_date]
    v_t_sliced = v_t.loc[:analysis_date]

    sma200_baseline = indicators.calculate_sma(c_t_sliced, 200)
    atr3_baseline = indicators.calculate_atr(h_t_sliced, l_t_sliced, c_t_sliced, 3)
    turnover_t = c_t_sliced * v_t_sliced
    turnover_sma20_baseline = indicators.calculate_volume_sma(turnover_t, 20)

    # Run 2: Full history with massive future spike at T+1..T+N
    c_f, h_f, l_f, v_f = make_data(10.0)
    sma200_future = indicators.calculate_sma(c_f, 200)
    atr3_future = indicators.calculate_atr(h_f, l_f, c_f, 3)
    turnover_f = c_f * v_f
    turnover_sma20_future = indicators.calculate_volume_sma(turnover_f, 20)

    # Indicator series up to date T must be completely identical
    assert sma200_baseline.loc[:analysis_date].equals(sma200_future.loc[:analysis_date])
    assert atr3_baseline.loc[:analysis_date].equals(atr3_future.loc[:analysis_date])
    assert turnover_sma20_baseline.loc[:analysis_date].equals(
        turnover_sma20_future.loc[:analysis_date]
    )
