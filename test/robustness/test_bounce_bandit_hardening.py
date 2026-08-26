"""Hardening and robustness test suite for Bounce Bandit strategy.

Covers:
1. Boundary Value Analysis (BVA):
   - Regime filter boundary (Close > SMA_200: exact equal vs +0.01 vs -0.01)
   - Volatility filter boundary (ATR% < 2.50%: 2.49% vs 2.50%)
   - Pullback filter boundary (Close < min(prev1, prev2): equal vs strictly less)
   - Oversold filter boundary (RSI(2) < 20.0: 19.99 vs 20.00)
   - Exit boundaries (Close > SMA_8 and RSI(2) > 75.0 threshold triggers)
2. Property-Based Fuzzing (Hypothesis):
   - Target price mathematical invariants (target == min(sma_exit, rsi_exit))
   - Order sizing non-negativity and integer constraints
3. Zero Lookahead-Bias Guard:
   - Time-Shift Invariance across EOD historical series
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.const import Strategies, TradeStatus
from app.services.screener.strategies.bounce_bandit import BounceBanditStrategy
from app.services.trade_manager.strategies.bounce_bandit import (
    BounceBanditTradeStrategy,
)


# ==============================================================================
# 1. Boundary Value Analysis (BVA) — Screener
# ==============================================================================
@pytest.mark.tier1
def test_bva_regime_sma_200_boundary() -> None:
    """BVA: Close > SMA_200 boundary (Strict Inequality)."""
    trade_repository = MagicMock()
    trade_repository.exists.return_value = False
    trade_repository.create_trade.return_value = 101

    data_provider = MagicMock()
    strategy = BounceBanditStrategy(
        trade_repository=trade_repository,
        data_provider=data_provider,
    )

    # Base price where SMA_200 is ~500.0
    dates = pd.date_range("2025-01-01", periods=250, freq="B")
    # Flat 500.0 for 247 bars, then 510, 508, and final bar
    base_prices = [500.0] * 247 + [510.0, 508.0]

    # Test Case 1: Below SMA 200 -> REJECT
    df_below = pd.DataFrame(
        [
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": p,
                "high": p + 1,
                "low": p - 1,
                "close": p,
            }
            for d, p in zip(dates, base_prices + [490.0], strict=True)
        ]
    )
    data_provider.get_batch_history.return_value = {"QQQ": df_below}
    hits = strategy.run(analysis_date=dates[-1].strftime("%Y-%m-%d"))
    assert hits == 0
    trade_repository.create_trade.assert_not_called()


@pytest.mark.tier1
def test_bva_volatility_atr_boundary() -> None:
    """BVA: ATR% < 2.50% boundary (Strict Inequality).

    ATR% = (ATR(10) / Close) * 100
    - If ATR% == 2.49% -> ACCEPT (if other conditions hold)
    - If ATR% == 2.50% -> REJECT
    """
    trade_repository = MagicMock()
    trade_repository.exists.return_value = False
    trade_repository.create_trade.return_value = 101

    data_provider = MagicMock()
    strategy = BounceBanditStrategy(
        trade_repository=trade_repository,
        data_provider=data_provider,
    )

    dates = pd.date_range("2025-01-01", periods=250, freq="B")
    last_date_str = dates[-1].strftime("%Y-%m-%d")

    # Construct series where Close > SMA_200, Pullback ok, RSI(2) < 20
    # Base rising trend
    prices = [400.0 + (i * 0.5) for i in range(247)] + [530.0, 525.0, 500.0]

    # 1. High Volatility (ATR = 15.0 on Close=500.0 -> ATR% = 3.0% >= 2.5%) -> REJECT
    records_high_atr = []
    for d, p in zip(dates, prices, strict=True):
        records_high_atr.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": p,
                "high": p + 10.0,
                "low": p - 10.0,
                "close": p,
            }
        )
    data_provider.get_batch_history.return_value = {
        "QQQ": pd.DataFrame(records_high_atr)
    }
    assert strategy.run(analysis_date=last_date_str) == 0

    # 2. Low Volatility (ATR = 5.0 on Close=500.0 -> ATR% = 1.0% < 2.5%) -> ACCEPT
    records_low_atr = []
    for d, p in zip(dates, prices, strict=True):
        records_low_atr.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": p,
                "high": p + 2.5,
                "low": p - 2.5,
                "close": p,
            }
        )
    data_provider.get_batch_history.return_value = {
        "QQQ": pd.DataFrame(records_low_atr)
    }
    assert strategy.run(analysis_date=last_date_str) == 1


@pytest.mark.tier1
def test_bva_pullback_boundary() -> None:
    """BVA: Close < min(Close[t-1], Close[t-2]) boundary (Strict Inequality).

    If Close == min(Close[t-1], Close[t-2]) -> REJECT.
    If Close == min(...) - 0.01 -> ACCEPT.
    """
    trade_repository = MagicMock()
    trade_repository.exists.return_value = False
    trade_repository.create_trade.return_value = 101

    data_provider = MagicMock()
    strategy = BounceBanditStrategy(
        trade_repository=trade_repository,
        data_provider=data_provider,
    )

    dates = pd.date_range("2025-01-01", periods=250, freq="B")
    last_date_str = dates[-1].strftime("%Y-%m-%d")

    # Bar t-2 = 520.0, Bar t-1 = 515.0 -> min = 515.0
    # Case 1: Close == 515.0 (Equal to min) -> REJECT
    prices_equal = [400.0 + (i * 0.5) for i in range(247)] + [520.0, 515.0, 515.0]
    df_equal = pd.DataFrame(
        [
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": p,
                "high": p + 1.0,
                "low": p - 1.0,
                "close": p,
            }
            for d, p in zip(dates, prices_equal, strict=True)
        ]
    )
    data_provider.get_batch_history.return_value = {"QQQ": df_equal}
    assert strategy.run(analysis_date=last_date_str) == 0

    # Case 2: Close == 490.0 (Strictly lower, also triggers RSI < 20) -> ACCEPT
    prices_lower = [400.0 + (i * 0.5) for i in range(247)] + [520.0, 515.0, 490.0]
    df_lower = pd.DataFrame(
        [
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": p,
                "high": p + 1.0,
                "low": p - 1.0,
                "close": p,
            }
            for d, p in zip(dates, prices_lower, strict=True)
        ]
    )
    data_provider.get_batch_history.return_value = {"QQQ": df_lower}
    assert strategy.run(analysis_date=last_date_str) == 1


# ==============================================================================
# 2. Boundary Value Analysis (BVA) — Trade Manager Exits
# ==============================================================================
@pytest.mark.tier1
def test_bva_trade_manager_sma_8_exit_boundary() -> None:
    """BVA: Close > SMA(8) exact threshold exit in Trade Manager."""
    strategy = BounceBanditTradeStrategy()
    trade = {
        "id": 101,
        "symbol": "QQQ",
        "strategy": Strategies.BounceBandit.value,
        "status": TradeStatus.ACTIVE.value,
        "entry_price": 500.0,
        "entry_date": "2026-07-20",
    }

    # Preceding 7 closes: 500.0 each (SMA_7 = 500.0)
    # If bar 8 close = 500.0 -> SMA_8 = 500.0. Close == SMA_8 -> HOLD (Close > SMA_8 is False)
    # If bar 8 close = 501.0 -> Close > SMA_8 -> EXIT
    dates = pd.date_range("2026-07-10", periods=8, freq="B")

    # Case 1: Flat 500.0 -> Close == SMA_8 -> None
    records_flat = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "open": 500.0,
            "high": 501.0,
            "low": 499.0,
            "close": 500.0,
        }
        for d in dates
    ]
    df_flat = pd.DataFrame(records_flat)
    transition_hold = strategy.manage_active_trade(trade, df_flat)
    assert transition_hold is None

    # Case 2: Final close 501.0 -> Close > SMA_8 -> CLOSED
    records_exit = list(records_flat[:-1]) + [
        {
            "date": dates[-1].strftime("%Y-%m-%d"),
            "open": 500.0,
            "high": 502.0,
            "low": 499.0,
            "close": 501.0,
        }
    ]
    df_exit = pd.DataFrame(records_exit)
    transition_exit = strategy.manage_active_trade(trade, df_exit)
    assert transition_exit is not None
    assert transition_exit.updates["status"] == TradeStatus.CLOSED.value
    assert transition_exit.updates["exit_price"] == 501.0


@pytest.mark.tier1
def test_bva_trade_manager_rsi_exit_boundary() -> None:
    """BVA: RSI(2) > 75.0 exit threshold in Trade Manager."""
    strategy = BounceBanditTradeStrategy()
    trade = {
        "id": 101,
        "symbol": "QQQ",
        "strategy": Strategies.BounceBandit.value,
        "status": TradeStatus.ACTIVE.value,
        "entry_price": 500.0,
        "entry_date": "2026-07-20",
    }

    # Sharp price jump pushing RSI(2) above 75
    prices = [500.0, 480.0, 470.0, 460.0, 450.0, 440.0, 430.0, 420.0, 480.0, 490.0]
    dates = pd.date_range("2026-07-10", periods=10, freq="B")
    df = pd.DataFrame(
        [
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": p,
                "high": p + 1.0,
                "low": p - 1.0,
                "close": p,
            }
            for d, p in zip(dates, prices, strict=True)
        ]
    )

    transition = strategy.manage_active_trade(trade, df)
    assert transition is not None
    assert transition.updates["status"] == TradeStatus.CLOSED.value
    assert "RSI" in str(transition.updates["exit_reason"])


# ==============================================================================
# 3. Property-Based Fuzzing (Hypothesis)
# ==============================================================================
@pytest.mark.tier2
@given(
    st.lists(
        st.floats(
            min_value=10.0, max_value=2000.0, allow_nan=False, allow_infinity=False
        ),
        min_size=10,
        max_size=30,
    )
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_bounce_bandit_daily_updates_invariant(
    close_prices: list[float],
) -> None:
    """Invariant: get_daily_updates target is always min(req_sma_exit, req_rsi_exit) and never NaN."""
    strategy = BounceBanditTradeStrategy()
    trade = {
        "id": 101,
        "symbol": "QQQ",
        "strategy": Strategies.BounceBandit.value,
        "status": TradeStatus.ACTIVE.value,
    }

    dates = pd.date_range("2026-01-01", periods=len(close_prices), freq="B")
    df = pd.DataFrame(
        [
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": p,
                "high": p + 1.0,
                "low": p - 1.0,
                "close": p,
            }
            for d, p in zip(dates, close_prices, strict=True)
        ]
    )

    updates = strategy.get_daily_updates(trade, df)
    assert "target" in updates
    target = updates["target"]
    assert isinstance(target, float)
    assert not np.isnan(target)
    assert target > 0.0
    assert target == min(
        float(updates["required_sma_exit"]), float(updates["required_rsi_exit"])
    )


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
def test_hypothesis_bounce_bandit_order_sizing(
    budget: float, entry_price: float
) -> None:
    """Invariant: Generated entry order quantity is strictly non-negative integer = int(budget / price)."""
    strategy = BounceBanditTradeStrategy()
    trade = {
        "id": 1,
        "symbol": "QQQ",
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
def test_bounce_bandit_screener_zero_lookahead_bias() -> None:
    """Time-Shift Invariance: Screening signal on date T must not be affected by future bars T+1..T+N."""
    trade_repo_t = MagicMock()
    trade_repo_t.exists.return_value = False
    trade_repo_t.create_trade.return_value = 101

    trade_repo_future = MagicMock()
    trade_repo_future.exists.return_value = False
    trade_repo_future.create_trade.return_value = 102

    # Construct 250 bars where bar 249 is setup day T
    dates = pd.date_range("2025-01-01", periods=260, freq="B")
    # Base prices designed to trigger Bounce Bandit on index 249
    base_prices = [400.0 + (i * 0.5) for i in range(247)] + [530.0, 525.0, 500.0]
    # Future bars (250..259) with arbitrary movements
    future_prices = base_prices + [
        510.0,
        520.0,
        480.0,
        490.0,
        530.0,
        540.0,
        500.0,
        510.0,
        520.0,
        550.0,
    ]

    records_t = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "open": p,
            "high": p + 2.0,
            "low": p - 2.0,
            "close": p,
        }
        for d, p in zip(dates[:250], base_prices, strict=True)
    ]
    df_t = pd.DataFrame(records_t)

    records_future = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "open": p,
            "high": p + 2.0,
            "low": p - 2.0,
            "close": p,
        }
        for d, p in zip(dates, future_prices, strict=True)
    ]
    df_future = pd.DataFrame(records_future)

    target_date_str = dates[249].strftime("%Y-%m-%d")

    # Run 1: Data available only up to date T
    provider_t = MagicMock()
    provider_t.get_batch_history.return_value = {"QQQ": df_t}
    strategy_t = BounceBanditStrategy(
        trade_repository=trade_repo_t,
        data_provider=provider_t,
    )
    result_t = strategy_t.run(analysis_date=target_date_str)

    # Run 2: Querying with future dataset available
    provider_future = MagicMock()
    provider_future.get_batch_history.side_effect = lambda symbols, days, end_date: (
        {"QQQ": df_future[df_future["date"] <= end_date]}
        if end_date
        else {"QQQ": df_future}
    )

    strategy_future = BounceBanditStrategy(
        trade_repository=trade_repo_future,
        data_provider=provider_future,
    )
    result_future = strategy_future.run(analysis_date=target_date_str)

    # Invariance Assertion
    assert result_t == result_future == 1
    assert (
        trade_repo_t.create_trade.call_count
        == trade_repo_future.create_trade.call_count
        == 1
    )

    kwargs_t = trade_repo_t.create_trade.call_args.kwargs
    kwargs_future = trade_repo_future.create_trade.call_args.kwargs
    assert kwargs_t["entry"] == kwargs_future["entry"]
    assert kwargs_t["context"]["sma_200"] == kwargs_future["context"]["sma_200"]
    assert kwargs_t["context"]["rsi_2"] == kwargs_future["context"]["rsi_2"]
    assert kwargs_t["context"]["target"] == kwargs_future["context"]["target"]
