"""Lookahead Bias & Data Leakage Prevention Tests.

Verifies that calculations and signals at time T do not access or depend on data
from time T+1..T+N (Time-Shift Invariance).
"""

import numpy as np
import pandas as pd
import pytest

from app.tools.indicators import (
    calculate_atr,
    calculate_ibs,
    calculate_rsi,
    calculate_sma,
)


@pytest.fixture
def sample_eod_price_history() -> pd.DataFrame:
    """Creates a deterministic 100-day price history."""
    dates = pd.date_range(start="2026-01-01", periods=100, freq="B")
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, size=100)
    close_prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": close_prices * 0.99,
            "high": close_prices * 1.02,
            "low": close_prices * 0.98,
            "close": close_prices,
            "volume": np.random.randint(100_000, 1_000_000, size=100),
        }
    ).set_index("date")
    return df


@pytest.mark.tier2
def test_sma_has_zero_lookahead_bias(sample_eod_price_history: pd.DataFrame) -> None:
    """Invariant: SMA(T) must be identical whether calculated on df[0:T] or full df."""
    full_series = sample_eod_price_history["close"]
    cutoff_idx = 50  # Day 50

    # Run on full dataset
    sma_full = calculate_sma(full_series, window=20)

    # Run on truncated dataset (strictly no future data available)
    truncated_series = full_series.iloc[:cutoff_idx]
    sma_truncated = calculate_sma(truncated_series, window=20)

    # Value at Day 50 must match to the exact float representation
    val_full = sma_full.iloc[cutoff_idx - 1]
    val_truncated = sma_truncated.iloc[-1]

    assert val_full == pytest.approx(val_truncated, abs=1e-12)


@pytest.mark.tier2
def test_rsi_has_zero_lookahead_bias(sample_eod_price_history: pd.DataFrame) -> None:
    """Invariant: RSI(T) must be identical whether calculated on df[0:T] or full df."""
    full_series = sample_eod_price_history["close"]
    cutoff_idx = 60

    rsi_full = calculate_rsi(full_series, window=14)
    rsi_truncated = calculate_rsi(full_series.iloc[:cutoff_idx], window=14)

    val_full = rsi_full.iloc[cutoff_idx - 1]
    val_truncated = rsi_truncated.iloc[-1]

    assert val_full == pytest.approx(val_truncated, abs=1e-12)


@pytest.mark.tier2
def test_atr_has_zero_lookahead_bias(sample_eod_price_history: pd.DataFrame) -> None:
    """Invariant: ATR(T) must be identical whether calculated on df[0:T] or full df."""
    df = sample_eod_price_history
    cutoff_idx = 50

    atr_full = calculate_atr(df["high"], df["low"], df["close"], window=14)
    atr_truncated = calculate_atr(
        df["high"].iloc[:cutoff_idx],
        df["low"].iloc[:cutoff_idx],
        df["close"].iloc[:cutoff_idx],
        window=14,
    )

    val_full = atr_full.iloc[cutoff_idx - 1]
    val_truncated = atr_truncated.iloc[-1]

    assert val_full == pytest.approx(val_truncated, abs=1e-12)


@pytest.mark.tier2
def test_ibs_has_zero_lookahead_bias(sample_eod_price_history: pd.DataFrame) -> None:
    """Invariant: IBS is calculated purely point-in-time per bar."""
    df = sample_eod_price_history
    ibs = calculate_ibs(df["high"], df["low"], df["close"])

    for i in range(10, 20):
        single_bar_ibs = calculate_ibs(
            df["high"].iloc[i : i + 1],
            df["low"].iloc[i : i + 1],
            df["close"].iloc[i : i + 1],
        ).iloc[0]
        assert single_bar_ibs == pytest.approx(ibs.iloc[i], abs=1e-12)
