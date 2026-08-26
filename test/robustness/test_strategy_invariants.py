"""Property-based fuzzing tests for financial indicators and strategy calculations.

Verifies mathematical invariants and resilience against extreme, randomized price feeds
using the Hypothesis framework.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.tools.indicators import (
    calculate_atr,
    calculate_ibs,
    calculate_max_close_for_rsi,
    calculate_rsi,
    calculate_sma,
    calculate_true_range,
)


@st.composite
def ohlcv_series_strategy(draw: st.DrawFn) -> pd.DataFrame:
    """Generates synthetic, structurally valid OHLCV price series."""
    size = draw(st.integers(min_value=20, max_value=200))

    # Generate realistic or extreme base close prices
    close_prices = draw(
        st.lists(
            st.floats(
                min_value=0.01, max_value=5000.0, allow_nan=False, allow_infinity=False
            ),
            min_size=size,
            max_size=size,
        )
    )

    df = pd.DataFrame({"close": close_prices})

    # Generate offsets for high, low, open
    high_offsets = draw(
        st.lists(
            st.floats(
                min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False
            ),
            min_size=size,
            max_size=size,
        )
    )
    low_offsets = draw(
        st.lists(
            st.floats(
                min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False
            ),
            min_size=size,
            max_size=size,
        )
    )

    df["high"] = df["close"] + high_offsets
    df["low"] = (df["close"] - low_offsets).clip(lower=0.001)
    df["open"] = (df["low"] + (df["high"] - df["low"]) * 0.5).clip(lower=0.001)
    df["volume"] = draw(
        st.lists(
            st.integers(min_value=0, max_value=100_000_000),
            min_size=size,
            max_size=size,
        )
    )

    # Invariant check: high >= low, high >= close, low <= close
    df["high"] = df[["high", "close", "open"]].max(axis=1)
    df["low"] = df[["low", "close", "open"]].min(axis=1)

    return df


@pytest.mark.tier2
@given(
    st.lists(
        st.floats(
            min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False
        ),
        min_size=20,
        max_size=150,
    )
)
@settings(max_examples=50, deadline=None)
def test_rsi_bounds_and_no_nan_post_warmup(prices: list[float]) -> None:
    """Invariant: RSI must strictly stay in [0.0, 100.0] and have no NaN post-warmup."""
    series = pd.Series(prices)
    window = 14
    rsi = calculate_rsi(series, window=window)

    valid_rsi = rsi.dropna()
    assert not valid_rsi.empty
    assert (valid_rsi >= 0.0).all(), f"RSI dropped below 0: {valid_rsi.min()}"
    assert (valid_rsi <= 100.0).all(), f"RSI exceeded 100: {valid_rsi.max()}"
    assert not np.isinf(valid_rsi).any(), "RSI produced Inf values"


@pytest.mark.tier2
@given(ohlcv_series_strategy())
@settings(max_examples=50, deadline=None)
def test_true_range_and_atr_invariants(df: pd.DataFrame) -> None:
    """Invariant: True Range and ATR must be strictly non-negative and finite."""
    tr = calculate_true_range(df["high"], df["low"], df["close"])
    assert (tr.dropna() >= 0.0).all(), "True Range produced negative values"

    atr = calculate_atr(df["high"], df["low"], df["close"], window=14)
    valid_atr = atr.dropna()
    assert (valid_atr >= 0.0).all(), "ATR produced negative values"
    assert not np.isinf(valid_atr).any(), "ATR produced Inf values"


@pytest.mark.tier2
@given(ohlcv_series_strategy())
@settings(max_examples=50, deadline=None)
def test_ibs_invariants(df: pd.DataFrame) -> None:
    """Invariant: IBS must be finite and well-defined even when High == Low."""
    ibs = calculate_ibs(df["high"], df["low"], df["close"])
    assert not ibs.isna().any(), "IBS produced NaN values"
    assert not np.isinf(ibs).any(), "IBS produced Inf values"


@pytest.mark.tier2
@given(
    st.lists(
        st.floats(
            min_value=0.01, max_value=5000.0, allow_nan=False, allow_infinity=False
        ),
        min_size=20,
        max_size=100,
    )
)
@settings(max_examples=50, deadline=None)
def test_sma_invariants(prices: list[float]) -> None:
    """Invariant: SMA must stay between min(prices) and max(prices) of each window."""
    series = pd.Series(prices)
    window = 10
    sma = calculate_sma(series, window=window)
    valid_sma = sma.dropna()

    assert len(valid_sma) == len(prices) - window + 1
    assert (valid_sma >= min(prices)).all()
    assert (valid_sma <= max(prices)).all()


@pytest.mark.tier2
@given(
    st.lists(
        st.floats(
            min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False
        ),
        min_size=15,
        max_size=60,
    ),
    st.floats(min_value=10.0, max_value=90.0),
)
@settings(max_examples=50, deadline=None)
def test_max_close_for_rsi_is_finite(prices: list[float], rsi_target: float) -> None:
    """Invariant: Max close required for RSI must return a finite float or NaN on insufficient data."""
    series = pd.Series(prices)
    max_close = calculate_max_close_for_rsi(series, window=2, rsi_target=rsi_target)

    assert not np.isinf(max_close), "calculate_max_close_for_rsi returned Inf"
