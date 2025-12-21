# tests/test_indicator.py - FINAL CORRECTED VERSION
import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.indicator import atr, ema, rma, sma


def _df_from_ohlc(high, low, close, index=None) -> pd.DataFrame:
    """Create DataFrame with LOWERCASE column names to match atr() expectations."""
    if index is None:
        index = pd.RangeIndex(len(close))
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=index)


# ==================== ATR TESTS ====================
def test_atr_sma_true_range_basic():
    """Test basic TR calculation with sma(1) - first value valid."""
    df = _df_from_ohlc([10, 12, 11], [9, 10, 10], [9.5, 11, 10.5])
    prev_close = df["close"].shift()
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    expected = tr.rolling(1).mean().round(2)  # min_periods=1 (default for window=1)
    out = atr(df, intervall=1, smoothing="sma")
    pdt.assert_series_equal(out, expected, check_names=False)


def test_atr_sma_constant_prices_returns_zeros():
    """Constant OHLC -> TR=0 everywhere."""
    df = _df_from_ohlc([10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10])
    out = atr(df, intervall=1, smoothing="sma")
    expected = pd.Series([0.0, 0.0, 0.0, 0.0], index=df.index)
    pdt.assert_series_equal(out, expected, check_names=False)


def test_atr_sma_window_2_matches_rolling_mean_of_tr():
    """Test sma(n=2) matches rolling(2).mean() - first value NaN."""
    df = _df_from_ohlc([10, 13, 12, 14], [9, 11, 11, 13], [9.5, 12, 11.5, 13.5])
    prev_close = df["close"].shift()
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    expected = (
        tr.rolling(2).mean().round(2)
    )  # FIXED: default min_periods=2 -> first NaN
    out = atr(df, intervall=2, smoothing="sma")
    pdt.assert_series_equal(out, expected, check_names=False)


def test_atr_unknown_smoothing_raises_valueerror_with_message():
    df = _df_from_ohlc([10, 11], [9, 10], [9.5, 10.5])
    with pytest.raises(ValueError, match=r"unknown smothing type"):
        atr(df, intervall=14, smoothing="wma")


@pytest.mark.parametrize("missing_col", ["high", "low", "close"])
def test_atr_missing_required_columns_raises_keyerror(missing_col):
    df = _df_from_ohlc([10, 11], [9, 10], [9.5, 10.5]).drop(columns=[missing_col])
    with pytest.raises(KeyError):
        atr(df)


def test_atr_non_numeric_data_raises_error():
    df = _df_from_ohlc(["a", "b"], ["c", "d"], ["e", "f"])
    with pytest.raises((TypeError, ValueError, pd.errors.DataError)):
        atr(df)


def test_atr_preserves_index():
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    df = _df_from_ohlc([10, 12, 11], [9, 10, 10], [9.5, 11, 10.5], index=idx)
    out = atr(df, intervall=1, smoothing="sma")
    assert out.index.equals(idx)


# ==================== SMA TESTS ====================
def test_sma_basic_window_and_rounding():
    """Test basic SMA calculation - first value NaN (min_periods=period)."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=list("abcd"))
    out = sma(s, period=2)
    expected = s.rolling(2).mean().round(2)  # FIXED: default min_periods=2
    pdt.assert_series_equal(out, expected)


def test_sma_period_greater_than_length_all_nan():
    s = pd.Series([1.0, 2.0, 3.0])
    out = sma(s, period=10)
    expected = s.rolling(10).mean().round(2)
    pdt.assert_series_equal(out, expected)


def test_sma_preserves_index():
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    s = pd.Series([1, 2, 3, 4, 5], index=idx, dtype=float)
    out = sma(s, period=3)
    assert out.index.equals(idx)


def test_sma_non_numeric_raises_dataerror():
    s = pd.Series(["a", "b", "c"])
    with pytest.raises(pd.errors.DataError):
        sma(s, period=2)


# ==================== EMA TESTS ====================
def test_ema_matches_pandas_ewm_params_and_rounding():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    period = 3
    out = ema(s, period=period)
    expected = (
        s.ewm(span=period, min_periods=period, adjust=False, ignore_na=False)
        .mean()
        .round(2)
    )
    pdt.assert_series_equal(out, expected)


def test_ema_min_periods_behavior_first_values_nan():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    out = ema(s, period=3)
    assert pd.isna(out.iloc[0])
    assert pd.isna(out.iloc[1])
    assert not pd.isna(out.iloc[2])


# ==================== RMA TESTS ====================
def test_rma_matches_pandas_ewm_alpha_params_and_rounding():
    s = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0])
    n = 3
    out = rma(s, intervall=n)
    expected = (
        s.ewm(alpha=1 / n, min_periods=n, adjust=False, ignore_na=False).mean().round(2)
    )
    pdt.assert_series_equal(out, expected)


def test_rma_intervall_one_equals_original_series_rounded():
    s = pd.Series([1.234, 2.345, 3.456])
    out = rma(s, intervall=1)
    expected = s.round(2)
    pdt.assert_series_equal(out, expected)
