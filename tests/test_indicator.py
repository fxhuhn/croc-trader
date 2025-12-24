"""
Unit tests for technical indicators.

Tests cover expected behavior, edge cases, error handling, and numerical
accuracy for all indicator functions.
"""

import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.indicator import atr, ema, ibs, performance, rma, rsi, sma

# ============================================================================
# Test Fixtures and Helpers
# ============================================================================


def create_ohlc_dataframe(
    high: list[float],
    low: list[float],
    close: list[float],
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """
    Create OHLC DataFrame for testing.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        index: Optional index (defaults to RangeIndex)

    Returns:
        DataFrame with high, low, and close columns
    """
    if index is None:
        index = pd.RangeIndex(len(close))

    return pd.DataFrame(
        {"high": high, "low": low, "close": close},
        index=index,
    )


def calculate_true_range(df: pd.DataFrame) -> pd.Series:
    """
    Calculate True Range manually for test validation.

    Args:
        df: DataFrame with high, low, close columns

    Returns:
        Series of True Range values
    """
    prev_close = df["close"].shift()
    tr_components = [
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ]
    return pd.concat(tr_components, axis=1).max(axis=1)


# ============================================================================
# ATR Tests
# ============================================================================


class TestATR:
    """Tests for Average True Range indicator."""

    def test_atr_sma_basic_calculation(self):
        """Test ATR with SMA smoothing on simple price data."""
        df = create_ohlc_dataframe([10, 12, 11], [9, 10, 10], [9.5, 11, 10.5])
        true_range = calculate_true_range(df)
        expected = true_range.rolling(1).mean().round(2)

        result = atr(df, period=1, smoothing="sma")

        pdt.assert_series_equal(result, expected, check_names=False)

    def test_atr_constant_prices_returns_zeros(self):
        """Test that constant prices produce zero ATR."""
        df = create_ohlc_dataframe(
            [10, 10, 10, 10],
            [10, 10, 10, 10],
            [10, 10, 10, 10],
        )

        result = atr(df, period=1, smoothing="sma")
        expected = pd.Series([0.0, 0.0, 0.0, 0.0], index=df.index)

        pdt.assert_series_equal(result, expected, check_names=False)

    def test_atr_sma_window_matches_rolling_mean(self):
        """Test ATR with 2-period SMA matches pandas rolling mean."""
        df = create_ohlc_dataframe(
            [10, 13, 12, 14],
            [9, 11, 11, 13],
            [9.5, 12, 11.5, 13.5],
        )
        true_range = calculate_true_range(df)
        expected = true_range.rolling(2).mean().round(2)

        result = atr(df, period=2, smoothing="sma")

        pdt.assert_series_equal(result, expected, check_names=False)

    def test_atr_unknown_smoothing_raises_error(self):
        """Test that invalid smoothing method raises ValueError."""
        df = create_ohlc_dataframe([10, 11], [9, 10], [9.5, 10.5])

        with pytest.raises(ValueError, match=r"Unknown smoothing type"):
            atr(df, period=14, smoothing="wma")

    @pytest.mark.parametrize("missing_col", ["high", "low", "close"])
    def test_atr_missing_columns_raises_error(self, missing_col):
        """Test that missing required columns raise KeyError."""
        df = create_ohlc_dataframe([10, 11], [9, 10], [9.5, 10.5])
        df = df.drop(columns=[missing_col])

        with pytest.raises(KeyError):
            atr(df)

    def test_atr_preserves_datetime_index(self):
        """Test that ATR preserves DatetimeIndex from input."""
        index = pd.date_range("2025-01-01", periods=3, freq="D")
        df = create_ohlc_dataframe(
            [10, 12, 11], [9, 10, 10], [9.5, 11, 10.5], index=index
        )

        result = atr(df, period=1, smoothing="sma")

        assert result.index.equals(index)

    @pytest.mark.parametrize("smoothing", ["ema", "rma", "sma"])
    def test_atr_all_smoothing_methods(self, smoothing):
        """Test that all smoothing methods execute without error."""
        df = create_ohlc_dataframe([10, 12, 11], [9, 10, 10], [9.5, 11, 10.5])

        result = atr(df, period=2, smoothing=smoothing)

        assert isinstance(result, pd.Series)
        assert len(result) == len(df)


# ============================================================================
# Moving Average Tests
# ============================================================================


class TestSMA:
    """Tests for Simple Moving Average."""

    def test_sma_basic_calculation(self):
        """Test basic SMA calculation with 2-period window."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0], index=list("abcd"))
        expected = series.rolling(2).mean().round(2)

        result = sma(series, period=2)

        pdt.assert_series_equal(result, expected)

    def test_sma_period_exceeds_length(self):
        """Test SMA when period is greater than series length."""
        series = pd.Series([1.0, 2.0, 3.0])

        result = sma(series, period=10)
        expected = series.rolling(10).mean().round(2)

        pdt.assert_series_equal(result, expected)

    def test_sma_preserves_datetime_index(self):
        """Test that SMA preserves DatetimeIndex."""
        index = pd.date_range("2025-01-01", periods=5, freq="D")
        series = pd.Series([1, 2, 3, 4, 5], index=index, dtype=float)

        result = sma(series, period=3)

        assert result.index.equals(index)

    def test_sma_accepts_list_input(self):
        """Test that SMA accepts list input and converts to Series."""
        prices = [10.0, 11.0, 12.0, 13.0]

        result = sma(prices, period=2)

        assert isinstance(result, pd.Series)
        assert len(result) == 4

    def test_sma_min_periods_parameter(self):
        """Test custom min_periods parameter."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0])

        # With min_periods=1, first value should not be NaN
        result = sma(series, period=3, min_periods=1)

        assert not pd.isna(result.iloc[0])


class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_ema_matches_pandas_ewm(self):
        """Test that EMA matches pandas ewm calculation."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        period = 3

        result = ema(series, period=period)
        expected = (
            series.ewm(span=period, min_periods=period, adjust=False, ignore_na=False)
            .mean()
            .round(2)
        )

        pdt.assert_series_equal(result, expected)

    def test_ema_min_periods_creates_leading_nans(self):
        """Test that EMA has NaN values before min_periods is met."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0])

        result = ema(series, period=3)

        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert not pd.isna(result.iloc[2])

    def test_ema_accepts_list_input(self):
        """Test that EMA accepts list input."""
        prices = [10.0, 11.0, 12.0, 13.0, 14.0]

        result = ema(prices, period=3)

        assert isinstance(result, pd.Series)
        assert len(result) == 5


class TestRMA:
    """Tests for Running Moving Average (Wilder's smoothing)."""

    def test_rma_matches_pandas_ewm_alpha(self):
        """Test that RMA matches pandas ewm with alpha parameter."""
        series = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0])
        period = 3

        result = rma(series, period=period)
        expected = (
            series.ewm(
                alpha=1 / period, min_periods=period, adjust=False, ignore_na=False
            )
            .mean()
            .round(2)
        )

        pdt.assert_series_equal(result, expected)

    def test_rma_period_one_returns_original_rounded(self):
        """Test that RMA with period=1 returns original series rounded."""
        series = pd.Series([1.234, 2.345, 3.456])

        result = rma(series, period=1)
        expected = series.round(2)

        pdt.assert_series_equal(result, expected)


# ============================================================================
# Momentum Indicator Tests
# ============================================================================


class TestRSI:
    """Tests for Relative Strength Index."""

    def test_rsi_range_bounds(self):
        """Test that RSI values are within 0-100 range."""
        prices = pd.Series([44, 44.5, 45, 43.5, 44, 45, 46, 45.5, 46.5, 47, 46, 45])

        result = rsi(prices, period=5)
        valid_values = result.dropna()

        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()

    def test_rsi_uptrend_high_values(self):
        """Test that consistently rising prices produce high RSI."""
        prices = pd.Series(range(1, 21))  # Continuous uptrend

        result = rsi(prices, period=14)

        # After initial period, RSI should be high (near 100)
        assert result.iloc[-1] > 90

    def test_rsi_downtrend_low_values(self):
        """Test that consistently falling prices produce low RSI."""
        prices = pd.Series(range(20, 0, -1))  # Continuous downtrend

        result = rsi(prices, period=14)

        # After initial period, RSI should be low (near 0)
        assert result.iloc[-1] < 10

    def test_rsi_accepts_list_input(self):
        """Test that RSI accepts list input."""
        prices = [44.0, 44.5, 45.0, 43.5, 44.0]

        result = rsi(prices, period=3)

        assert isinstance(result, pd.Series)


class TestPerformance:
    """Tests for percentage performance indicator."""

    def test_performance_basic_calculation(self):
        """Test basic percentage change calculation."""
        prices = pd.Series([100.0, 110.0, 121.0, 115.0])

        result = performance(prices, period=1)

        # First value NaN, then 10%, 10%, -4.96%
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 10.0
        assert result.iloc[2] == 10.0

    def test_performance_multi_period(self):
        """Test performance over multiple periods."""
        prices = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0])

        result = performance(prices, period=2)

        # 2-period change: NaN, NaN, 10%, 9.52%, 9.09% (rounded to 0 decimals)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 10.0

    def test_performance_negative_change(self):
        """Test performance calculation with price decline."""
        prices = pd.Series([100.0, 90.0])

        result = performance(prices, period=1)

        assert result.iloc[1] == -10.0


# ============================================================================
# Strength Indicator Tests
# ============================================================================


class TestIBS:
    """Tests for Internal Bar Strength."""

    def test_ibs_close_at_high(self):
        """Test IBS when close equals high."""
        df = pd.DataFrame({"high": [10, 11], "low": [9, 10], "close": [10, 11]})

        result = ibs(df)

        assert result.iloc[0] == 1.0
        assert result.iloc[1] == 1.0

    def test_ibs_close_at_low(self):
        """Test IBS when close equals low."""
        df = pd.DataFrame({"high": [10, 11], "low": [9, 10], "close": [9, 10]})

        result = ibs(df)

        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 0.0

    def test_ibs_close_at_midpoint(self):
        """Test IBS when close is at midpoint of range."""
        df = pd.DataFrame({"high": [10], "low": [8], "close": [9]})

        result = ibs(df)

        assert result.iloc[0] == 0.5

    def test_ibs_zero_range_returns_nan(self):
        """Test IBS when high equals low (zero range)."""
        df = pd.DataFrame({"high": [10, 10], "low": [10, 10], "close": [10, 10]})

        result = ibs(df)

        # When range is zero, result should be NaN
        assert pd.isna(result.iloc[0])

    def test_ibs_missing_columns_raises_error(self):
        """Test that missing required columns raise KeyError."""
        df = pd.DataFrame({"high": [10], "low": [9]})  # Missing 'close'

        with pytest.raises(KeyError, match="missing"):
            ibs(df)


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for input validation and error handling."""

    @pytest.mark.parametrize("func", [sma, ema, rma, rsi, performance])
    def test_invalid_period_raises_error(self, func):
        """Test that invalid period values raise ValueError."""
        series = pd.Series([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="positive integer"):
            func(series, period=0)

        with pytest.raises(ValueError, match="positive integer"):
            func(series, period=-5)

    def test_atr_invalid_period_raises_error(self):
        """Test that ATR raises error for invalid period."""
        df = create_ohlc_dataframe([10, 11], [9, 10], [9.5, 10.5])

        with pytest.raises(ValueError, match="positive integer"):
            atr(df, period=0)
