"""Unit tests for early exit validation checks in app/tools/indicators.py."""

import pandas as pd
import pytest

from app.tools.indicators import (
    calculate_atr,
    calculate_ibs,
    calculate_max_close_for_rsi,
    calculate_rsi,
    calculate_sma,
    calculate_true_range,
    calculate_volume_sma,
)


def test_calculate_sma_raises_value_error_on_empty_series() -> None:
    """Verifies calculate_sma raises ValueError on empty series."""
    empty_series = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="Cannot calculate SMA: series is empty."):
        calculate_sma(empty_series, window=8)


def test_calculate_true_range_raises_value_error_on_empty_series() -> None:
    """Verifies calculate_true_range raises ValueError on empty series."""
    empty_series = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="Cannot calculate True Range"):
        calculate_true_range(empty_series, empty_series, empty_series)


def test_calculate_atr_raises_value_error_on_empty_series() -> None:
    """Verifies calculate_atr raises ValueError on empty series."""
    empty_series = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="Cannot calculate ATR"):
        calculate_atr(empty_series, empty_series, empty_series, window=14)


def test_calculate_volume_sma_raises_value_error_on_empty_series() -> None:
    """Verifies calculate_volume_sma raises ValueError on empty series."""
    empty_series = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="Cannot calculate Volume SMA"):
        calculate_volume_sma(empty_series, window=20)


def test_calculate_ibs_raises_value_error_on_empty_series() -> None:
    """Verifies calculate_ibs raises ValueError on empty series."""
    empty_series = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="Cannot calculate IBS"):
        calculate_ibs(empty_series, empty_series, empty_series)


def test_calculate_rsi_raises_value_error_on_empty_series() -> None:
    """Verifies calculate_rsi raises ValueError on empty series."""
    empty_series = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="Cannot calculate RSI"):
        calculate_rsi(empty_series, window=14)


def test_calculate_max_close_for_rsi_returns_nan_on_insufficient_length() -> None:
    """Verifies calculate_max_close_for_rsi returns NaN on insufficient length."""
    short_series = pd.Series([100.0])
    max_c = calculate_max_close_for_rsi(short_series, window=2, rsi_target=40.0)
    assert pd.isna(max_c)
