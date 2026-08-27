import pandas as pd
import pytest

from app.tools.indicators import (
    calculate_max_close_for_rsi,
    calculate_rsi,
    calculate_rsi_exit_target,
    extract_safe_float,
)


def test_calculate_max_close_for_rsi_falling_case():
    """Tests calculate_max_close_for_rsi when price must fall to meet RSI target."""
    prices = pd.Series([350.0, 345.0, 360.0, 365.0, 370.0])
    max_c = calculate_max_close_for_rsi(prices, window=2, rsi_target=40.0)

    # Validate that appending max_c results in exact target RSI
    extended_prices = pd.concat([prices, pd.Series([max_c])], ignore_index=True)
    resulting_rsi = calculate_rsi(extended_prices, window=2).iloc[-1]

    assert resulting_rsi == pytest.approx(40.0, abs=1e-4)


def test_calculate_max_close_for_rsi_rising_case():
    """Tests calculate_max_close_for_rsi when price can rise to meet RSI target."""
    prices = pd.Series([350.0, 355.0, 340.0, 330.0, 320.0])
    max_c = calculate_max_close_for_rsi(prices, window=2, rsi_target=40.0)

    # Validate that appending max_c results in exact target RSI
    extended_prices = pd.concat([prices, pd.Series([max_c])], ignore_index=True)
    resulting_rsi = calculate_rsi(extended_prices, window=2).iloc[-1]

    assert resulting_rsi == pytest.approx(40.0, abs=1e-4)


def test_calculate_max_close_for_rsi_insufficient_data():
    """Tests calculate_max_close_for_rsi with insufficient price history."""
    prices = pd.Series([100.0, 105.0])
    max_c = calculate_max_close_for_rsi(prices, window=2, rsi_target=40.0)
    assert pd.isna(max_c)


def test_calculate_rsi_exit_target():
    """Tests calculate_rsi_exit_target returns expected minimum exit target."""
    prices = pd.Series([100.0, 95.0, 90.0, 85.0])
    target = calculate_rsi_exit_target(prices, window=2, rsi_target=75.0)
    assert target > 85.0
    # Extending series with target price should achieve RSI >= 75
    extended = pd.concat([prices, pd.Series([target])], ignore_index=True)
    assert calculate_rsi(extended, window=2).iloc[-1] >= 74.99


def test_extract_safe_float():
    """Tests extract_safe_float handles normal, string, NaN, and None values."""
    assert extract_safe_float(42.5) == 42.5
    assert extract_safe_float("123.45") == 123.45
    assert extract_safe_float(None, default=0.0) == 0.0
    assert extract_safe_float(float("nan"), default=-1.0) == -1.0
    assert extract_safe_float("invalid", default=99.0) == 99.0
