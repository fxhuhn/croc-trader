import pandas as pd
import pytest

from app.tools.indicators import calculate_max_close_for_rsi, calculate_rsi


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
