import datetime

import pandas as pd
import pytest

from app.tools.indicators import (
    calculate_max_close_for_rsi,
    calculate_mtd_return,
    calculate_rsi,
    calculate_rsi_exit_target,
    calculate_up_days_count,
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


def test_calculate_roc_series():
    """Tests calculate_roc on a price series."""
    from app.tools.indicators import calculate_roc

    prices = pd.Series([100.0, 110.0, 105.0])
    roc = calculate_roc(prices, window=1)
    assert pd.isna(roc.iloc[0])
    assert roc.iloc[1] == pytest.approx(10.0)
    assert roc.iloc[2] == pytest.approx(-4.54545, rel=1e-3)


def test_calculate_roc_dataframe():
    """Tests calculate_roc on a multi-symbol price DataFrame."""
    from app.tools.indicators import calculate_roc

    data = pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, 121.0],
            "MSFT": [200.0, 220.0, 242.0],
        }
    )
    roc = calculate_roc(data, window=1)
    assert pd.isna(roc["AAPL"].iloc[0])
    assert roc["AAPL"].iloc[1] == pytest.approx(10.0)
    assert roc["AAPL"].iloc[2] == pytest.approx(10.0)
    assert roc["MSFT"].iloc[1] == pytest.approx(10.0)
    assert roc["MSFT"].iloc[2] == pytest.approx(10.0)


def test_calculate_sma_dataframe():
    """Tests calculate_sma on a multi-symbol price DataFrame."""
    from app.tools.indicators import calculate_sma

    data = pd.DataFrame(
        {
            "AAPL": [10.0, 20.0, 30.0],
            "MSFT": [100.0, 200.0, 300.0],
        }
    )
    sma = calculate_sma(data, window=2)
    assert pd.isna(sma["AAPL"].iloc[0])
    assert sma["AAPL"].iloc[1] == pytest.approx(15.0)
    assert sma["AAPL"].iloc[2] == pytest.approx(25.0)
    assert sma["MSFT"].iloc[1] == pytest.approx(150.0)
    assert sma["MSFT"].iloc[2] == pytest.approx(250.0)


def test_calculate_up_days_count_insufficient_data() -> None:
    """Tests calculate_up_days_count returns None when history < lookback + 1."""
    assert calculate_up_days_count(pd.Series([], dtype=float), lookback=225) is None
    # 225 items with lookback 225 -> not enough to compute 225 changes (needs 226)
    short_series = pd.Series([100.0 + i for i in range(225)])
    assert calculate_up_days_count(short_series, lookback=225) is None


def test_calculate_up_days_count_normal() -> None:
    """Tests calculate_up_days_count accurately counts positive daily changes."""
    # Lookback = 5, requires 6 prices
    prices = pd.Series([10.0, 12.0, 11.0, 13.0, 14.0, 12.0])
    # 10->12 (up), 12->11 (down), 11->13 (up), 13->14 (up), 14->12 (down) => 3 up-days
    assert calculate_up_days_count(prices, lookback=5) == 3

    # Default lookback = 225, exactly 226 bars: all strictly increasing
    full_up = pd.Series([100.0 + i for i in range(226)])
    assert calculate_up_days_count(full_up, lookback=225) == 225

    # Exactly 226 bars: all strictly decreasing
    full_down = pd.Series([500.0 - i for i in range(226)])
    assert calculate_up_days_count(full_down, lookback=225) == 0


def test_calculate_mtd_return_normal() -> None:
    """Tests calculate_mtd_return with standard prior month and current month bars."""
    dates = pd.Series(
        [
            datetime.date(2026, 6, 29),
            datetime.date(2026, 6, 30),
            datetime.date(2026, 7, 1),
            datetime.date(2026, 7, 2),
        ]
    )
    prices = pd.Series([100.0, 102.0, 105.0, 104.04])
    ref_date = datetime.date(2026, 7, 2)

    # Base close is 102.0 (2026-06-30), current close is 104.04 (2026-07-02)
    # Return = 104.04 / 102.0 - 1.0 = 0.02 (2.0%)
    result = calculate_mtd_return(prices, dates, ref_date)
    assert result == pytest.approx(0.02, abs=1e-5)


def test_calculate_mtd_return_with_datetime_index() -> None:
    """Tests calculate_mtd_return when dates are provided via DatetimeIndex."""
    datetime_index = pd.to_datetime(["2026-06-30", "2026-07-01", "2026-07-02"])
    prices = pd.Series([100.0, 105.0, 110.0], index=datetime_index)
    ref_date = datetime.date(2026, 7, 2)

    result = calculate_mtd_return(prices, datetime_index, ref_date)
    assert result == pytest.approx(0.10, abs=1e-5)


def test_calculate_mtd_return_missing_prior_month() -> None:
    """Tests calculate_mtd_return returns 0.0 when no prior month data exists."""
    dates = pd.Series([datetime.date(2026, 7, 1), datetime.date(2026, 7, 2)])
    prices = pd.Series([100.0, 105.0])
    ref_date = datetime.date(2026, 7, 2)

    assert calculate_mtd_return(prices, dates, ref_date) == 0.0


def test_calculate_mtd_return_empty_or_zero_base() -> None:
    """Tests calculate_mtd_return handles empty series and zero base close gracefully."""
    empty_series = pd.Series([], dtype=float)
    empty_dates = pd.Series([], dtype="object")
    assert (
        calculate_mtd_return(empty_series, empty_dates, datetime.date(2026, 7, 2))
        == 0.0
    )

    dates = pd.Series([datetime.date(2026, 6, 30), datetime.date(2026, 7, 1)])
    prices_zero = pd.Series([0.0, 10.0])
    assert calculate_mtd_return(prices_zero, dates, datetime.date(2026, 7, 1)) == 0.0


def test_calculate_mtd_return_lookahead_guard() -> None:
    """Tests that bars after reference_date are strictly excluded (Zero Lookahead-Bias)."""
    dates = pd.Series(
        [
            datetime.date(2026, 6, 30),
            datetime.date(2026, 7, 1),
            datetime.date(2026, 7, 2),
            datetime.date(2026, 7, 3),  # Future bar relative to ref_date
            datetime.date(2026, 7, 6),  # Future bar relative to ref_date
        ]
    )
    prices = pd.Series([100.0, 102.0, 104.0, 120.0, 150.0])
    ref_date = datetime.date(2026, 7, 2)

    # Must calculate using 2026-07-02 (104.0) vs 2026-06-30 (100.0) -> +4%
    result = calculate_mtd_return(prices, dates, ref_date)
    assert result == pytest.approx(0.04, abs=1e-5)
