"""
Technical indicators for trading analysis.

This module provides common technical indicators including moving averages,
momentum indicators, and volatility measures. All indicators follow a consistent
API pattern and return pandas Series with appropriate rounding.
"""

from collections.abc import Sequence
from typing import Literal

import pandas as pd

# Type alias for smoothing methods supported by ATR
Smoothing = Literal["ema", "rma", "sma"]


def _as_series(
    values: pd.Series | Sequence[float], *, name: str | None = None
) -> pd.Series:
    """
    Convert input to pandas Series without copying if already a Series.

    Args:
        values: Input data as Series or sequence of floats
        name: Optional name for the resulting Series

    Returns:
        pandas Series containing the input values
    """
    if isinstance(values, pd.Series):
        return values
    return pd.Series(values, name=name)


def _validate_period(period: int, *, param_name: str = "period") -> None:
    """
    Validate that a period parameter is a positive integer.

    Args:
        period: The period value to validate
        param_name: Name of the parameter for error messages

    Raises:
        ValueError: If period is not a positive integer
    """
    if not isinstance(period, int) or period <= 0:
        raise ValueError(f"{param_name} must be a positive integer, got {period!r}.")


def _validate_dataframe_columns(
    df: pd.DataFrame, required_columns: tuple[str, ...]
) -> None:
    """
    Validate that DataFrame contains required columns.

    Args:
        df: DataFrame to validate
        required_columns: Tuple of required column names

    Raises:
        KeyError: If any required columns are missing
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(
            f"DataFrame requires columns {required_columns}, missing: {missing}"
        )


# ============================================================================
# Moving Averages
# ============================================================================


def sma(
    close: pd.Series | Sequence[float],
    period: int = 200,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Calculate Simple Moving Average.

    Args:
        close: Price series (typically closing prices)
        period: Number of periods for moving average (default: 200)
        min_periods: Minimum periods required for calculation (default: period)

    Returns:
        Series of SMA values rounded to 2 decimal places

    Example:
        >>> prices = pd.Series([10, 11, 12, 13, 14])
        >>> sma(prices, period=3)
    """
    close_series = _as_series(close, name="close")

    if min_periods is None:
        min_periods = period

    return close_series.rolling(window=period, min_periods=min_periods).mean().round(2)


def ema(
    close: pd.Series | Sequence[float],
    period: int = 200,
) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Uses pandas ewm with span parameter. More weight is given to recent prices
    compared to SMA.

    Args:
        close: Price series (typically closing prices)
        period: Number of periods for EMA calculation (default: 200)

    Returns:
        Series of EMA values rounded to 2 decimal places

    Example:
        >>> prices = pd.Series([10, 11, 12, 13, 14])
        >>> ema(prices, period=5)
    """
    close_series = _as_series(close, name="close")

    return (
        close_series.ewm(
            span=period,
            min_periods=period,
            adjust=False,
            ignore_na=False,
        )
        .mean()
        .round(2)
    )


def rma(
    close: pd.Series | Sequence[float],
    intervall: int,
) -> pd.Series:
    """
    Calculate Running Moving Average (Wilder's smoothing).

    Uses exponential weighting with alpha = 1/intervall. This is commonly used
    in indicators like RSI and ATR.

    Args:
        close: Price series
        intervall: Number of periods for RMA calculation

    Returns:
        Series of RMA values rounded to 2 decimal places

    Example:
        >>> prices = pd.Series([10, 11, 12, 13, 14])
        >>> rma(prices, intervall=5)
    """
    close_series = _as_series(close, name="close")

    return (
        close_series.ewm(
            alpha=1 / intervall,
            min_periods=intervall,
            adjust=False,
            ignore_na=False,
        )
        .mean()
        .round(2)
    )


# ============================================================================
# Momentum Indicators
# ============================================================================


def rsi(
    close: pd.Series | Sequence[float],
    period: int = 7,
) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    RSI measures the magnitude of recent price changes to evaluate overbought
    or oversold conditions. Values range from 0 to 100.

    Args:
        close: Price series (typically closing prices)
        period: Lookback period for RSI calculation (default: 7)

    Returns:
        Series of RSI values rounded to 0 decimal places

    Example:
        >>> prices = pd.Series([44, 44.5, 45, 43.5, 44])
        >>> rsi(prices, period=3)
    """
    close_series = _as_series(close, name="close")

    # Calculate price changes
    delta = close_series.diff(1)

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    # Calculate exponentially weighted averages
    avg_gains = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    avg_losses = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    # Calculate RSI
    rs = avg_gains / avg_losses
    rsi_values = 100.0 - (100.0 / (1.0 + rs))

    return rsi_values.round(0)


def performance(
    close: pd.Series | Sequence[float],
    period: int = 10,
) -> pd.Series:
    """
    Calculate percentage price change over a specified period.

    Args:
        close: Price series (typically closing prices)
        period: Lookback period for performance calculation (default: 10)

    Returns:
        Series of percentage changes rounded to 0 decimal places

    Raises:
        ValueError: If period is not a positive integer

    Example:
        >>> prices = pd.Series([100, 102, 105, 103, 107])
        >>> performance(prices, period=2)
    """
    _validate_period(period)
    close_series = _as_series(close, name="close")

    return close_series.pct_change(periods=period).mul(100).round(0)


# ============================================================================
# Volatility and Strength Indicators
# ============================================================================


def ibs(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Internal Bar Strength.

    IBS measures where the close is relative to the day's range:
    IBS = (close - low) / (high - low)

    Values near 1.0 indicate closes near the high, values near 0.0 indicate
    closes near the low. Useful for mean reversion strategies.

    Args:
        df: DataFrame with 'high', 'low', and 'close' columns

    Returns:
        Series of IBS values rounded to 2 decimal places

    Raises:
        KeyError: If required columns are missing

    Example:
        >>> df = pd.DataFrame({
        ...     'high': [10, 11, 12],
        ...     'low': [9, 10, 11],
        ...     'close': [9.5, 10.8, 11.2]
        ... })
        >>> ibs(df)
    """
    required_columns = ("high", "low", "close")
    _validate_dataframe_columns(df, required_columns)

    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Calculate range, avoiding division by zero
    price_range = high - low
    ibs_values = (close - low).div(price_range.where(price_range != 0.0))

    return ibs_values.round(2)


def atr(
    df: pd.DataFrame,
    intervall: int = 9,
    smoothing: Smoothing = "ema",
) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    ATR measures market volatility by decomposing the entire range of price
    movement. True Range is the maximum of:
    - Current high - current low
    - Absolute value of current high - previous close
    - Absolute value of current low - previous close

    Args:
        df: DataFrame with 'high', 'low', and 'close' columns
        intervall: Lookback period for averaging (default: 9)
        smoothing: Smoothing method - 'ema', 'rma', or 'sma' (default: 'ema')

    Returns:
        Series of ATR values rounded to 2 decimal places

    Raises:
        KeyError: If required columns are missing
        ValueError: If smoothing method is unknown

    Example:
        >>> df = pd.DataFrame({
        ...     'high': [50, 52, 51],
        ...     'low': [48, 49, 49],
        ...     'close': [49, 51, 50]
        ... })
        >>> atr(df, intervall=2, smoothing='sma')

    Reference:
        https://stackoverflow.com/a/74282809/
    """
    required_columns = ("high", "low", "close")
    _validate_dataframe_columns(df, required_columns)

    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift()

    # Calculate True Range components
    tr_components = [
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ]

    # True Range is the maximum of the three components
    true_range = pd.concat(tr_components, axis=1).max(axis=1)

    # Apply smoothing method
    if smoothing == "rma":
        return rma(true_range, intervall=intervall)
    elif smoothing == "ema":
        return ema(true_range, period=intervall)
    elif smoothing == "sma":
        return sma(true_range, period=intervall)
    else:
        raise ValueError(f"unknown smothing type {smoothing}")
