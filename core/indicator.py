from collections.abc import Sequence
from typing import Literal, cast

import pandas as pd

Smoothing = Literal["ema", "rma", "sma"]

### Helper to ensure input is a pandas Series


def _as_series(
    values: pd.Series | Sequence[float], *, name: str | None = None
) -> pd.Series:
    """Return `values` as a pandas Series without copying when already a Series."""
    if isinstance(values, pd.Series):
        return values
    return pd.Series(values, name=name)


def _validate_period(period: int, *, param_name: str = "period") -> None:
    if not isinstance(period, int) or period <= 0:
        raise ValueError(f"{param_name} must be a positive integer, got {period!r}.")


### Indicators


def ema(Close: pd.Series, period: int = 200) -> pd.Series:
    if not isinstance(Close, pd.Series):
        Close = pd.Series(Close)
    return round(
        Close.ewm(
            span=period, min_periods=period, adjust=False, ignore_na=False
        ).mean(),
        2,
    )


def sma(
    Close: pd.Series, period: int = 200, min_periods: int | None = None
) -> pd.Series:
    if not isinstance(Close, pd.Series):
        Close = pd.Series(Close)
    if min_periods is None:
        min_periods = period
    return round(Close.rolling(period, min_periods=min_periods).mean(), 2)


def rma(close: pd.Series, intervall: int) -> pd.Series:
    return (
        close.ewm(
            alpha=1 / intervall, min_periods=intervall, adjust=False, ignore_na=False
        )
        .mean()
        .round(2)
    )


def performance(Close: pd.Series | Sequence[float], period: int = 10) -> pd.Series:
    """Percent performance over `period` bars (rounded to 0 decimals to match original behavior)."""
    _validate_period(period)
    close = _as_series(Close, name="close")
    return close.pct_change(periods=period).mul(100).round(0)


def rsi(Close: pd.Series, period: int = 7) -> pd.Series:
    if not isinstance(Close, pd.Series):
        Close = pd.Series(Close)

    # Get rid of the first row, which is NaN since it did not have a previous
    # row to calculate the differences
    delta = Close.diff(1)

    # Make the positive gains (up) and negative gains (down) Series
    up = delta.where(delta > 0, 0.0)
    down = -delta.where(delta < 0, 0.0)

    # Calculate the EWMA
    roll_up = up.ewm(min_periods=period, adjust=False, alpha=(1 / period)).mean()
    roll_down = down.ewm(min_periods=period, adjust=False, alpha=1 / period).mean()

    # Calculate the RSI based on EWMA
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return round(rsi, 0)


def ibs(df: pd.DataFrame) -> pd.Series:
    """Internal Bar Strength: (close - low) / (high - low), rounded to 2 decimals."""
    required = ("high", "low", "close")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"ibs() requires columns {required}, missing: {missing}")

    high = cast(pd.Series, df["high"])
    low = cast(pd.Series, df["low"])
    close = cast(pd.Series, df["close"])

    rng = high - low
    # Avoid division by zero when high == low.
    value = (close - low).div(rng.where(rng.ne(0.0)))
    return value.round(2)


def atr(df: pd.DataFrame, intervall: int = 9, smoothing: str = "ema") -> pd.Series:
    # Ref: https://stackoverflow.com/a/74282809/

    high, low, prev_close = df["high"], df["low"], df["close"].shift()
    tr_all = [high - low, high - prev_close, low - prev_close]
    tr_all = [tr.abs() for tr in tr_all]
    tr = pd.concat(tr_all, axis=1).max(axis=1)

    if smoothing == "rma":
        return rma(tr, intervall)
    if smoothing == "ema":
        return ema(tr, intervall)
    if smoothing == "sma":
        return sma(tr, intervall)
    raise ValueError(f"unknown smothing type {smoothing}")
