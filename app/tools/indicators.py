import pandas as pd


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Calculates the Simple Moving Average (SMA)."""
    if series.empty:
        raise ValueError("Cannot calculate SMA: series is empty.")
    return series.rolling(window=window).mean()


def calculate_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Calculates the True Range (TR).

    TR = Max(High - Low, |High - PrevClose|, |Low - PrevClose|)
    """
    if close.empty:
        raise ValueError("Cannot calculate True Range: close series is empty.")
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    # Vectorized Max
    tr = tr1.where(tr1 > tr2, tr2).where(lambda x: x > tr3, tr3)
    return tr


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """Calculates the Average True Range (ATR) using Wilder's Smoothing (RMA)."""
    if close.empty:
        raise ValueError("Cannot calculate ATR: close series is empty.")
    tr = calculate_true_range(high, low, close)
    return tr.ewm(span=(2 * window) - 1, adjust=False).mean()


def calculate_volume_sma(volume: pd.Series, window: int) -> pd.Series:
    """Calculates the Volume SMA."""
    if volume.empty:
        raise ValueError("Cannot calculate Volume SMA: volume series is empty.")
    return volume.rolling(window=window).mean()


def calculate_ibs(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculates the Internal Bar Strength (IBS).

    IBS = (Close - Low) / (High - Low)
    Handles division by zero by replacing 0 range with 0.01.
    """
    if close.empty:
        raise ValueError("Cannot calculate IBS: close series is empty.")
    high_low_range = (high - low).replace(0, 0.01)
    return (close - low) / high_low_range


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI)."""
    if series.empty:
        raise ValueError("Cannot calculate RSI: series is empty.")
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Wilder's Smoothing (Optional, but standard RSI often uses it)
    # Here we stick to Simple Moving Average for simplicity unless Wilder is requested.
    # Standard RSI usually uses Wilder's. Let's use Wilder's to be precise.
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle zero loss/gain edge cases (e.g. flat line or monotonic gains)
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where((avg_gain != 0) | (avg_loss != 0), 50.0)
    return rsi


def calculate_max_close_for_rsi(
    close_series: pd.Series, window: int = 2, rsi_target: float = 40.0
) -> float:
    """Calculates the maximum Close price required today for RSI(window) <= rsi_target.

    Uses Wilder's EWM smoothing matching calculate_rsi. The close_series parameter
    must contain price history up to yesterday (t-1).
    """
    min_required_len = window + 1
    if close_series.empty or len(close_series) < min_required_len:
        return float("nan")

    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain_series = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss_series = loss.ewm(alpha=1 / window, adjust=False).mean()

    ag_prev = float(avg_gain_series.iloc[-1])
    al_prev = float(avg_loss_series.iloc[-1])
    close_prev = float(close_series.iloc[-1])

    rs_target = rsi_target / (100.0 - rsi_target)
    rsi_decay = (
        (100.0 * ag_prev / (ag_prev + al_prev)) if (ag_prev + al_prev) > 0 else 50.0
    )

    if rsi_decay >= rsi_target:
        return close_prev + al_prev - ag_prev * (100.0 / rsi_target - 1.0)

    return close_prev + rs_target * al_prev - ag_prev


def calculate_rsi_exit_target(
    close_series: pd.Series, window: int = 2, rsi_target: float = 75.0
) -> float:
    """Calculates the minimum Close price required to exceed rsi_target today.

    Uses Wilder's EWM smoothing matching calculate_rsi.
    """
    min_required_len = window + 1
    if close_series.empty or len(close_series) < min_required_len:
        return float("nan")

    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain_series = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss_series = loss.ewm(alpha=1 / window, adjust=False).mean()

    last_avg_gain = float(avg_gain_series.iloc[-1])
    last_avg_loss = float(avg_loss_series.iloc[-1])
    last_close = float(close_series.iloc[-1])

    rs_multiplier = rsi_target / (100.0 - rsi_target)
    required_delta_rsi = max(0.0, (rs_multiplier * last_avg_loss) - last_avg_gain)
    return last_close + required_delta_rsi + 0.01


def extract_safe_float(value: object, default: float = 0.0) -> float:
    """Safely extracts a float value from a potentially null, NaN, or non-numeric object."""
    if pd.isna(value) or value is None:
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return default
