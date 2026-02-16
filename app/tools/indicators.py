
import pandas as pd
import numpy as np

def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Calculates the Simple Moving Average (SMA)."""
    return series.rolling(window=window).mean()

def calculate_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculates the True Range (TR).
    
    TR = Max(High - Low, |High - PrevClose|, |Low - PrevClose|)
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    # Vectorized Max
    tr = tr1.where(tr1 > tr2, tr2).where(lambda x: x > tr3, tr3)
    return tr

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Calculates the Average True Range (ATR) using Wilder's Smoothing (RMA)."""
    tr = calculate_true_range(high, low, close)
    return tr.ewm(span=(2 * window) - 1, adjust=False).mean()

def calculate_volume_sma(volume: pd.Series, window: int) -> pd.Series:
    """Calculates the Volume SMA."""
    return volume.rolling(window=window).mean()

def calculate_ibs(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculates the Internal Bar Strength (IBS).
    
    IBS = (Close - Low) / (High - Low)
    Handles division by zero by replacing 0 range with 0.01.
    """
    high_low_range = (high - low).replace(0, 0.01)
    return (close - low) / high_low_range

def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Wilder's Smoothing (Optional, but standard RSI often uses it)
    # Here we stick to Simple Moving Average for simplicity unless Wilder is requested.
    # Standard RSI usually uses Wilder's. Let's use Wilder's to be precise.
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(series: pd.Series, window: int) -> pd.Series:
    """Calculates the Exponential Moving Average (EMA)."""
    return series.ewm(span=window, adjust=False).mean()

def calculate_roc(series: pd.Series, period: int) -> pd.Series:
    """Calculates the Rate of Change (ROC).
    
    ROC = ((Price - Price_n_periods_ago) / Price_n_periods_ago) * 100
    """
    return (series - series.shift(period)) / series.shift(period) * 100
