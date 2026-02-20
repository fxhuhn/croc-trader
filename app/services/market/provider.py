import logging
import threading
from functools import wraps


import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Constants
REQUEST_TIMEOUT = 30
_provider_lock = threading.Lock()


def require_lock(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not _provider_lock.acquire(blocking=False):
            logger.warning(f"SKIP {f.__name__}: Provider is busy (Lock active).")
            return None
        try:
            return f(*args, **kwargs)
        finally:
            _provider_lock.release()

    return wrapper


class YahooDataProvider:
    """
    Encapsulates interaction with Yahoo Finance API.
    Handles MultiIndex issues, timeouts, and batch processing.
    """

    def fetch_batch_raw(
        self, symbols: list[str], start_date: str
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Downloads data for a list of symbols.
        Returns:
            (DataFrame, List of Failed Symbols)
        """
        if not symbols:
            return pd.DataFrame(), []

        failed = []
        try:
            # force simple index doesn't always work with yfinance groups, so we handle it below
            df = yf.download(
                tickers=" ".join(symbols),
                start=start_date,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
                timeout=REQUEST_TIMEOUT,
                ignore_tz=True,
            )
        except Exception as e:
            logger.error(f"YFinance Download Error: {e}")
            return pd.DataFrame(), symbols

        if df.empty:
            return pd.DataFrame(), symbols

        return df, failed

    def extract_symbol_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Safely extracts data for a single symbol from the batch DataFrame.
        Handles MultiIndex structures typical of yfinance batch downloads.
        """
        if df.empty:
            return pd.DataFrame()

        # Case 1: MultiIndex (Level 0 = Symbol, Level 1 = OHLCV)
        if isinstance(df.columns, pd.MultiIndex):
            # Check if symbol is in top level
            if symbol in df.columns.get_level_values(0):
                # .copy() is vital to avoid SettingWithCopy warnings later
                return df[symbol].copy()
            else:
                return pd.DataFrame()

        # Case 2: Single Index (Single Symbol Download)
        # Verify if columns match expectations (open, close, etc.)
        if "close" in df.columns:
            # If we downloaded 1 symbol, yfinance gives a flat DF.
            # We assume it matches the requested symbol.
            return df.copy()

        return pd.DataFrame()
