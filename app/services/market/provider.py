import logging
import threading
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Constants
REQUEST_TIMEOUT = 30
_provider_lock = threading.Lock()

P = ParamSpec("P")
R = TypeVar("R")


def require_lock[**P, R](func: Callable[P, R]) -> Callable[P, R | None]:
    """Decorator that acquires a global lock before executing the function.

    If the lock is already held, the function call is skipped and None is returned.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
        if not _provider_lock.acquire(blocking=False):
            logger.warning("SKIP %s: Provider is busy (Lock active).", func.__name__)
            return None
        try:
            return func(*args, **kwargs)
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
                repair=True,
                progress=False,
                threads=True,
                timeout=REQUEST_TIMEOUT,
                ignore_tz=True,
            )
        except Exception as e:
            logger.error("YFinance Download Error: %s", e)
            return pd.DataFrame(), symbols

        if df.empty:
            return pd.DataFrame(), symbols

        # Identify requested symbols that are missing from the returned columns
        if isinstance(df.columns, pd.MultiIndex):
            downloaded_symbols = set(df.columns.get_level_values(0).unique())
        else:
            downloaded_symbols = {symbols[0]}

        for symbol in symbols:
            if symbol not in downloaded_symbols:
                failed.append(symbol)

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

    def reconcile_eod_candle(
        self,
        symbol: str,
        symbol_dataframe: pd.DataFrame,
        *,
        drift_threshold: float = 0.002,
    ) -> pd.DataFrame:
        """Reconciles the latest daily bar close against Yahoo fast_info regularMarketPrice.

        If post-market drift (> drift_threshold) is detected between bar close
        and official regular market price, corrects the latest bar's close
        to the regular market price.

        Args:
            symbol: Equity ticker symbol.
            symbol_dataframe: Single-symbol historical prices DataFrame.
            drift_threshold: Relative discrepancy threshold (default 0.2%).

        Returns:
            pd.DataFrame: Reconciled DataFrame.
        """
        if symbol_dataframe.empty:
            return symbol_dataframe

        close_column = next(
            (c for c in ("close", "Close") if c in symbol_dataframe.columns),
            None,
        )
        if close_column is None:
            return symbol_dataframe

        try:
            fast_info = getattr(yf.Ticker(symbol), "fast_info", None)
            regular_price = fast_info.get("lastPrice") if fast_info else None
            raw_close = float(symbol_dataframe[close_column].iloc[-1])

            if regular_price is not None and regular_price > 0 and raw_close > 0:
                relative_diff = abs(raw_close - regular_price) / regular_price
                if relative_diff > drift_threshold:
                    logger.info(
                        "Reconciled post-market close for %s: %.4f -> %.4f (diff: %.2f%%)",
                        symbol,
                        raw_close,
                        regular_price,
                        relative_diff * 100,
                    )
                    reconciled_dataframe = symbol_dataframe.copy()
                    reconciled_dataframe.iloc[
                        -1, reconciled_dataframe.columns.get_loc(close_column)
                    ] = regular_price
                    return reconciled_dataframe

        except Exception as error:
            logger.debug("EOD reconciliation skipped for %s: %s", symbol, error)

        return symbol_dataframe
