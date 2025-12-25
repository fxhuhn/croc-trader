from typing import List, Protocol

import pandas as pd


class MarketDataLoader(Protocol):
    """Protocol for fetching market data."""

    def fetch_daily_history(
        self, symbols: List[str], lookback_years: int = 4
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data.
        Returns DataFrame with columns:
        [symbol, date, open, high, low, close, volume, exchange]
        """
        ...
