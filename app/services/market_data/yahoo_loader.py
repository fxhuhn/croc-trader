from datetime import date, timedelta
from typing import Dict, List

import pandas as pd
import yfinance as yf

from .base import MarketDataLoader


class YahooLoader(MarketDataLoader):
    def __init__(self, exchange_map: Dict[str, str]):
        self.exchange_map = exchange_map

    def fetch_daily_history(
        self, symbols: List[str], lookback_years: int = 4
    ) -> pd.DataFrame:
        print(f"📡 Yahoo: Fetching {len(symbols)} symbols...")

        # Determine start date
        start_date = (date.today() - timedelta(days=lookback_years * 365)).isoformat()

        # 1. Download
        df = yf.download(
            symbols,
            start=start_date,
            group_by="ticker",
            rounding=True,
            threads=True,
            auto_adjust=True,  # Handles splits implicitly
            progress=False,
        )

        if df.empty:
            return pd.DataFrame()

        # 2. Transform MultiIndex (Ticker, OHLCV) -> Stacked
        # If single symbol, yf doesn't use MultiIndex columns same way
        if len(symbols) == 1:
            df.columns = df.columns.str.lower()
            df["symbol"] = symbols[0]
            df = df.reset_index()
        else:
            # Stack level 0 (Ticker) to become a column
            df = df.stack(level=0, future_stack=True)
            df.index.names = ["date", "symbol"]
            df = df.reset_index()
            df.columns = df.columns.str.lower()

        # 3. Clean & Validate
        df = self._clean_ohlcv(df)

        # 4. Map Exchange
        df["exchange"] = df["symbol"].map(lambda x: self.exchange_map.get(x, "SMART"))

        return df

    def _clean_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove invalid rows (negative prices, zero volume if strict, etc)."""
        # Ensure numeric types
        cols = ["open", "high", "low", "close", "volume"]
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=cols)

        # Logic checks
        mask = (
            (df["open"] > 0)
            & (df["high"] > 0)
            & (df["low"] > 0)
            & (df["close"] > 0)
            & (df["volume"] >= 0)
            & (df["high"] >= df["low"])
        )
        return df[mask].copy()
