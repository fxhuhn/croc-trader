"""TradingView market data provider module.

Encapsulates retrieval of historical daily price quotes from TradingView
via the TvDatafeed API library.
"""

import logging
import time
from typing import TypedDict

import pandas as pd
from tvDatafeed import Interval, TvDatafeed

from app.mapping import ExchangeMapper, mapper

logger = logging.getLogger(__name__)


class TradingViewBarRecord(TypedDict, total=False):
    """Typed representation of a daily price bar record from TradingView."""

    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float | int


class TradingViewDataProvider:
    """Encapsulates interaction with TradingView API via TvDatafeed (Anonymous Mode).

    Handles symbol exchange resolution, symbol format conversion (e.g. BRK-B -> BRK.B),
    and rate-limiting.
    """

    def __init__(self, exchange_mapper: ExchangeMapper | None = None) -> None:
        """Initializes the TradingView provider with an optional ExchangeMapper instance."""
        self._exchange_mapper: ExchangeMapper = exchange_mapper or mapper
        self._tv: TvDatafeed | None = None

    def _get_instance(self) -> TvDatafeed:
        """Lazily instantiates and returns the TvDatafeed client connection."""
        if self._tv is None:
            self._tv = TvDatafeed()
        return self._tv

    def fetch_symbol_history(
        self,
        symbol: str,
        number_of_bars: int = 100,
    ) -> list[TradingViewBarRecord]:
        """Fetches historical daily price records for a single symbol from TradingView.

        Args:
            symbol: Equity symbol in standard Yahoo format (e.g., 'BRK-B').
            number_of_bars: Number of daily bars to retrieve.

        Returns:
            List of typed dictionary records ready for MarketPrice parsing.
        """
        standard_symbol = symbol.strip().upper()
        # Remove Yahoo's exchange suffix (e.g., .DE) if present
        base_symbol = standard_symbol.split(".")[0]
        # Convert Yahoo's share class dash to TradingView's dot (e.g., BRK-B -> BRK.B)
        tv_symbol = base_symbol.replace("-", ".")
        exchange_name = self._exchange_mapper.get_exchange(standard_symbol)
        exchanges_to_try = (
            [exchange_name] if exchange_name else ["NASDAQ", "NYSE", "AMEX"]
        )

        history_dataframe = self._query_historical_dataframe(
            standard_symbol=standard_symbol,
            tv_symbol=tv_symbol,
            exchanges_to_try=exchanges_to_try,
            number_of_bars=number_of_bars,
        )

        if history_dataframe is None or history_dataframe.empty:
            logger.warning(
                "TradingView returned empty data for %s",
                standard_symbol,
            )
            return []

        cleaned_records = self._standardize_dataframe_records(
            dataframe=history_dataframe,
            standard_symbol=standard_symbol,
        )

        # Rate limiting delay to respect TradingView anonymous query limits
        time.sleep(0.5)
        return cleaned_records

    def _query_historical_dataframe(
        self,
        standard_symbol: str,
        tv_symbol: str,
        exchanges_to_try: list[str],
        number_of_bars: int,
    ) -> pd.DataFrame | None:
        """Queries TvDatafeed historical price bars across candidate exchanges.

        Args:
            standard_symbol: Standardized ticker symbol for logging context.
            tv_symbol: TradingView-formatted ticker symbol (e.g. 'BRK.B').
            exchanges_to_try: List of target exchange codes to query.
            number_of_bars: Number of daily price bars to fetch.

        Returns:
            Pandas DataFrame containing raw price bars, or None if download failed.
        """
        for exchange_name in exchanges_to_try:
            try:
                datafeed_instance = self._get_instance()
                dataframe = datafeed_instance.get_hist(
                    symbol=tv_symbol,
                    exchange=exchange_name,
                    interval=Interval.in_daily,
                    n_bars=number_of_bars,
                )
                if dataframe is not None and not dataframe.empty:
                    return dataframe
            except Exception as error:
                logger.debug(
                    "TradingView Download Error for %s (%s): %s",
                    standard_symbol,
                    exchange_name,
                    error,
                )
                # Force auto-reconnect on next request if connection dropped
                self._tv = None

        return None

    def _standardize_dataframe_records(
        self,
        dataframe: pd.DataFrame,
        standard_symbol: str,
    ) -> list[TradingViewBarRecord]:
        """Transforms raw TradingView DataFrame into standardized records.

        Args:
            dataframe: Raw DataFrame returned by TvDatafeed.
            standard_symbol: Standard Yahoo symbol to assign to all output rows.

        Returns:
            List of standardized dictionary records.
        """
        cleaned_dataframe = dataframe.copy()
        cleaned_dataframe.columns = cleaned_dataframe.columns.str.lower()

        # Handle datetime index or datetime column alignment
        cleaned_dataframe = cleaned_dataframe.reset_index()
        rename_map: dict[str, str] = {}
        for column_name in cleaned_dataframe.columns:
            if str(column_name).lower() in ("datetime", "index"):
                rename_map[str(column_name)] = "date"

        if rename_map:
            cleaned_dataframe = cleaned_dataframe.rename(columns=rename_map)

        # Vectorized assignment of standard symbol across all rows
        cleaned_dataframe["symbol"] = standard_symbol

        return cleaned_dataframe.to_dict("records")  # type: ignore[return-value]
