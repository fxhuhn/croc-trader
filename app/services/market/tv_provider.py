import logging
import time
from typing import Any

from tvDatafeed import Interval, TvDatafeed

from app.mapping import mapper

logger = logging.getLogger(__name__)


class TradingViewDataProvider:
    """
    Encapsulates interaction with TradingView API via TvDatafeed (Anonymous Mode).
    Handles symbol exchange resolution, symbol format conversion (e.g. BRK-B -> BRK.B),
    and rate-limiting.
    """

    def __init__(self) -> None:
        self._tv: TvDatafeed | None = None

    def _get_instance(self) -> TvDatafeed:
        if self._tv is None:
            self._tv = TvDatafeed()
        return self._tv

    def fetch_symbol_history(
        self, symbol: str, n_bars: int = 100
    ) -> list[dict[str, Any]]:
        """
        Fetches historical daily price records for a single symbol from TradingView.

        :param symbol: Equity symbol in standard Yahoo format (e.g. 'BRK-B').
        :param n_bars: Number of daily bars to retrieve.
        :return: List of dictionary records ready for MarketPrice parsing.
        """
        standard_symbol = symbol.strip().upper()
        tv_symbol = standard_symbol.replace("-", ".")
        exchange = mapper.get_exchange(standard_symbol)
        exchanges_to_try = [exchange] if exchange else ["NASDAQ", "NYSE", "AMEX"]

        df = None
        for ex in exchanges_to_try:
            try:
                tv = self._get_instance()
                df = tv.get_hist(
                    symbol=tv_symbol,
                    exchange=ex,
                    interval=Interval.in_daily,
                    n_bars=n_bars,
                )
                if df is not None and not df.empty:
                    break
            except Exception as error:
                logger.debug(
                    "TradingView Download Error for %s (%s): %s",
                    standard_symbol,
                    ex,
                    error,
                )

        if df is None or df.empty:
            logger.warning(
                "TradingView returned empty data for %s",
                standard_symbol,
            )
            return []

        # Standardize columns and structure
        df_clean = df.copy()
        df_clean.columns = df_clean.columns.str.lower()

        # Handle datetime index/column
        df_clean = df_clean.reset_index()
        rename_map = {}
        for col in df_clean.columns:
            if str(col).lower() in ("datetime", "index"):
                rename_map[col] = "date"
        if rename_map:
            df_clean = df_clean.rename(columns=rename_map)

        records: list[dict[str, Any]] = []
        for row in df_clean.to_dict("records"):
            # Ensure output record retains the standard Yahoo-style symbol
            row["symbol"] = standard_symbol
            records.append(row)

        # Rate limiting delay
        time.sleep(0.5)
        return records
