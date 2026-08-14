"""Market data provider service module.

Provides high-performance, cached, and pivoted daily market data access
for quantitative screeners and backtesting modules.
"""

import logging

import pandas as pd

from ..session import DatabaseSession

logger = logging.getLogger(__name__)

# Type alias for return structure (Open, High, Low, Close, Volume as DataFrames)
type MarketDataDict = dict[str, pd.DataFrame]


class MarketDataProvider:
    """Central service for accessing market data with in-memory caching and pivoting capabilities."""

    DEFAULT_UNIVERSE_CHUNK_SIZE: int = 500
    MAX_CACHE_ENTRIES: int = 4

    def __init__(self, session: DatabaseSession) -> None:
        """Initializes MarketDataProvider with a shared DatabaseSession.

        Args:
            session: Shared DatabaseSession instance for database queries.
        """
        self.session: DatabaseSession = session
        self._in_memory_cache: MarketDataDict | None = None
        self._cache_lookback: int = 0
        self._all_daily_data_cache: dict[int, MarketDataDict] = {}

    def _build_ranked_query(
        self,
        where_filter: str,
        include_symbol_in_select: bool = True,
        partition_by_symbol: bool = True,
    ) -> str:
        """Constructs a deduplicated, ranked OHLCV query prioritizing Yahoo over TradingView."""
        partition_clause = (
            "PARTITION BY symbol, date" if partition_by_symbol else "PARTITION BY date"
        )
        symbol_col = "symbol, " if include_symbol_in_select else ""
        query_str = (
            "WITH ranked AS ("  # nosec B608
            f"    SELECT date, {symbol_col}open, high, low, close, volume, "
            "           ROW_NUMBER() OVER ( "
            f"               {partition_clause} "
            "               ORDER BY CASE WHEN provider = 'yahoo' THEN 1 WHEN provider = 'tradingview' THEN 2 ELSE 3 END "
            "           ) as rank_idx "
            "    FROM market_prices "
            f"   WHERE {where_filter} AND timeframe = '1D'"  # nosec B608
            ") "
            f"SELECT date, {symbol_col}open, high, low, close, volume "
            "FROM ranked "
            "WHERE rank_idx = 1 "
            "ORDER BY date ASC"
        )
        return query_str

    def preload_all_data(self, days: int = 1000) -> None:
        """Loads all market price data into memory for high-speed analysis.

        Args:
            days: Lookback period in days for preloading records.
        """
        logger.info(
            "[MarketData] Preloading ALL data into memory (Lookback: %dd)...",
            days,
        )
        try:
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime(
                "%Y-%m-%d"
            )
            with self.session.connect() as connection:
                query = self._build_ranked_query("date >= ?")
                dataframe = pd.read_sql_query(query, connection, params=(start_date,))

            if not dataframe.empty:
                self._in_memory_cache = self._pivot_data(dataframe)
                self._cache_lookback = days
                logger.info("[MarketData] Cache Warm! Loaded %d rows.", len(dataframe))
            else:
                logger.warning("[MarketData] Preload returned empty dataset.")
        except Exception as error:
            logger.error("[MarketData] Preload Failed: %s", error)

    def get_all_daily_data(self, days: int) -> MarketDataDict | None:
        """Loads and pivots daily OHLCV data for all available symbols.

        Args:
            days: Number of lookback days to retrieve.

        Returns:
            MarketDataDict containing pivoted OHLCV DataFrames, or None if empty/error.
        """
        if self._in_memory_cache and days <= self._cache_lookback:
            return self._in_memory_cache

        if days in self._all_daily_data_cache:
            return self._all_daily_data_cache[days]

        start_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        logger.info("[MarketData] Loading data from DB (Lookback: %dd)...", days)

        try:
            with self.session.connect() as connection:
                query = self._build_ranked_query("date >= ?")
                dataframe = pd.read_sql_query(query, connection, params=(start_date,))
        except Exception as error:
            logger.error("[MarketData] DB Error: %s", error)
            return None

        if dataframe.empty:
            logger.warning("[MarketData] No data found.")
            return None

        pivoted_result = self._pivot_data(dataframe)
        if len(self._all_daily_data_cache) >= self.MAX_CACHE_ENTRIES:
            first_key = next(iter(self._all_daily_data_cache))
            self._all_daily_data_cache.pop(first_key)
        self._all_daily_data_cache[days] = pivoted_result
        return pivoted_result

    def get_universe_daily_data(
        self, symbols: list[str], days: int
    ) -> MarketDataDict | None:
        """Loads and pivots data for a specific list of ticker symbols.

        Args:
            symbols: List of ticker symbols to fetch.
            days: Lookback period in days.

        Returns:
            MarketDataDict containing pivoted DataFrames, or None if empty/error.
        """
        if not symbols:
            return None

        if self._in_memory_cache and days <= self._cache_lookback:
            filtered_dict: MarketDataDict = {}
            for column_name, dataframe in self._in_memory_cache.items():
                available_columns = [
                    symbol for symbol in symbols if symbol in dataframe.columns
                ]
                if available_columns:
                    filtered_dict[column_name] = dataframe[available_columns]
                else:
                    filtered_dict[column_name] = pd.DataFrame(index=dataframe.index)
            return filtered_dict

        start_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        logger.info(
            "[MarketData] Loading universe data (%d symbols, %dd)...",
            len(symbols),
            days,
        )

        all_dataframes = self._fetch_universe_chunks(
            symbols=symbols, start_date=start_date
        )

        if not all_dataframes:
            logger.warning("[MarketData] No data found for universe.")
            return None

        full_dataframe = pd.concat(all_dataframes, ignore_index=True)
        return self._pivot_data(full_dataframe)

    def _fetch_universe_chunks(
        self, symbols: list[str], start_date: str
    ) -> list[pd.DataFrame]:
        """Fetches market price chunks for large symbol universes to avoid SQLite variable limits.

        Args:
            symbols: Complete list of ticker symbols.
            start_date: Range start date string (YYYY-MM-DD).

        Returns:
            List of non-empty DataFrames fetched per chunk.
        """
        all_dataframes: list[pd.DataFrame] = []
        chunk_size = self.DEFAULT_UNIVERSE_CHUNK_SIZE

        try:
            with self.session.connect() as connection:
                for index in range(0, len(symbols), chunk_size):
                    symbol_chunk = symbols[index : index + chunk_size]
                    placeholders = ",".join("?" for _ in symbol_chunk)
                    query = self._build_ranked_query(  # nosec B608
                        f"symbol IN ({placeholders}) AND date >= ?"
                    )
                    query_params = tuple(symbol_chunk) + (start_date,)
                    chunk_dataframe = pd.read_sql_query(
                        query, connection, params=query_params
                    )
                    if not chunk_dataframe.empty:
                        all_dataframes.append(chunk_dataframe)
        except Exception as error:
            logger.error("[MarketData] Universe Fetch Error: %s", error)

        return all_dataframes

    def _pivot_data(self, dataframe: pd.DataFrame) -> MarketDataDict:
        """Pivots raw daily price records into separate OHLCV attribute DataFrames.

        Args:
            dataframe: Raw un-pivoted market price DataFrame.

        Returns:
            Dictionary mapping OHLCV metric names to pivoted DataFrames.
        """
        dataframe["date"] = pd.to_datetime(dataframe["date"])
        metric_columns = ["open", "high", "low", "close", "volume"]
        return {
            column_name: dataframe.pivot(
                index="date", columns="symbol", values=column_name
            )
            for column_name in metric_columns
        }

    def clear_cache(self) -> None:
        """Clears internal memory and daily data caches."""
        self._all_daily_data_cache.clear()
        self._in_memory_cache = None
        self._cache_lookback = 0
        logger.info("[MarketData] Cache cleared.")

    def get_symbol_history(self, symbol: str, days: int = 400) -> pd.DataFrame:
        """Loads OHLCV price history for a single symbol.

        Args:
            symbol: Ticker symbol.
            days: Lookback period in days.

        Returns:
            DataFrame containing daily price history.
        """
        if self._in_memory_cache and days <= self._cache_lookback:
            try:
                cached_data = {}
                has_data = False
                # Anchor relative to latest available date in cached series
                anchor_series = self._in_memory_cache.get("close")
                anchor_date = (
                    anchor_series.index.max()
                    if anchor_series is not None and not anchor_series.empty
                    else pd.Timestamp.now()
                )
                cutoff_timestamp = anchor_date - pd.Timedelta(days=days)

                for column_name in ["open", "high", "low", "close", "volume"]:
                    if symbol in self._in_memory_cache[column_name].columns:
                        series = self._in_memory_cache[column_name][symbol]
                        filtered = series[series.index >= cutoff_timestamp]
                        if not filtered.empty:
                            cached_data[column_name] = filtered
                            has_data = True

                if has_data and "close" in cached_data:
                    dataframe = pd.DataFrame(cached_data)
                    dataframe.index.name = "date"
                    return dataframe.reset_index()
            except Exception as error:
                logger.warning("Failed to extract %s from cache: %s", symbol, error)

        with self.session.connect() as connection:
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime(
                "%Y-%m-%d"
            )

            query = self._build_ranked_query(
                "symbol = ? AND date >= ?",
                include_symbol_in_select=False,
                partition_by_symbol=False,
            )
            dataframe = pd.read_sql_query(
                query, connection, params=(symbol, start_date)
            )
            if not dataframe.empty:
                dataframe["date"] = pd.to_datetime(dataframe["date"])
            return dataframe

    def get_batch_history(
        self,
        symbols: list[str],
        days: int = 100,
        end_date: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Loads price history for multiple symbols grouped into a dictionary of DataFrames.

        Args:
            symbols: List of ticker symbols.
            days: Lookback period in days.
            end_date: Optional end date string (YYYY-MM-DD).

        Returns:
            Dictionary mapping symbol strings to daily price DataFrames.
        """
        if not symbols:
            return {}

        with self.session.connect() as connection:
            if not end_date:
                end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
            start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=days)).strftime(
                "%Y-%m-%d"
            )

            placeholders = ",".join("?" for _ in symbols)
            query = self._build_ranked_query(  # nosec B608
                f"symbol IN ({placeholders}) AND date >= ? AND date <= ?"
            )

            dataframe = pd.read_sql(
                query, connection, params=symbols + [start_date, end_date]
            )

        result_dict: dict[str, pd.DataFrame] = {}
        if not dataframe.empty:
            dataframe["date"] = pd.to_datetime(dataframe["date"])
            for symbol_name, group_dataframe in dataframe.groupby("symbol"):
                result_dict[str(symbol_name)] = group_dataframe
        return result_dict

    def get_available_dates(self, start_date: str, end_date: str) -> list[pd.Timestamp]:
        """Retrieves a list of distinct trading dates within a given date range.

        Args:
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).

        Returns:
            List of pandas Timestamps representing trading days.
        """
        with self.session.connect() as connection:
            query = """
                SELECT DISTINCT date
                FROM market_prices
                WHERE date >= ? AND date <= ? AND timeframe='1D'
                ORDER BY date ASC
            """
            rows = connection.execute(query, (start_date, end_date)).fetchall()

        return [pd.Timestamp(row[0]) for row in rows]

    def get_latest_date(self) -> str | None:
        """Returns the latest available date string on the daily timeframe.

        Returns:
            Date string (YYYY-MM-DD), or None if no records exist.
        """
        try:
            with self.session.connect() as connection:
                query = "SELECT MAX(date) FROM market_prices WHERE timeframe='1D'"
                row = connection.execute(query).fetchone()
                if row and row[0]:
                    return str(row[0]).split(" ")[0]
        except Exception as error:
            logger.error("[MarketData] Could not determine latest date: %s", error)
        return None
