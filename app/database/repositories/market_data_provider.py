import logging

import pandas

from ..session import DatabaseSession  # NEU: Zentrale Session

logger = logging.getLogger(__name__)

# Type alias for return structure (Open, High, Low, Close, Volume as DataFrames)
type MarketDataDict = dict[str, pandas.DataFrame]


class MarketDataProvider:
    """
    Central service for accessing market data (High-Performance Read).
    Uses DatabaseSession for consistent connections and WAL-mode.
    """

    def __init__(self, session: DatabaseSession) -> None:
        # Store the session instead of the file path
        self.session = session
        self._in_memory_cache: MarketDataDict | None = None
        self._cache_lookback = 0
        self._all_daily_data_cache: dict[int, MarketDataDict] = {}

    def preload_all_data(self, days: int = 1000) -> None:
        """
        Loads ALL market data into memory for high-speed access.
        Critical for backtesting performance.
        """
        logger.info(
            "[MarketData] Preloading ALL data into memory (Lines: %dd)...", days
        )
        try:
            start_date = (
                pandas.Timestamp.now() - pandas.Timedelta(days=days)
            ).strftime("%Y-%m-%d")
            with self.session.connect() as connection:
                query = """
                    WITH ranked AS (
                        SELECT date, symbol, open, high, low, close, volume,
                               ROW_NUMBER() OVER (
                                   PARTITION BY symbol, date
                                   ORDER BY CASE WHEN provider = 'yahoo' THEN 1 WHEN provider = 'tradingview' THEN 2 ELSE 3 END
                               ) as rank_idx
                        FROM market_prices
                        WHERE date >= ? AND timeframe = '1D'
                    )
                    SELECT date, symbol, open, high, low, close, volume
                    FROM ranked
                    WHERE rank_idx = 1
                    ORDER BY date ASC
                """
                df = pandas.read_sql_query(query, connection, params=(start_date,))

            if not df.empty:
                self._in_memory_cache = self._pivot_data(df)
                self._cache_lookback = days
                logger.info("[MarketData] Cache Warm! Loaded %d rows.", len(df))
            else:
                logger.warning("[MarketData] Preload returned empty.")
        except Exception as error:
            logger.error("[MarketData] Preload Failed: %s", error)

    def get_all_daily_data(self, days: int) -> MarketDataDict | None:
        """
        Loads OHLCV data for all symbols and pivots them.
        The result is cached on the instance.

        :param days: Number of days for lookback (e.g. 400).
        """
        # Check explicit memory cache first
        if self._in_memory_cache and days <= self._cache_lookback:
            return self._in_memory_cache

        if days in self._all_daily_data_cache:
            return self._all_daily_data_cache[days]

        start_date = (pandas.Timestamp.now() - pandas.Timedelta(days=days)).strftime(
            "%Y-%m-%d"
        )
        logger.info("[MarketData] Loading data from DB (Lookback: %dd)...", days)

        try:
            with self.session.connect() as connection:
                query = """
                    WITH ranked AS (
                        SELECT date, symbol, open, high, low, close, volume,
                               ROW_NUMBER() OVER (
                                   PARTITION BY symbol, date
                                   ORDER BY CASE WHEN provider = 'yahoo' THEN 1 WHEN provider = 'tradingview' THEN 2 ELSE 3 END
                               ) as rank_idx
                        FROM market_prices
                        WHERE date >= ? AND timeframe = '1D'
                    )
                    SELECT date, symbol, open, high, low, close, volume
                    FROM ranked
                    WHERE rank_idx = 1
                    ORDER BY date ASC
                """
                df = pandas.read_sql_query(query, connection, params=(start_date,))
        except Exception as error:
            logger.error("[MarketData] DB Error: %s", error)
            return None

        if df.empty:
            logger.warning("[MarketData] No data found.")
            return None

        result = self._pivot_data(df)
        if len(self._all_daily_data_cache) >= 4:
            first_key = next(iter(self._all_daily_data_cache))
            self._all_daily_data_cache.pop(first_key)
        self._all_daily_data_cache[days] = result
        return result

    def get_universe_daily_data(
        self, symbols: list[str], days: int
    ) -> MarketDataDict | None:
        """
        Loads data only for a specific list of symbols (Pre-Filtering).
        Uses memory cache if available.
        """
        if not symbols:
            return None

        # 1. Try Memory Cache
        if self._in_memory_cache and days <= self._cache_lookback:
            filtered = {}
            for col, df in self._in_memory_cache.items():
                available_cols = [symbol for symbol in symbols if symbol in df.columns]
                if available_cols:
                    filtered[col] = df[available_cols]
                else:
                    filtered[col] = pandas.DataFrame(index=df.index)
            return filtered

        start_date = (pandas.Timestamp.now() - pandas.Timedelta(days=days)).strftime(
            "%Y-%m-%d"
        )
        logger.info(
            "[MarketData] Loading universe data (%d symbols, %dd)...",
            len(symbols),
            days,
        )

        all_dfs = []
        chunk_size = 500

        try:
            with self.session.connect() as connection:
                for i in range(0, len(symbols), chunk_size):
                    chunk = symbols[i : i + chunk_size]
                    placeholders = ",".join("?" for _ in chunk)
                    query = (
                        "WITH ranked AS ("
                        "    SELECT date, symbol, open, high, low, close, volume, "
                        "           ROW_NUMBER() OVER ( "
                        "               PARTITION BY symbol, date "
                        "               ORDER BY CASE WHEN provider = 'yahoo' THEN 1 WHEN provider = 'tradingview' THEN 2 ELSE 3 END "
                        "           ) as rank_idx "
                        "    FROM market_prices "
                        f"   WHERE symbol IN ({placeholders}) AND date >= ? AND timeframe = '1D'"  # nosec B608
                        ") "
                        "SELECT date, symbol, open, high, low, close, volume "
                        "FROM ranked "
                        "WHERE rank_idx = 1 "
                        "ORDER BY date ASC"
                    )
                    params = tuple(chunk) + (start_date,)
                    chunk_df = pandas.read_sql_query(query, connection, params=params)
                    if not chunk_df.empty:
                        all_dfs.append(chunk_df)

        except Exception as error:
            logger.error("[MarketData] Universe Fetch Error: %s", error)
            return None

        if not all_dfs:
            logger.warning("[MarketData] No data found for universe.")
            return None

        full_df = pandas.concat(all_dfs, ignore_index=True)
        return self._pivot_data(full_df)

    def _pivot_data(self, df: pandas.DataFrame) -> MarketDataDict:
        """Helper to pivot raw dataframe into MarketDataDict structure."""
        df["date"] = pandas.to_datetime(df["date"])
        pivoted_data = {
            col: df.pivot(index="date", columns="symbol", values=col)
            for col in ["open", "high", "low", "close", "volume"]
        }
        return pivoted_data

    def clear_cache(self) -> None:
        """Clears the internal LRU cache (e.g. after a database update)."""
        self._all_daily_data_cache.clear()
        self._in_memory_cache = None
        self._cache_lookback = 0
        logger.info("[MarketData] Cache cleared.")

    def get_symbol_history(self, symbol: str, days: int = 400) -> pandas.DataFrame:
        """Loads OHLCV history for a symbol without pivot. Uses cache if possible."""
        if self._in_memory_cache and days <= self._cache_lookback:
            try:
                data = {}
                cutoff = pandas.Timestamp.now() - pandas.Timedelta(days=days)
                has_data = False
                for col in ["open", "high", "low", "close", "volume"]:
                    if symbol in self._in_memory_cache[col].columns:
                        series = self._in_memory_cache[col][symbol]
                        data[col] = series[series.index >= cutoff]
                        has_data = True

                if has_data:
                    df = pandas.DataFrame(data)
                    df.index.name = "date"
                    return df.reset_index()
            except Exception as error:
                logger.warning("Failed to extract %s from cache: %s", symbol, error)

        with self.session.connect() as connection:
            start_date = (
                pandas.Timestamp.now() - pandas.Timedelta(days=days)
            ).strftime("%Y-%m-%d")

            sql = """
                WITH ranked AS (
                    SELECT date, open, high, low, close, volume,
                           ROW_NUMBER() OVER (
                               PARTITION BY date
                               ORDER BY CASE WHEN provider = 'yahoo' THEN 1 WHEN provider = 'tradingview' THEN 2 ELSE 3 END
                           ) as rank_idx
                    FROM market_prices
                    WHERE symbol = ? AND date >= ? AND timeframe = '1D'
                )
                SELECT date, open, high, low, close, volume
                FROM ranked
                WHERE rank_idx = 1
                ORDER BY date ASC
            """
            df = pandas.read_sql_query(sql, connection, params=(symbol, start_date))
            if not df.empty:
                df["date"] = pandas.to_datetime(df["date"])
            return df

    def get_batch_history(
        self,
        symbols: list[str],
        days: int = 100,
        end_date: str | None = None,
    ) -> dict[str, pandas.DataFrame]:
        """Loads history for multiple symbols."""
        if not symbols:
            return {}

        with self.session.connect() as connection:
            if not end_date:
                end_date = pandas.Timestamp.now().strftime("%Y-%m-%d")
            start_date = (
                pandas.Timestamp(end_date) - pandas.Timedelta(days=days)
            ).strftime("%Y-%m-%d")

            placeholders = ",".join("?" for _ in symbols)
            sql = (
                "WITH ranked AS ("
                "    SELECT symbol, date, open, high, low, close, volume, "
                "           ROW_NUMBER() OVER ( "
                "               PARTITION BY symbol, date "
                "               ORDER BY CASE WHEN provider = 'yahoo' THEN 1 WHEN provider = 'tradingview' THEN 2 ELSE 3 END "
                "           ) as rank_idx "
                "    FROM market_prices "
                f"   WHERE symbol IN ({placeholders}) AND date >= ? AND date <= ? AND timeframe = '1D'"  # nosec B608
                ") "
                "SELECT symbol, date, open, high, low, close, volume "
                "FROM ranked "
                "WHERE rank_idx = 1 "
                "ORDER BY date ASC"
            )

            df = pandas.read_sql(
                sql, connection, params=symbols + [start_date, end_date]
            )

        result = {}
        if not df.empty:
            df["date"] = pandas.to_datetime(df["date"])
            for symbol, group in df.groupby("symbol"):
                result[symbol] = group
        return result

    def get_available_dates(
        self, start_date: str, end_date: str
    ) -> list[pandas.Timestamp]:
        """Retrieves a list of all available trading days in the range (fallback for missing SPY)."""
        with self.session.connect() as connection:
            query = """
                SELECT DISTINCT date
                FROM market_prices
                WHERE date >= ? AND date <= ? AND timeframe='1D'
                ORDER BY date ASC
            """
            rows = connection.execute(query, (start_date, end_date)).fetchall()

        return [pandas.Timestamp(row[0]) for row in rows]

    def get_latest_date(self) -> str | None:
        """Returns the date of the last available record (Timeframe '1D').

        Serves as the 'Global Analysis Date' for the screener.
        """
        try:
            with self.session.connect() as connection:
                query = "SELECT MAX(date) FROM market_prices WHERE timeframe='1D'"
                row = connection.execute(query).fetchone()
                if row and row[0]:
                    # Strip timestamp if present: "2026-02-04 00:00:00" -> "2026-02-04"
                    return str(row[0]).split(" ")[0]
        except Exception as error:
            logger.error("[MarketData] Could not determine the latest date: %s", error)
        return None
