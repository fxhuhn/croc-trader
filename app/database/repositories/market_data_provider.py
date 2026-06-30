import logging
from functools import lru_cache

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

    def preload_all_data(self, days: int = 1000) -> None:
        """
        Loads ALL market data into memory for high-speed access.
        Critical for backtesting performance.
        """
        logger.info(
            "[MarketData] Preloading ALL data into memory (Lines: %dd)...", days
        )
        # Reuse existing logic but store it
        # We assume universe is all symbols in DB or we fetch all.
        try:
            start_date = (
                pandas.Timestamp.now() - pandas.Timedelta(days=days)
            ).strftime("%Y-%m-%d")
            with self.session.connect() as connection:
                # Fetch EVERYTHING
                query = "SELECT date, symbol, open, high, low, close, volume FROM market_prices WHERE date >= ? AND timeframe='1D' ORDER BY date ASC"
                df = pandas.read_sql_query(query, connection, params=(start_date,))

            if not df.empty:
                self._in_memory_cache = self._pivot_data(df)
                self._cache_lookback = days
                logger.info("[MarketData] Cache Warm! Loaded %d rows.", len(df))
            else:
                logger.warning("[MarketData] Preload returned empty.")
        except Exception as error:
            logger.error("[MarketData] Preload Failed: %s", error)

    @lru_cache(maxsize=4)
    def get_all_daily_data(self, days: int) -> MarketDataDict | None:
        """
        Loads OHLCV data for all symbols and pivots them.
        The result is cached (LRU).

        :param days: Number of days for lookback (e.g. 400).
        """
        # Check explicit memory cache first
        if self._in_memory_cache and days <= self._cache_lookback:
            return self._in_memory_cache

        start_date = (pandas.Timestamp.now() - pandas.Timedelta(days=days)).strftime(
            "%Y-%m-%d"
        )
        logger.info("[MarketData] Loading data from DB (Lookback: %dd)...", days)

        try:
            # Use the central session
            # Yields a configured connection (WAL-mode active)
            with self.session.connect() as connection:
                # Optimized query on '1D'
                query = """
                    SELECT date, symbol, open, high, low, close, volume
                    FROM market_prices
                    WHERE date >= ? AND timeframe = '1D'
                    ORDER BY date ASC
                """
                # Pandas can work directly with the sqlite3 connection object
                df = pandas.read_sql_query(query, connection, params=(start_date,))
        except Exception as error:
            logger.error("[MarketData] DB Error: %s", error)
            return None

        if df.empty:
            logger.warning("[MarketData] No data found.")
            return None

        return self._pivot_data(df)

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
            # Filter the pivoted cache for requested symbols
            filtered = {}
            for col, df in self._in_memory_cache.items():
                available_cols = [symbol for symbol in symbols if symbol in df.columns]
                if available_cols:
                    filtered[col] = df[available_cols]  # Slice columns
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
        chunk_size = 500  # Safe limit for SQLite variables

        try:
            with self.session.connect() as connection:
                for i in range(0, len(symbols), chunk_size):
                    chunk = symbols[i : i + chunk_size]
                    placeholders = ",".join("?" for _ in chunk)
                    query = f"""
                        SELECT date, symbol, open, high, low, close, volume
                        FROM market_prices
                        WHERE symbol IN ({placeholders})
                          AND date >= ? 
                          AND timeframe = '1D'
                        ORDER BY date ASC
                    """
                    # Params: symbols + start_date
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
        # Data type conversion
        df["date"] = pandas.to_datetime(df["date"])

        # Pivot logic for vectorized strategies
        # Generates a dict with DataFrames (Index=Date, Columns=Symbols)
        pivoted_data = {
            col: df.pivot(index="date", columns="symbol", values=col)
            for col in ["open", "high", "low", "close", "volume"]
        }

        return pivoted_data

    def clear_cache(self) -> None:
        """Clears the internal LRU cache (e.g. after a database update)."""
        self.get_all_daily_data.cache_clear()
        self._in_memory_cache = None
        logger.info("[MarketData] Cache cleared.")

    def get_symbol_history(self, symbol: str, days: int = 400) -> pandas.DataFrame:
        """Loads OHLCV history for a symbol without pivot. Uses cache if possible."""
        # 1. Try Memory Cache
        if self._in_memory_cache and days <= self._cache_lookback:
            # Reconstruct DataFrame from Pivoted Data
            try:
                data = {}
                # Start date filter
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
                # Fallback to DB

        # Since BaseRepository allows SQL (layer boundary), this is acceptable.
        with self.session.connect() as connection:
            # end_date not used in query
            start_date = (
                pandas.Timestamp.now() - pandas.Timedelta(days=days)
            ).strftime("%Y-%m-%d")

            df = pandas.read_sql_query(
                "SELECT date, open, high, low, close, volume FROM market_prices WHERE symbol = ? AND date >= ? AND timeframe='1D' ORDER BY date ASC",
                connection,
                params=(symbol, start_date),
            )
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
            sql = f"""SELECT symbol, date, open, high, low, close, volume FROM market_prices 
                      WHERE symbol IN ({placeholders}) AND date >= ? AND date <= ? AND timeframe='1D' ORDER BY date ASC"""

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
