"""Market repository module for managing persistence of market price data and symbol blacklists.

Provides database access operations for market_prices and ignored_symbols tables in stocks.db.
"""

import logging
from datetime import date
from typing import Any

import pandas as pd

from .base import BaseRepository

logger = logging.getLogger(__name__)


class MarketRepository(BaseRepository):
    """Repository for querying and persisting market price data and symbol blacklist records."""

    def init_schema(self) -> None:
        """Creates tables and indexes for market price data and ignored symbol records."""
        with self.session.connect() as connection:
            self.execute(
                """
                CREATE TABLE IF NOT EXISTS market_prices (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    provider TEXT NOT NULL DEFAULT 'yahoo',
                    timeframe TEXT NOT NULL DEFAULT '1D',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, date, timeframe, provider)
                )
            """,
                connection=connection,
            )

            self.execute(
                """
                CREATE TABLE IF NOT EXISTS ignored_symbols (
                    symbol TEXT PRIMARY KEY,
                    reason TEXT,
                    ignored_since TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
                connection=connection,
            )

            self.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_date ON market_prices(date)",
                connection=connection,
            )
            self.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_sym_tf ON market_prices(symbol, timeframe)",
                connection=connection,
            )

    # --- Blacklist Logic ---

    def ignore_symbol(self, symbol: str, reason: str) -> None:
        """Adds or updates a symbol in the ignored symbols blacklist with a reason.

        Args:
            symbol: Ticker symbol to blacklist.
            reason: Text explanation for blacklisting the symbol.
        """
        self.execute(
            "INSERT OR REPLACE INTO ignored_symbols (symbol, reason) VALUES (?, ?)",
            (symbol, reason),
        )

    def get_ignored_symbols(self) -> set[str]:
        """Fetches all blacklisted symbols from the database.

        Returns:
            Set of blacklisted ticker symbols.
        """
        rows = self.fetch_all("SELECT symbol FROM ignored_symbols")
        return {row["symbol"] for row in rows}

    def remove_ignored_symbol(self, symbol: str) -> None:
        """Removes a symbol from the ignored symbols blacklist.

        Args:
            symbol: Ticker symbol to remove from the blacklist.
        """
        self.execute(
            "DELETE FROM ignored_symbols WHERE symbol = ?",
            (symbol,),
        )

    def clear_ignored_symbols(self) -> None:
        """Clears all blacklisted symbols from the database."""
        self.execute("DELETE FROM ignored_symbols")

    def get_all_known_symbols(self) -> list[str]:
        """Fetches all unique ticker symbols present in the market_prices table.

        Returns:
            List of distinct ticker symbols.
        """
        rows = self.fetch_all("SELECT DISTINCT symbol FROM market_prices")
        return [row["symbol"] for row in rows]

    def get_outdated_symbols(
        self, reference_date: str, provider: str | None = None
    ) -> list[str]:
        """Finds symbols whose latest price date is older than the reference date.

        Args:
            reference_date: Cutoff date string (YYYY-MM-DD).
            provider: Optional provider filter ('yahoo' or 'tradingview').

        Returns:
            List of ticker symbols requiring market data updates.
        """
        if provider:
            sql_query = """
                SELECT symbol FROM market_prices
                WHERE provider = ? AND symbol NOT IN (SELECT symbol FROM ignored_symbols)
                GROUP BY symbol HAVING MAX(date) < ?
            """
            rows = self.fetch_all(sql_query, (provider, reference_date))
        else:
            sql_query = """
                SELECT symbol FROM market_prices
                WHERE symbol NOT IN (SELECT symbol FROM ignored_symbols)
                GROUP BY symbol HAVING MAX(date) < ?
            """
            rows = self.fetch_all(sql_query, (reference_date,))
        return [row["symbol"] for row in rows]

    def get_symbols_with_missing_history(
        self, cutoff_date: str = "2020-01-01"
    ) -> list[str]:
        """Finds symbols whose historical price data starts after the cutoff date.

        Args:
            cutoff_date: Required earliest date string (YYYY-MM-DD). Defaults to '2020-01-01'.

        Returns:
            List of ticker symbols with insufficient historical depth.
        """
        sql_query = """
            SELECT symbol FROM market_prices
            WHERE symbol NOT IN (SELECT symbol FROM ignored_symbols)
            GROUP BY symbol HAVING MIN(date) > ?
        """
        rows = self.fetch_all(sql_query, (cutoff_date,))
        return [row["symbol"] for row in rows]

    # --- Data Access Logic (Single Value) ---

    def get_latest_updated_at(self) -> str | None:
        """Fetches the latest updated_at timestamp from market_prices in stocks.db.

        Returns:
            Formatted timestamp string (YYYY-MM-DD HH:MM), or None if database is empty.
        """
        latest_timestamp = self.fetch_value(
            "SELECT updated_at FROM market_prices WHERE updated_at IS NOT NULL ORDER BY updated_at DESC LIMIT 1"
        )
        if latest_timestamp:
            raw_timestamp = str(latest_timestamp).strip()
            timestamp_parts = raw_timestamp.replace("T", " ").split(" ")
            if len(timestamp_parts) >= 2:
                date_part = timestamp_parts[0]
                time_part = timestamp_parts[1].split(".")[0][:5]
                return f"{date_part} {time_part}"
            return timestamp_parts[0]
        return None

    def get_latest_price(self, symbol: str) -> float | None:
        """Fetches the most recent closing price for a symbol on the daily timeframe.

        Args:
            symbol: Ticker symbol.

        Returns:
            Latest closing price float, or None if symbol data is missing.
        """
        sql_query = """
            WITH ranked AS (
                SELECT close,
                       ROW_NUMBER() OVER (
                           PARTITION BY symbol, date
                           ORDER BY CASE WHEN provider = 'yahoo' THEN 1 WHEN provider = 'tradingview' THEN 2 ELSE 3 END
                       ) as rank_idx,
                       date
                FROM market_prices
                WHERE symbol = ? AND timeframe = '1D'
            )
            SELECT close FROM ranked WHERE rank_idx = 1 ORDER BY date DESC LIMIT 1
        """
        return self.fetch_value(sql_query, (symbol,))

    def get_trading_days_count(
        self,
        symbol: str,
        start_date: str = "2020-01-01",
        end_date: str | None = None,
    ) -> int:
        """Counts distinct trading days available for a symbol within a date range.

        Args:
            symbol: Ticker symbol.
            start_date: Range start date string. Defaults to '2020-01-01'.
            end_date: Range end date string. Defaults to today's date string.

        Returns:
            Count of distinct trading days.
        """
        if end_date is None:
            end_date = date.today().strftime("%Y-%m-%d")
        start_date_string = str(start_date).split(" ")[0]
        end_date_string = str(end_date).split(" ")[0]
        sql_query = "SELECT COUNT(DISTINCT date) FROM market_prices WHERE symbol = ? AND date >= ? AND date <= ? AND timeframe = '1D'"
        return (
            self.fetch_value(sql_query, (symbol, start_date_string, end_date_string))
            or 0
        )

    # --- Helper for Validation ---

    def get_ohlcv(self, symbol: str, date: str) -> dict[str, object] | None:
        """Fetches a single OHLCV record for a symbol on a specific date.

        Args:
            symbol: Ticker symbol.
            date: Quote date string (YYYY-MM-DD).

        Returns:
            Dictionary containing price record fields, or None if record is missing.
        """
        sql_query = """
            SELECT * FROM market_prices
            WHERE symbol = ? AND date = ? AND timeframe = '1D'
            ORDER BY CASE WHEN provider = 'yahoo' THEN 1 WHEN provider = 'tradingview' THEN 2 ELSE 3 END
            LIMIT 1
        """
        row = self.fetch_one(sql_query, (symbol, date))
        return dict(row) if row else None

    # --- Data Access Logic (Bulk / Pandas) ---

    def get_data_for_lookback(self, start_date: str = "2020-01-01") -> pd.DataFrame:
        """Loads all daily price data starting from a given date.

        Args:
            start_date: Start date string (YYYY-MM-DD). Defaults to '2020-01-01'.

        Returns:
            DataFrame containing ranked daily price records.
        """
        sql_query = """
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
        with self.session.connect() as connection:
            dataframe = pd.read_sql_query(sql_query, connection, params=(start_date,))
            if not dataframe.empty:
                dataframe["date"] = pd.to_datetime(dataframe["date"])
            return dataframe

    def get_symbol_history_raw(
        self, symbol: str, start_date: str = "2020-01-01"
    ) -> pd.DataFrame:
        """Loads price history for a single symbol starting from a given date.

        Args:
            symbol: Ticker symbol.
            start_date: Start date string (YYYY-MM-DD). Defaults to '2020-01-01'.

        Returns:
            DataFrame containing ranked daily price records for the symbol.
        """
        sql_query = """
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
        with self.session.connect() as connection:
            dataframe = pd.read_sql_query(
                sql_query, connection, params=(symbol, start_date)
            )
            if not dataframe.empty:
                dataframe["date"] = pd.to_datetime(dataframe["date"])
            return dataframe

    def get_batch_history_raw(
        self,
        symbols: list[str],
        start_date: str = "2020-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Loads price history for multiple symbols within a date range.

        Args:
            symbols: List of ticker symbols.
            start_date: Start date string (YYYY-MM-DD). Defaults to '2020-01-01'.
            end_date: End date string (YYYY-MM-DD). Defaults to today's date string.

        Returns:
            DataFrame containing ranked daily price records for the requested symbols.
        """
        if end_date is None:
            end_date = date.today().strftime("%Y-%m-%d")
        if not symbols:
            return pd.DataFrame()
        placeholders = ",".join("?" for _ in symbols)
        sql_query = (
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
        params = symbols + [start_date, end_date]
        with self.session.connect() as connection:
            dataframe = pd.read_sql_query(sql_query, connection, params=params)
            if not dataframe.empty:
                dataframe["date"] = pd.to_datetime(dataframe["date"])
            return dataframe

    def save_bulk_prices(self, records: list[Any]) -> None:
        """Saves a list of MarketPrice objects or raw tuples into the database.

        Args:
            records: List of MarketPrice instances or row tuples matching table layout.
        """
        if not records:
            return

        first_record = records[0]
        data_to_insert: list[Any] = []

        if hasattr(first_record, "to_db_row"):
            data_to_insert = [record.to_db_row() for record in records]
        else:
            data_to_insert = records

        sql_query = "INSERT OR REPLACE INTO market_prices (symbol, date, open, high, low, close, volume, provider, timeframe) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        with self.session.connect() as connection:
            connection.executemany(sql_query, data_to_insert)
