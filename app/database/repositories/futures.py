"""Futures price repository for managing persistence of 30-minute and daily futures data.

Manages the futures.db database with two tables:
- futures_prices: Raw 30-minute bars from TradingView
- futures_daily: Cash-session aggregated daily bars for comparison with equity ETFs
"""

import logging
from typing import Any

import pandas as pd

from .base import BaseRepository

logger = logging.getLogger(__name__)


class FuturesRepository(BaseRepository):
    """Repository for querying and persisting futures price data."""

    def init_schema(self) -> None:
        """Creates tables and indexes for futures price data."""
        with self.session.connect() as connection:
            self.execute(
                """
                CREATE TABLE IF NOT EXISTS futures_prices (
                    symbol TEXT NOT NULL,
                    contract TEXT NOT NULL,
                    datetime TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    timeframe TEXT NOT NULL DEFAULT '30min',
                    provider TEXT NOT NULL DEFAULT 'tradingview',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (contract, datetime, timeframe)
                )
            """,
                connection=connection,
            )

            self.execute(
                """
                CREATE TABLE IF NOT EXISTS futures_daily (
                    symbol TEXT NOT NULL,
                    contract TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    session TEXT NOT NULL DEFAULT 'cash',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (contract, date, session)
                )
            """,
                connection=connection,
            )

            self.execute(
                "CREATE INDEX IF NOT EXISTS idx_futures_sym ON futures_prices(symbol)",
                connection=connection,
            )
            self.execute(
                "CREATE INDEX IF NOT EXISTS idx_futures_contract ON futures_prices(contract)",
                connection=connection,
            )
            self.execute(
                "CREATE INDEX IF NOT EXISTS idx_futures_daily_sym ON futures_daily(symbol)",
                connection=connection,
            )

    # --- Intraday (30-min) Data ---

    def save_bulk_futures_prices(self, records: list[Any]) -> None:
        """Saves a list of FuturesPrice objects into the futures_prices table.

        Args:
            records: List of FuturesPrice instances with a to_db_row() method.
        """
        if not records:
            return

        data_to_insert = [record.to_db_row() for record in records]
        sql = (
            "INSERT OR REPLACE INTO futures_prices "
            "(symbol, contract, datetime, open, high, low, close, volume, timeframe, provider) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        with self.session.connect() as connection:
            connection.executemany(sql, data_to_insert)

    def get_latest_datetime(self, contract: str) -> str | None:
        """Returns the most recent bar datetime for a given contract.

        Args:
            contract: Full contract identifier (e.g. "MNQU2026").

        Returns:
            ISO datetime string of the latest bar, or None if no data exists.
        """
        raw_value = self.fetch_value(
            "SELECT MAX(datetime) FROM futures_prices WHERE contract = ?",
            (contract,),
        )
        return str(raw_value) if raw_value is not None else None

    def get_intraday_bar_count(self, contract: str) -> int:
        """Returns the total number of intraday bars for a contract.

        Args:
            contract: Full contract identifier (e.g. "MNQU2026").
        """
        raw_count = self.fetch_value(
            "SELECT COUNT(*) FROM futures_prices WHERE contract = ?",
            (contract,),
        )
        return int(raw_count) if raw_count is not None else 0

    # --- Daily (Cash Session) Data ---

    def save_bulk_daily_bars(self, records: list[Any]) -> None:
        """Saves a list of CashSessionDailyBar objects into futures_daily.

        Args:
            records: List of CashSessionDailyBar instances with a to_db_row() method.
        """
        if not records:
            return

        data_to_insert = [record.to_db_row() for record in records]
        sql = (
            "INSERT OR REPLACE INTO futures_daily "
            "(symbol, contract, date, open, high, low, close, volume, session) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        with self.session.connect() as connection:
            connection.executemany(sql, data_to_insert)

    def get_daily_history(
        self,
        symbol: str,
        session: str = "cash",
    ) -> pd.DataFrame:
        """Loads daily bar history for a symbol across all contracts.

        Results are ordered by date ascending, useful for comparison with
        equity ETF daily data (e.g. QQQ, SPY).

        Args:
            symbol: Internal base symbol (e.g. "MNQ").
            session: Session type filter. Defaults to "cash".

        Returns:
            DataFrame with columns: date, contract, open, high, low, close, volume.
        """
        sql = """
            SELECT date, contract, open, high, low, close, volume
            FROM futures_daily
            WHERE symbol = ? AND session = ?
            ORDER BY date ASC
        """
        with self.session.connect() as connection:
            dataframe = pd.read_sql_query(sql, connection, params=(symbol, session))
            if not dataframe.empty:
                dataframe["date"] = pd.to_datetime(dataframe["date"])
            return dataframe

    def get_contract_history(
        self,
        contract: str,
    ) -> pd.DataFrame:
        """Loads all 30-minute bars for a specific contract.

        Args:
            contract: Full contract identifier (e.g. "MNQU2026").

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume.
        """
        sql = """
            SELECT datetime, open, high, low, close, volume
            FROM futures_prices
            WHERE contract = ?
            ORDER BY datetime ASC
        """
        with self.session.connect() as connection:
            dataframe = pd.read_sql_query(sql, connection, params=(contract,))
            if not dataframe.empty:
                dataframe["datetime"] = pd.to_datetime(dataframe["datetime"])
            return dataframe
