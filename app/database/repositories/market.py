import logging

import pandas

from .base import BaseRepository

logger = logging.getLogger(__name__)


class MarketRepository(BaseRepository):
    def init_schema(self) -> None:
        """Creates the table for market data."""
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
        self.execute(
            "INSERT OR REPLACE INTO ignored_symbols (symbol, reason) VALUES (?, ?)",
            (symbol, reason),
        )

    def get_ignored_symbols(self) -> set[str]:
        rows = self.fetch_all("SELECT symbol FROM ignored_symbols")
        return {row["symbol"] for row in rows}

    def remove_ignored_symbol(self, symbol: str) -> None:
        """Removes a symbol from the ignored symbols blacklist."""
        self.execute(
            "DELETE FROM ignored_symbols WHERE symbol = ?",
            (symbol,),
        )

    def get_all_known_symbols(self) -> list[str]:
        rows = self.fetch_all("SELECT DISTINCT symbol FROM market_prices")
        return [row["symbol"] for row in rows]

    def get_outdated_symbols(
        self, reference_date: str, provider: str = "yahoo"
    ) -> list[str]:
        sql = """
            SELECT symbol FROM market_prices
            WHERE provider = ? AND symbol NOT IN (SELECT symbol FROM ignored_symbols)
            GROUP BY symbol HAVING MAX(date) < ?
        """
        rows = self.fetch_all(sql, (provider, reference_date))
        return [row["symbol"] for row in rows]

    def get_symbols_with_missing_history(self, cutoff_date: str) -> list[str]:
        """Finds symbols whose history starts AFTER the cutoff_date (insufficient data)."""
        sql = """
            SELECT symbol FROM market_prices
            WHERE symbol NOT IN (SELECT symbol FROM ignored_symbols)
            GROUP BY symbol HAVING MIN(date) > ?
        """
        rows = self.fetch_all(sql, (cutoff_date,))
        return [row["symbol"] for row in rows]

    # --- Data Access Logic (Single Value) ---
    def get_latest_price(self, symbol: str) -> float | None:
        return self.fetch_value(
            "SELECT close FROM market_prices WHERE symbol = ? AND timeframe = '1D' ORDER BY date DESC LIMIT 1",
            (symbol,),
        )

    def get_trading_days_count(
        self, symbol: str, start_date: str, end_date: str
    ) -> int:
        start_date_string = str(start_date).split(" ")[0]
        end_date_string = str(end_date).split(" ")[0]
        sql = "SELECT COUNT(*) FROM market_prices WHERE symbol = ? AND date >= ? AND date <= ? AND timeframe = '1D'"
        return self.fetch_value(sql, (symbol, start_date_string, end_date_string)) or 0

    # --- Helper for Validation ---
    def get_ohlcv(self, symbol: str, date: str) -> dict[str, object] | None:
        """Fetches a single OHLCV record."""
        sql = "SELECT * FROM market_prices WHERE symbol = ? AND date = ? AND timeframe = '1D'"
        row = self.fetch_one(sql, (symbol, date))
        return dict(row) if row else None

    # --- Data Access Logic (Bulk / Pandas) ---
    def get_data_for_lookback(self, start_date: str) -> pandas.DataFrame:
        """Loads all data from start_date for pivot operations."""
        sql = """
            SELECT date, symbol, open, high, low, close, volume
            FROM market_prices
            WHERE date >= ? AND timeframe = '1D'
            ORDER BY date ASC
        """
        with self.session.connect() as connection:
            df = pandas.read_sql_query(sql, connection, params=(start_date,))
            if not df.empty:
                df["date"] = pandas.to_datetime(df["date"])  # FIX: Type Conversion
            return df

    def get_symbol_history_raw(self, symbol: str, start_date: str) -> pandas.DataFrame:
        """Loads history for a single symbol (IMPORTANT for TradeManager)."""
        sql = """
            SELECT date, open, high, low, close, volume 
            FROM market_prices 
            WHERE symbol = ? AND date >= ? AND timeframe='1D' 
            ORDER BY date ASC
        """
        with self.session.connect() as connection:
            df = pandas.read_sql_query(sql, connection, params=(symbol, start_date))
            if not df.empty:
                df["date"] = pandas.to_datetime(df["date"])  # FIX: Type Conversion
            return df

    def get_batch_history_raw(
        self, symbols: list[str], start_date: str, end_date: str
    ) -> pandas.DataFrame:
        """Loads history for multiple symbols."""
        if not symbols:
            return pandas.DataFrame()
        placeholders = ",".join("?" for _ in symbols)
        sql = f"""
            SELECT symbol, date, open, high, low, close, volume 
            FROM market_prices 
            WHERE symbol IN ({placeholders}) 
            AND date >= ? AND date <= ? AND timeframe='1D' 
            ORDER BY date ASC
        """
        params = symbols + [start_date, end_date]
        with self.session.connect() as connection:
            df = pandas.read_sql_query(sql, connection, params=params)
            if not df.empty:
                df["date"] = pandas.to_datetime(df["date"])  # FIX: Type Conversion
            return df

    def save_bulk_prices(self, records: list[object]) -> None:
        """Saves a list of MarketPrice objects (or tuples for backward compat)."""
        if not records:
            return

        # Prepare Data
        # Check if first item is an object (MarketPrice) or tuple
        first = records[0]
        data_to_insert = []

        if hasattr(first, "to_db_row"):
            # It's a MarketPrice object
            data_to_insert = [record.to_db_row() for record in records]
        else:
            # It's likely already a tuple (Legacy support)
            data_to_insert = records

        sql = "INSERT OR REPLACE INTO market_prices (symbol, date, open, high, low, close, volume, provider, timeframe) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        with self.session.connect() as connection:
            connection.executemany(sql, data_to_insert)
