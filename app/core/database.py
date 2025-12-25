import sqlite3
from datetime import date
from typing import Optional

import pandas as pd


class OHLCVRepository:
    def __init__(self, db_path: str = "market_data.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        # Performance tuning for bulk writes
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA cache_size = -64000;")  # 64MB cache
        return conn

    def _init_db(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL NOT NULL CHECK(open > 0),
            high REAL NOT NULL CHECK(high > 0),
            low REAL NOT NULL CHECK(low > 0),
            close REAL NOT NULL CHECK(close > 0),
            volume INTEGER NOT NULL CHECK(volume >= 0),
            source TEXT NOT NULL DEFAULT 'yahoo',
            exchange TEXT NOT NULL DEFAULT 'SMART',

            PRIMARY KEY (symbol, date, source),
            CHECK(high >= low),
            CHECK(high >= open),
            CHECK(high >= close)
        ) WITHOUT ROWID;
        """
        # Optimized indexes
        idx1 = (
            "CREATE INDEX IF NOT EXISTS idx_date_symbol ON ohlcv (date DESC, symbol);"
        )
        idx2 = (
            "CREATE INDEX IF NOT EXISTS idx_symbol_date ON ohlcv (symbol, date DESC);"
        )

        with self._connect() as conn:
            conn.execute(ddl)
            conn.execute(idx1)
            conn.execute(idx2)

    def upsert_dataframe(
        self,
        df: pd.DataFrame,
        source: str = "yahoo",
        exchange_default: str = "SMART",
        chunk_size: int = 10_000,
    ) -> int:
        if df.empty:
            return 0

        df = df.copy().reset_index()
        # Ensure required columns
        required = {"symbol", "date", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            # Basic handling if index names were used
            if "symbol" in df.index.names and "date" in df.index.names:
                df = df.reset_index()
            elif missing:
                raise ValueError(f"Missing columns for upsert: {missing}")

        # Normalize types
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
        df["source"] = source
        if "exchange" not in df.columns:
            df["exchange"] = exchange_default

        # Fill NaNs
        num_cols = ["open", "high", "low", "close", "volume"]
        df[num_cols] = df[num_cols].fillna(0)

        records = list(
            df[
                [
                    "symbol",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "source",
                    "exchange",
                ]
            ].itertuples(index=False, name=None)
        )

        sql = """
        INSERT INTO ohlcv (symbol, date, open, high, low, close, volume, source, exchange)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, date, source) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            volume=excluded.volume,
            exchange=excluded.exchange;
        """

        with self._connect() as conn:
            total = 0
            for i in range(0, len(records), chunk_size):
                chunk = records[i : i + chunk_size]
                conn.executemany(sql, chunk)
                total += len(chunk)
            conn.commit()

        return total

    def get_latest_date_for_symbol(self, symbol: str, source: str) -> Optional[date]:
        query = "SELECT MAX(date) FROM ohlcv WHERE symbol = ? AND source = ?"
        with self._connect() as conn:
            row = conn.execute(query, (symbol, source)).fetchone()
            if row and row[0]:
                return date.fromisoformat(row[0])
        return None
