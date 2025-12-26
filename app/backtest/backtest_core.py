from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Add project root to sys.path so we can import 'app', 'core', 'services'
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from app.database.sqlite_repo import SQLiteRepository


@dataclass
class BacktestTrade:
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    return_pct: float
    hold_days: int


class BacktestRepository(SQLiteRepository):
    def init_tables(self, clear_existing: bool = True):
        with self._get_connection() as conn:
            if clear_existing:
                conn.execute("DROP TABLE IF EXISTS backtest_trades")
                conn.execute("DROP TABLE IF EXISTS backtest_equity")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    entry_date TEXT NOT NULL,
                    exit_date TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    shares INTEGER NOT NULL,
                    pnl REAL NOT NULL,
                    return_pct REAL NOT NULL,
                    hold_days INTEGER NOT NULL,
                    strategy_name TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_equity (
                    date TEXT PRIMARY KEY,
                    total_equity REAL NOT NULL,
                    cash REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    drawdown_pct REAL NOT NULL
                )
            """)
            conn.commit()

    def log_trade(self, t: BacktestTrade, strategy_name: str):
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO backtest_trades
                (symbol, entry_date, exit_date, entry_price, exit_price,
                 shares, pnl, return_pct, hold_days, strategy_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    t.symbol,
                    t.entry_date,
                    t.exit_date,
                    t.entry_price,
                    t.exit_price,
                    t.shares,
                    t.pnl,
                    t.return_pct,
                    t.hold_days,
                    strategy_name,
                ),
            )
            conn.commit()

    def log_equity(self, date_str, equity, cash, pos_val, dd):
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO backtest_equity VALUES (?,?,?,?,?)",
                (date_str, equity, cash, pos_val, dd),
            )
            conn.commit()

    def get_equity_curve(self) -> pd.DataFrame:
        with self._get_connection() as conn:
            df = pd.read_sql("SELECT * FROM backtest_equity ORDER BY date", conn)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date")
        return df

    def get_trades(self) -> pd.DataFrame:
        with self._get_connection() as conn:
            return pd.read_sql("SELECT * FROM backtest_trades", conn)
