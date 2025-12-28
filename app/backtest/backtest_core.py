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
    def cleanup_strategy(self, strategy_name: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM backtest_trades WHERE strategy_name = ?",
                (strategy_name,),
            )
            conn.execute(
                "DELETE FROM backtest_equity WHERE strategy_name = ?", (strategy_name,)
            )
            conn.commit()

    def init_tables(self, clear_existing: bool = False):
        with self._get_connection() as conn:
            """
            conn.execute("DROP TABLE IF EXISTS backtest_trades")
            conn.execute("DROP TABLE IF EXISTS backtest_equity")
            """
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
                    strategy_name TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_equity (
                    date TEXT PRIMARY KEY,
                    strategy_name TEXT NOT NULL,
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

    def log_equity(self, date_str, equity, cash, pos_val, dd, strategy_name):
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO backtest_equity VALUES (?,?,?,?,?,?)",
                (date_str, strategy_name, equity, cash, pos_val, dd),
            )
            conn.commit()

    def get_equity_curve(self, strategy_name) -> pd.DataFrame:
        with self._get_connection() as conn:
            # Filtert explizit nach der übergebenen Strategie
            df = pd.read_sql(
                "SELECT * FROM backtest_equity WHERE strategy_name = ? ORDER BY date",
                conn,
                params=(strategy_name,),
            )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date")
        return df

    def get_trades(self, strategy_name) -> pd.DataFrame:
        with self._get_connection() as conn:
            return pd.read_sql(
                "SELECT * FROM backtest_trades WHERE strategy_name = ?",
                conn,
                params=(strategy_name,),
            )
