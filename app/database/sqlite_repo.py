from __future__ import annotations

import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.protocols import SignalRepository


class SQLiteRepository(SignalRepository):
    def __init__(self, db_path: str = "signals.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    close REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    wuk REAL NOT NULL,
                    status TEXT NOT NULL,
                    kerze TEXT NOT NULL,
                    wolke TEXT,
                    trend TEXT NOT NULL,
                    setter TEXT NOT NULL,
                    welle TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timestamp_covering
                ON signals(symbol, timestamp DESC, signal)
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    signal TEXT NOT NULL,

                    buy_limit REAL,
                    stop_loss REAL,
                    take_profit REAL,

                    entry_time TEXT,
                    entry_price REAL,

                    exit_time TEXT,
                    exit_price REAL,

                    current_price REAL,
                    state TEXT DEFAULT 'pending',

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE(symbol, timestamp, signal)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_state
                ON signal_trades(state, symbol)
            """)
            conn.commit()

    def get_all_trades(self, limit: int = 500) -> list[dict]:
        """Return all tracked trades, newest first."""
        query = """
            SELECT *
            FROM signal_trades
            ORDER BY created_at DESC
            LIMIT ?
        """
        with self._get_conn() as conn:
            rows = conn.execute(query, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def save_signal(self, data: dict[str, Any]) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO signals
                (symbol, timeframe, timestamp, signal, close, high, low, wuk,
                 status, kerze, wolke, trend, setter, welle)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["symbol"],
                    data["timeframe"],
                    data["timestamp"].isoformat()
                    if hasattr(data["timestamp"], "isoformat")
                    else data["timestamp"],
                    data["signal"],
                    data["close"],
                    data["high"],
                    data["low"],
                    data["wuk"],
                    data["status"],
                    data["kerze"],
                    data.get("wolke"),
                    data["trend"],
                    data["setter"],
                    data["setter"] if "setter" in data else data["setter"],
                    data["welle"],
                ),
            )
            conn.commit()

    def get_signals(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        signal: Optional[str] = None,
        day: Optional[str] = None,
        limit: int = 500,
    ) -> list[dict]:
        query = """
            SELECT s.*,
                   t.id as trade_id,
                   t.state as trade_state,
                   t.buy_limit, t.stop_loss, t.take_profit,
                   t.entry_time, t.entry_price,
                   t.exit_time, t.exit_price,
                   t.current_price,
                   CASE WHEN t.id IS NOT NULL THEN 1 ELSE 0 END as is_tracked
            FROM signals s
            LEFT JOIN signal_trades t
              ON s.symbol = t.symbol AND s.timestamp = t.timestamp AND s.signal = t.signal
            WHERE 1=1
        """
        params: list[Any] = []

        if symbol:
            query += " AND s.symbol = ?"
            params.append(symbol)
        if timeframe:
            query += " AND s.timeframe = ?"
            params.append(timeframe)
        if signal:
            query += " AND s.signal = ?"
            params.append(signal)
        if day:
            query += " AND date(s.timestamp) = ?"
            params.append(day)

        query += " ORDER BY s.timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_distinct(self, column: str) -> list[str]:
        if column not in ("symbol", "timeframe", "signal"):
            return []
        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT {column} FROM signals ORDER BY {column}"  # nosec B608
            ).fetchall()
            return [r[0] for r in rows if r[0]]

    def toggle_trade_tracking(self, symbol: str, timestamp: str, signal: str) -> dict:
        with self._get_conn() as conn:
            exists = conn.execute(
                "SELECT id FROM signal_trades WHERE symbol=? AND timestamp=? AND signal=?",
                (symbol, timestamp, signal),
            ).fetchone()

            if exists:
                conn.execute(
                    "DELETE FROM signal_trades WHERE symbol=? AND timestamp=? AND signal=?",
                    (symbol, timestamp, signal),
                )
                conn.commit()
                return {"is_tracked": False}

            conn.execute(
                "INSERT INTO signal_trades (symbol, timestamp, signal, state) VALUES (?, ?, ?, 'pending')",
                (symbol, timestamp, signal),
            )
            conn.commit()
            return {"is_tracked": True, "state": "pending"}

    def get_trade(self, trade_id: int) -> Optional[dict]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM signal_trades WHERE id = ?", (trade_id,)
            ).fetchone()
            return dict(row) if row else None

    def update_trade(self, trade_id: int, data: dict) -> bool:
        allowed = {
            "buy_limit",
            "stop_loss",
            "take_profit",
            "entry_time",
            "entry_price",
            "exit_time",
            "exit_price",
            "current_price",
            "state",
        }
        fields = []
        values: list[Any] = []

        for k in allowed:
            if k in data:
                fields.append(f"{k} = ?")
                values.append(data[k])

        if not fields:
            return False

        fields.append("updated_at = ?")
        values.append(datetime.now(UTC).isoformat())
        values.append(trade_id)

        with self._get_conn() as conn:
            conn.execute(
                f"UPDATE signal_trades SET {', '.join(fields)} WHERE id = ?",  # nosec B608
                values,
            )
            conn.commit()
        return True
