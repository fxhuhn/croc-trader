"""
SQLite repository implementation for trading signals and trade tracking.

This module provides a SQLite-based implementation of the SignalRepository protocol,
handling persistence for trading signals, trade tracking, and signal statistics.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generator

from app.core.protocols import SignalRepository


class SQLiteRepository(SignalRepository):
    """
    SQLite-based repository for trading signals and trades.

    This repository manages three main tables:
    - signals: Raw trading signals with technical indicators
    - signal_trades: Trade tracking and lifecycle management
    - signal_statistic: Historical performance statistics for signal patterns

    Attributes:
        db_path: Path to the SQLite database file
    """

    # SQL queries as class constants for better maintainability
    _CREATE_SIGNALS_TABLE = """
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
    """

    _CREATE_SIGNALS_INDEX = """
        CREATE INDEX IF NOT EXISTS idx_symbol_timestamp_covering
        ON signals(symbol, timestamp DESC, signal)
    """

    _CREATE_TRADES_TABLE = """
        CREATE TABLE IF NOT EXISTS signal_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            signal TEXT NOT NULL,
            signal_timeframe TEXT,
            signal_timestamp TEXT,
            buy_limit REAL,
            stop_loss REAL,
            take_profit REAL,
            quantity INTEGER,
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
    """

    _CREATE_TRADES_INDEX = """
        CREATE INDEX IF NOT EXISTS idx_trades_state
        ON signal_trades(state, symbol)
    """

    _CREATE_STATISTICS_TABLE = """
        CREATE TABLE IF NOT EXISTS signal_statistic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            wolke TEXT,
            welle TEXT,
            trend TEXT,
            setter TEXT,
            tp_3r INTEGER NOT NULL,
            sl_1r INTEGER NOT NULL,
            rejected_0r INTEGER NOT NULL,
            level TEXT NOT NULL,
            total_signals INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """

    _CREATE_STATISTICS_INDEX = """
        CREATE INDEX IF NOT EXISTS idx_signal_statistic_key
        ON signal_statistic(signal, symbol, timeframe, wolke, welle, trend, setter)
    """

    # Allowed columns for dynamic queries
    _DISTINCT_COLUMNS = frozenset(["symbol", "timeframe", "signal"])

    # Allowed fields for trade updates
    _UPDATABLE_TRADE_FIELDS = frozenset(
        [
            "buy_limit",
            "stop_loss",
            "take_profit",
            "entry_time",
            "entry_price",
            "exit_time",
            "exit_price",
            "current_price",
            "state",
            "quantity",
        ]
    )

    def __init__(self, db_path: str | Path = "signals.db"):
        """
        Initialize the repository and create database schema.

        Args:
            db_path: Path to SQLite database file (created if doesn't exist)
        """
        self.db_path = str(db_path)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for database connections.

        Yields:
            Database connection with Row factory enabled

        Example:
            >>> with repo._get_connection() as conn:
            ...     cursor = conn.execute("SELECT * FROM signals")
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """
        Initialize database schema with all required tables and indexes.

        Creates:
            - signals table with covering index
            - signal_trades table with state index
            - signal_statistic table with composite key index
        """
        with self._get_connection() as conn:
            # Create signals table and index
            conn.execute(self._CREATE_SIGNALS_TABLE)
            conn.execute(self._CREATE_SIGNALS_INDEX)

            # Create trades table and index
            conn.execute(self._CREATE_TRADES_TABLE)
            conn.execute(self._CREATE_TRADES_INDEX)

            # Create statistics table and index
            conn.execute(self._CREATE_STATISTICS_TABLE)
            conn.execute(self._CREATE_STATISTICS_INDEX)

            conn.commit()

    def save_signal(self, data: dict[str, Any]) -> None:
        """
        Save a trading signal to the database.

        Args:
            data: Signal data dictionary with required fields:
                - symbol: Stock/instrument symbol
                - timeframe: Trading timeframe (e.g., "5m", "1h")
                - timestamp: Signal timestamp (datetime or ISO string)
                - signal: Signal type (e.g., "buy", "sell")
                - close, high, low: Price data
                - wuk: Wuk indicator value
                - status: Signal status
                - kerze: Candle pattern
                - trend: Market trend
                - setter: Signal setter indicator
                - welle: Wave indicator
                - wolke: Optional cloud indicator

        Example:
            >>> repo.save_signal({
            ...     "symbol": "AAPL",
            ...     "timeframe": "1h",
            ...     "timestamp": datetime.now(),
            ...     "signal": "buy",
            ...     "close": 150.0,
            ...     "high": 151.0,
            ...     "low": 149.0,
            ...     "wuk": 0.5,
            ...     "status": "active",
            ...     "kerze": "bullish",
            ...     "trend": "up",
            ...     "setter": "ema",
            ...     "welle": "wave1"
            ... })
        """
        # Convert timestamp to ISO format if datetime object
        timestamp = data["timestamp"]
        if hasattr(timestamp, "isoformat"):
            timestamp = timestamp.isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO signals (
                    symbol, timeframe, timestamp, signal, close, high, low, wuk,
                    status, kerze, wolke, trend, setter, welle
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["symbol"],
                    data["timeframe"],
                    timestamp,
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
                    data["welle"],
                ),
            )
            conn.commit()

    def get_signals(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
        signal: str | None = None,
        day: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """
        Retrieve signals with optional filtering and trade tracking information.

        Args:
            symbol: Filter by symbol (e.g., "AAPL")
            timeframe: Filter by timeframe (e.g., "1h")
            signal: Filter by signal type (e.g., "buy")
            day: Filter by date in YYYY-MM-DD format
            limit: Maximum number of results (default: 500)

        Returns:
            List of signal dictionaries with trade tracking info joined

        Example:
            >>> signals = repo.get_signals(symbol="AAPL", limit=10)
            >>> for sig in signals:
            ...     print(f"{sig['symbol']} - {sig['signal']}")
        """
        query = """
            SELECT s.*,
                   t.id as trade_id,
                   t.state as trade_state,
                   t.quantity,
                   t.buy_limit, t.stop_loss, t.take_profit,
                   t.entry_time, t.entry_price,
                   t.exit_time, t.exit_price,
                   t.current_price,
                   CASE WHEN t.id IS NOT NULL THEN 1 ELSE 0 END as is_tracked
            FROM signals s
            LEFT JOIN signal_trades t
              ON s.symbol = t.symbol
              AND s.timestamp = t.timestamp
              AND s.signal = t.signal
            WHERE 1=1
        """
        params: list[Any] = []

        # Build dynamic WHERE clause
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

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_distinct(self, column: str) -> list[str]:
        """
        Get distinct values for a specific column.

        Args:
            column: Column name (must be one of: symbol, timeframe, signal)

        Returns:
            Sorted list of distinct values, excluding None

        Example:
            >>> symbols = repo.get_distinct("symbol")
            >>> print(symbols)  # ['AAPL', 'GOOGL', 'MSFT']
        """
        if column not in self._DISTINCT_COLUMNS:
            return []

        # Safe to use f-string here since column is validated against whitelist
        query = f"SELECT DISTINCT {column} FROM signals ORDER BY {column}"

        with self._get_connection() as conn:
            rows = conn.execute(query).fetchall()
            return [row[0] for row in rows if row[0] is not None]

    def toggle_trade_tracking(
        self,
        symbol: str,
        timestamp: str,
        signal: str,
        timeframe: str | None = None,
        signal_timestamp: str | None = None,
    ) -> dict:
        """
        Toggle trade tracking for a signal (mark/unmark for monitoring).

        If the trade is already tracked, removes it. If not tracked, adds it.

        Args:
            symbol: Instrument symbol
            timestamp: Signal timestamp
            signal: Signal type
            timeframe: Optional timeframe for reference
            signal_timestamp: Optional original signal timestamp

        Returns:
            Dictionary with keys:
                - tracked: Boolean indicating new state
                - trade_id: ID of the trade record

        Example:
            >>> result = repo.toggle_trade_tracking("AAPL", "2025-01-01T10:00:00", "buy")
            >>> print(result)  # {'tracked': True, 'trade_id': 42}
        """
        with self._get_connection() as conn:
            # Check if trade already exists
            existing = conn.execute(
                """
                SELECT id FROM signal_trades
                WHERE symbol = ? AND timestamp = ? AND signal = ?
                """,
                (symbol, timestamp, signal),
            ).fetchone()

            if existing:
                # Remove tracking
                conn.execute(
                    "DELETE FROM signal_trades WHERE id = ?",
                    (existing["id"],),
                )
                conn.commit()
                return {"tracked": False, "trade_id": existing["id"]}

            # Add tracking
            cursor = conn.execute(
                """
                INSERT INTO signal_trades (
                    symbol, timestamp, signal, signal_timeframe, signal_timestamp
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (symbol, timestamp, signal, timeframe, signal_timestamp or timestamp),
            )
            conn.commit()
            return {"tracked": True, "trade_id": cursor.lastrowid}

    def get_trade(self, trade_id: int) -> dict | None:
        """
        Retrieve a specific trade by ID.

        Args:
            trade_id: Trade record ID

        Returns:
            Trade dictionary or None if not found

        Example:
            >>> trade = repo.get_trade(42)
            >>> if trade:
            ...     print(f"State: {trade['state']}")
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM signal_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_all_trades(self, limit: int = 500) -> list[dict]:
        """
        Retrieve all tracked trades, newest first.

        Args:
            limit: Maximum number of trades to return (default: 500)

        Returns:
            List of trade dictionaries ordered by creation time (descending)

        Example:
            >>> trades = repo.get_all_trades(limit=100)
            >>> for trade in trades:
            ...     print(f"{trade['symbol']} - {trade['state']}")
        """
        query = """
            SELECT id, symbol, signal, state,quantity,
                   buy_limit, stop_loss, take_profit,
                   entry_time, entry_price,
                   exit_time, exit_price,
                   current_price,
                   signal_timeframe, signal_timestamp
            FROM signal_trades
            ORDER BY created_at DESC
            LIMIT ?
        """

        with self._get_connection() as conn:
            rows = conn.execute(query, (limit,)).fetchall()
            return [dict(row) for row in rows]

    def update_trade(self, trade_id: int, data: dict[str, Any]) -> bool:
        """
        Update a tracked trade with new information.

        Args:
            trade_id: Trade record ID
            data: Dictionary of fields to update (only allowed fields are used)

        Returns:
            True if any fields were updated, False otherwise

        Example:
            >>> repo.update_trade(42, {
            ...     "state": "filled",
            ...     "entry_price": 150.5,
            ...     "entry_time": "2025-01-01T10:30:00"
            ... })
        """
        # Filter to only allowed fields
        fields_to_update = []
        values: list[Any] = []

        for field in self._UPDATABLE_TRADE_FIELDS:
            if field in data:
                fields_to_update.append(f"{field} = ?")
                values.append(data[field])

        if not fields_to_update:
            return False

        # Add updated_at timestamp
        fields_to_update.append("updated_at = ?")
        values.append(datetime.now(UTC).isoformat())

        # Add trade_id for WHERE clause
        values.append(trade_id)

        query = f"UPDATE signal_trades SET {', '.join(fields_to_update)} WHERE id = ?"

        with self._get_connection() as conn:
            conn.execute(query, values)
            conn.commit()

        return True

    def replace_signal_statistics(self, rows: list[dict[str, Any]]) -> None:
        """
        Replace all signal statistics with new data (bulk operation).

        This performs a DELETE followed by INSERT, useful for periodic
        recalculation of statistics.

        Args:
            rows: List of statistic dictionaries with keys:
                - signal, symbol, timeframe
                - wolke, welle, trend, setter (optional)
                - TP(3R), SL(-1R), Rejected(0R): outcome counts
                - level: Quality level indicator

        Example:
            >>> stats = [
            ...     {
            ...         "signal": "buy",
            ...         "symbol": "AAPL",
            ...         "timeframe": "1h",
            ...         "wolke": "green",
            ...         "TP(3R)": 10,
            ...         "SL(-1R)": 2,
            ...         "Rejected(0R)": 1,
            ...         "level": "A"
            ...     }
            ... ]
            >>> repo.replace_signal_statistics(stats)
        """
        with self._get_connection() as conn:
            # Clear existing statistics
            conn.execute("DELETE FROM signal_statistic")

            # Bulk insert new statistics
            conn.executemany(
                """
                INSERT INTO signal_statistic (
                    signal, symbol, timeframe, wolke, welle, trend, setter,
                    tp_3r, sl_1r, rejected_0r, level, total_signals
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row["signal"],
                        row["symbol"],
                        row["timeframe"],
                        row.get("wolke"),
                        row.get("welle"),
                        row.get("trend"),
                        row.get("setter"),
                        row["TP(3R)"],
                        row["SL(-1R)"],
                        row["Rejected(0R)"],
                        row["level"],
                        row["TP(3R)"] + row["SL(-1R)"] + row["Rejected(0R)"],
                    )
                    for row in rows
                ],
            )
            conn.commit()

    def get_best_stat_for_signal(
        self,
        signal: str,
        symbol: str,
        timeframe: str,
        wolke: str | None,
        welle: str | None,
        trend: str | None,
        setter: str | None,
    ) -> dict | None:
        """
        Find best matching statistic using hierarchical matching strategy.

        Searches for statistics with progressively less specific criteria,
        returning the most specific match with at least 3 total signals.

        Matching hierarchy (most to least specific):
        1. Exact match (all parameters)
        2. Wolke + Welle + Trend
        3. Wolke + Welle
        4. Wolke only
        5. Signal + Symbol + Timeframe only

        Args:
            signal: Signal type
            symbol: Instrument symbol
            timeframe: Trading timeframe
            wolke: Cloud indicator (optional)
            welle: Wave indicator (optional)
            trend: Trend indicator (optional)
            setter: Setter indicator (optional)

        Returns:
            Best matching statistic dict with 'match_quality' key, or None

        Example:
            >>> stat = repo.get_best_stat_for_signal(
            ...     signal="buy",
            ...     symbol="AAPL",
            ...     timeframe="1h",
            ...     wolke="green",
            ...     welle="up",
            ...     trend="bullish",
            ...     setter="ema"
            ... )
            >>> if stat:
            ...     print(f"Match: {stat['match_quality']}")
        """

        def _query_stat(
            conn: sqlite3.Connection,
            where_clause: str,
            params: tuple,
            quality_label: str,
        ) -> dict | None:
            """Helper to query statistics with quality labeling."""
            query = f"""
                SELECT * FROM signal_statistic
                WHERE {where_clause}
                AND total_signals >= 3
                ORDER BY total_signals DESC
                LIMIT 1
            """
            row = conn.execute(query, params).fetchone()
            if row:
                result = dict(row)
                result["match_quality"] = quality_label
                return result
            return None

        with self._get_connection() as conn:
            # Try most specific match first
            result = _query_stat(
                conn,
                "signal=? AND symbol=? AND timeframe=? "
                "AND wolke=? AND welle=? AND trend=? AND setter=?",
                (signal, symbol, timeframe, wolke, welle, trend, setter),
                "Exakte Übereinstimmung (Wolke, Welle, Trend, Setter)",
            )
            if result:
                return result

            # Try without setter
            result = _query_stat(
                conn,
                "signal=? AND symbol=? AND timeframe=? "
                "AND wolke=? AND welle=? AND trend=?",
                (signal, symbol, timeframe, wolke, welle, trend),
                "Wolke + Welle + Trend",
            )
            if result:
                return result

            # Try wolke + welle only
            result = _query_stat(
                conn,
                "signal=? AND symbol=? AND timeframe=? AND wolke=? AND welle=?",
                (signal, symbol, timeframe, wolke, welle),
                "Wolke + Welle",
            )
            if result:
                return result

            # Try wolke only
            result = _query_stat(
                conn,
                "signal=? AND symbol=? AND timeframe=? AND wolke=?",
                (signal, symbol, timeframe, wolke),
                "nur Wolke",
            )
            if result:
                return result

            # Fall back to basic match
            return _query_stat(
                conn,
                "signal=? AND symbol=? AND timeframe=?",
                (signal, symbol, timeframe),
                "nur Signal + Symbol + Timeframe",
            )

    def enrich_signals_with_stats(self, signals: list[dict]) -> list[dict]:
        """
        Enrich signal list with historical performance statistics.

        For each signal, finds best matching statistics and adds performance
        metrics to the signal dictionary.

        Args:
            signals: List of signal dictionaries

        Returns:
            Same list with added fields:
                - tp_3r: Take profit count
                - sl_1r: Stop loss count
                - rej_0r: Rejected count
                - stats_total: Total historical signals
                - match_quality: Quality of statistical match

        Example:
            >>> signals = repo.get_signals(symbol="AAPL")
            >>> enriched = repo.enrich_signals_with_stats(signals)
            >>> for sig in enriched:
            ...     if sig['stats_total'] > 0:
            ...         print(f"{sig['symbol']}: {sig['tp_3r']}/{sig['stats_total']}")
        """
        enriched = []

        for signal in signals:
            stat = self.get_best_stat_for_signal(
                signal=signal.get("signal"),
                symbol=signal.get("symbol"),
                timeframe=signal.get("timeframe"),
                wolke=signal.get("wolke"),
                welle=signal.get("welle"),
                trend=signal.get("trend"),
                setter=signal.get("setter"),
            )

            if stat:
                signal["tp_3r"] = stat["tp_3r"]
                signal["sl_1r"] = stat["sl_1r"]
                signal["rej_0r"] = stat["rejected_0r"]
                signal["stats_total"] = stat["total_signals"]
                signal["match_quality"] = stat.get("match_quality", "")
            else:
                signal["tp_3r"] = None
                signal["sl_1r"] = None
                signal["rej_0r"] = None
                signal["stats_total"] = 0
                signal["match_quality"] = ""

            enriched.append(signal)

        return enriched
