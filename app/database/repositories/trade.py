import json
import logging
import sqlite3
from enum import Enum
from typing import Any

from .base import BaseRepository

logger = logging.getLogger(__name__)


class TradeRepository(BaseRepository):
    VALID_COLUMNS = {
        "symbol",
        "strategy",
        "status",
        "initial_size",
        "current_size",
        "entry_price",
        "entry_date",
        "current_price",
        "current_stop_loss",
        "current_target",
        "avg_exit_price",
        "realized_pnl",
        "exit_price",
        "exit_date",
        "exit_reason",
        "signal_context",
        "created_at",
        "updated_at",
    }

    def init_schema(self) -> None:
        """Recreates the DB schema (Unified Table)."""
        with self.session.connect() as connection:
            # self.execute("DROP TABLE IF EXISTS active_trades", connection=connection)
            # self.execute("DROP TABLE IF EXISTS trades_croc", connection=connection)
            # self.execute("DROP TABLE IF EXISTS trades_dip_buyer", connection=connection)
            # self.execute("DROP TABLE IF EXISTS trades", connection=connection)
            # self.execute("DROP TABLE IF EXISTS trade_logs", connection=connection)

            self.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    status TEXT DEFAULT 'CREATED',
                    
                    -- Size Management
                    initial_size REAL DEFAULT 0,
                    current_size REAL DEFAULT 0,
                    
                    -- Preise & Limits
                    entry_price REAL,
                    entry_date TIMESTAMP,
                    current_price REAL,
                    current_stop_loss REAL,
                    current_target REAL,
                    
                    -- Performance
                    avg_exit_price REAL,
                    realized_pnl REAL DEFAULT 0,
                    
                    -- Exit Details
                    exit_price REAL,
                    exit_date TIMESTAMP,
                    exit_reason TEXT,
                    
                    -- Meta Infos
                    
                    -- Context (JSON)
                    signal_context TEXT,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
                connection=connection,
            )

            self.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)",
                connection=connection,
            )
            self.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
                connection=connection,
            )

            self.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER,
                    event_type TEXT,
                    old_value TEXT,
                    new_value TEXT,
                    reason TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(trade_id) REFERENCES trades(id)
                )
            """,
                connection=connection,
            )

    def clear_trades(self) -> None:
        """Truncates the trades and trade_logs tables to restart a backtest run."""
        with self.session.connect() as connection:
            self.execute("DELETE FROM trade_logs", connection=connection)
            self.execute("DELETE FROM trades", connection=connection)
            connection.commit()
            logger.info("Trade history cleared for new backtest run.")

    def get_trade(
        self,
        trade_id: int,
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, Any] | None:
        row = self.fetch_one(
            "SELECT * FROM trades WHERE id = ?", (trade_id,), connection=connection
        )
        return dict(row) if row else None

    def get_active_trades(self) -> list[dict[str, object]]:
        """Legacy Helper: Fetches CREATED and ACTIVE."""
        rows = self.fetch_all(
            "SELECT * FROM trades WHERE status IN ('CREATED', 'ACTIVE')"
        )
        return [dict(row) for row in rows]

    def get_by_status(
        self, status: str | Enum | list[str | Enum]
    ) -> list[dict[str, object]]:
        """Fetches trades based on status. Accepts enums or strings."""
        if isinstance(status, list):
            # Convert enums to strings
            statuses = [s.value if isinstance(s, Enum) else s for s in status]
            placeholders = ",".join("?" for _ in statuses)
            sql = f"SELECT * FROM trades WHERE status IN ({placeholders})"
            rows = self.fetch_all(sql, tuple(statuses))
        else:
            status_value = status.value if isinstance(status, Enum) else status
            rows = self.fetch_all(
                "SELECT * FROM trades WHERE status = ?", (status_value,)
            )

        return [dict(row) for row in rows]

    def get_all_traded_symbols(self) -> list[str]:
        """Returns a list of all distinct symbols ever traded."""
        rows = self.fetch_all("SELECT DISTINCT symbol FROM trades")
        return [row["symbol"] for row in rows if row["symbol"]]

    def get_all_by_strategy(self, strategy: str | Enum) -> list[dict[str, object]]:
        """Fetches all trades for a specific strategy."""
        strategy_value = strategy.value if isinstance(strategy, Enum) else strategy
        rows = self.fetch_all(
            "SELECT * FROM trades WHERE strategy = ?", (strategy_value,)
        )
        results = []
        for row in rows:
            trade_dict = dict(row)
            signal_context_raw = trade_dict.get("signal_context")
            if signal_context_raw:
                try:
                    trade_dict["signal_context"] = json.loads(str(signal_context_raw))
                except (json.JSONDecodeError, TypeError) as parse_error:
                    trade_dict["signal_context"] = {}
                    logger.warning(
                        "Failed to decode signal_context for strategy trade: %s",
                        parse_error,
                    )
            results.append(trade_dict)
        return results

    def _validate_financial_inputs(
        self, symbol: str, entry: float, stop_loss: float, target: float
    ) -> None:
        """Security Hardening: Validate financial inputs."""
        import math

        for value, name in [
            (entry, "entry"),
            (stop_loss, "stop_loss"),
            (target, "target"),
        ]:
            if not math.isfinite(value) or value < 0:
                logger.error(
                    "❌ SECURITY: Invalid financial input for %s: %s=%f",
                    symbol,
                    name,
                    value,
                )
                raise ValueError(
                    f"Value for {name} must be a finite non-negative number"
                )

    def _fetch_existing_trade_info(
        self,
        connection: sqlite3.Connection,
        symbol: str,
        strategy: str,
        signal_date: str,
    ) -> tuple[int, str] | None:
        """Fetches existing trade ID and status based on context date."""
        check_sql = """
            SELECT id, status FROM trades 
            WHERE symbol = ? 
            AND strategy = ? 
            AND json_extract(signal_context, '$.date') = ?
        """
        row = connection.execute(check_sql, (symbol, strategy, signal_date)).fetchone()
        return (row[0], row[1]) if row else None

    def _reset_existing_trade(
        self,
        connection: sqlite3.Connection,
        trade_id: int,
        quantity: int,
        entry: float,
        stop_loss: float,
        target: float,
        context_json: str,
        current_status: str,
    ) -> None:
        """Resets a non-active trade candidate's fields."""
        update_sql = """
            UPDATE trades SET 
                status = 'CREATED',
                initial_size = ?, current_size = ?,
                entry_price = ?, current_stop_loss = ?, current_target = ?,
                signal_context = ?,
                exit_price = NULL, exit_date = NULL, exit_reason = NULL,
                entry_date = NULL, realized_pnl = 0,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """
        connection.execute(
            update_sql,
            (
                quantity,
                quantity,
                entry,
                stop_loss,
                target,
                context_json,
                trade_id,
            ),
        )

        connection.execute(
            "INSERT INTO trade_logs (trade_id, event_type, old_value, new_value, reason) VALUES (?, ?, ?, ?, ?)",
            (
                trade_id,
                "RESET",
                current_status,
                "CREATED",
                "Strategy Re-Sync",
            ),
        )

    def _insert_new_trade(
        self,
        connection: sqlite3.Connection,
        symbol: str,
        strategy: str,
        quantity: int,
        entry: float,
        stop_loss: float,
        target: float,
        context_json: str,
    ) -> int:
        """Inserts a new trade candidate into the trades table."""
        sql = """
            INSERT INTO trades (
                symbol, strategy, status,
                initial_size, current_size,
                entry_price, current_stop_loss, current_target,
                signal_context
            ) VALUES (?, ?, 'CREATED', ?, ?, ?, ?, ?, ?)
        """
        cursor = connection.execute(
            sql,
            (
                symbol,
                strategy,
                quantity,
                quantity,
                entry,
                stop_loss,
                target,
                context_json,
            ),
        )
        trade_id = cursor.lastrowid

        connection.execute(
            "INSERT INTO trade_logs (trade_id, event_type, old_value, new_value, reason) VALUES (?, ?, ?, ?, ?)",
            (
                trade_id,
                "ENTRY",
                None,
                f"Quantity: {quantity} @ {entry}",
                "Setup Found",
            ),
        )
        return trade_id

    def create_trade(
        self,
        symbol: str,
        strategy: str,
        size: float,
        entry: float,
        stop_loss: float,
        target: float,
        context: dict[str, object],
    ) -> int:
        self._validate_financial_inputs(symbol, entry, stop_loss, target)

        context_json = json.dumps(context, default=str, ensure_ascii=False)
        quantity = int(size)
        signal_date = context.get("date")

        with self.session.connect() as connection:
            if signal_date:
                existing_trade = self._fetch_existing_trade_info(
                    connection, symbol, strategy, str(signal_date)
                )

                if existing_trade:
                    trade_id, current_status = existing_trade

                    # If the trade is already ACTIVE, we don't want to reset it!
                    if current_status == "ACTIVE":
                        logger.debug(
                            "Trade %d for %s is already ACTIVE. Skipping reset.",
                            trade_id,
                            symbol,
                        )
                        return trade_id

                    # UPDATE existing trade (reset to starting conditions)
                    self._reset_existing_trade(
                        connection,
                        trade_id,
                        quantity,
                        entry,
                        stop_loss,
                        target,
                        context_json,
                        current_status,
                    )
                    return trade_id

            return self._insert_new_trade(
                connection,
                symbol,
                strategy,
                quantity,
                entry,
                stop_loss,
                target,
                context_json,
            )

    def update_trade(
        self,
        trade_id: int,
        updates: dict[str, object],
        reason: str | None = None,
    ) -> None:
        """Generic update."""
        if not updates:
            return

        # Enums in updates behandeln
        safe_updates = {}
        for key, value in updates.items():
            if key not in self.VALID_COLUMNS:
                logger.error("❌ SECURITY: Attempted update to invalid column: %s", key)
                raise ValueError(f"Invalid column: {key}")

            # Enum conversion
            safe_value = value.value if isinstance(value, Enum) else value

            # Robust Date Normalization for entry_date and exit_date
            # Ensures "YYYY-MM-DD" instead of "YYYY-MM-DD HH:MM:SS"
            if key in ("entry_date", "exit_date"):
                if isinstance(safe_value, str) and " " in safe_value:
                    safe_value = safe_value.split(" ")[0]
                elif hasattr(safe_value, "strftime"):
                    safe_value = safe_value.strftime("%Y-%m-%d")

            safe_updates[key] = safe_value

        with self.session.connect() as connection:
            trade = self.get_trade(trade_id, connection=connection)
            if not trade:
                return

            set_clauses = []
            values = []
            changes = []

            for key, new_value in safe_updates.items():
                old_value = trade.get(key)
                # Compare as string to avoid type issues
                if str(old_value) != str(new_value):
                    set_clauses.append(f"{key} = ?")
                    values.append(new_value)
                    changes.append((key, old_value, new_value))

            if not set_clauses:
                return

            values.append(trade_id)
            sql = f"UPDATE trades SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
            self.execute(sql, values, connection=connection)

            for key, old_value, new_value in changes:
                self._log_event_conn(
                    connection,
                    trade_id,
                    f"UPDATE_{key.upper()}",
                    old_value,
                    new_value,
                    reason,
                )

    def _log_event(
        self,
        trade_id: int,
        event_type: object,
        old_value: object,
        new_value: object,
        reason: str | None,
    ) -> None:
        self.execute(
            "INSERT INTO trade_logs (trade_id, event_type, old_value, new_value, reason) VALUES (?, ?, ?, ?, ?)",
            (trade_id, str(event_type), str(old_value), str(new_value), reason),
        )

    def _log_event_conn(
        self,
        connection: sqlite3.Connection,
        trade_id: int,
        event_type: object,
        old_value: object,
        new_value: object,
        reason: str | None,
    ) -> None:
        self.execute(
            "INSERT INTO trade_logs (trade_id, event_type, old_value, new_value, reason) VALUES (?, ?, ?, ?, ?)",
            (trade_id, str(event_type), str(old_value), str(new_value), reason),
            connection=connection,
        )

    # --- missing method for TurnoverTimingStrategy ---
    def exists(self, symbol: str, strategy: str, date: str) -> bool:
        """
        Checks if a trade exists by looking up the date in signal_context.
        """
        # Find the date cleanly as a JSON extract.
        wildcard_strat = f"{strategy}%"

        sql = """
            SELECT 1 FROM trades 
            WHERE symbol = ? 
            AND strategy LIKE ? 
            AND (json_extract(signal_context, '$.date') = ? OR json_extract(signal_context, '$.setup_date') = ?)
            LIMIT 1
        """
        row = self.fetch_one(sql, (symbol, wildcard_strat, date, date))
        return row is not None
