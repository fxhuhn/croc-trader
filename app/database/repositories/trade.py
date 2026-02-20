import json
import logging
from enum import Enum
from typing import Any

from .base import BaseRepository

logger = logging.getLogger(__name__)


class TradeRepository(BaseRepository):
    def init_schema(self):
        """Erstellt das DB-Schema neu (Unified Table)."""
        with self.session.connect() as conn:
            # self.execute("DROP TABLE IF EXISTS active_trades", conn=conn)
            # self.execute("DROP TABLE IF EXISTS trades_croc", conn=conn)
            # self.execute("DROP TABLE IF EXISTS trades_dip_buyer", conn=conn)
            # self.execute("DROP TABLE IF EXISTS trades", conn=conn)
            # self.execute("DROP TABLE IF EXISTS trade_logs", conn=conn)

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
                conn=conn,
            )

            self.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)",
                conn=conn,
            )
            self.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
                conn=conn,
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
                conn=conn,
            )

    def clear_trades(self):
        """Truncates the trades and trade_logs tables to restart a backtest run."""
        with self.session.connect() as conn:
            self.execute("DELETE FROM trade_logs", conn=conn)
            self.execute("DELETE FROM trades", conn=conn)
            conn.commit()
            logger.info("Trade history cleared for new backtest run.")

    def get_trade(self, trade_id: int, conn=None) -> dict[str, Any] | None:
        row = self.fetch_one(
            "SELECT * FROM trades WHERE id = ?", (trade_id,), conn=conn
        )
        return dict(row) if row else None

    def get_active_trades(self) -> list[dict[str, Any]]:
        """Legacy Helper: Holt CREATED und ACTIVE."""
        rows = self.fetch_all(
            "SELECT * FROM trades WHERE status IN ('CREATED', 'ACTIVE')"
        )
        return [dict(r) for r in rows]

    def get_by_status(
        self, status: str | Enum | list[str | Enum]
    ) -> list[dict[str, Any]]:
        """
        Holt Trades basierend auf Status. Akzeptiert Enums oder Strings.
        """
        if isinstance(status, list):
            # Enums in Strings wandeln
            statuses = [s.value if isinstance(s, Enum) else s for s in status]
            placeholders = ",".join("?" for _ in statuses)
            sql = f"SELECT * FROM trades WHERE status IN ({placeholders})"
            rows = self.fetch_all(sql, tuple(statuses))
        else:
            s_val = status.value if isinstance(status, Enum) else status
            rows = self.fetch_all("SELECT * FROM trades WHERE status = ?", (s_val,))

        return [dict(r) for r in rows]

    def get_all_traded_symbols(self) -> list[str]:
        """Returns a list of all distinct symbols ever traded."""
        rows = self.fetch_all("SELECT DISTINCT symbol FROM trades")
        return [r["symbol"] for r in rows if r["symbol"]]

    def get_all_by_strategy(self, strategy: str | Enum) -> list[dict[str, Any]]:
        """Holt alle Trades für eine bestimmte Strategie."""
        strategy_value = strategy.value if isinstance(strategy, Enum) else strategy
        rows = self.fetch_all(
            "SELECT * FROM trades WHERE strategy = ?", (strategy_value,)
        )
        results = []
        for row in rows:
            trade_dict = dict(row)
            if trade_dict.get("signal_context"):
                try:
                    trade_dict["signal_context"] = json.loads(
                        trade_dict["signal_context"]
                    )
                except (json.JSONDecodeError, TypeError):
                    trade_dict["signal_context"] = {}
            results.append(trade_dict)
        return results

    def create_trade(
        self,
        symbol: str,
        strategy: str,
        size: float,
        entry: float,
        stop_loss: float,
        target: float,
        context: dict,
    ) -> int:
        # Security Hardening: Validate financial inputs
        import math

        for val, name in [
            (entry, "entry"),
            (stop_loss, "stop_loss"),
            (target, "target"),
        ]:
            if not math.isfinite(val) or val < 0:
                logger.error(
                    f"❌ SECURITY: Invalid financial input for {symbol}: {name}={val}"
                )
                raise ValueError(
                    f"Value for {name} must be a finite non-negative number"
                )

        context_json = json.dumps(context, default=str)
        quantity = int(size)

        # 1. Datum extrahieren für den Eindeutigkeits-Check
        signal_date = context.get("date")

        with self.session.connect() as conn:
            trade_id = None

            # 2. Prüfen: Gibt es diesen Trade schon? (Symbol + Strategy + Datum im JSON)
            if signal_date:
                check_sql = """
                    SELECT id, status FROM trades 
                    WHERE symbol = ? 
                    AND strategy = ? 
                    AND json_extract(signal_context, '$.date') = ?
                """
                existing = conn.execute(
                    check_sql, (symbol, strategy, signal_date)
                ).fetchone()

                if existing:
                    trade_id, current_status = existing

                    # If the trade is already ACTIVE, we don't want to reset it!
                    if current_status == "ACTIVE":
                        logger.debug(
                            f"Trade {trade_id} for {symbol} is already ACTIVE. Skipping reset."
                        )
                        return trade_id

                    # UPDATE existierenden Trade (Reset auf Startbedingungen)
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
                    conn.execute(
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

                    conn.execute(
                        "INSERT INTO trade_logs (trade_id, event_type, old_value, new_value, reason) VALUES (?, ?, ?, ?, ?)",
                        (
                            trade_id,
                            "RESET",
                            current_status,
                            "CREATED",
                            "Strategy Re-Sync",
                        ),
                    )

                    conn.commit()
                    return trade_id

            # 3. INSERT (falls neu)
            sql = """
                INSERT INTO trades (
                    symbol, strategy, status,
                    initial_size, current_size,
                    entry_price, current_stop_loss, current_target,
                    signal_context
                ) VALUES (?, ?, 'CREATED', ?, ?, ?, ?, ?, ?)
            """

            cursor = conn.execute(
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

            conn.execute(
                "INSERT INTO trade_logs (trade_id, event_type, old_value, new_value, reason) VALUES (?, ?, ?, ?, ?)",
                (
                    trade_id,
                    "ENTRY",
                    None,
                    f"Quantity: {quantity} @ {entry}",
                    "Setup Found",
                ),
            )

            conn.commit()
            return trade_id

    def update_trade(self, trade_id: int, updates: dict, reason: str = None):
        """Generisches Update."""
        if not updates:
            return

        # Enums in updates behandeln
        safe_updates = {}
        for k, v in updates.items():
            # Enum conversion
            val = v.value if isinstance(v, Enum) else v

            # Robust Date Normalization for entry_date and exit_date
            # Ensures "YYYY-MM-DD" instead of "YYYY-MM-DD HH:MM:SS"
            if k in ("entry_date", "exit_date"):
                if isinstance(val, str) and " " in val:
                    val = val.split(" ")[0]
                elif hasattr(val, "strftime"):
                    val = val.strftime("%Y-%m-%d")

            safe_updates[k] = val

        with self.session.connect() as conn:
            trade = self.get_trade(trade_id, conn=conn)
            if not trade:
                return

            set_clauses = []
            values = []
            changes = []

            for key, new_val in safe_updates.items():
                old_val = trade.get(key)
                # Vergleich als String um Typ-Probleme zu vermeiden
                if str(old_val) != str(new_val):
                    set_clauses.append(f"{key} = ?")
                    values.append(new_val)
                    changes.append((key, old_val, new_val))

            if not set_clauses:
                return

            values.append(trade_id)
            sql = f"UPDATE trades SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
            self.execute(sql, values, conn=conn)

            for key, old, new in changes:
                self._log_event_conn(
                    conn, trade_id, f"UPDATE_{key.upper()}", old, new, reason
                )

    def _log_event(self, trade_id, event, old, new, reason):
        self.execute(
            "INSERT INTO trade_logs (trade_id, event_type, old_value, new_value, reason) VALUES (?, ?, ?, ?, ?)",
            (trade_id, str(event), str(old), str(new), reason),
        )

    def _log_event_conn(self, conn, trade_id, event, old, new, reason):
        self.execute(
            "INSERT INTO trade_logs (trade_id, event_type, old_value, new_value, reason) VALUES (?, ?, ?, ?, ?)",
            (trade_id, str(event), str(old), str(new), reason),
            conn=conn,
        )

    # --- NEU: Die fehlende Methode für TurnoverTimingStrategy ---
    def exists(self, symbol: str, strategy: str, date: str) -> bool:
        """
        Prüft, ob ein Trade existiert, indem im signal_context nach dem Datum gesucht wird.
        """
        # Wir suchen das Datum sauber als JSON extract.
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
