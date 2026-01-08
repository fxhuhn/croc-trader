import logging
import sqlite3
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class StrategyDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout = 3000;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Erstellt die Tabelle und fügt bei Bedarf Spalten hinzu."""
        schema = """
        CREATE TABLE IF NOT EXISTS strategy_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            symbol TEXT,
            strategy TEXT,
            timeframe TEXT,
            limit_stp REAL,     -- NEU: Stop Buy (für Breakouts)
            limit_lmt REAL,     -- Limit Buy (für Dips)
            stop_loss REAL,
            take_profit REAL,
            qty INTEGER,
            status TEXT DEFAULT 'PENDING',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, symbol, strategy)
        );
        """
        try:
            with self._get_conn() as conn:
                conn.executescript(schema)

                # MIGRATION: Falls die Tabelle schon existiert, prüfen wir auf fehlende Spalten
                try:
                    conn.execute(
                        "ALTER TABLE strategy_trades ADD COLUMN limit_stp REAL"
                    )
                    logger.info("Migration: Spalte 'limit_stp' hinzugefügt.")
                except sqlite3.OperationalError:
                    pass  # Existiert schon

                try:
                    conn.execute(
                        "ALTER TABLE strategy_trades ADD COLUMN status TEXT DEFAULT 'PENDING'"
                    )
                except sqlite3.OperationalError:
                    pass  # Existiert schon

                conn.commit()
        except Exception as e:
            logger.error(f"Fehler DB Init (Strategies): {e}")

    def save_trade(self, trade_data: Dict):
        """Speichert einen neuen Trade-Vorschlag."""
        # Wir nutzen INSERT OR REPLACE, damit Updates (z.B. geänderter Preis am selben Tag) möglich sind
        # Oder INSERT OR IGNORE, wenn der erste Wurf zählt. Hier nehmen wir REPLACE für Updates.
        sql = """
        INSERT OR REPLACE INTO strategy_trades
        (date, symbol, strategy, timeframe, limit_stp, limit_lmt, stop_loss, take_profit, qty, status)
        VALUES (:date, :symbol, :strategy, :timeframe, :limit_stp, :limit_lmt, :stop_loss, :take_profit, :qty, :status)
        """

        # Sicherstellen, dass fehlende Werte NULL (None) sind
        if "limit_stp" not in trade_data:
            trade_data["limit_stp"] = None
        if "limit_lmt" not in trade_data:
            trade_data["limit_lmt"] = None
        if "status" not in trade_data:
            trade_data["status"] = "PENDING"

        try:
            with self._get_conn() as conn:
                conn.execute(sql, trade_data)
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Speichern des Strategy-Trades: {e}")

    def get_latest_trades(self, limit=50):
        import pandas as pd

        try:
            with self._get_conn() as conn:
                df = pd.read_sql_query(
                    f"SELECT * FROM strategy_trades ORDER BY date DESC, created_at DESC LIMIT {limit}",
                    conn,
                )
            return df
        except Exception as e:
            logger.error(f"Fehler beim Lesen der Trades: {e}")
            return pd.DataFrame()

    def update_status(self, trade_id: int, new_status: str):
        sql = "UPDATE strategy_trades SET status = ? WHERE id = ?"
        with self._get_conn() as conn:
            conn.execute(sql, (new_status, trade_id))
            conn.commit()
