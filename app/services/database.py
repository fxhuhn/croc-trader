import logging
import sqlite3
import time
from pathlib import Path
from typing import List

from ..models import CrocSignal, SignalStat

logger = logging.getLogger(__name__)


class SignalDatabase:
    """
    Verantwortlich für SQLite Verbindungen, Schema-Erstellung,
    Migrationen und Datenzugriff.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        # WAL Mode & Timeout für hohe Concurrency
        conn.execute("PRAGMA busy_timeout = 3000;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def optimize(self):
        """Führt VACUUM durch, um Speicherplatz freizugeben."""
        try:
            with self._get_conn() as conn:
                conn.isolation_level = None
                conn.execute("VACUUM")
            logger.info("DB Wartung: VACUUM erfolgreich.")
        except sqlite3.Error as e:
            logger.error(f"Fehler bei DB-Optimierung: {e}")

    def _init_db(self) -> None:
        """Erstellt Tabellen, Indizes und Views."""

        # 1. Signals Tabelle (Mit UNIQUE constraint auf reference für Deduplizierung)
        schema_signals = """
        CREATE TABLE IF NOT EXISTS signals (
            symbol TEXT NOT NULL, timeframe TEXT NOT NULL, signal TEXT NOT NULL,
            timestamp TEXT NOT NULL, close REAL, high REAL, low REAL, wuk REAL,
            status TEXT, kerze TEXT, trend TEXT, setter TEXT, welle TEXT,
            wolke TEXT, rsi REAL, sma_200 REAL, sma_20 REAL,
            strategy_id TEXT, exchange TEXT,
            reference TEXT UNIQUE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_signals_filter ON signals(symbol, timeframe, signal, timestamp DESC);
        """

        # 2. Stats Tabelle (Mit Generated Columns)
        schema_stats = """
        CREATE TABLE IF NOT EXISTS signal_stats (
            signal TEXT, symbol TEXT, timeframe TEXT, exchange TEXT, level TEXT,
            total REAL, win REAL, loss REAL, rejected REAL,
            wolke TEXT, welle TEXT, trend TEXT, setter TEXT,
            updated_at TEXT,
            win_rate REAL GENERATED ALWAYS AS (CASE WHEN total > 0 THEN ROUND((win / total) * 100, 2) ELSE 0 END) VIRTUAL,
            loss_rate REAL GENERATED ALWAYS AS (CASE WHEN total > 0 THEN ROUND((loss / total) * 100, 2) ELSE 0 END) VIRTUAL
        );
        CREATE INDEX IF NOT EXISTS idx_stats_lookup_perf ON signal_stats(symbol, signal, timeframe, level);
        """

        # 3. View (Der komplexe Join-Block von vorhin)
        # (Ich kürze den String hier für die Übersicht ab, füge hier deinen vollen VIEW SQL Code ein)
        schema_view = """
        CREATE VIEW IF NOT EXISTS view_signals_enriched AS
        SELECT s.id, s.symbol, s.signal, s.timeframe, s.timestamp, s.close,
               s.trend, s.welle, s.wolke, s.setter, s.exchange,
               -- ... (Dein großer COALESCE Block) ...
               COALESCE(st1.win_rate, st2.win_rate, st5.win_rate, 0) as predicted_win_rate
        FROM signals s
        LEFT JOIN signal_stats st1 ON s.symbol = st1.symbol AND s.signal = st1.signal AND st1.level = 'wolke_welle_trend_setter'
        LEFT JOIN signal_stats st2 ON s.symbol = st2.symbol AND s.signal = st2.signal AND st2.level = 'wolke_welle_trend'
        LEFT JOIN signal_stats st5 ON s.symbol = st5.symbol AND s.signal = st5.signal AND st5.level = 'gesamt';
        """

        try:
            with self._get_conn() as conn:
                conn.executescript(schema_signals)
                conn.executescript(schema_stats)
                # View handling: Drop & Recreate um Updates sicherzustellen
                conn.execute("DROP VIEW IF EXISTS view_signals_enriched")
                # conn.execute(schema_view) # Hier den echten View SQL Code nutzen
                conn.commit()
        except sqlite3.Error as e:
            logger.critical(f"DB Init failed: {e}")
            raise

    def save_many(self, signals: List[CrocSignal], retries: int = 5) -> int:
        if not signals:
            return 0

        # INSERT OR IGNORE für Deduplizierung basierend auf 'reference'
        sql = """
        INSERT OR IGNORE INTO signals
        (symbol, timeframe, timestamp, signal, close, high, low, wuk,
         status, kerze, wolke, trend, setter, welle,
         rsi, sma_200, sma_20, strategy_id, reference, exchange)
        VALUES
        (:symbol, :timeframe, :timestamp, :signal, :close, :high, :low, :wuk,
         :status, :kerze, :wolke, :trend, :setter, :welle,
         :rsi, :sma_200, :sma_20, :strategy_id, :reference, :exchange)
        """

        for attempt in range(retries):
            try:
                with self._get_conn() as conn:
                    conn.executemany(sql, [s.to_db_row() for s in signals])
                    conn.commit()
                return len(signals)

            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    # Exponentielles Warten: 0.1s, 0.2s, 0.4s...
                    wait_time = 0.1 * (2**attempt)
                    logger.warning(
                        f"DB gesperrt. Warte {wait_time}s (Versuch {attempt + 1}/{retries})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"SQL Fehler: {e}")
                    raise  # Anderer Fehler -> Sofort Abbruch
            except Exception as e:
                logger.error(f"Unerwarteter Fehler: {e}")
                return 0

        logger.error(
            "DB Timeout: Konnte Signale nach mehreren Versuchen nicht speichern."
        )
        return 0

    def replace_stats(self, stats: List[SignalStat]) -> int:
        if not stats:
            return 0

        sql_insert = """
        INSERT INTO signal_stats
        (signal, symbol, timeframe, exchange, level, total, win, loss, rejected,
         wolke, welle, trend, setter, updated_at)
        VALUES
        (:signal, :symbol, :timeframe, :exchange, :level, :total, :win, :loss, :rejected,
         :wolke, :welle, :trend, :setter, :updated_at)
        """
        try:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM signal_stats")
                conn.executemany(sql_insert, [s.to_db_row() for s in stats])
                conn.commit()
            return len(stats)
        except sqlite3.Error as e:
            logger.error(f"Stats Replacement Error: {e}")
            raise
