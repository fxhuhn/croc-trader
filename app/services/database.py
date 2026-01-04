import logging
import sqlite3
import time
from pathlib import Path
from typing import List

from ..extensions import cache
from ..mapping import mapper
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
        SELECT
            s.rowid AS id,  -- <--- HIER WAR DER FEHLER (statt s.id)
            s.symbol, s.signal, s.timeframe, s.timestamp,
            s.close, s.trend, s.welle, s.wolke, s.setter, s.exchange,

            -- 1. INDIVIDUELLE STATISTIK
            COALESCE(st1.win_rate, st2.win_rate, st3.win_rate, st4.win_rate, st5.win_rate) as specific_win_rate,
            COALESCE(st1.total, st2.total, st3.total, st4.total, st5.total) as specific_samples,

            CASE
                WHEN st1.win_rate IS NOT NULL THEN 'full'
                WHEN st2.win_rate IS NOT NULL THEN 'no_setter'
                WHEN st3.win_rate IS NOT NULL THEN 'no_trend'
                WHEN st4.win_rate IS NOT NULL THEN 'no_welle'
                WHEN st5.win_rate IS NOT NULL THEN 'base'
                ELSE NULL
            END as specific_match_level,

            -- 2. GLOBALE STATISTIK
            COALESCE(gl1.win_rate, gl2.win_rate, gl3.win_rate, gl4.win_rate, gl5.win_rate) as global_win_rate,
            COALESCE(gl1.total, gl2.total, gl3.total, gl4.total, gl5.total) as global_samples

        FROM signals s

        -- BLOCK A: Spezifische Joins
        LEFT JOIN signal_stats st1 ON s.symbol = st1.symbol AND s.signal = st1.signal AND s.timeframe = st1.timeframe AND st1.level = 'wolke_welle_trend_setter' AND s.wolke = st1.wolke AND s.welle = st1.welle AND s.trend = st1.trend AND s.setter = st1.setter
        LEFT JOIN signal_stats st2 ON s.symbol = st2.symbol AND s.signal = st2.signal AND s.timeframe = st2.timeframe AND st2.level = 'wolke_welle_trend' AND s.wolke = st2.wolke AND s.welle = st2.welle AND s.trend = st2.trend
        LEFT JOIN signal_stats st3 ON s.symbol = st3.symbol AND s.signal = st3.signal AND s.timeframe = st3.timeframe AND st3.level = 'wolke_welle' AND s.wolke = st3.wolke AND s.welle = st3.welle
        LEFT JOIN signal_stats st4 ON s.symbol = st4.symbol AND s.signal = st4.signal AND s.timeframe = st4.timeframe AND st4.level = 'wolke' AND s.wolke = st4.wolke
        LEFT JOIN signal_stats st5 ON s.symbol = st5.symbol AND s.signal = st5.signal AND s.timeframe = st5.timeframe AND st5.level = 'gesamt'

        -- BLOCK B: Globale Joins
        LEFT JOIN signal_stats gl1 ON gl1.symbol = 'ALL_SYMBOLS' AND s.signal = gl1.signal AND s.timeframe = gl1.timeframe AND gl1.level = 'wolke_welle_trend_setter' AND s.wolke = gl1.wolke AND s.welle = gl1.welle AND s.trend = gl1.trend AND s.setter = gl1.setter
        LEFT JOIN signal_stats gl2 ON gl2.symbol = 'ALL_SYMBOLS' AND s.signal = gl2.signal AND s.timeframe = gl2.timeframe AND gl2.level = 'wolke_welle_trend' AND s.wolke = gl2.wolke AND s.welle = gl2.welle AND s.trend = gl2.trend
        LEFT JOIN signal_stats gl3 ON gl3.symbol = 'ALL_SYMBOLS' AND s.signal = gl3.signal AND s.timeframe = gl3.timeframe AND gl3.level = 'wolke_welle' AND s.wolke = gl3.wolke AND s.welle = gl3.welle
        LEFT JOIN signal_stats gl4 ON gl4.symbol = 'ALL_SYMBOLS' AND s.signal = gl4.signal AND s.timeframe = gl4.timeframe AND gl4.level = 'wolke' AND s.wolke = gl4.wolke
        LEFT JOIN signal_stats gl5 ON gl5.symbol = 'ALL_SYMBOLS' AND s.signal = gl5.signal AND s.timeframe = gl5.timeframe AND gl5.level = 'gesamt';
        """
        schema_screener = """
        CREATE TABLE IF NOT EXISTS screener_results (
            strategy TEXT NOT NULL,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            close REAL,
            setup_score REAL,
            atr_r3 REAL,
            ibs REAL,
            atr5 REAL,
            entry_limit REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (strategy, symbol, date)
        );
        CREATE INDEX IF NOT EXISTS idx_screener_lookup ON screener_results(strategy, date DESC);
        """

        schema_trades = """
        CREATE TABLE IF NOT EXISTS active_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            entry_date TEXT NOT NULL,    -- YYYY-MM-DD
            entry_price REAL NOT NULL,
            atr_at_entry REAL NOT NULL,  -- Wichtig für Kursziel Berechnung
            quantity INTEGER DEFAULT 1,
            status TEXT DEFAULT 'OPEN',  -- OPEN, CLOSED
            exit_reason TEXT,            -- TIME, TARGET, LOC, MANUAL
            closed_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(symbol, entry_date)
        );
        CREATE INDEX IF NOT EXISTS idx_trades_status ON active_trades(status);
        """

        try:
            with self._get_conn() as conn:
                conn.executescript(schema_signals)
                conn.executescript(schema_stats)
                # View handling: Drop & Recreate um Updates sicherzustellen
                # conn.execute("DROP TABLE screener_results")
                # conn.execute("DROP TABLE active_trades")
                # conn.execute("DROP VIEW IF EXISTS view_signals_enriched")
                conn.execute(schema_view)  # Hier den echten View SQL Code nutzen
                conn.executescript(schema_screener)
                conn.executescript(schema_trades)
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

    @cache.memoize(timeout=10)
    def get_latest_signals_with_stats(
        self, limit: int = 50, symbol: str = None
    ) -> List[dict]:
        """
        Holt die neuesten Signale inklusive der angereicherten Statistiken (Win-Rates)
        aus dem View 'view_signals_enriched'.
        """
        try:
            with self._get_conn() as conn:
                if symbol:
                    # Fall A: Filter nach Symbol
                    sql = """
                    SELECT * FROM view_signals_enriched
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """
                    cursor = conn.execute(sql, (symbol, limit))
                else:
                    # Fall B: Alle Signale
                    sql = """
                    SELECT * FROM view_signals_enriched
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """
                    cursor = conn.execute(sql, (limit,))

                # Rows in Dicts umwandeln für JSON-Output
                return [dict(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            logger.error(f"Fehler beim Abrufen der Croc-Signale: {e}")
            return []

    def clean_batz_exchanges(self) -> int:
        """
        Bereinigt die Datenbank: Ersetzt BATS (und andere) durch
        die korrekten Börsen aus der symbol_exchange.json.
        """
        # 1. Mapping holen (Dict: {'AAPL': 'NASDAQ', ...})
        mapping_dict = mapper._mapping

        if not mapping_dict:
            logger.warning("Kein Mapping geladen. Abruch der Bereinigung.")
            return 0

        # 2. Liste für Batch-Update vorbereiten
        # Format für executemany: [(neuer_exchange, symbol), (neuer_exchange, symbol), ...]
        update_data = []
        for symbol, real_exchange in mapping_dict.items():
            update_data.append((real_exchange, symbol))

        logger.info(f"Starte Exchange-Bereinigung für {len(update_data)} Symbole...")

        sql = """
        UPDATE signals
        SET exchange = ?
        WHERE symbol = ?
        -- Optional: Nur ändern, wenn es vorher BATS war oder leer?
        -- Wenn wir IMMER korrigieren wollen (auch NYSE -> NASDAQ Fehler),
        -- lassen wir die AND Klausel weg.
        -- Für BATS-Only Fokus: AND (exchange = 'BATS' OR exchange IS NULL)
        """

        try:
            with self._get_conn() as conn:
                # executemany führt das Update tausendfach extrem schnell aus
                conn.executemany(sql, update_data)

                # Wir können zählen, wie viele Zeilen tatsächlich geändert wurden,
                # aber bei executemany gibt rowcount oft die Gesamtzahl der Versuche zurück.
                changes = conn.total_changes
                conn.commit()

            logger.info(f"Datenbank bereinigt. Betroffene Zeilen (ca.): {changes}")
            return changes

        except sqlite3.Error as e:
            logger.error(f"Fehler bei Exchange-Bereinigung: {e}")
            return 0

    def save_screener_results(self, results: List[dict]):
        if not results:
            return

        sql = """
        INSERT OR REPLACE INTO screener_results
        (strategy, symbol, date, close, setup_score, atr_r3, ibs, atr5, entry_limit)
        VALUES (:strategy, :symbol, :date, :close, :setup_score, :atr_r3, :ibs, :atr5, :entry_limit)
        """
        try:
            with self._get_conn() as conn:
                conn.executemany(sql, results)
                conn.commit()
            logger.info(f"Screener: {len(results)} Treffer gespeichert.")
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Speichern der Screener-Daten: {e}")

        cache.delete_memoized(self.get_screener_results)
        logger.info("Screener Cache invalidiert.")

    # Neue Methode zum Abrufen für die API
    @cache.memoize(timeout=3600)
    def get_screener_results(self, strategy: str, limit: int = 50):
        sql = """
        SELECT * FROM screener_results
        WHERE strategy = ?
        ORDER BY date DESC, setup_score DESC
        LIMIT ?
        """
        with self._get_conn() as conn:
            return [
                dict(row) for row in conn.execute(sql, (strategy, limit)).fetchall()
            ]

    # --- NEUE METHODEN FÜR TRADES ---

    def add_trade(self, symbol, entry_date, entry_price, atr_at_entry, quantity=1):
        """
        Fügt einen Trade hinzu, ABER NUR, wenn nicht bereits eine offene Position existiert.
        """
        with self._get_conn() as conn:
            # 1. Check: Gibt es schon eine laufende Position (ACTIVE oder CREATED)?
            check_running = "SELECT id FROM active_trades WHERE symbol = ? AND status IN ('CREATED', 'ACTIVE')"
            existing = conn.execute(check_running, (symbol,)).fetchone()

            if existing:
                return existing[0]

            # 2. Insert mit Status CREATED
            sql = """
            INSERT INTO active_trades (symbol, entry_date, entry_price, atr_at_entry, quantity, status)
            VALUES (?, ?, ?, ?, ?, 'CREATED')  -- <--- ÄNDERUNG HIER
            """
            cursor = conn.execute(
                sql, (symbol, entry_date, entry_price, atr_at_entry, quantity)
            )
            conn.commit()
            return cursor.lastrowid

    def get_open_trades(self) -> List[dict]:
        sql = "SELECT * FROM active_trades WHERE status = 'OPEN'"
        with self._get_conn() as conn:
            return [dict(row) for row in conn.execute(sql).fetchall()]

    def close_trade(self, trade_id, reason="MANUAL"):
        sql = """
        UPDATE active_trades
        SET status = 'CLOSED', exit_reason = ?, closed_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """
        with self._get_conn() as conn:
            conn.execute(sql, (reason, trade_id))
            conn.commit()

    def update_trade_status(self, trade_id, new_status, exit_reason=None):
        """Hilfsfunktion zum Ändern des Status (z.B. CREATED -> ACTIVE)."""
        sql = "UPDATE active_trades SET status = ? WHERE id = ?"
        params = [new_status, trade_id]

        if exit_reason:
            sql = "UPDATE active_trades SET status = ?, exit_reason = ?, closed_at = CURRENT_TIMESTAMP WHERE id = ?"
            params = [new_status, exit_reason, trade_id]

        with self._get_conn() as conn:
            conn.execute(sql, tuple(params))
            conn.commit()

    def get_all_managed_trades(self) -> List[dict]:
        """Holt alles was überwacht werden muss (CREATED und ACTIVE)."""
        sql = "SELECT * FROM active_trades WHERE status IN ('CREATED', 'ACTIVE')"
        with self._get_conn() as conn:
            return [dict(row) for row in conn.execute(sql).fetchall()]
