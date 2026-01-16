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

        # 1. Signals Tabelle
        # NEU: dist_sma_20 und dist_sma_200 als GENERATED COLUMNS
        schema_signals = """
        CREATE TABLE IF NOT EXISTS signals (
            symbol TEXT NOT NULL, timeframe TEXT NOT NULL, signal TEXT NOT NULL,
            timestamp TEXT NOT NULL, close REAL, high REAL, low REAL, wuk REAL,
            status TEXT, kerze TEXT, trend TEXT, setter TEXT, welle TEXT,
            wolke TEXT, deluxe TEXT, rsi REAL, sma_200 REAL, sma_20 REAL,
            strategy_id TEXT, exchange TEXT,
            reference TEXT UNIQUE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            dist_sma_20 REAL GENERATED ALWAYS AS (
                CASE WHEN sma_20 IS NOT NULL AND sma_20 != 0
                THEN ROUND(((close - sma_20) / sma_20) * 100, 2)
                ELSE NULL END
            ) VIRTUAL,
            dist_sma_200 REAL GENERATED ALWAYS AS (
                CASE WHEN sma_200 IS NOT NULL AND sma_200 != 0
                THEN ROUND(((close - sma_200) / sma_200) * 100, 2)
                ELSE NULL END
            ) VIRTUAL
        );
        CREATE INDEX IF NOT EXISTS idx_signals_filter ON signals(symbol, timeframe, signal, timestamp DESC);
        """

        # 2. Stats Tabelle
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

        # 3. View
        schema_view = """
        CREATE VIEW IF NOT EXISTS view_signals_enriched AS
        SELECT
            s.rowid AS id,
            s.symbol, s.signal, s.timeframe, s.timestamp,
            s.close, s.trend, s.welle, s.wolke, s.setter, s.exchange,
            COALESCE(st1.win_rate, st5.win_rate) as specific_win_rate,
            COALESCE(st1.total, st5.total) as specific_samples,
            COALESCE(gl1.win_rate, gl5.win_rate) as global_win_rate,
            COALESCE(gl1.total, gl5.total) as global_samples
        FROM signals s
        LEFT JOIN signal_stats st1 ON s.symbol = st1.symbol AND s.signal = st1.signal AND s.timeframe = st1.timeframe AND st1.level = 'wolke_welle_trend_setter' AND s.wolke = st1.wolke AND s.welle = st1.welle AND s.trend = st1.trend AND s.setter = st1.setter
        LEFT JOIN signal_stats st5 ON s.symbol = st5.symbol AND s.signal = st5.signal AND s.timeframe = st5.timeframe AND st5.level = 'gesamt'
        LEFT JOIN signal_stats gl1 ON gl1.symbol = 'ALL_SYMBOLS' AND s.signal = gl1.signal AND s.timeframe = gl1.timeframe AND gl1.level = 'wolke_welle_trend_setter' AND s.wolke = gl1.wolke AND s.welle = gl1.welle AND s.trend = gl1.trend AND s.setter = gl1.setter
        LEFT JOIN signal_stats gl5 ON gl5.symbol = 'ALL_SYMBOLS' AND s.signal = gl5.signal AND s.timeframe = gl5.timeframe AND gl5.level = 'gesamt';
        """

        # 4. TABELLE: DIP BUYER SCREENER
        schema_dip_buyer = """
        CREATE TABLE IF NOT EXISTS screener_dip_buyer (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            symbol TEXT,
            exchange TEXT,
            timeframe TEXT,

            close REAL,
            high REAL,
            atr_r3 REAL,
            setup_score REAL,
            entry_limit REAL,
            atr5 REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, symbol)
        );
        """

        # 5. TABELLE: WEBHOOK SCREENER
        # Auch hier fügen wir die virtuellen Spalten hinzu, damit der Screener konsistent ist
        schema_webhook = """
        CREATE TABLE IF NOT EXISTS screener_webhook (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            symbol TEXT,
            exchange TEXT,
            timeframe TEXT,
            strategy TEXT,
            signal TEXT,

            close REAL,
            high REAL,
            low REAL,
            rsi REAL,
            sma_200 REAL,
            sma_20 REAL,

            rank INTEGER,
            filter_details TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            dist_sma_20 REAL GENERATED ALWAYS AS (
                CASE WHEN sma_20 IS NOT NULL AND sma_20 != 0
                THEN ROUND(((close - sma_20) / sma_20) * 100, 2)
                ELSE NULL END
            ) VIRTUAL,
            dist_sma_200 REAL GENERATED ALWAYS AS (
                CASE WHEN sma_200 IS NOT NULL AND sma_200 != 0
                THEN ROUND(((close - sma_200) / sma_200) * 100, 2)
                ELSE NULL END
            ) VIRTUAL,
            UNIQUE(date, symbol, strategy)
        );
        """

        # 6. TABELLE: TURNOVER TIMING SCREENER
        schema_turnover = """
        CREATE TABLE IF NOT EXISTS screener_turnover_timing (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            symbol TEXT,
            exchange TEXT,
            timeframe TEXT,
            source_index TEXT,

            close REAL,
            atr3 REAL,
            turnover_sma20 REAL,

            entry_1 REAL,
            entry_2 REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, symbol)
        );
        """

        # 7. Trades
        schema_trades = """
        CREATE TABLE IF NOT EXISTS active_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            entry_date TEXT NOT NULL,
            entry_price REAL NOT NULL,
            atr_at_entry REAL NOT NULL,
            quantity INTEGER DEFAULT 1,
            status TEXT DEFAULT 'OPEN',
            strategy TEXT,
            exit_price REAL,
            exit_reason TEXT,
            closed_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, entry_date)
        );
        """

        # 8. TABELLE: CROC SETUP SCREENER
        schema_screener_croc = """
        CREATE TABLE IF NOT EXISTS screener_croc (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            exchange TEXT,
            timeframe TEXT,
            signal TEXT,
            rank INTEGER,
            r_per_trade REAL,
            recommended_strategy TEXT,
            close REAL,
            high REAL,
            low REAL,
            rsi REAL,
            dist_ema REAL,
            match_filter TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, symbol, signal)
        );
        CREATE INDEX IF NOT EXISTS idx_screener_croc_date ON screener_croc(date);
        """

        try:
            with self._get_conn() as conn:
                conn.executescript(schema_signals)
                conn.executescript(schema_stats)
                conn.execute(schema_view)
                conn.executescript(schema_dip_buyer)
                conn.executescript(schema_webhook)
                conn.executescript(schema_turnover)
                conn.executescript(schema_trades)
                conn.executescript(schema_screener_croc)

                # --- MIGRATIONEN ---

                # Migration für exit_price in active trades Spalte
                try:
                    conn.execute(
                        "ALTER TABLE active_trades ADD COLUMN exit_price REAL;"
                    )
                    logger.info(
                        "Migration: Spalte 'exit_price' zu Tabelle 'active_trades' hinzugefügt."
                    )
                except sqlite3.OperationalError:
                    pass

                # Migration für DELUXE Spalte
                try:
                    conn.execute("ALTER TABLE signals ADD COLUMN deluxe TEXT")
                    logger.info(
                        "Migration: Spalte 'deluxe' zu Tabelle 'signals' hinzugefügt."
                    )
                except sqlite3.OperationalError:
                    pass

                # Migration: dist_sma_20 (Virtual Column)
                # Löst den Fehler "no such column: dist_sma_20"
                try:
                    conn.execute("""
                        ALTER TABLE signals
                        ADD COLUMN dist_sma_20 REAL GENERATED ALWAYS AS (
                            CASE WHEN sma_20 IS NOT NULL AND sma_20 != 0
                            THEN ROUND(((close - sma_20) / sma_20) * 100, 2)
                            ELSE NULL END
                        ) VIRTUAL
                    """)
                    logger.info(
                        "Migration: Spalte 'dist_sma_20' zu 'signals' hinzugefügt."
                    )
                except sqlite3.OperationalError:
                    pass  # Existiert schon

                # Migration: dist_sma_200 (Virtual Column)
                try:
                    conn.execute("""
                        ALTER TABLE signals
                        ADD COLUMN dist_sma_200 REAL GENERATED ALWAYS AS (
                            CASE WHEN sma_200 IS NOT NULL AND sma_200 != 0
                            THEN ROUND(((close - sma_200) / sma_200) * 100, 2)
                            ELSE NULL END
                        ) VIRTUAL
                    """)
                    logger.info(
                        "Migration: Spalte 'dist_sma_200' zu 'signals' hinzugefügt."
                    )
                except sqlite3.OperationalError:
                    pass  # Existiert schon

                # Migration für HIGH/LOW in screener_croc
                try:
                    conn.execute("ALTER TABLE screener_croc ADD COLUMN high REAL")
                    logger.info(
                        "Migration: Spalte 'high' zu Tabelle 'screener_croc' hinzugefügt."
                    )
                except sqlite3.OperationalError:
                    pass

                try:
                    conn.execute("ALTER TABLE screener_croc ADD COLUMN low REAL")
                    logger.info(
                        "Migration: Spalte 'low' zu Tabelle 'screener_croc' hinzugefügt."
                    )
                except sqlite3.OperationalError:
                    pass

                conn.commit()

        except sqlite3.Error as e:
            logger.critical(f"DB Init failed: {e}")
            raise

    def save_many(self, signals: List[CrocSignal], retries: int = 5) -> int:
        if not signals:
            return 0

        sql = """
        INSERT OR IGNORE INTO signals
        (symbol, timeframe, timestamp, signal, close, high, low, wuk, status, kerze, wolke, trend, setter, welle, deluxe, rsi, sma_200, sma_20, strategy_id, reference, exchange)
        VALUES
        (:symbol, :timeframe, :timestamp, :signal, :close, :high, :low, :wuk, :status, :kerze, :wolke, :trend, :setter, :welle, :deluxe, :rsi, :sma_200, :sma_20, :strategy_id, :reference, :exchange)
        """
        for attempt in range(retries):
            try:
                with self._get_conn() as conn:
                    conn.executemany(sql, [s.to_db_row() for s in signals])
                    conn.commit()
                return len(signals)
            except sqlite3.OperationalError:
                time.sleep(0.1 * (2**attempt))
            except Exception as e:
                logger.warning(f"DB save_many failed: {e}")
                return 0
        return 0

    def replace_stats(self, stats: List[SignalStat]) -> int:
        if not stats:
            return 0
        sql = "INSERT INTO signal_stats (signal, symbol, timeframe, exchange, level, total, win, loss, rejected, wolke, welle, trend, setter, updated_at) VALUES (:signal, :symbol, :timeframe, :exchange, :level, :total, :win, :loss, :rejected, :wolke, :welle, :trend, :setter, :updated_at)"
        with self._get_conn() as conn:
            conn.execute("DELETE FROM signal_stats")
            conn.executemany(sql, [s.to_db_row() for s in stats])
            conn.commit()
        return len(stats)

    @cache.memoize(timeout=10)
    def get_latest_signals_with_stats(self, limit=50, symbol=None):
        # todo: Cleanup function
        logger.warning("Aktell nicht im einsatz")

        with self._get_conn() as conn:
            if symbol:
                cursor = conn.execute(
                    "SELECT * FROM view_signals_enriched WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                    (symbol, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM view_signals_enriched ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
            return [dict(row) for row in cursor.fetchall()]

    def clean_batz_exchanges(self):
        mapping_dict = mapper._mapping
        if not mapping_dict:
            return 0
        update_data = [(real, sym) for sym, real in mapping_dict.items()]
        with self._get_conn() as conn:
            conn.executemany(
                "UPDATE signals SET exchange = ? WHERE symbol = ?", update_data
            )
            return conn.total_changes

    def save_screener_dip_buyer(self, results: List[dict]):
        if not results:
            return
        sql = "INSERT OR REPLACE INTO screener_dip_buyer (date, symbol, exchange, timeframe, close, high, atr_r3, setup_score, entry_limit, atr5) VALUES (:date, :symbol, :exchange, :timeframe, :close, :high, :atr_r3, :setup_score, :entry_limit, :atr5)"
        with self._get_conn() as conn:
            conn.executemany(sql, results)
            conn.commit()

    def save_screener_webhook(self, results: List[dict]):
        if not results:
            return
        sql = """
        INSERT OR IGNORE INTO screener_webhook
        (date, symbol, exchange, timeframe, strategy, signal, close, high, low, rsi, sma_200, sma_20, rank, filter_details)
        VALUES
        (:date, :symbol, :exchange, :timeframe, :strategy, :signal, :close, :high, :low, :rsi, :sma_200, :sma_20, :rank, :filter_details)
        """
        try:
            with self._get_conn() as conn:
                conn.executemany(sql, results)
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Fehler save_screener_webhook: {e}")

    # --- TURNOVER TIMING SPEICHERN ---
    def save_screener_turnover_timing(self, results: List[dict]):
        if not results:
            return
        sql = """
        INSERT OR REPLACE INTO screener_turnover_timing
        (date, symbol, exchange, timeframe, source_index, close, atr3, turnover_sma20, entry_1, entry_2)
        VALUES (:date, :symbol, :exchange, :timeframe, :source_index, :close, :atr3, :turnover_sma20, :entry_1, :entry_2)
        """
        try:
            with self._get_conn() as conn:
                conn.executemany(sql, results)
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Fehler save_screener_turnover_timing: {e}")

    # --- CROC SETUP SPEICHERN ---
    def save_screener_croc(self, results: List[dict]):
        if not results:
            return

        # SQL Update: high und low hinzugefügt
        sql = """
        INSERT OR REPLACE INTO screener_croc
        (date, symbol, exchange, timeframe, signal, rank, r_per_trade,
         recommended_strategy, close, high, low, rsi, dist_ema, match_filter)
        VALUES
        (:date, :symbol, :exchange, :timeframe, :signal, :rank, :r_per_trade,
         :recommended_strategy, :close, :high, :low, :rsi, :dist_ema, :match_filter)
        """
        try:
            with self._get_conn() as conn:
                conn.executemany(sql, results)
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Fehler save_screener_croc: {e}")

    def get_dip_buyer_results(self, limit: int = 50):
        sql = "SELECT * FROM screener_dip_buyer ORDER BY date DESC, setup_score DESC LIMIT ?"
        with self._get_conn() as conn:
            return [dict(row) for row in conn.execute(sql, (limit,)).fetchall()]

    def get_webhook_results(self, limit: int = 50):
        sql = "SELECT * FROM screener_webhook ORDER BY date DESC, rank ASC, created_at DESC LIMIT ?"
        with self._get_conn() as conn:
            return [dict(row) for row in conn.execute(sql, (limit,)).fetchall()]

    # --- TURNOVER TIMING LESEN ---
    def get_turnover_timing_results(self, limit: int = 50):
        sql = "SELECT * FROM screener_turnover_timing ORDER BY date DESC, turnover_sma20 DESC LIMIT ?"
        with self._get_conn() as conn:
            return [dict(row) for row in conn.execute(sql, (limit,)).fetchall()]

    # --- CROC SETUP LESEN ---
    def get_croc_results(self, limit: int = 100) -> List[dict]:
        sql = """
        SELECT * FROM screener_croc
        ORDER BY date DESC, rank ASC, r_per_trade DESC
        LIMIT ?
        """
        with self._get_conn() as conn:
            return [dict(row) for row in conn.execute(sql, (limit,)).fetchall()]

    def clear_screener_webhook(self):
        try:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM screener_webhook")
                conn.commit()
            logger.info("Tabelle screener_webhook erfolgreich geleert.")
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Leeren von screener_webhook: {e}")

    # --- Trades ---

    def add_trade(
        self,
        symbol,
        entry_date,
        entry_price,
        atr_at_entry,
        strategy="MANUAL",
        quantity=1,
    ):
        with self._get_conn() as conn:
            existing = conn.execute(
                "SELECT id FROM active_trades WHERE symbol = ? AND entry_date = ?",
                (symbol, entry_date),
            ).fetchone()

            if existing:
                return existing[0]

            try:
                cursor = conn.execute(
                    """
                    INSERT INTO active_trades
                    (symbol, entry_date, entry_price, atr_at_entry, quantity, status, strategy)
                    VALUES (?, ?, ?, ?, ?, 'CREATED', ?)
                    """,
                    (symbol, entry_date, entry_price, atr_at_entry, quantity, strategy),
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                logger.debug(
                    f"Trade existiert bereits (IntegrityError): {symbol} am {entry_date}"
                )
                return None

    def get_open_trades(self):
        # Todo: cleanup
        logger.warning("Status OPEN wird nicht verwendet")

        with self._get_conn() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM active_trades WHERE status = 'OPEN'"
                ).fetchall()
            ]

    def get_trades_history(self, limit: int = 100):
        sql = "SELECT * FROM active_trades ORDER BY entry_date DESC, created_at DESC LIMIT ?"
        with self._get_conn() as conn:
            return [dict(row) for row in conn.execute(sql, (limit,)).fetchall()]

    def close_trade(self, trade_id, reason="MANUAL"):
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE active_trades SET status = 'CLOSED', exit_reason = ?, closed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (reason, trade_id),
            )
            conn.commit()

    def update_trade_status(self, trade_id, new_status, exit_reason=None):
        sql = "UPDATE active_trades SET status = ? WHERE id = ?"
        params = [new_status, trade_id]
        if exit_reason:
            sql = "UPDATE active_trades SET status = ?, exit_reason = ?, closed_at = CURRENT_TIMESTAMP WHERE id = ?"
            params = [new_status, exit_reason, trade_id]
        with self._get_conn() as conn:
            conn.execute(sql, tuple(params))
            conn.commit()

    def get_all_managed_trades(self):
        with self._get_conn() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM active_trades WHERE status IN ('CREATED', 'ACTIVE')"
                ).fetchall()
            ]

    def update_trade_quantity(self, trade_id: int, quantity: int):
        """Aktualisiert die Stückzahl eines Trades."""
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "UPDATE active_trades SET quantity = ? WHERE id = ?",
                    (quantity, trade_id),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Fehler beim Update der Quantity für Trade {trade_id}: {e}")
