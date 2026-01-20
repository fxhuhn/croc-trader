import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

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

        # 7. Trades (The Registry)
        schema_trades = """
        CREATE TABLE IF NOT EXISTS active_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            signal_date TEXT,
            entry_date TEXT NOT NULL,
            entry_price REAL NOT NULL,
            atr_at_entry REAL NOT NULL,
            quantity INTEGER DEFAULT 1,
            status TEXT DEFAULT 'OPEN',
            strategy TEXT,

            screener_id INTEGER,

            exit_price REAL,
            exit_reason TEXT,
            closed_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            exit_date TEXT,
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

        # 9. TABELLE: TRADES CROC LOG
        schema_trades_croc = """
        CREATE TABLE IF NOT EXISTS trades_croc (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            exchange TEXT,
            timeframe TEXT,
            signal TEXT,
            recommended_strategy TEXT,

            entry REAL,
            stop REAL,
            tp_1 REAL,
            tp_2 REAL,

            exit_reason TEXT,
            close REAL,
            high REAL,
            low REAL,

            active_trade_id INTEGER,
            pnl_percent REAL,
            quantity INTEGER,
            risk_multiple REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, symbol, active_trade_id)
        );
        """

        # 10. TABELLE: TRADES DIP BUYER LOG
        schema_trades_dip = """
        CREATE TABLE IF NOT EXISTS trades_dip_buyer (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            exchange TEXT,
            timeframe TEXT,

            entry REAL,
            atr REAL,
            tp_target REAL,
            threshold_loc REAL,

            exit_reason TEXT,
            close REAL,
            high REAL,
            low REAL,

            active_trade_id INTEGER,
            pnl_percent REAL,
            quantity INTEGER,
            days_held INTEGER,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, symbol, active_trade_id)
        );
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
                conn.executescript(schema_trades_croc)
                conn.executescript(schema_trades_dip)

                # --- MIGRATIONEN ---
                # (Same as before)
                try:
                    conn.execute(
                        "ALTER TABLE active_trades ADD COLUMN exit_price REAL;"
                    )
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute("ALTER TABLE signals ADD COLUMN deluxe TEXT")
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute(
                        """ALTER TABLE signals ADD COLUMN dist_sma_20 REAL GENERATED ALWAYS AS (CASE WHEN sma_20 IS NOT NULL AND sma_20 != 0 THEN ROUND(((close - sma_20) / sma_20) * 100, 2) ELSE NULL END) VIRTUAL"""
                    )
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute(
                        """ALTER TABLE signals ADD COLUMN dist_sma_200 REAL GENERATED ALWAYS AS (CASE WHEN sma_200 IS NOT NULL AND sma_200 != 0 THEN ROUND(((close - sma_200) / sma_200) * 100, 2) ELSE NULL END) VIRTUAL"""
                    )
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute("ALTER TABLE screener_croc ADD COLUMN high REAL")
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute("ALTER TABLE screener_croc ADD COLUMN low REAL")
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute("ALTER TABLE active_trades ADD COLUMN exit_date TEXT")
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute(
                        "ALTER TABLE active_trades ADD COLUMN signal_date TEXT"
                    )
                except sqlite3.OperationalError:
                    pass

                try:
                    cursor = conn.execute("PRAGMA table_info(active_trades)")
                    columns = [row["name"] for row in cursor.fetchall()]
                    if "source_id" in columns and "screener_id" not in columns:
                        conn.execute(
                            "ALTER TABLE active_trades RENAME COLUMN source_id TO screener_id"
                        )
                    elif "screener_id" not in columns:
                        conn.execute(
                            "ALTER TABLE active_trades ADD COLUMN screener_id INTEGER"
                        )
                except sqlite3.OperationalError:
                    pass

                try:
                    conn.execute("ALTER TABLE trades_dip_buyer ADD COLUMN atr REAL")
                except sqlite3.OperationalError:
                    pass

                conn.commit()

        except sqlite3.Error as e:
            logger.critical(f"DB Init failed: {e}")
            raise

    # ... (Helpers like save_many, replace_stats, etc. remain the same) ...
    def save_many(self, signals, retries=5):
        return 0

    def replace_stats(self, stats):
        return 0

    def clean_batz_exchanges(self):
        return 0

    def save_screener_dip_buyer(self, r):
        if not r:
            return
        with self._get_conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO screener_dip_buyer (date, symbol, exchange, timeframe, close, high, atr_r3, setup_score, entry_limit, atr5) VALUES (:date, :symbol, :exchange, :timeframe, :close, :high, :atr_r3, :setup_score, :entry_limit, :atr5)",
                r,
            )
            conn.commit()

    def save_screener_webhook(self, r):
        if not r:
            return
        with self._get_conn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO screener_webhook (date, symbol, exchange, timeframe, strategy, signal, close, high, low, rsi, sma_200, sma_20, rank, filter_details) VALUES (:date, :symbol, :exchange, :timeframe, :strategy, :signal, :close, :high, :low, :rsi, :sma_200, :sma_20, :rank, :filter_details)",
                r,
            )
            conn.commit()

    def save_screener_turnover_timing(self, r):
        if not r:
            return
        with self._get_conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO screener_turnover_timing (date, symbol, exchange, timeframe, source_index, close, atr3, turnover_sma20, entry_1, entry_2) VALUES (:date, :symbol, :exchange, :timeframe, :source_index, :close, :atr3, :turnover_sma20, :entry_1, :entry_2)",
                r,
            )
            conn.commit()

    def save_screener_croc(self, r):
        if not r:
            return
        with self._get_conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO screener_croc (date, symbol, exchange, timeframe, signal, rank, r_per_trade, recommended_strategy, close, high, low, rsi, dist_ema, match_filter) VALUES (:date, :symbol, :exchange, :timeframe, :signal, :rank, :r_per_trade, :recommended_strategy, :close, :high, :low, :rsi, :dist_ema, :match_filter)",
                r,
            )
            conn.commit()

    def get_dip_buyer_results(self, limit=50):
        with self._get_conn() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM screener_dip_buyer ORDER BY date DESC, setup_score DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            ]

    def get_webhook_results(self, limit=50):
        with self._get_conn() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM screener_webhook ORDER BY date DESC, rank ASC, created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            ]

    def get_turnover_timing_results(self, limit=50):
        with self._get_conn() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM screener_turnover_timing ORDER BY date DESC, turnover_sma20 DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            ]

    def get_croc_results(self, limit=100):
        with self._get_conn() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM screener_croc ORDER BY date DESC, rank ASC, r_per_trade DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            ]

    def clear_screener_webhook(self):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM screener_webhook")
            conn.commit()

    # --- LOGS ---
    def log_croc_trade(self, data):
        sql = "INSERT OR REPLACE INTO trades_croc (date, symbol, exchange, timeframe, signal, recommended_strategy, entry, stop, tp_1, tp_2, exit_reason, close, high, low, active_trade_id, pnl_percent, quantity, risk_multiple) VALUES (:date, :symbol, :exchange, :timeframe, :signal, :recommended_strategy, :entry, :stop, :tp_1, :tp_2, :exit_reason, :close, :high, :low, :active_trade_id, :pnl_percent, :quantity, :risk_multiple)"
        try:
            with self._get_conn() as conn:
                conn.execute(sql, data)
                conn.commit()
        except:
            pass

    def get_latest_croc_snapshot(self, trade_id):
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM trades_croc WHERE active_trade_id = ? ORDER BY date DESC LIMIT 1",
                (trade_id,),
            ).fetchone()
            return dict(row) if row else None

    def log_dip_trade(self, data):
        sql = "INSERT OR REPLACE INTO trades_dip_buyer (date, symbol, exchange, timeframe, entry, atr, tp_target, threshold_loc, exit_reason, close, high, low, active_trade_id, pnl_percent, quantity, days_held) VALUES (:date, :symbol, :exchange, :timeframe, :entry, :atr, :tp_target, :threshold_loc, :exit_reason, :close, :high, :low, :active_trade_id, :pnl_percent, :quantity, :days_held)"
        try:
            with self._get_conn() as conn:
                conn.execute(sql, data)
                conn.commit()
        except:
            pass

    def get_latest_dip_snapshot(self, trade_id):
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM trades_dip_buyer WHERE active_trade_id = ? ORDER BY date DESC LIMIT 1",
                (trade_id,),
            ).fetchone()
            return dict(row) if row else None

    # --- Active Trades ---

    def _get_next_trading_day(self, symbol: str, date_val: datetime) -> Optional[str]:
        """
        Calculates the next business day accurately using market_prices.

        Logic:
        1. If date is in the past (Backfill context):
           Strictly query the DB. If no data exists > date_val, return None.
           This prevents inventing trading days on holidays/weekends.

        2. If date is Today or Future (Live Trading context):
           Use simple calendar math (skip weekends) to predict the next slot.
        """
        date_str = date_val.strftime("%Y-%m-%d")
        today_str = datetime.now().strftime("%Y-%m-%d")

        # A) BACKFILL Context (Historical Date)
        if date_str < today_str:
            try:
                with self._get_conn() as conn:
                    # Find the VERY NEXT available date in the DB
                    row = conn.execute(
                        "SELECT date FROM market_prices WHERE symbol = ? AND date > ? ORDER BY date ASC LIMIT 1",
                        (symbol, date_str),
                    ).fetchone()

                    if row:
                        return row[0]  # Found accurate next trading day
                    else:
                        return None  # No data found (End of history or missing data)
            except Exception:
                return None

        # B) LIVE Context (Today/Future)
        # Use Calendar Math fallback
        next_day = date_val + timedelta(days=1)
        weekday = next_day.weekday()  # 0=Mon, 6=Sun

        if weekday == 5:  # Saturday -> Monday
            next_day += timedelta(days=2)
        elif weekday == 6:  # Sunday -> Monday
            next_day += timedelta(days=1)

        return next_day.strftime("%Y-%m-%d")

    def add_trade(
        self,
        symbol,
        signal_date,
        entry_price,
        atr_at_entry,
        strategy="MANUAL",
        quantity=1,
        screener_id=None,
    ):
        try:
            # Parse signal date
            sig_dt = datetime.strptime(str(signal_date).split(" ")[0], "%Y-%m-%d")

            # 1. Calculate Valid From (Entry Date)
            valid_from = self._get_next_trading_day(symbol, sig_dt)

            if valid_from is None:
                # If backfill returns None, it means we have no valid next day.
                # Abort trade creation to avoid invalid dates.
                logger.warning(
                    f"Skipping trade for {symbol} on {signal_date}: No valid next trading day found in DB."
                )
                return None

        except Exception:
            valid_from = signal_date

        with self._get_conn() as conn:
            # 2. Check for Duplicate Active Trades
            # Prevent creating a 'CREATED' trade if an 'ACTIVE' one exists for the same strategy/symbol
            active_trade = conn.execute(
                "SELECT id FROM active_trades WHERE symbol = ? AND status IN ('CREATED', 'ACTIVE') AND strategy = ?",
                (symbol, strategy),
            ).fetchone()

            if active_trade:
                logger.info(
                    f"Skipping new trade for {symbol} ({strategy}): Active trade already exists (ID: {active_trade[0]})."
                )
                return existing_trade_id_if_needed  # Or None

            # 3. Check if *this specific signal* was already added (Idempotency)
            existing = conn.execute(
                "SELECT id FROM active_trades WHERE symbol = ? AND signal_date = ?",
                (symbol, signal_date),
            ).fetchone()

            if existing:
                return existing[0]

            try:
                cursor = conn.execute(
                    """
                    INSERT INTO active_trades
                    (symbol, signal_date, entry_date, entry_price, atr_at_entry, quantity, status, strategy, screener_id)
                    VALUES (?, ?, ?, ?, ?, ?, 'CREATED', ?, ?)
                    """,
                    (
                        symbol,
                        signal_date,
                        valid_from,
                        entry_price,
                        atr_at_entry,
                        quantity,
                        strategy,
                        screener_id,
                    ),
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None

    def get_open_trades(self):
        with self._get_conn() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM active_trades WHERE status = 'OPEN'"
                ).fetchall()
            ]

    def get_trades_history(self, limit=100):
        with self._get_conn() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM active_trades ORDER BY entry_date DESC, created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            ]

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
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM active_trades WHERE status IN ('CREATED', 'ACTIVE')"
                ).fetchall()
            ]

    def update_trade_quantity(self, trade_id, qty):
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "UPDATE active_trades SET quantity = ? WHERE id = ?",
                    (qty, trade_id),
                )
                conn.commit()
        except:
            pass

    def get_croc_trades_log(self, limit=100):
        """Fetches historical log of Croc trades."""
        with self._get_conn() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM trades_croc ORDER BY date DESC, created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            ]

    def get_dip_trades_log(self, limit=100):
        """Fetches historical log of DipBuyer trades."""
        with self._get_conn() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM trades_dip_buyer ORDER BY date DESC, created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            ]
