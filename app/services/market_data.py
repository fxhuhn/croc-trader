import logging
import random
import sqlite3
import threading
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import pytz
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ..tools.symbol_lists import ExchangeSymbol

logger = logging.getLogger(__name__)

type SymbolList = List[str]
type MarketRecord = Dict[str, Any]

BATCH_SIZE = 500
DB_TIMEOUT = 60
REQUEST_TIMEOUT = 30

# --- HELPER: THREAD LOCK DECORATOR ---
# Verhindert, dass Sync manuell und per Scheduler gleichzeitig läuft
_service_lock = threading.Lock()


def require_lock(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        # acquire(blocking=False) gibt False zurück, wenn schon gelockt
        if not _service_lock.acquire(blocking=False):
            logger.warning(f"SKIP {f.__name__}: Ein Update läuft bereits!")
            return
        try:
            return f(*args, **kwargs)
        finally:
            _service_lock.release()

    return wrapper


# --------------------------------------------------------------------------
# Teil 1: Datenbank Layer
# --------------------------------------------------------------------------


class MarketDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=DB_TIMEOUT)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS market_prices (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            provider TEXT NOT NULL,
            timeframe TEXT NOT NULL DEFAULT '1D',
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date, provider)
        );
        -- Tabelle für ignorierte Symbole (Delisted / Broken)
        CREATE TABLE IF NOT EXISTS ignored_symbols (
            symbol TEXT PRIMARY KEY,
            reason TEXT,
            ignored_since TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_market_lookup ON market_prices(symbol, date);
        CREATE INDEX IF NOT EXISTS idx_market_date ON market_prices(date);
        """
        try:
            with self._get_conn() as conn:
                conn.executescript(schema)
        except sqlite3.Error as e:
            logger.critical(f"DB Schema Init failed: {e}")
            raise

    def optimize_db(self):
        """Führt Maintenance-Tasks durch (VACUUM, Analyze)."""
        try:
            logger.info("Führe DB Maintenance (VACUUM) durch...")
            with self._get_conn() as conn:
                conn.execute("VACUUM;")
                conn.execute("ANALYZE;")
        except sqlite3.Error as e:
            logger.error(f"DB Maintenance failed: {e}")

    def upsert_bulk(self, records: List[MarketRecord]) -> int:
        if not records:
            return 0
        sql = """
        INSERT OR REPLACE INTO market_prices
        (symbol, date, provider, timeframe, open, high, low, close, volume)
        VALUES (:symbol, :date, :provider, :timeframe, :open, :high, :low, :close, :volume)
        """
        try:
            with self._get_conn() as conn:
                conn.executemany(sql, records)
                conn.commit()
            return len(records)
        except sqlite3.Error as e:
            logger.error(f"Bulk Insert fehlgeschlagen: {e}")
            return 0

    def ignore_symbol(self, symbol: str, reason: str):
        """Setzt ein Symbol auf die Blacklist."""
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO ignored_symbols (symbol, reason) VALUES (?, ?)",
                    (symbol, reason),
                )
                conn.commit()
            logger.warning(f"Symbol '{symbol}' wird ignoriert. Grund: {reason}")
        except sqlite3.Error:
            pass

    def get_ignored_symbols(self) -> Set[str]:
        """Holt die Blacklist."""
        try:
            with self._get_conn() as conn:
                res = conn.execute("SELECT symbol FROM ignored_symbols").fetchall()
                return {r["symbol"] for r in res}
        except sqlite3.Error:
            return set()

    def get_outdated_symbols(
        self, reference_date: str, provider: str = "yahoo"
    ) -> SymbolList:
        # Ignorierte Symbole direkt ausschließen
        sql = """
        SELECT symbol FROM market_prices
        WHERE provider = ?
        AND symbol NOT IN (SELECT symbol FROM ignored_symbols)
        GROUP BY symbol HAVING MAX(date) < ?
        """
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(sql, (provider, reference_date))
                return [row["symbol"] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Gap Check Query fehlgeschlagen: {e}")
            return []

    def get_all_known_symbols(self) -> SymbolList:
        try:
            with self._get_conn() as conn:
                cursor = conn.execute("SELECT DISTINCT symbol FROM market_prices")
                return [row["symbol"] for row in cursor.fetchall()]
        except sqlite3.Error:
            return []

    def get_candle_at_date(self, symbol: str, date: str) -> Optional[Dict[str, float]]:
        sql = "SELECT open, high, low, close, volume FROM market_prices WHERE symbol = ? AND date = ?"
        try:
            with self._get_conn() as conn:
                row = conn.execute(sql, (symbol, date)).fetchone()
                if row:
                    return {
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"],
                    }
                return None
        except sqlite3.Error:
            return None


# --------------------------------------------------------------------------
# Teil 2: Service Layer
# --------------------------------------------------------------------------


class MarketDataService:
    def __init__(self, db_path: Path):
        self.db = MarketDatabase(db_path)

    @require_lock
    def update_market_data(
        self, full_reload: bool = False, specific_symbols: Optional[SymbolList] = None
    ) -> None:
        start_time = datetime.now()
        mode_label = "FULL RELOAD" if full_reload else "INCREMENTAL"

        ignored = self.db.get_ignored_symbols()

        # 1. Symbol Merge & Clean
        if specific_symbols:
            raw_symbols = set(specific_symbols)
            logger.info(f"[{mode_label}] Explizite Liste: {len(raw_symbols)} Symbole.")
        else:
            index_syms = set(ExchangeSymbol().all)
            db_syms = set(self.db.get_all_known_symbols())
            raw_symbols = index_syms.union(db_syms)
            logger.info(f"[{mode_label}] Auto-Discovery: {len(raw_symbols)} Symbole.")

        # Filter Ignored
        symbols = list(raw_symbols - ignored)
        if len(raw_symbols) != len(symbols):
            logger.info(
                f"Ignoriere {len(raw_symbols) - len(symbols)} Symbole (Blacklist)."
            )

        if not symbols:
            logger.warning("Keine Symbole zu verarbeiten.")
            return

        # 2. Download
        start_date = (
            "2023-01-01"
            if full_reload
            else (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        )
        total_records = 0

        # Batch Loop
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i : i + BATCH_SIZE]
            try:
                records, failures = self.fetch_and_process_batch(batch, start_date)
                count = self.db.upsert_bulk(records)
                total_records += count

                # Handling Failures (Dead Letter Logic)
                if failures and full_reload:
                    # Wenn bei einem Full Reload keine Daten kommen, ist das Symbol vermutlich tot.
                    for fail_sym in failures:
                        self.db.ignore_symbol(
                            fail_sym, "No Data found during Full Reload"
                        )

                if (i // BATCH_SIZE) % 5 == 0:
                    logger.info(
                        f"Progress: {i + len(batch)}/{len(symbols)} verarbeitet."
                    )
            except Exception as e:
                logger.error(f"Fehler im Batch ab Index {i}: {e}")

        duration = datetime.now() - start_time
        logger.info(f"Update fertig: {total_records} Records in {duration}.")

    def perform_gap_check(self) -> None:
        """Prüft auf Lücken. (Locking passiert innerhalb update_market_data bei Repair)"""
        # Hinweis: perform_gap_check selbst braucht keinen Lock, da es nur liest,
        # aber wenn es update aufruft, greift dort der Lock.
        logger.info("Führe Gap-Check durch...")
        threshold_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

        try:
            outdated = self.db.get_outdated_symbols(threshold_date)
            if outdated:
                logger.warning(
                    f"Gap Check: {len(outdated)} Symbole veraltet. Starte Repair."
                )
                # Hier rufen wir update auf, welches gelockt ist.
                # Achtung: Da wir bereits im Thread sein könnten, ist RLock besser oder wir vertrauen auf Queueing.
                # Da unser Lock blocking=False ist, würde es hier fehlschlagen, wenn update schon läuft.
                # Das ist gewollt! Wir wollen keine parallelen Updates.
                self.update_market_data(full_reload=True, specific_symbols=outdated)
            else:
                logger.info("Gap Check: Alles aktuell.")
        except Exception as e:
            logger.error(f"Gap Check Error: {e}")

    def fetch_and_process_batch(
        self, symbols: SymbolList, start_date: str
    ) -> tuple[List[MarketRecord], List[str]]:
        """Gibt Records UND Liste fehlgeschlagener Symbole zurück."""
        if not symbols:
            return [], []

        records = []
        found_symbols = set()

        try:
            # shared_errors='ignore' verhindert, dass yfinance Exceptions wirft bei leeren Daten
            df = yf.download(
                " ".join(symbols),
                start=start_date,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
                timeout=REQUEST_TIMEOUT,
                ignore_tz=True,
            )
        except Exception as e:
            logger.error(f"yfinance Batch Error: {e}")
            return [], symbols  # Alle fehlgeschlagen

        if df.empty:
            return [], symbols

        def process_single_df(sym: str, d: pd.DataFrame):
            d.columns = d.columns.str.lower()
            d = d.dropna(subset=["open", "high", "low", "close"])
            d = d[(d["open"] > 0) & (d["high"] >= d["low"]) & (d["close"] > 0)]

            if not d.empty:
                found_symbols.add(sym)
                for ts, row in d.iterrows():
                    records.append(
                        {
                            "symbol": sym,
                            "date": ts.strftime("%Y-%m-%d"),
                            "provider": "yahoo",
                            "timeframe": "1D",
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": float(row.get("volume", 0)),
                        }
                    )

        if len(symbols) == 1:
            process_single_df(symbols[0], df)
        else:
            for sym in symbols:
                try:
                    if sym in df.columns.get_level_values(0):
                        process_single_df(sym, df[sym].copy())
                except KeyError:
                    pass

        # Welche Symbole aus dem Batch haben gar keine Daten geliefert?
        failures = list(set(symbols) - found_symbols)
        return records, failures


# --------------------------------------------------------------------------
# Teil 3: Data Validator
# --------------------------------------------------------------------------


class DataValidator:
    def __init__(self, service: MarketDataService):
        self.service = service
        self.db = service.db

    def run_logical_checks(self):
        logger.info("Validierung: Starte Logische Checks...")
        checks = {
            "neg_price": "low < 0",
            "zero_close": "close = 0",
            "inv_ohlc": "high < low",
        }
        anomalies = []

        with self.db._get_conn() as conn:
            for name, condition in checks.items():
                sql = f"SELECT DISTINCT symbol FROM market_prices WHERE {condition}"
                try:
                    rows = conn.execute(sql).fetchall()
                    if rows:
                        anomalies.append(f"{name}: {len(rows)} Symbole")
                except Exception:
                    pass

        if anomalies:
            logger.warning(f"Anomalien: {', '.join(anomalies)}")
        else:
            logger.info("Validierung: Logik OK.")

    def run_spot_check(self, sample_size: int = 20, lookback_days: int = 5):
        logger.info(f"Validierung: Spot Check (n={sample_size})...")
        all_syms = self.db.get_all_known_symbols()
        if not all_syms:
            return

        # Ignorierte Symbole nicht checken
        ignored = self.db.get_ignored_symbols()
        valid_syms = list(set(all_syms) - ignored)

        sample = random.sample(valid_syms, min(len(valid_syms), sample_size))

        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%d"
        )
        live_records, _ = self.service.fetch_and_process_batch(sample, start_date)

        errors = []
        for rec in live_records:
            db_candle = self.db.get_candle_at_date(rec["symbol"], rec["date"])
            if db_candle:
                # Close Preis Check
                if (
                    abs(rec["close"] - db_candle["close"])
                    / (db_candle["close"] + 0.001)
                    > 0.005
                ):
                    errors.append(rec["symbol"])

        unique_errors = list(set(errors))
        if unique_errors:
            logger.warning(f"Spot Check Failed: {len(unique_errors)} Symbole. Repair.")
            self.service.update_market_data(
                full_reload=True, specific_symbols=unique_errors
            )
        else:
            logger.info("✅ Spot Check OK.")


# --------------------------------------------------------------------------
# Teil 4: Scheduler
# --------------------------------------------------------------------------

_scheduler = BackgroundScheduler()


def task_daily_routine(db_path: Path):
    svc = MarketDataService(db_path)
    # 1. Update
    svc.update_market_data(full_reload=False)
    # 2. Validate
    val = DataValidator(svc)
    val.run_logical_checks()
    val.run_spot_check(sample_size=30)
    # 3. Gap Check
    svc.perform_gap_check()


def task_maintenance(db_path: Path):
    """Wöchentliche DB-Hygiene."""
    db = MarketDatabase(db_path)
    db.optimize_db()


class MarketDataScheduler:
    def __init__(self, db_path: Path, run_on_start: bool = True):
        self.db_path = db_path
        self.run_on_start = run_on_start

    def start(self):
        if not _scheduler.running:
            # Tägliches Update
            _scheduler.add_job(
                task_daily_routine,
                args=[self.db_path],
                trigger=CronTrigger(
                    hour=17, minute=0, timezone=pytz.timezone("America/New_York")
                ),
                id="md_daily",
                replace_existing=True,
            )

            # Wöchentliche Maintenance (Sonntags)
            _scheduler.add_job(
                task_maintenance,
                args=[self.db_path],
                trigger=CronTrigger(day_of_week="sun", hour=4, minute=0),
                id="md_maintenance",
                replace_existing=True,
            )

            _scheduler.start()
            logger.info("Scheduler gestartet.")

            if self.run_on_start:
                threading.Thread(
                    target=task_daily_routine, args=[self.db_path], daemon=True
                ).start()

    def stop(self):
        _scheduler.shutdown()
