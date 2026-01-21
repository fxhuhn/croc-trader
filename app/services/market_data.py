import logging
import random
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ..tools.symbol_lists import ExchangeSymbol

logger = logging.getLogger(__name__)

# Typ-Aliase
type SymbolList = List[str]
type MarketRecord = Dict[str, Any]

# Konstanten
BATCH_SIZE = 500
DB_TIMEOUT = 60  # Erhöht für große Operationen
REQUEST_TIMEOUT = 30


# --------------------------------------------------------------------------
# Teil 1: Datenbank Layer
# --------------------------------------------------------------------------


class MarketDatabase:
    """Verwaltet den SQLite-Zugriff. Optimiert für Bulk-Operationen."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        # Timeout erhöht, um Locks bei großen Schreibvorgängen zu tolerieren
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
        CREATE INDEX IF NOT EXISTS idx_market_lookup ON market_prices(symbol, date);
        CREATE INDEX IF NOT EXISTS idx_market_date ON market_prices(date);
        """
        try:
            with self._get_conn() as conn:
                conn.executescript(schema)
        except sqlite3.Error as e:
            logger.critical(f"DB Schema Init failed: {e}")
            raise

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

    def get_outdated_symbols(
        self, reference_date: str, provider: str = "yahoo"
    ) -> SymbolList:
        """Findet Symbole, deren letztes Datum älter ist als reference_date."""
        # Hier ist GROUP BY notwendig, aber wir filtern erst nach Provider
        # Der Index (symbol, date) hilft hier oft nicht direkt, wenn provider nicht im Index ist.
        # Optimierung: Wir vertrauen auf die PK-Struktur.
        sql = "SELECT symbol FROM market_prices WHERE provider = ? GROUP BY symbol HAVING MAX(date) < ?"
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(sql, (provider, reference_date))
                return [row["symbol"] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Gap Check Query fehlgeschlagen: {e}")
            return []

    def get_all_known_symbols(self) -> SymbolList:
        """Holt ALLE Symbole aus der DB (Legacy Support)."""
        try:
            with self._get_conn() as conn:
                cursor = conn.execute("SELECT DISTINCT symbol FROM market_prices")
                return [row["symbol"] for row in cursor.fetchall()]
        except sqlite3.Error:
            return []

    def get_candle_at_date(self, symbol: str, date: str) -> Optional[Dict[str, float]]:
        """Holt die komplette OHLCV Kerze für einen Tag."""
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
    """Business Logik für Download und Sync."""

    def __init__(self, db_path: Path):
        self.db = MarketDatabase(db_path)

    def update_market_data(
        self, full_reload: bool = False, specific_symbols: Optional[SymbolList] = None
    ) -> None:
        start_time = datetime.now()
        mode_label = "FULL RELOAD" if full_reload else "INCREMENTAL"

        # --- 1. Symbol Merge Logik ---
        if specific_symbols:
            symbols = list(set(specific_symbols))
            logger.info(
                f"[{mode_label}] Starte Update für {len(symbols)} explizite Symbole."
            )
        else:
            index_syms = ExchangeSymbol().all
            db_syms = self.db.get_all_known_symbols()
            unique_pool = set(index_syms + db_syms)
            symbols = list(unique_pool)
            logger.info(
                f"[{mode_label}] Starte Update für {len(symbols)} Symbole (Index + Legacy)."
            )

        if not symbols:
            logger.warning("Keine Symbole gefunden.")
            return

        # --- 2. Download ---
        start_date = (
            "2023-01-01"
            if full_reload
            else (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        )
        total_records = 0

        total_batches = (len(symbols) // BATCH_SIZE) + 1
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i : i + BATCH_SIZE]
            try:
                records = self.fetch_and_process_batch(batch, start_date)
                count = self.db.upsert_bulk(records)
                total_records += count

                if (i // BATCH_SIZE) % 5 == 0:
                    logger.info(
                        f"Progress: {i + len(batch)}/{len(symbols)} verarbeitet."
                    )
            except Exception as e:
                logger.error(f"Fehler im Batch ab Index {i}: {e}")

        duration = datetime.now() - start_time
        logger.info(f"Update fertig: {total_records} Records in {duration}.")

    def perform_gap_check(self) -> None:
        """Prüft auf Lücken und repariert automatisch."""
        logger.info("Führe Gap-Check durch...")
        # Daten sollten bis vor 3 Tagen vorliegen (Wochenende/Feiertage Puffer)
        threshold_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

        try:
            outdated = self.db.get_outdated_symbols(threshold_date)
            if outdated:
                logger.warning(
                    f"Gap Check: {len(outdated)} Symbole veraltet (Letzter Stand < {threshold_date}). Starte Repair."
                )
                self.update_market_data(full_reload=True, specific_symbols=outdated)
            else:
                logger.info("Gap Check: Alles aktuell.")
        except Exception as e:
            logger.error(f"Gap Check abgestürzt: {e}")

    def fetch_and_process_batch(
        self, symbols: SymbolList, start_date: str
    ) -> List[MarketRecord]:
        if not symbols:
            return []

        try:
            df = yf.download(
                " ".join(symbols),
                start=start_date,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
                timeout=REQUEST_TIMEOUT,
            )
        except Exception as e:
            logger.error(f"yfinance Fehler: {e}")
            return []

        if df.empty:
            return []

        records = []

        def process_single_df(sym: str, d: pd.DataFrame):
            d.columns = d.columns.str.lower()
            d = d.dropna(subset=["open", "high", "low", "close"])
            d = d[(d["open"] > 0) & (d["high"] >= d["low"]) & (d["close"] > 0)]

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
                    continue

        return records


# --------------------------------------------------------------------------
# Teil 3: Data Validator (Optimiert)
# --------------------------------------------------------------------------


class DataValidator:
    """Verantwortlich für Qualitätsprüfung der Daten."""

    def __init__(self, service: MarketDataService):
        self.service = service
        self.db = service.db

    def run_logical_checks(self):
        """
        SQL-basierte Prüfung auf unmögliche Werte.
        OPTIMIERT: Nutzt WHERE statt GROUP BY/HAVING für massive Performance-Steigerung.
        """
        logger.info("Validierung: Starte Logische Checks...")

        # Mapping: Name -> SQL WHERE Bedingung (NICHT Aggregation!)
        checks = {
            "neg_price": "low < 0",
            "zero_close": "close = 0",
            "inv_ohlc": "high < low",
        }

        anomalies = []

        with self.db._get_conn() as conn:
            for name, condition in checks.items():
                # Wir suchen nur UNIQUE Symbole, die mindestens EINEN fehlerhaften Eintrag haben.
                # Das vermeidet das Zählen der gesamten Tabelle.
                sql = f"SELECT DISTINCT symbol FROM market_prices WHERE {condition}"
                try:
                    logger.info(f"Prüfe Regel: {name}...")
                    cursor = conn.execute(sql)
                    rows = cursor.fetchall()
                    if rows:
                        count = len(rows)
                        sample = [r["symbol"] for r in rows[:5]]
                        anomalies.append(
                            f"{name}: {count} Symbole betroffen (z.B. {', '.join(sample)})"
                        )
                except Exception as e:
                    logger.error(f"Fehler bei Check '{name}': {e}")

        if anomalies:
            logger.warning("Logik-Fehler gefunden:\n" + "\n".join(anomalies))
        else:
            logger.info("Validierung: Keine logischen Fehler gefunden.")

    def run_spot_check(self, sample_size: int = 20, lookback_days: int = 5):
        """Zieht Stichprobe und vergleicht komplettes OHLCV mit Live-Daten."""
        logger.info(f"Validierung: OHLCV Spot Check (n={sample_size})...")

        all_syms = self.db.get_all_known_symbols()
        if not all_syms:
            logger.info("Spot Check übersprungen (Keine Symbole in DB).")
            return

        sample = random.sample(all_syms, min(len(all_syms), sample_size))

        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%d"
        )
        live_records = self.service.fetch_and_process_batch(sample, start_date)

        errors = []
        checked_count = 0

        for rec in live_records:
            sym = rec["symbol"]
            date = rec["date"]

            db_candle = self.db.get_candle_at_date(sym, date)
            if not db_candle:
                continue

            checked_count += 1
            is_valid = True
            msg_parts = []

            # Preis-Check (0.5% Toleranz)
            for f in ["open", "high", "low", "close"]:
                live_v = rec[f]
                db_v = db_candle[f]
                if abs(live_v - db_v) / (db_v + 0.0001) > 0.005:
                    is_valid = False
                    msg_parts.append(f"{f.upper()}: Live={live_v:.2f}/DB={db_v:.2f}")

            if not is_valid:
                errors.append(sym)
                logger.warning(f"❌ Abweichung {sym} am {date}: {', '.join(msg_parts)}")

        unique_errors = list(set(errors))
        if unique_errors:
            logger.warning(
                f"Spot Check Failed für {len(unique_errors)} Symbole. Trigger Repair..."
            )
            self.service.update_market_data(
                full_reload=True, specific_symbols=unique_errors
            )
        else:
            logger.info(f"✅ Spot Check OK ({checked_count} Datenpunkte geprüft).")


# --------------------------------------------------------------------------
# Teil 4: Scheduler & Task Wrapper
# --------------------------------------------------------------------------

_scheduler = BackgroundScheduler()


def task_daily_routine(db_path: Path):
    """Update + Validierung."""
    svc = MarketDataService(db_path)
    svc.update_market_data(full_reload=False)

    val = DataValidator(svc)
    val.run_logical_checks()
    val.run_spot_check(sample_size=30)

    # Gap Check auch hier am Ende
    svc.perform_gap_check()


def task_gap_repair(db_path: Path):
    """Findet Löcher und stopft sie (Nachts)."""
    svc = MarketDataService(db_path)
    svc.perform_gap_check()


class MarketDataScheduler:
    def __init__(self, db_path: Path, run_on_start: bool = True):
        self.db_path = db_path
        self.run_on_start = run_on_start

    def start(self):
        if not _scheduler.running:
            # 1. Haupt-Update (Mo-Fr 18:00 NYC)
            _scheduler.add_job(
                task_daily_routine,
                args=[self.db_path],
                trigger=CronTrigger(
                    hour=17, minute=0, timezone=pytz.timezone("America/New_York")
                ),
                id="md_daily",
                replace_existing=True,
            )

            # 2. Gap Check (Nachts 02:00 NYC als Safety Net)
            _scheduler.add_job(
                task_gap_repair,
                args=[self.db_path],
                trigger=CronTrigger(
                    hour=2, minute=0, timezone=pytz.timezone("America/New_York")
                ),
                id="md_gap",
                replace_existing=True,
            )

            _scheduler.start()
            logger.info("Market Data Scheduler gestartet.")

            if self.run_on_start:
                threading.Thread(
                    target=task_daily_routine, args=[self.db_path], daemon=True
                ).start()

    def stop(self):
        _scheduler.shutdown()
