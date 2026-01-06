import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
import pytz
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ..tools.symbol_lists import ExchangeSymbol

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Teil A: Deine Helper Funktionen (angepasst für Klassennutzung)
# --------------------------------------------------------------------------


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Entfernt Zeilen mit ungültigen OHLCV Daten."""
    # Sicherstellen, dass Spalten lowercase sind
    df.columns = df.columns.str.lower()
    return df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        # Volume kann 0 sein bei manchen Feiertagen/Assets, daher >= 0
        & (df["volume"] >= 0)
        # Plausibilitäts-Checks
        & (df["high"] >= df["low"])
    ].copy()


def convert_to_flat_format(
    df: pd.DataFrame, provider: str, timeframe: str = "1D"
) -> pd.DataFrame:
    """
    Wandelt den yfinance MultiIndex in ein flaches Format für die DB um.
    Fügt die 'provider' Spalte hinzu.
    """
    if df.empty:
        return df

    # Falls wir nur ein Symbol geladen haben, ist es kein MultiIndex in den Columns
    if isinstance(df.columns, pd.MultiIndex):
        # Stack moves Symbol into Index (Date, Symbol)
        df = df.stack(level=0, future_stack=True)
    else:
        # Bei einem Symbol müssen wir das Symbol manuell hinzufügen,
        # da yfinance es dann oft weglässt im Index
        pass

    # Reset Index um Date und Symbol als Spalten zu haben
    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Spalten bereinigen
    df.columns = df.columns.str.lower()

    # Provider hinzufügen
    df["provider"] = provider

    # Timeframe hinzufügen
    df["timeframe"] = timeframe

    # Datentypen erzwingen
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Benötigte Spalten filtern/ordnen
    cols = ["symbol", "date", "provider", "open", "high", "low", "close", "volume"]
    # Falls 'ticker' statt 'symbol' heißt (passiert je nach Pandas version)
    if "ticker" in df.columns:
        df = df.rename(columns={"ticker": "symbol"})

    return df[cols]


# --------------------------------------------------------------------------
# Teil B: Die Datenbank Klasse für stocks.db
# --------------------------------------------------------------------------


class MarketDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
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
        """
        with self._get_conn() as conn:
            conn.executescript(schema)

    def get_last_entry(self, symbol: str, provider: str, timeframe: str = "1D"):
        """Holt das letzte Datum für spezifischen Timeframe."""
        sql = """
        SELECT date, close FROM market_prices
        WHERE symbol = ? AND provider = ? AND timeframe = ?
        ORDER BY date DESC LIMIT 1
        """
        with self._get_conn() as conn:
            row = conn.execute(sql, (symbol, provider, timeframe)).fetchone()
            if row:
                return {"date": row["date"], "close": row["close"]}
            return None

    def delete_symbol_data(self, symbol: str, provider: str, timeframe: str = "1D"):
        """Löscht Daten nur für diesen Timeframe."""
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM market_prices WHERE symbol = ? AND provider = ? AND timeframe = ?",
                (symbol, provider, timeframe),
            )

    def upsert_data(self, df: pd.DataFrame):
        if df.empty:
            return

        data = df.to_dict(orient="records")
        # SQL erweitert um timeframe
        sql = """
        INSERT OR REPLACE INTO market_prices
        (symbol, date, provider, timeframe, open, high, low, close, volume)
        VALUES (:symbol, :date, :provider, :timeframe, :open, :high, :low, :close, :volume)
        """
        with self._get_conn() as conn:
            conn.executemany(sql, data)


# --------------------------------------------------------------------------
# Teil C: Der Worker (Scheduler & Logik)
# --------------------------------------------------------------------------


class MarketDataWorker:
    def __init__(self, db_path: Path, run_on_start: bool = True):
        self.db = MarketDatabase(db_path)

        # Scheduler einrichten
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            self.run_update_job,
            trigger=CronTrigger(
                hour=17, minute=0, timezone=pytz.timezone("America/New_York")
            ),
            id="market_update_job",
            replace_existing=True,
        )

        if run_on_start:
            # Wir planen einen Job für "jetzt gleich" ('date' trigger)
            # APScheduler führt diesen im Hintergrund-Thread aus,
            # daher startet Flask sofort weiter durch.
            self.scheduler.add_job(
                self.run_update_job,
                trigger="date",
                run_date=datetime.now(),
                id="market_update_startup",
            )

    def start(self):
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Market Data Scheduler gestartet (18:00 NYC).")

    def stop(self):
        self.scheduler.shutdown()

    def _load_symbols(self) -> List[str]:
        symbols = ExchangeSymbol().all
        if symbols is None:
            logger.warning("Keine Symbole gefunden.")
            return []
        return symbols

    def run_update_job(self):
        """Hauptfunktion, die täglich aufgerufen wird."""
        logger.info("Starte tägliches Market Data Update...")
        symbols = self._load_symbols()
        if not symbols:
            return

        # Wir nutzen Yahoo als Provider
        PROVIDER = "yahoo"

        for symbol in symbols:
            try:
                self._process_symbol(symbol, PROVIDER)
            except Exception as e:
                logger.error(f"Fehler bei Update für {symbol}: {e}")

        logger.info("Market Data Update abgeschlossen.")

    def _process_symbol(self, symbol: str, provider: str):
        TIMEFRAME = "1D"

        # 1. Check was wir in der DB haben
        last_entry = self.db.get_last_entry(symbol, provider)

        start_date = "2020-01-01"
        is_full_update = False

        if last_entry:
            # Wir laden ein paar Tage Überlappung, um Splits zu prüfen
            # String Datum zu Objekt wandeln
            last_date_obj = datetime.strptime(last_entry["date"], "%Y-%m-%d")
            overlap_date = last_date_obj - timedelta(days=5)
            start_date = overlap_date.strftime("%Y-%m-%d")

        # 2. Download via yfinance
        # auto_adjust=True ist wichtig für Split-Bereinigung!
        df = yf.download(
            symbol, start=start_date, progress=False, auto_adjust=True, rounding=True
        )

        if df.empty:
            logger.warning(f"Keine Daten für {symbol} gefunden.")
            return

        # 3. Bereinigen & Formatieren
        # Flatten yfinance structure
        df = df.reset_index()
        # Falls MultiIndex Columns (bei single download manchmal anders), fixen:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Lowercase und Formatierung für unsere Helper
        df.columns = df.columns.str.lower()
        df = clean_ohlcv(df)

        # Spalte 'symbol' und 'provider' hinzufügen für den DB Import
        df["symbol"] = symbol
        df["provider"] = provider
        df["timeframe"] = TIMEFRAME

        # Datum String sicherstellen
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        # 4. SPLIT / CHANGE DETECTION
        if last_entry:
            # Suche den Eintrag in den neuen Daten, der dem letzten DB-Eintrag entspricht
            mask = df["date"] == last_entry["date"]
            if mask.any():
                new_price = df.loc[mask, "close"].values[0]
                old_price = last_entry["close"]

                # Wenn der Preis signifikant abweicht (> 1%), gehen wir von einem Split/Korrektur aus
                # (oder yfinance hat rückwirkend Daten geändert)
                if abs(new_price - old_price) / old_price > 0.01:
                    logger.warning(
                        f"Split/Korrektur erkannt bei {symbol} ({last_entry['date']}). Old: {old_price}, New: {new_price}. Full Reload."
                    )
                    is_full_update = True
                else:
                    # Alles okay, wir speichern nur Daten, die NEUER sind als der letzte DB Eintrag
                    df = df[df["date"] > last_entry["date"]]

        # 5. Speichern
        if is_full_update:
            # Wenn Split erkannt: Alles löschen, dann alles neu speichern
            # Dafür müssen wir aber den KOMPLETTEN Download machen, falls wir oben nur
            # den Overlap geladen hatten.
            # Da wir oben start="2020..." oder overlap hatten, müssen wir prüfen.
            # Sicherheitshalber bei Full Update nochmal ab 2020 laden, falls wir im Overlap-Modus waren.
            if start_date != "2020-01-01":
                df = yf.download(
                    symbol,
                    start="2020-01-01",
                    progress=False,
                    auto_adjust=True,
                    rounding=True,
                )
                df = df.reset_index()
                df.columns = df.columns.str.lower()
                df = clean_ohlcv(df)
                df["symbol"] = symbol
                df["provider"] = provider
                df["timeframe"] = TIMEFRAME
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

            self.db.delete_symbol_data(symbol, provider)
            self.db.upsert_data(df)
            logger.info(f"{symbol}: Full Reload durchgeführt.")
        else:
            if not df.empty:
                self.db.upsert_data(df)
                logger.info(f"{symbol}: {len(df)} neue Tage angefügt.")
            else:
                logger.debug(f"{symbol}: Keine neuen Daten.")
