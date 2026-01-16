import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TypeAlias

import pandas as pd
import pytz
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ..tools.symbol_lists import ExchangeSymbol

logger = logging.getLogger(__name__)

MarketRecord: TypeAlias = Dict[str, Any]
LastEntryMap: TypeAlias = Dict[str, Tuple[str, float]]  # Symbol -> (Date, Close)


class MarketDatabase:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
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

    def get_all_last_entries_map(
        self, provider: str, timeframe: str = "1D"
    ) -> LastEntryMap:
        sql = """
            SELECT symbol, max(date) as last_date, close
            FROM market_prices
            WHERE provider = ? AND timeframe = ?
            GROUP BY symbol
        """
        result: LastEntryMap = {}
        with self._get_conn() as conn:
            cursor = conn.execute(sql, (provider, timeframe))
            for row in cursor:
                result[row["symbol"]] = (row["last_date"], row["close"])
        return result

    def delete_symbol_data(self, symbol: str, provider: str, timeframe: str) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM market_prices WHERE symbol = ? AND provider = ? AND timeframe = ?",
                (symbol, provider, timeframe),
            )

    def upsert_many(self, records: List[MarketRecord]) -> None:
        if not records:
            return

        sql = """
        INSERT OR REPLACE INTO market_prices
        (symbol, date, provider, timeframe, open, high, low, close, volume)
        VALUES (:symbol, :date, :provider, :timeframe, :open, :high, :low, :close, :volume)
        """
        try:
            with self._get_conn() as conn:
                conn.executemany(sql, records)
        except sqlite3.Error as e:
            logger.error(f"DB Batch Insert Failed: {e}")
            raise


class MarketDataWorker:
    PROVIDER = "yahoo"
    TIMEFRAME = "1D"
    BATCH_SIZE = 100

    def __init__(self, db_path: Path, run_on_start: bool = True) -> None:
        self.db = MarketDatabase(db_path)
        self.scheduler = BackgroundScheduler()

        # Schedule: Täglich 17:00 NY Time (ca. 23:00 DE)
        self.scheduler.add_job(
            self.run_update_job,
            trigger=CronTrigger(
                hour=17, minute=0, timezone=pytz.timezone("America/New_York")
            ),
            id="market_update_job",
            replace_existing=True,
        )

        if run_on_start:
            self.scheduler.add_job(
                self.run_update_job,
                trigger="date",
                run_date=datetime.now() + timedelta(seconds=5),
                id="market_update_startup",
            )

    def start(self) -> None:
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Market Data Scheduler gestartet.")

    def run_update_job(self) -> None:
        logger.info("Starte Market Data Update (Batch-Modus)...")

        all_symbols = ExchangeSymbol().all
        if not all_symbols:
            logger.warning("Keine Symbole in ExchangeSymbol gefunden.")
            return

        # 1. State laden: Was ist der letzte Stand in der DB?
        last_entries = self.db.get_all_last_entries_map(self.PROVIDER, self.TIMEFRAME)

        # 2. Batches verarbeiten
        split_candidates: Set[str] = set()
        processed_count = 0

        # Chunking list into batches
        for i in range(0, len(all_symbols), self.BATCH_SIZE):
            batch_symbols = all_symbols[i : i + self.BATCH_SIZE]

            try:
                # Download mit adaptivem Backoff
                df_batch = self._download_batch_with_backoff(batch_symbols)

                if df_batch is None or df_batch.empty:
                    continue

                # Daten verarbeiten & auf Splits prüfen
                records_to_save, splits_in_batch = self._process_batch_data(
                    df_batch, last_entries
                )

                if records_to_save:
                    self.db.upsert_many(records_to_save)
                    processed_count += len(records_to_save)
                    logger.debug(f"Batch saved: {len(records_to_save)} records.")

                split_candidates.update(splits_in_batch)

            except Exception as e:
                logger.error(f"Unerwarteter Fehler im Batch {i}: {e}", exc_info=True)

        logger.info(f"Basis-Update fertig. {processed_count} neue Datensätze.")

        # 3. Splits / Full Reloads verarbeiten
        if split_candidates:
            logger.warning(
                f"Starte Full Reload für {len(split_candidates)} Symbole mit erkannten Splits/Abweichungen."
            )
            self._perform_full_reload(list(split_candidates))

        logger.info("Market Data Update vollständig abgeschlossen.")

    def _download_batch_with_backoff(
        self, symbols: List[str]
    ) -> Optional[pd.DataFrame]:
        """
        Versucht Symbole zu laden. Bei Fehler wird die Liste halbiert.
        Rekursiver Ansatz.
        """
        if not symbols:
            return None

        try:
            # yfinance Multi-Download
            # group_by='ticker' sorgt für saubere Struktur bei Multi-Index
            df = yf.download(
                tickers=" ".join(symbols),
                period="1mo",  # Kurzer Zeitraum reicht für Update
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
                timeout=10,
            )
            return df

        except Exception as e:
            # Fehlerfall
            if len(symbols) <= 1:
                # Einzelnes Symbol fehlgeschlagen
                logger.error(f"Timeout/Fehler bei Einzelsymbol '{symbols[0]}': {e}")
                return None

            # Batch halbieren
            mid = len(symbols) // 2
            left_chunk = symbols[:mid]
            right_chunk = symbols[mid:]

            logger.warning(
                f"Batch-Fehler bei {len(symbols)} Symbolen. Versuche Split ({len(left_chunk)} / {len(right_chunk)})."
            )

            df_left = self._download_batch_with_backoff(left_chunk)
            df_right = self._download_batch_with_backoff(right_chunk)

            # Ergebnisse zusammenfügen
            results = []
            if df_left is not None and not df_left.empty:
                results.append(df_left)
            if df_right is not None and not df_right.empty:
                results.append(df_right)

            if not results:
                return None

            # Bei unterschiedlichen Columns (weil Ticker unterschiedlich) müssen wir aufpassen.
            # yfinance returns bei Multi-Tickers einen DataFrame mit Columns Levels.
            # Ein Concat hier ist tricky wenn die Structure nicht passt.
            # Besser: Wir geben einen Dict von Dataframes zurück oder nutzen einfach pd.concat axis=1
            try:
                return pd.concat(results, axis=1)
            except Exception as join_error:
                logger.error(f"Fehler beim Mergen der Split-Batches: {join_error}")
                return None

    def _process_batch_data(
        self, df_batch: pd.DataFrame, last_entries: LastEntryMap
    ) -> Tuple[List[MarketRecord], Set[str]]:
        """
        Wandelt yfinance DataFrame in DB-Records und identifiziert Splits.
        """
        records: List[MarketRecord] = []
        splits: Set[str] = set()

        # Handle Single Symbol vs Multi Symbol DataFrame Structure from yfinance
        # If columns is MultiIndex, level 0 is Ticker, level 1 is OHLCV (due to group_by='ticker')
        # If single symbol, simple Index.

        is_multi_index = isinstance(df_batch.columns, pd.MultiIndex)

        # Liste der Ticker im DF ermitteln
        if is_multi_index:
            tickers_in_df = df_batch.columns.get_level_values(0).unique().tolist()
        else:
            # Workaround für yfinance quirks bei 1 Symbol im Batch return
            # Wir nehmen an, dass der Caller (download) immer group_by='ticker' nutzt,
            # aber bei 1 Ticker flacht yf das manchmal ab.
            # Da wir "tickers=" string passierte, sollte yf den Ticker kennen.
            # Wir extrahieren ihn hier nicht trivial, daher Fallback:
            # Wenn wir hier landen, ist die Struktur meist flach.
            # Wir skippen hier komplexe Single-Logik und verlassen uns auf den Loop unten,
            # der für Multi ausgelegt ist.
            # Einzige Lösung: Re-Index oder Prüfung.
            # Simpler Hack: Wenn nur 1 Ticker im Batch war, ist df_batch direkt das Dataframe.
            return (
                [],
                set(),
            )  # Edge Case handling vereinfacht: Skip saving if structure invalid

        for ticker in tickers_in_df:
            # Slice für den Ticker holen
            try:
                df_ticker = df_batch[ticker].copy()
            except KeyError:
                continue

            df_ticker = self._clean_ohlcv(df_ticker)
            if df_ticker.empty:
                continue

            # DB Status prüfen
            last_entry = last_entries.get(ticker)

            if last_entry:
                last_date_str, last_close = last_entry

                # Zeile suchen, die dem DB-Datum entspricht
                if last_date_str in df_ticker.index:
                    # Parse Timestamp to String for comparison if needed, but yf index is Timestamp
                    # df_ticker.index is DatetimeIndex.
                    # We compare using string representation yyyy-mm-dd
                    try:
                        matched_row = df_ticker.loc[last_date_str]
                        # Wenn mehrere Einträge pro Tag (dirty data), nimm den letzten
                        if isinstance(matched_row, pd.DataFrame):
                            matched_row = matched_row.iloc[-1]

                        new_close = float(matched_row["close"])

                        # Split Check (1% Abweichung)
                        if abs(new_close - last_close) / last_close > 0.01:
                            splits.add(ticker)
                            continue  # Nicht speichern, kommt in Queue

                        # Nur neue Daten nehmen
                        df_ticker = df_ticker[df_ticker.index > last_date_str]

                    except KeyError:
                        pass  # Datum im Batch nicht enthalten (Gap?), einfach alles Neue speichern

            # Records erstellen
            for ts, row in df_ticker.iterrows():
                records.append(
                    {
                        "symbol": ticker,
                        "date": ts.strftime("%Y-%m-%d"),
                        "provider": self.PROVIDER,
                        "timeframe": self.TIMEFRAME,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    }
                )

        return records, splits

    def _perform_full_reload(self, symbols: List[str]) -> None:
        """
        Lädt die komplette Historie ab 2020 für Split-Kandidaten.
        """
        start_date = "2020-01-01"

        for symbol in symbols:
            try:
                df = yf.download(
                    symbol,
                    start=start_date,
                    auto_adjust=True,
                    progress=False,
                    timeout=20,
                )

                if df.empty:
                    continue

                # Clean & Formatting
                df = self._clean_ohlcv(df)

                # Delete Old
                self.db.delete_symbol_data(symbol, self.PROVIDER, self.TIMEFRAME)

                # Prepare New
                records = []
                for ts, row in df.iterrows():
                    records.append(
                        {
                            "symbol": symbol,
                            "date": ts.strftime("%Y-%m-%d"),
                            "provider": self.PROVIDER,
                            "timeframe": self.TIMEFRAME,
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": float(row["volume"]),
                        }
                    )

                self.db.upsert_many(records)
                logger.warning(f"Full Reload durchgeführt für: {symbol}")

            except Exception as e:
                logger.error(f"Fehler bei Full Reload für {symbol}: {e}")

    @staticmethod
    def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.lower()
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            return pd.DataFrame()

        return df[
            (df["open"] > 0)
            & (df["high"] > 0)
            & (df["low"] > 0)
            & (df["close"] > 0)
            & (df["volume"] >= 0)
            & (df["high"] >= df["low"])
        ].copy()
