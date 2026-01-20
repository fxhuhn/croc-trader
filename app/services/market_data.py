import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        # WAL Mode für bessere Concurrency (Lesen blockiert Schreiben nicht)
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
            conn.commit()

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
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"DB Batch Insert Failed: {e}")
            raise


class MarketDataWorker:
    PROVIDER = "yahoo"
    TIMEFRAME = "1D"
    BATCH_SIZE = 100
    MAX_WORKERS = 2

    def __init__(self, db_path: Path, run_on_start: bool = True) -> None:
        self.db = MarketDatabase(db_path)
        self.scheduler = BackgroundScheduler()

        # Schedule 1: Regular Update (Last 30 days) - 17:00 NY Time
        self.scheduler.add_job(
            self.run_update_job,
            trigger=CronTrigger(
                hour=17, minute=0, timezone=pytz.timezone("America/New_York")
            ),
            id="market_update_job",
            replace_existing=True,
        )

        # Schedule 2: Gap Check & Repair - 02:00 European Time (Berlin)
        # This checks for any stocks lagging behind and forces a full reload for them.
        self.scheduler.add_job(
            self.run_gap_check,
            trigger=CronTrigger(
                hour=2, minute=0, timezone=pytz.timezone("Europe/Berlin")
            ),
            id="market_gap_check_job",
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

    def run_gap_check(self) -> Dict[str, Any]:
        """
        Checks all stocks against the latest available date in the DB.
        Triggers a full reload for any missing or lagging symbols.
        Returns a summary dictionary.
        """
        logger.info("Starting Daily Market Gap Check...")

        last_entries = self.db.get_all_last_entries_map(self.PROVIDER, self.TIMEFRAME)

        # If DB is empty, we can't determine a target date.
        # (Alternatively, could default to yesterday, but let's assume at least some data exists)
        if not last_entries:
            logger.warning("Gap Check Skipped: No market data found in DB.")
            return {"status": "skipped", "reason": "empty_db"}

        # Determine the target date (max date in DB)
        all_dates = [meta[0] for meta in last_entries.values()]
        target_date = max(all_dates)

        all_symbols = ExchangeSymbol().all
        outdated_symbols = []
        stock_status_map = {}

        for symbol in all_symbols:
            entry = last_entries.get(symbol)

            if not entry:
                # Symbol missing completely
                stock_status_map[symbol] = "MISSING"
                outdated_symbols.append(symbol)
            else:
                last_date = entry[0]
                stock_status_map[symbol] = last_date

                if last_date < target_date:
                    outdated_symbols.append(symbol)

        if outdated_symbols:
            logger.info(
                f"Gap Check: Found {len(outdated_symbols)} outdated symbols. Scheduling reload."
            )

            # Schedule the heavy reload job
            self.scheduler.add_job(
                self._perform_full_reload,
                args=[outdated_symbols],
                trigger="date",
                run_date=datetime.now(),
                id="auto_gap_fill_job",
                replace_existing=True,
            )
        else:
            logger.info("Gap Check: All symbols are up to date.")

        return {
            "status": "queued" if outdated_symbols else "synced",
            "target_max_date": target_date,
            "outdated_count": len(outdated_symbols),
            "triggered_symbols": outdated_symbols,
            "all_stocks_status": stock_status_map,
        }

    def run_update_job(self) -> None:
        logger.info("Starte Market Data Update (High-Performance Mode)...")

        all_symbols = ExchangeSymbol().all
        if not all_symbols:
            logger.warning("Keine Symbole gefunden.")
            return

        # 1. State laden (vermeidet N+1 Queries)
        last_entries = self.db.get_all_last_entries_map(self.PROVIDER, self.TIMEFRAME)

        # 2. Batches vorbereiten
        batches = [
            all_symbols[i : i + self.BATCH_SIZE]
            for i in range(0, len(all_symbols), self.BATCH_SIZE)
        ]

        split_candidates: Set[str] = set()
        processed_records = 0
        processed_batches = 0

        # 3. Parallel Processing Pipeline
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            future_to_idx = {
                executor.submit(self._download_batch_with_backoff, batch): i
                for i, batch in enumerate(batches)
            }

            for future in as_completed(future_to_idx):
                batch_idx = future_to_idx[future]
                try:
                    df_batch = future.result()

                    if df_batch is None or df_batch.empty:
                        continue

                    records, splits = self._process_batch_data(df_batch, last_entries)

                    if records:
                        self.db.upsert_many(records)
                        processed_records += len(records)
                        del records

                    split_candidates.update(splits)
                    processed_batches += 1

                    if processed_batches % 5 == 0:
                        logger.info(
                            f"Progress: {processed_batches}/{len(batches)} Batches verarbeitet."
                        )

                except Exception as e:
                    logger.error(
                        f"Kritischer Fehler bei Batch {batch_idx}: {e}", exc_info=True
                    )

        logger.info(
            f"Basis-Update fertig. {processed_records} neue Datensätze importiert."
        )

        # 4. Splits / Full Reloads verarbeiten
        if split_candidates:
            logger.warning(
                f"Starte Full Reload für {len(split_candidates)} Symbole (Splits/Inkonsistenzen)."
            )
            self._perform_full_reload(list(split_candidates))

        logger.info("Market Data Update vollständig abgeschlossen.")

    def _download_batch_with_backoff(
        self, symbols: List[str]
    ) -> Optional[pd.DataFrame]:
        if not symbols:
            return None

        try:
            df = yf.download(
                tickers=" ".join(symbols),
                period="1mo",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
                timeout=15,
            )
            return df

        except Exception as e:
            if len(symbols) <= 1:
                logger.error(f"Timeout/Fehler bei Einzelsymbol '{symbols[0]}': {e}")
                return None

            mid = len(symbols) // 2
            left_chunk = symbols[:mid]
            right_chunk = symbols[mid:]

            logger.warning(
                f"Batch-Fehler bei {len(symbols)} Symbolen. Splitte in {len(left_chunk)} + {len(right_chunk)}."
            )

            df_left = self._download_batch_with_backoff(left_chunk)
            df_right = self._download_batch_with_backoff(right_chunk)

            results = []
            if df_left is not None and not df_left.empty:
                results.append(df_left)
            if df_right is not None and not df_right.empty:
                results.append(df_right)

            if not results:
                return None

            try:
                return pd.concat(results, axis=1)
            except Exception as join_error:
                logger.error(f"Fehler beim Mergen der Split-Batches: {join_error}")
                return None

    def _process_batch_data(
        self, df_batch: pd.DataFrame, last_entries: LastEntryMap
    ) -> Tuple[List[MarketRecord], Set[str]]:
        records: List[MarketRecord] = []
        splits: Set[str] = set()

        is_multi_index = isinstance(df_batch.columns, pd.MultiIndex)

        if is_multi_index:
            tickers_in_df = df_batch.columns.get_level_values(0).unique().tolist()
        else:
            return [], set()

        for ticker in tickers_in_df:
            try:
                df_ticker = df_batch[ticker]
            except KeyError:
                continue

            df_ticker = self._clean_ohlcv(df_ticker.copy())
            if df_ticker.empty:
                continue

            last_entry = last_entries.get(ticker)

            if last_entry:
                last_date_str, last_close = last_entry

                if last_date_str in df_ticker.index:
                    try:
                        matched_row = df_ticker.loc[last_date_str]
                        if isinstance(matched_row, pd.DataFrame):
                            matched_row = matched_row.iloc[-1]

                        new_close = float(matched_row["close"])

                        if abs(new_close - last_close) / last_close > 0.01:
                            splits.add(ticker)
                            continue

                        df_ticker = df_ticker[df_ticker.index > last_date_str]

                    except KeyError:
                        pass

            if df_ticker.empty:
                continue

            for row in df_ticker.itertuples():
                records.append(
                    {
                        "symbol": ticker,
                        "date": row.Index.strftime("%Y-%m-%d"),
                        "provider": self.PROVIDER,
                        "timeframe": self.TIMEFRAME,
                        "open": float(row.open),
                        "high": float(row.high),
                        "low": float(row.low),
                        "close": float(row.close),
                        "volume": float(row.volume),
                    }
                )

        return records, splits

    def _perform_full_reload(self, symbols: List[str]) -> None:
        start_date = "2020-01-01"
        logger.info(f"Starting Full Reload for {len(symbols)} symbols...")

        for symbol in symbols:
            try:
                df = yf.download(
                    symbol,
                    start=start_date,
                    auto_adjust=True,
                    progress=False,
                    timeout=20,
                )

                df = self._clean_ohlcv(df)
                if df.empty:
                    continue

                self.db.delete_symbol_data(symbol, self.PROVIDER, self.TIMEFRAME)

                records = []
                for row in df.itertuples():
                    records.append(
                        {
                            "symbol": symbol,
                            "date": row.Index.strftime("%Y-%m-%d"),
                            "provider": self.PROVIDER,
                            "timeframe": self.TIMEFRAME,
                            "open": float(row.open),
                            "high": float(row.high),
                            "low": float(row.low),
                            "close": float(row.close),
                            "volume": float(row.volume),
                        }
                    )

                self.db.upsert_many(records)
                logger.info(f"Full Reload success: {symbol}")

            except Exception as e:
                logger.error(f"Fehler bei Full Reload für {symbol}: {e}")

        logger.info("Full Reload batch completed.")

    @staticmethod
    def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            found_level = False
            for i in range(df.columns.nlevels):
                level_vals = df.columns.get_level_values(i)
                if "Close" in level_vals or "close" in level_vals:
                    df.columns = level_vals
                    found_level = True
                    break

            if not found_level:
                df.columns = df.columns.get_level_values(0)

        df.columns = df.columns.str.lower()
        mask = (
            (df["open"] > 0)
            & (df["high"] > 0)
            & (df["low"] > 0)
            & (df["close"] > 0)
            & (df["volume"] >= 0)
            & (df["high"] >= df["low"])
        )
        return df[mask]
