import logging
import threading
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from app.database.repositories.market import MarketRepository
from app.database.session import DatabaseSession
from ..tools.symbol_lists import ExchangeSymbol

logger = logging.getLogger(__name__)

type SymbolList = list[str]
REQUEST_TIMEOUT = 30
BATCH_SIZE = 500

# --- Helper: Thread Lock ---
_service_lock = threading.Lock()

def require_lock(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not _service_lock.acquire(blocking=False):
            logger.warning(f"SKIP {f.__name__}: Ein Update läuft bereits!")
            return
        try:
            return f(*args, **kwargs)
        finally:
            _service_lock.release()
    return wrapper

class MarketDataService:
    """Service für das Herunterladen und Schreiben von Marktdaten."""
    
    def __init__(self, db_path: Path):
        self.session = DatabaseSession(str(db_path))
        self.repo = MarketRepository(self.session)
        self.repo.init_schema()

    @require_lock
    def update_market_data(self, full_reload: bool = False, specific_symbols: SymbolList | None = None) -> None:
        start_time = datetime.now()
        
        # Symbole ermitteln
        ignored = self.repo.get_ignored_symbols()
        if specific_symbols:
            raw_symbols = set(specific_symbols)
        else:
            raw_symbols = set(ExchangeSymbol().all).union(set(self.repo.get_all_known_symbols()))

        symbols = list(raw_symbols - ignored)
        if not symbols:
            logger.warning("Keine Symbole zu verarbeiten.")
            return

        logger.info(f"Starte Update für {len(symbols)} Symbole (Full={full_reload})...")

        # Download Startdatum
        start_date = "2022-01-01" if full_reload else (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        total_records = 0

        # Batch Processing
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i : i + BATCH_SIZE]
            try:
                records, failures = self.fetch_and_process_batch(batch, start_date)
                self.repo.save_bulk_prices(records)
                total_records += len(records)
                
                if failures and full_reload:
                    for f in failures: self.repo.ignore_symbol(f, "No Data (Full Reload)")
            except Exception as e:
                logger.error(f"Fehler im Batch {i}: {e}")

        logger.info(f"Update fertig: {total_records} Records in {datetime.now() - start_time}.")

    def perform_gap_check(self) -> None:
        logger.info("Führe Gap-Check durch...")
        thresh = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        try:
            outdated = self.repo.get_outdated_symbols(thresh)
            if outdated:
                logger.warning(f"Repariere {len(outdated)} veraltete Symbole.")
                self.update_market_data(full_reload=True, specific_symbols=outdated)
            else:
                logger.info("Gap Check: Alles aktuell.")
        except Exception as e:
            logger.error(f"Gap Check Error: {e}")

    def fetch_and_process_batch(self, symbols: SymbolList, start_date: str) -> tuple[list[tuple], list[str]]:
        if not symbols: return [], []
        records = []
        found = set()
        
        try:
            df = yf.download(
                " ".join(symbols), start=start_date, group_by="ticker", 
                auto_adjust=True, progress=False, threads=True, timeout=REQUEST_TIMEOUT, ignore_tz=True
            )
        except Exception: return [], symbols

        if df.empty: return [], symbols

        def process(sym, d):
            d.columns = d.columns.str.lower()
            d.dropna(subset=["close"], inplace=True)
            if d.empty: return
            found.add(sym)
            for ts, row in d.iterrows():
                # (symbol, date, open, high, low, close, volume, provider, timeframe)
                records.append((
                    sym, ts.strftime("%Y-%m-%d"), 
                    float(row.get("open", 0)), float(row.get("high", 0)), float(row.get("low", 0)), float(row.get("close", 0)), 
                    int(row.get("volume", 0)), "yahoo", "1D"
                ))

        if len(symbols) == 1: process(symbols[0], df)
        else:
            for sym in symbols:
                if sym in df.columns.get_level_values(0): process(sym, df[sym].copy())

        return records, list(set(symbols) - found)

class DataValidator:
    def __init__(self, service: MarketDataService):
        self.service = service
        self.repo = service.repo

    def run_checks(self):
        logger.info("Validierung läuft...")
        # Beispielhafter Check
        self.repo.fetch_all("SELECT symbol FROM market_prices WHERE low < 0")