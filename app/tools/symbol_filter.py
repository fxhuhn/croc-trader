import logging
import threading
import json
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import yfinance as yf
import time
from .symbol_lists import ExchangeSymbol

# Setup Logger
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CACHE_FILE = CACHE_DIR / "preferred_symbols.json"

class SymbolFilter:
    """
    Singleton class to filter generic symbols based on volume/popularity.
    Example: Prefer GOOG over GOOGL if GOOG has higher volume.
    Loads mapping in background and caches it to disk.
    """

    _instance: Optional["SymbolFilter"] = None
    _initialized: bool = False
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if SymbolFilter._initialized:
            return

        with SymbolFilter._lock:
            if SymbolFilter._initialized:
                return

            self._mapping: Dict[str, List[str]] = {} # Winner -> [Losers]
            
            # 1. Try to load from cache
            self._load_from_cache()

            # 2. Start background thread to refresh mapping
            # We need a list of symbols to filter. 
            # Ideally, we get this from ExchangeSymbol, but that might be empty initially.
            # We will lazy-load or periodically refresh based on ExchangeSymbol.all
            # For now, we start a thread that waits for ExchangeSymbol to be ready or just runs.
            threading.Thread(target=self._refresh_mapping_background, daemon=True).start()

            SymbolFilter._initialized = True

    def _load_from_cache(self) -> None:
        """Loads preferred symbol mapping from local cache."""
        if not CACHE_FILE.exists():
            return

        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                self._mapping = json.load(f)
            logger.info("✓ Loaded symbol preference mapping with %d rules", len(self._mapping))
        except Exception as e:
            logger.error("Failed to load symbol preference cache: %s", e)

    def _save_to_cache(self) -> None:
        """Saves current mapping to local cache."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._mapping, f, indent=2)
            logger.info("✓ Symbol preference cache saved to %s", CACHE_FILE)
        except Exception as e:
            logger.error("Failed to save symbol preference cache: %s", e)

    def _refresh_mapping_background(self) -> None:
        """Fetches volume data for all symbols and builds preference mapping."""

        # Wait a bit for ExchangeSymbol to populate (heuristic)
        time.sleep(5) 
        
        exchange = ExchangeSymbol()
        all_symbols = exchange.all
        
        if not all_symbols:
            logger.warning("No symbols found in ExchangeSymbol to build filter mapping.")
            return

        logger.info("Starting background symbol preference analysis for %d symbols...", len(all_symbols))
        
        try:
            new_mapping = self._build_mapping(all_symbols)
            if new_mapping:
                self._mapping = new_mapping
                self._save_to_cache()
                logger.info("✓ Symbol preference analysis complete. Found %d duplicate groups.", len(self._mapping))
        except Exception as e:
            logger.error("Symbol preference analysis failed: %s", e)

    def _build_mapping(self, symbols: List[str]) -> Dict[str, List[str]]:
        """
        Fetches metadata and determines winners.
        """
        raw_data = []
        # Chunking to avoid overwhelming yfinance? 
        # yfinance can handle multiple tickers string, but 'info' is usually per ticker.
        # We'll do it sequentially or batched. For >2000 symbols this is SLOW.
        # We only do this if we really need to.
        # Optimization: Only fetch if we don't have a cache?
        # Or just do it. The user asked for "background thread".

        count = 0
        for symbol in symbols:
            try:
                # This is IO bound and slow. 
                # Ideally we use batch endpoint if available, but yfinance Ticker("A B C").tickers is an option.
                # But extracting 'longName' and 'averageVolume' from bulk might be tricky.
                # Let's stick to simple loop for robustness in background.
                info = yf.Ticker(symbol).info
                raw_data.append({
                    'symbol': symbol,
                    # Fallback to shortName or symbol if longName missing
                    'longName': info.get('longName') or info.get('shortName') or symbol,
                    'averageVolume': info.get('averageVolume', 0)
                })
                count += 1
                if count % 100 == 0:
                    logger.debug("Processed %d/%d symbols for preference mapping", count, len(symbols))
            except Exception as e:
                logger.warning("Failed to fetch info for %s: %s", symbol, e)
                continue

        if not raw_data:
            return {}

        df = pd.DataFrame(raw_data)
        
        # Filter duplicates based on Name
        # We are looking for DIFFERENT symbols that share the SAME Corp Name
        duplicates = df[df.duplicated(subset=['longName'], keep=False)]
        
        if duplicates.empty:
            return {}

        # Sort by Name (asc) and Volume (desc) -> Winner is first
        duplicates = duplicates.sort_values(['longName', 'averageVolume'], ascending=[True, False])

        mapping = {}
        for _, group in duplicates.groupby('longName', sort=False):
            if len(group) > 1:
                winner = group['symbol'].iloc[0]
                losers = group['symbol'].iloc[1:].tolist()
                mapping[winner] = losers
        
        return mapping

    def filter_symbols(self, candidates: List[str]) -> List[str]:
        """
        Filters the candidate list.
        If a 'Loser' is present, check if its 'Winner' is also present (or if we just hate losers?).
        
        Strategy:
        If we have a mapping Winner -> [Losers], and we see a Loser in candidates:
        - If Winner is ALSO in candidates, definitely drop Loser.
        - If Winner is NOT in candidates, do we keep Loser? 
          User said: "if key goog from filter_symbols exists delete the values like googl if applicable"
          Context: "perform these kind of filtering at the end of _filter_market_state" (which implies we have a list of valid signals).
          
        Interpretation: Only show the BEST symbol for the company if multiple are signalled.
        So: Drop Loser ONLY IF Winner is in candidates.
        """
        candidates_set = set(candidates)
        to_remove = set()

        for winner, losers in self._mapping.items():
            # Check if any loser matches a candidate
            for loser in losers:
                if loser in candidates_set:
                    # Found a loser. Is the winner also here?
                    if winner in candidates_set:
                        to_remove.add(loser)
                    # Use Case 2: Maybe we simply NEVER want the loser if we prefer the winner generally?
                    # But if the winner didn't trigger a signal (e.g. slight price diff), maybe we take the loser?
                    # SAFE BET: Only drop if we have a better alternative (Winner) in the list.
        
        return [c for c in candidates if c not in to_remove]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    f = SymbolFilter()
    print("Filter initialized. Waiting for background thread won't work well in script unless we sleep.")
