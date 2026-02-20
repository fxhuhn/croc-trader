import json
import logging
import threading
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

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

    _instance: "SymbolFilter | None" = None
    _initialized: bool = False
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "SymbolFilter":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if SymbolFilter._initialized:
            return

        with SymbolFilter._lock:
            if SymbolFilter._initialized:
                return

            self._mapping: dict[str, list[str]] = {}  # Winner -> [Losers]

            # 1. Try to load from cache
            self._load_from_cache()

            # 2. Start background thread to refresh mapping
            threading.Thread(
                target=self._refresh_mapping_background, daemon=True
            ).start()

            SymbolFilter._initialized = True

    def _load_from_cache(self) -> None:
        """
        Loads preferred symbol mapping from local cache.

        Reads the JSON mapping file and updates the internal state.
        Errors are logged but do not crash the initialization.
        """
        if not CACHE_FILE.exists():
            return

        try:
            with CACHE_FILE.open("r", encoding="utf-8") as cache_file:
                self._mapping = json.load(cache_file)
            logger.debug(
                "✓ Loaded symbol preference mapping with %d rules", len(self._mapping)
            )
        except Exception as error:
            logger.error("Failed to load symbol preference cache: %s", error)

    def _save_to_cache(self) -> None:
        """
        Saves current mapping to local cache.

        Writes the internal mapping to disk in JSON format.
        Creates parent directories if necessary.
        """
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with CACHE_FILE.open("w", encoding="utf-8") as cache_file:
                json.dump(self._mapping, cache_file, indent=2)
            logger.debug("✓ Symbol preference cache saved to %s", CACHE_FILE)
        except Exception as error:
            logger.error("Failed to save symbol preference cache: %s", error)

    def _refresh_mapping_background(self) -> None:
        """
        Fetches volume data for all symbols and builds preference mapping.

        Orchestrates the background process of gathering ticker metadata
        and identifying preferred symbols (winners) based on liquidity.
        """
        # Wait a bit for ExchangeSymbol to populate (heuristic)
        time.sleep(5)

        exchange: ExchangeSymbol = ExchangeSymbol()
        all_symbols: list[str] = exchange.all

        if not all_symbols:
            logger.warning(
                "No symbols found in ExchangeSymbol to build filter mapping."
            )
            return

        logger.debug(
            "Starting background symbol preference analysis for %d symbols...",
            len(all_symbols),
        )

        try:
            new_mapping: dict[str, list[str]] = self._build_mapping(all_symbols)
            if new_mapping:
                self._mapping = new_mapping
                self._save_to_cache()
                logger.debug(
                    "✓ Symbol preference analysis complete. Found %d duplicate groups.",
                    len(self._mapping),
                )
            else:
                logger.warning(
                    "Symbol preference analysis returned no results or was aborted."
                )
        except Exception as error:
            logger.error("Symbol preference analysis failed: %s", error)

    def _build_mapping(self, symbols: list[str]) -> dict[str, list[str]]:
        """
        Fetches metadata and determines winner symbols for duplicate companies.

        Args:
            symbols: A list of tickers to evaluate.

        Returns:
            A dictionary mapping winner symbols to lists of excluded (loser) tickers.
        """
        raw_metadata_list: list[dict] = []
        processed_count: int = 0

        for symbol in symbols:
            try:
                ticker_metadata: dict = yf.Ticker(symbol).info
                raw_metadata_list.append(
                    {
                        "symbol": symbol,
                        "longName": ticker_metadata.get("longName")
                        or ticker_metadata.get("shortName")
                        or symbol,
                        "averageVolume": ticker_metadata.get("averageVolume", 0),
                    }
                )
                processed_count += 1
                if processed_count % 100 == 0:
                    logger.debug(
                        "Processed %d/%d symbols for preference mapping",
                        processed_count,
                        len(symbols),
                    )
            except Exception as error:
                error_message: str = str(error)
                if "Too Many Requests" in error_message or "429" in error_message:
                    logger.error(
                        "ERROR: Yahoo Finance rate limit (429) hit. "
                        "Aborting refresh to preserve existing cache."
                    )
                    return {}

                logger.warning("Failed to fetch info for %s: %s", symbol, error)
                continue

        if not raw_metadata_list:
            return {}

        metadata_dataframe: pd.DataFrame = pd.DataFrame(raw_metadata_list)

        # Filter duplicates based on Name
        duplicates: pd.DataFrame = metadata_dataframe[
            metadata_dataframe.duplicated(subset=["longName"], keep=False)
        ]

        if duplicates.empty:
            return {}

        # Sort by Name (asc) and Volume (desc) -> Winner is first
        duplicates = duplicates.sort_values(
            ["longName", "averageVolume"], ascending=[True, False]
        )

        mapping: dict[str, list[str]] = {}
        for _, group in duplicates.groupby("longName", sort=False):
            if len(group) > 1:
                winner_ticker: str = group["symbol"].iloc[0]
                loser_tickers: list[str] = group["symbol"].iloc[1:].tolist()
                mapping[winner_ticker] = loser_tickers

        return mapping

    def filter_symbols(self, candidates: list[str]) -> list[str]:
        """
        Analyzes a list of symbols and removes less liquid duplicates.

        Drops "loser" tickers (e.g., GOOGL) only if their preferred "winner"
        counterpart (e.g., GOOG) is also present in the candidate list.

        Args:
            candidates: A list of ticker symbols to filter.

        Returns:
            A filtered list containing only the preferred tickers.
        """
        candidates_set: set[str] = set(candidates)
        to_remove: set[str] = set()

        for winner_ticker, loser_tickers in self._mapping.items():
            for loser_ticker in loser_tickers:
                if loser_ticker in candidates_set and winner_ticker in candidates_set:
                    to_remove.add(loser_ticker)

        return [candidate for candidate in candidates if candidate not in to_remove]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    filter_instance: SymbolFilter = SymbolFilter()
    logger.info("SymbolFilter initialized for stand-alone testing.")
