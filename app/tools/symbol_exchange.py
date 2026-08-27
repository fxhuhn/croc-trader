import json
import logging
import threading
import time
from pathlib import Path

import requests

# Setup Logger
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CACHE_FILE = CACHE_DIR / "symbol_exchange.json"


class SymbolExchange:
    """
    Singleton class to manage symbol-to-exchange mapping.
    Loads mapping from local cache and refreshes from GitHub in the background.
    """

    _instance: "SymbolExchange | None" = None
    _initialized: bool = False
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "SymbolExchange":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        with SymbolExchange._lock:
            if SymbolExchange._initialized:
                return

            self._mapping: dict[str, str] = {}  # Symbol -> Exchange

            # 1. Try to load from cache
            self._load_from_cache()

            # 2. Start background thread to refresh mapping
            threading.Thread(
                target=self._refresh_mapping_background, daemon=True
            ).start()

            SymbolExchange._initialized = True

    def _load_from_cache(self) -> None:
        """Loads symbol-to-exchange mapping from local cache."""
        if not CACHE_FILE.exists():
            return

        try:
            with CACHE_FILE.open("r", encoding="utf-8") as cache_file:
                self._mapping = json.load(cache_file)
            logger.debug(
                "✓ Loaded symbol-exchange mapping with %d symbols", len(self._mapping)
            )
        except Exception as error:
            logger.error("Failed to load symbol-exchange cache: %s", error)

    def _save_to_cache(self) -> None:
        """Saves current mapping to local cache."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with CACHE_FILE.open("w", encoding="utf-8") as cache_file:
                json.dump(self._mapping, cache_file, indent=2)
            logger.debug(
                "✓ Symbol-exchange cache saved up to %d symbols", len(self._mapping)
            )
        except Exception as error:
            logger.error("Failed to save symbol-exchange cache: %s", error)

    def _refresh_mapping_background(self) -> None:
        """Fetches ticker lists from GitHub and updates the local mapping."""
        # Small delay to let other services initialize
        time.sleep(2)

        base_url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main"
        exchanges = {
            "NASDAQ": f"{base_url}/nasdaq/nasdaq_tickers.json",
            "NYSE": f"{base_url}/nyse/nyse_tickers.json",
            "AMEX": f"{base_url}/amex/amex_tickers.json",
        }

        new_mapping: dict[str, str] = {}
        logger.debug("Starting background symbol-exchange refresh from GitHub...")

        success_count = 0
        for exchange_name, url in exchanges.items():
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                tickers = response.json()
                for ticker in tickers:
                    new_mapping[ticker] = exchange_name
                logger.debug("Loaded %d symbols for %s", len(tickers), exchange_name)
                success_count += 1
            except Exception as e:
                logger.warning(
                    "Failed to load %s tickers from GitHub: %s", exchange_name, e
                )

        if success_count > 0:
            self._mapping = new_mapping
            self._save_to_cache()
            logger.debug(
                "✓ Symbol-exchange refresh complete. Total symbols: %d",
                len(self._mapping),
            )
        else:
            logger.error("Failed to refresh any symbol-exchange data from GitHub.")

    def get_exchange(self, symbol: str) -> str:
        """
        Returns the exchange for a given symbol.
        Returns 'NASDAQ' as default if not found.
        """
        return self._mapping.get(symbol, "NASDAQ")

    @property
    def mapping(self) -> dict[str, str]:
        """Returns a copy of the symbol-to-exchange mapping."""
        return self._mapping.copy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exchange_manager = SymbolExchange()
    # Wait for background thread for a bit if running as script
    time.sleep(10)
    logger.info("Total symbols: %d", len(exchange_manager.mapping))
    logger.info("AAPL exchange: %s", exchange_manager.get_exchange("AAPL"))
