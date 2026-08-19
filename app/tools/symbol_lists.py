import json
import logging
import threading
from pathlib import Path

import pandas as pd

# Setup Logger
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CACHE_FILE = CACHE_DIR / "symbol_cache.json"


class ExchangeSymbol:
    """
    Singleton class to fetch and cache stock symbols from Wikipedia.
    Uses dynamic table search to be robust against page layout changes.
    Thread-safe implementation to prevent race conditions during initialization.
    Supports background loading and local caching to speed up startup.
    """

    _instance: "ExchangeSymbol | None" = None
    _initialized: bool = False
    _lock = threading.Lock()  # Lock for thread-safety

    def __new__(cls) -> "ExchangeSymbol":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # Quick check without lock (Performance)
        if ExchangeSymbol._initialized:
            return

        # Critical section: Only one thread initializes
        with ExchangeSymbol._lock:
            if ExchangeSymbol._initialized:
                return

            logger.debug("Initializing ExchangeSymbol singleton...")

            # Initialize empty lists
            self._sp_500: list[str] = []
            self._nasdaq_100: list[str] = []
            self._dow_30: list[str] = []
            self._russell_1000: list[str] = []
            self._special_symbols: list[str] = ["SPY", "QQQ", "SXRV.DE", "DIA", "^VIX"]

            # 1. Try to load from cache immediately
            self._load_from_cache()

            # 2. Start background thread to refresh data
            refresh_thread = threading.Thread(target=self._refresh_data, daemon=True)
            refresh_thread.start()

            ExchangeSymbol._initialized = True

    def _load_from_cache(self) -> None:
        """Loads symbol lists from local JSON cache if available."""
        if not CACHE_FILE.exists():
            logger.debug("No symbol cache found at %s", CACHE_FILE)
            return

        try:
            with open(CACHE_FILE, encoding="utf-8") as f:
                data = json.load(f)

            self._sp_500 = data.get("sp_500", [])
            self._nasdaq_100 = data.get("nasdaq_100", [])
            self._dow_30 = data.get("dow_30", [])
            self._russell_1000 = data.get("russell_1000", [])

            logger.debug(
                "✓ Loaded symbols from cache: SPX=%d, NDX=%d, DOW=%d, RUI=%d",
                len(self._sp_500),
                len(self._nasdaq_100),
                len(self._dow_30),
                len(self._russell_1000),
            )
        except Exception as e:
            logger.error("Failed to load symbol cache: %s", e)

    def _save_to_cache(self) -> None:
        """Saves current symbol lists to local JSON cache."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "sp_500": self._sp_500,
                "nasdaq_100": self._nasdaq_100,
                "dow_30": self._dow_30,
                "russell_1000": self._russell_1000,
            }
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug("✓ Symbol cache saved to %s", CACHE_FILE)
        except Exception as e:
            logger.error("Failed to save symbol cache: %s", e)

    def _refresh_data(self) -> None:
        """Background task to fetch fresh data from Wikipedia."""
        logger.debug("Starting background symbol refresh...")

        try:
            # 1. S&P 500
            sp_500 = self._fetch_from_wikipedia(
                url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                search_columns=["Symbol", "Ticker"],
                name="S&P 500",
            )

            # 2. NASDAQ-100
            nasdaq_100 = self._fetch_from_wikipedia(
                url=[
                    "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
                    "https://en.wikipedia.org/wiki/Nasdaq-100",
                ],
                search_columns=["Ticker", "Symbol"],
                name="NASDAQ-100",
            )

            # 3. Dow Jones 30
            dow_30 = self._fetch_from_wikipedia(
                url=[
                    "https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies",
                    "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
                ],
                search_columns=["Symbol", "Ticker"],
                name="Dow Jones 30",
            )

            # 4. Russell 1000
            russell_1000 = self._fetch_from_wikipedia(
                url=[
                    "https://en.wikipedia.org/wiki/List_of_Russell_1000_companies",
                    "https://en.wikipedia.org/wiki/Russell_1000_Index",
                ],
                search_columns=["Symbol", "Ticker"],
                name="Russell 1000",
            )

            # Update internal state (Atomic assignment is thread-safe in Python for lists)
            if sp_500:
                self._sp_500 = sp_500
            if nasdaq_100:
                self._nasdaq_100 = nasdaq_100
            if dow_30:
                self._dow_30 = dow_30
            if russell_1000:
                self._russell_1000 = russell_1000

            logger.debug(
                "✓ Symbol refresh complete: S&P 500=%d, NASDAQ-100=%d, Dow 30=%d, Russell 1000=%d",
                len(self._sp_500),
                len(self._nasdaq_100),
                len(self._dow_30),
                len(self._russell_1000),
            )

            self._save_to_cache()

        except Exception as e:
            logger.error("Background symbol refresh failed: %s", e)

    def _fetch_from_wikipedia(
        self, url: str | list[str], search_columns: list[str], name: str
    ) -> list[str]:
        """
        Fetches all tables from Wikipedia page(s) and identifies the correct one
        by checking if one of the 'search_columns' is present in the table.
        Accepts a single URL or a list of fallback URLs.
        """
        urls = [url] if isinstance(url, str) else url

        for target_url in urls:
            try:
                logger.debug("Fetching %s from %s...", name, target_url)

                # Load all tables on the page
                try:
                    tables = pd.read_html(
                        target_url, storage_options={"User-Agent": "Mozilla/5.0"}
                    )
                except Exception as e:
                    logger.error("Error reading HTML from %s: %s", target_url, e)
                    continue

                target_df = None
                found_col = None

                # Search all discovered tables
                for _i, df in enumerate(tables):
                    # Check if one of the target columns (e.g. "Symbol") exists
                    for col_candidate in search_columns:
                        # Case-insensitive check of column names
                        match = next(
                            (
                                c
                                for c in df.columns
                                if str(c).strip().lower() == col_candidate.lower()
                            ),
                            None,
                        )
                        if match:
                            target_df = df
                            found_col = match
                            break

                    if target_df is not None:
                        break

                if target_df is None:
                    logger.warning(
                        "Could not find a table with columns %s for %s at %s. Found %d tables.",
                        search_columns,
                        name,
                        target_url,
                        len(tables),
                    )
                    continue

                # Extract and clean symbols
                symbols = target_df[found_col].astype(str).str.strip()

                # Clean symbols: replace dots with dashes (BRK.B -> BRK-B), filter empty/nan
                clean_symbols = [
                    s.replace(".", "-")
                    for s in symbols
                    if len(s) > 0 and s.lower() != "nan"
                ]

                # Deduplicate and sort
                result = sorted(set(clean_symbols))
                if result:
                    return result

            except Exception as e:
                logger.error("Failed to load %s from %s: %s", name, target_url, e)

        return []

    @property
    def sp_500(self) -> list[str]:
        return self._sp_500.copy()

    @property
    def nasdaq_100(self) -> list[str]:
        return self._nasdaq_100.copy()

    @property
    def dow_30(self) -> list[str]:
        return self._dow_30.copy()

    @property
    def russell_1000(self) -> list[str]:
        return self._russell_1000.copy()

    @property
    def special_symbols(self) -> list[str]:
        return self._special_symbols.copy()

    @property
    def russell_1000_exclusive(self) -> list[str]:
        """
        Russell 1000 EXCLUDING constituents from S&P 500, Nasdaq 100, and Dow 30.
        Helps identify 'smaller' large caps that are not in the premier indices.
        """
        all_others = set(self._sp_500) | set(self._nasdaq_100) | set(self._dow_30)
        rus_excl = set(self._russell_1000) - all_others
        return sorted(rus_excl)

    @property
    def all(self) -> list[str]:
        combined = set(
            self._dow_30
            + self._nasdaq_100
            + self._sp_500
            + self._russell_1000
            + self._special_symbols
        )
        return sorted(combined)


if __name__ == "__main__":
    # Configure logging to see output
    logging.basicConfig(level=logging.INFO)
    exchange = ExchangeSymbol()

    # Wait for thread to finish to see results in a script run
    import time

    logger.info("Waiting for background thread (max 30s)...")
    time.sleep(5)

    logger.info("Total Unique Symbols: %d", len(exchange.all))
