import logging
import threading
import json
from pathlib import Path
from typing import Optional
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

    _instance: Optional["ExchangeSymbol"] = None
    _initialized: bool = False
    _lock = threading.Lock()  # Lock for thread-safety

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
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
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
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
                url="https://en.wikipedia.org/wiki/Nasdaq-100",
                search_columns=["Ticker", "Symbol"],
                name="NASDAQ-100",
            )

            # 3. Dow Jones 30
            dow_30 = self._fetch_from_wikipedia(
                url="https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
                search_columns=["Symbol", "Ticker"],
                name="Dow Jones 30",
            )

            # 4. Russell 1000
            russell_1000 = self._fetch_from_wikipedia(
                url="https://en.wikipedia.org/wiki/Russell_1000_Index",
                search_columns=["Symbol", "Ticker", "Company"],
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
                f"✓ Symbol refresh complete: "
                f"S&P 500={len(self._sp_500)}, "
                f"NASDAQ-100={len(self._nasdaq_100)}, "
                f"Dow 30={len(self._dow_30)}, "
                f"Russell 1000={len(self._russell_1000)}"
            )

            self._save_to_cache()

        except Exception as e:
            logger.error("Background symbol refresh failed: %s", e)

    def _fetch_from_wikipedia(
        self, url: str, search_columns: list[str], name: str
    ) -> list[str]:
        """
        Lädt alle Tabellen einer Wikipedia-Seite und sucht die richtige heraus,
        indem geprüft wird, ob eine der 'search_columns' existiert.
        """
        try:
            logger.debug(f"Fetching {name} from {url}...")

            # Alle Tabellen der Seite laden
            try:
                tables = pd.read_html(
                    url, storage_options={"User-Agent": "Mozilla/5.0"}
                )
            except Exception as e:
                logger.error(f"Error reading HTML from {url}: {e}")
                return []

            target_df = None
            found_col = None

            # Durchsuche alle gefundenen Tabellen
            for i, df in enumerate(tables):
                # Prüfe ob eine der gesuchten Spalten (z.B. "Symbol") existiert
                for col_candidate in search_columns:
                    # Case-insensitive Suche in den Spaltennamen
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
                        # logger.debug(f"Found '{col_candidate}' in table index {i} for {name}")
                        break

                if target_df is not None:
                    break

            if target_df is None:
                logger.warning(
                    f"Could not find a table with columns {search_columns} for {name}. Found {len(tables)} tables."
                )
                return []

            # Symbole extrahieren und bereinigen
            symbols = target_df[found_col].astype(str).str.strip()

            # Bereinigung: Punkte durch Striche ersetzen (BRK.B -> BRK-B), leere entfernen
            clean_symbols = [
                s.replace(".", "-")
                for s in symbols
                if len(s) > 0 and s.lower() != "nan"
            ]

            # Duplikate entfernen und sortieren
            result = sorted(list(set(clean_symbols)))

            # logger.info(
            #     f"✓ Loaded {len(result)} {name} symbols (found in table with col '{found_col}')"
            # )
            return result

        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
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
        Russell 1000 OHNE die Titel aus S&P 500, Nasdaq 100 und Dow 30.
        Dient dazu, 'kleinere' Large Caps zu finden, die nicht in den Top-Indizes sind.
        """
        all_others = set(self._sp_500) | set(self._nasdaq_100) | set(self._dow_30)
        rus_excl = set(self._russell_1000) - all_others
        return sorted(list(rus_excl))

    @property
    def all(self) -> list[str]:
        combined = set(
            self._dow_30
            + self._nasdaq_100
            + self._sp_500
            + self._russell_1000
            + self._special_symbols
        )
        return sorted(list(combined))


if __name__ == "__main__":
    # Configure logging to see output
    logging.basicConfig(level=logging.INFO)
    exchange = ExchangeSymbol()

    # Wait for thread to finish to see results in a script run
    import time

    print("Waiting for background thread (max 30s)...")
    time.sleep(5)

    print(f"Total Unique Symbols: {len(exchange.all)}")
