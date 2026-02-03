import logging
import threading
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExchangeSymbol:
    """
    Singleton class to fetch and cache stock symbols from Wikipedia.
    Uses dynamic table search to be robust against page layout changes.
    Thread-safe implementation to prevent race conditions during initialization.
    """

    _instance: Optional["ExchangeSymbol"] = None
    _initialized: bool = False
    _lock = threading.Lock()  # Lock für Thread-Safety

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking für die Instanz
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Schneller Check ohne Lock (Performance)
        if ExchangeSymbol._initialized:
            return

        # Kritischer Abschnitt: Nur ein Thread darf initialisieren
        with ExchangeSymbol._lock:
            # Zweiter Check innerhalb des Locks (falls ein anderer Thread gerade fertig wurde)
            if ExchangeSymbol._initialized:
                return

            logger.info("Initializing ExchangeSymbol singleton...")

            self._sp_500: list[str] = []
            self._nasdaq_100: list[str] = []
            self._dow_30: list[str] = []
            self._russell_1000: list[str] = []
            self._special_symbols: list[str] = ["SPY", "QQQ", "SXRV", "DIA"]

            # 1. S&P 500
            self._sp_500 = self._fetch_from_wikipedia(
                url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                search_columns=["Symbol", "Ticker"],
                name="S&P 500",
            )

            # 2. NASDAQ-100
            self._nasdaq_100 = self._fetch_from_wikipedia(
                url="https://en.wikipedia.org/wiki/Nasdaq-100",
                search_columns=["Ticker", "Symbol"],
                name="NASDAQ-100",
            )

            # 3. Dow Jones 30
            self._dow_30 = self._fetch_from_wikipedia(
                url="https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
                search_columns=["Symbol", "Ticker"],
                name="Dow Jones 30",
            )

            # 4. Russell 1000
            # Wir suchen dynamisch nach der Tabelle mit 'Ticker' oder 'Symbol'
            self._russell_1000 = self._fetch_from_wikipedia(
                url="https://en.wikipedia.org/wiki/Russell_1000_Index",
                search_columns=["Symbol", "Ticker", "Company"],
                name="Russell 1000",
            )

            ExchangeSymbol._initialized = True
            logger.info(
                f"✓ ExchangeSymbol initialized: "
                f"S&P 500={len(self._sp_500)}, "
                f"NASDAQ-100={len(self._nasdaq_100)}, "
                f"Dow 30={len(self._dow_30)}, "
                f"Russell 1000={len(self._russell_1000)}, "
                f"Special={len(self._special_symbols)}"
            )

    def _fetch_from_wikipedia(
        self, url: str, search_columns: list[str], name: str
    ) -> list[str]:
        """
        Lädt alle Tabellen einer Wikipedia-Seite und sucht die richtige heraus,
        indem geprüft wird, ob eine der 'search_columns' existiert.
        """
        try:
            logger.info(f"Fetching {name} from {url}...")

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

            logger.info(
                f"✓ Loaded {len(result)} {name} symbols (found in table with col '{found_col}')"
            )
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
            self._dow_30 + self._nasdaq_100 + self._sp_500 + self._russell_1000 + self._special_symbols
        )
        return sorted(list(combined))


if __name__ == "__main__":
    exchange = ExchangeSymbol()
    print(f"Total Unique Symbols: {len(exchange.all)}")
