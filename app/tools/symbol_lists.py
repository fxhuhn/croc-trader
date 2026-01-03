import logging
from typing import List, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExchangeSymbol:
    """
    Singleton class to fetch and cache stock symbols from Wikipedia.
    Data is loaded once during initialization.
    """

    _instance: Optional["ExchangeSymbol"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once (Singleton pattern)
        if ExchangeSymbol._initialized:
            return

        logger.info("Initializing ExchangeSymbol singleton...")

        self._sp_500: List[str] = []
        self._nasdaq_100: List[str] = []
        self._dow_30: List[str] = []

        # Load all symbols during initialization
        self._sp_500 = self._fetch_symbols(
            url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            table_index=0,
            column_name="Symbol",
            name="S&P 500",
        )

        self._nasdaq_100 = self._fetch_symbols(
            url="https://en.wikipedia.org/wiki/Nasdaq-100",
            table_index=4,
            column_name="Ticker",
            name="NASDAQ-100",
        )

        self._dow_30 = self._fetch_symbols(
            url="https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
            table_index=2,
            column_name="Symbol",
            name="Dow Jones 30",
        )

        ExchangeSymbol._initialized = True
        logger.info(
            f"✓ ExchangeSymbol initialized successfully: "
            f"S&P 500={len(self._sp_500)}, "
            f"NASDAQ-100={len(self._nasdaq_100)}, "
            f"Dow 30={len(self._dow_30)} symbols"
        )

    def _fetch_symbols(
        self, url: str, table_index: int, column_name: str, name: str
    ) -> List[str]:
        """
        Fetch stock symbols from a Wikipedia table.
        """
        try:
            logger.info(f"Fetching {name} symbols from {url}...")

            # Read table from Wikipedia
            df = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})[
                table_index
            ]

            # Extract and clean symbols
            symbols = df[column_name].str.replace(".", "-", regex=False).tolist()
            symbols = [s for s in symbols if s and isinstance(s, str) and len(s) > 0]

            logger.info(f"✓ Loaded {len(symbols)} {name} symbols")
            return symbols

        except Exception as e:
            logger.error(f"Failed to load {name} symbols from {url}: {e}")
            return []

    @property
    def sp_500(self) -> List[str]:
        """Get S&P 500 symbols."""
        return self._sp_500.copy()

    @property
    def nasdaq_100(self) -> List[str]:
        """Get NASDAQ-100 symbols."""
        return self._nasdaq_100.copy()

    @property
    def dow_30(self) -> List[str]:
        """Get Dow Jones 30 symbols."""
        return self._dow_30.copy()

    @property
    def all(self) -> List[str]:
        """Get Dow Jones 30 symbols."""
        return list(set(self._dow_30 + self.nasdaq_100 + self.sp_500)).copy()


if __name__ == "__main__":
    exchange = ExchangeSymbol()
    symbols = exchange.all
    print(symbols)
