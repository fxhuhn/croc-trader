import logging
from functools import lru_cache
from typing import List

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=3)
def _get_symbols_from_wikipedia_table(
    url: str, table_index: int, column_name: str
) -> List[str]:
    """
    Generic function to get stock symbols from a Wikipedia table.
    """
    try:
        logger.info(f"Fetching symbols from Wikipedia table at {url}...")

        # Read table from Wikipedia
        df = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})[
            table_index
        ]

        # Extract and clean symbols
        symbols = df[column_name].str.replace(".", "-", regex=False).tolist()
        symbols = [s for s in symbols if s and isinstance(s, str) and len(s) > 0]

        logger.info(f"✓ Loaded {len(symbols)} symbols from {url}")
        return symbols

    except Exception as e:
        logger.error(f"Failed to load symbols from {url}: {e}")
        return []


def nasdaq100_symbols() -> List[str]:
    return _get_symbols_from_wikipedia_table(
        url="https://en.wikipedia.org/wiki/Nasdaq-100",
        table_index=4,
        column_name="Ticker",
    )


def dow30_symbols() -> List[str]:
    return _get_symbols_from_wikipedia_table(
        url="https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
        table_index=2,
        column_name="Symbol",
    )


def sp500_symbols() -> List[str]:
    return _get_symbols_from_wikipedia_table(
        url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        table_index=0,
        column_name="Symbol",
    )
