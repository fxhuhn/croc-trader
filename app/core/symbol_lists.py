"""
Stock symbol list fetchers from Wikipedia.

This module provides functions to fetch stock symbols for major US indices
(NASDAQ-100, Dow Jones Industrial Average, S&P 500) by scraping Wikipedia tables.
Results are cached to minimize network requests.
"""

import logging
from typing import Final

import pandas as pd
from cachetools import TTLCache, cached

# ============================================================================
# Configuration
# ============================================================================

# Wikipedia URLs for stock indices
NASDAQ_100_URL: Final[str] = "https://en.wikipedia.org/wiki/Nasdaq-100"
DOW_30_URL: Final[str] = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
SP_500_URL: Final[str] = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# User agent for Wikipedia requests
USER_AGENT: Final[str] = "Mozilla/5.0 (compatible; StockSymbolFetcher/1.0)"

# Cache configuration
CACHE_SIZE: Final[int] = 3  # Cache results for all three indices

# ============================================================================
# Logging Setup
# ============================================================================

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================================
# Core Functionality
# ============================================================================


# Cache expires after 24 hours
@cached(cache=TTLCache(maxsize=3, ttl=86400))
def _get_symbols_from_wikipedia_table(
    url: str,
    table_index: int,
    column_name: str,
) -> tuple[str, ...]:
    """
    Fetch and parse stock symbols from a Wikipedia table.

    This function scrapes a Wikipedia page, extracts a specific table,
    and returns cleaned stock symbols from the specified column. Results
    are cached to avoid repeated network requests.

    Args:
        url: Wikipedia page URL containing the table
        table_index: Zero-based index of the table on the page
        column_name: Name of the column containing stock symbols

    Returns:
        Tuple of cleaned stock symbols (immutable for caching)

    Raises:
        Does not raise exceptions; returns empty tuple on failure

    Example:
        >>> symbols = _get_symbols_from_wikipedia_table(
        ...     "https://en.wikipedia.org/wiki/Nasdaq-100",
        ...     4,
        ...     "Ticker"
        ... )
        >>> "AAPL" in symbols
        True

    Note:
        - Dots in symbols are replaced with hyphens (e.g., "BRK.B" -> "BRK-B")
        - Empty strings and non-string values are filtered out
        - Returns tuple (immutable) to work with lru_cache
    """
    try:
        logger.info(f"Fetching symbols from Wikipedia table at {url}...")

        # Fetch tables from Wikipedia with custom user agent
        tables = pd.read_html(
            url,
            storage_options={"User-Agent": USER_AGENT},
        )

        # Extract the target table
        df = tables[table_index]

        # Extract symbols and clean them
        # Replace dots with hyphens to match Yahoo Finance symbol format
        symbols_series = df[column_name].str.replace(".", "-", regex=False)

        # Convert to list and filter out invalid entries
        symbols_list = symbols_series.tolist()
        cleaned_symbols = [
            symbol
            for symbol in symbols_list
            if symbol and isinstance(symbol, str) and len(symbol.strip()) > 0
        ]

        # Convert to tuple for immutability and better caching
        symbols_tuple = tuple(cleaned_symbols)

        logger.info(f"✓ Successfully loaded {len(symbols_tuple)} symbols from {url}")
        return symbols_tuple

    except IndexError:
        logger.error(
            f"Failed to load symbols from {url}: Table index {table_index} out of range"
        )
        return ()

    except KeyError:
        logger.error(
            f"Failed to load symbols from {url}: "
            f"Column '{column_name}' not found in table"
        )
        return ()

    except Exception as e:
        logger.error(f"Failed to load symbols from {url}: {type(e).__name__}: {e}")
        return ()


# ============================================================================
# Public API Functions
# ============================================================================


def nasdaq100_symbols() -> list[str]:
    """
    Fetch NASDAQ-100 stock symbols from Wikipedia.

    The NASDAQ-100 includes the 100 largest non-financial companies
    listed on the NASDAQ stock exchange.

    Returns:
        List of NASDAQ-100 stock ticker symbols

    Example:
        >>> symbols = nasdaq100_symbols()
        >>> len(symbols) > 90  # Should have ~100 symbols
        True
        >>> "AAPL" in symbols
        True

    Note:
        Results are cached. Subsequent calls return cached data without
        making additional network requests.
    """
    return list(
        _get_symbols_from_wikipedia_table(
            url=NASDAQ_100_URL,
            table_index=4,
            column_name="Ticker",
        )
    )


def dow30_symbols() -> list[str]:
    """
    Fetch Dow Jones Industrial Average (Dow 30) stock symbols from Wikipedia.

    The Dow 30 consists of 30 large, publicly-owned blue-chip companies
    trading on the New York Stock Exchange and NASDAQ.

    Returns:
        List of Dow 30 stock ticker symbols

    Example:
        >>> symbols = dow30_symbols()
        >>> len(symbols) == 30
        True
        >>> "AAPL" in symbols
        True

    Note:
        Results are cached. Subsequent calls return cached data without
        making additional network requests.
    """
    return list(
        _get_symbols_from_wikipedia_table(
            url=DOW_30_URL,
            table_index=2,
            column_name="Symbol",
        )
    )


def sp500_symbols() -> list[str]:
    """
    Fetch S&P 500 stock symbols from Wikipedia.

    The S&P 500 is a stock market index tracking the performance of
    500 large companies listed on US stock exchanges.

    Returns:
        List of S&P 500 stock ticker symbols

    Example:
        >>> symbols = sp500_symbols()
        >>> len(symbols) > 490  # Should have ~500 symbols
        True
        >>> "AAPL" in symbols
        True

    Note:
        Results are cached. Subsequent calls return cached data without
        making additional network requests.
    """
    return list(
        _get_symbols_from_wikipedia_table(
            url=SP_500_URL,
            table_index=0,
            column_name="Symbol",
        )
    )


def get_all_symbols() -> dict[str, list[str]]:
    """
    Fetch symbols for all supported indices.

    Returns:
        Dictionary mapping index names to their symbol lists

    Example:
        >>> all_symbols = get_all_symbols()
        >>> "nasdaq100" in all_symbols
        True
        >>> len(all_symbols["dow30"]) == 30
        True
    """
    return {
        "nasdaq100": nasdaq100_symbols(),
        "dow30": dow30_symbols(),
        "sp500": sp500_symbols(),
    }


def clear_cache() -> None:
    """
    Clear the symbol cache.

    Use this to force a fresh fetch from Wikipedia on the next call.
    Useful for testing or when you need updated symbol lists.

    Example:
        >>> clear_cache()
        >>> # Next call will fetch from Wikipedia again
        >>> symbols = nasdaq100_symbols()
    """
    _get_symbols_from_wikipedia_table.cache_clear()
    logger.info("Symbol cache cleared")
