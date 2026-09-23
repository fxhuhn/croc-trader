"""Provides equity symbol lists for major market indices fetched from Wikipedia.

Maintains local cache persistence and thread-safe memory state for S&P 500, S&P 100,
NASDAQ-100, Dow Jones 30, and Russell 1000 indices.
"""

import json
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical directory and cache file paths
DEFAULT_CACHE_DIR: Path = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_CACHE_FILE: Path = DEFAULT_CACHE_DIR / "symbol_cache.json"
CACHE_DIR: Path = DEFAULT_CACHE_DIR
CACHE_FILE: Path = DEFAULT_CACHE_FILE


@dataclass(frozen=True)
class IndexSourceDefinition:
    """Immutable definition of an index Wikipedia data source."""

    name: str
    cache_key: str
    urls: tuple[str, ...]
    search_columns: tuple[str, ...]


DEFAULT_INDEX_SOURCES: tuple[IndexSourceDefinition, ...] = (
    IndexSourceDefinition(
        name="S&P 500",
        cache_key="sp_500",
        urls=("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",),
        search_columns=("Symbol", "Ticker"),
    ),
    IndexSourceDefinition(
        name="S&P 100",
        cache_key="sp_100",
        urls=("https://en.wikipedia.org/wiki/S%26P_100",),
        search_columns=("Symbol", "Ticker"),
    ),
    IndexSourceDefinition(
        name="NASDAQ-100",
        cache_key="nasdaq_100",
        urls=(
            "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
            "https://en.wikipedia.org/wiki/Nasdaq-100",
        ),
        search_columns=("Ticker", "Symbol"),
    ),
    IndexSourceDefinition(
        name="Dow Jones 30",
        cache_key="dow_30",
        urls=(
            "https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies",
            "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
        ),
        search_columns=("Symbol", "Ticker"),
    ),
    IndexSourceDefinition(
        name="Russell 1000",
        cache_key="russell_1000",
        urls=(
            "https://en.wikipedia.org/wiki/List_of_Russell_1000_companies",
            "https://en.wikipedia.org/wiki/Russell_1000_Index",
        ),
        search_columns=("Symbol", "Ticker"),
    ),
)

DEFAULT_SPECIAL_SYMBOLS: tuple[str, ...] = ("SPY", "QQQ", "SXRV.DE", "DIA", "^VIX")


# --- Functional Core (Pure Transformations) ---


def clean_symbol(raw_symbol: str) -> str | None:
    """Sanitizes an equity ticker symbol by trimming, hyphenating share classes, and filtering invalid values.

    Args:
        raw_symbol: Raw string ticker from table parsing.

    Returns:
        Cleaned uppercase ticker symbol, or None if invalid or empty.
    """
    stripped = raw_symbol.strip().replace(".", "-")
    if not stripped or stripped.lower() == "nan":
        return None
    return stripped


def extract_symbols_from_tables(
    tables: Sequence[pd.DataFrame],
    search_columns: Sequence[str],
) -> list[str]:
    """Finds the first table matching candidate column headers and extracts cleaned symbols.

    Args:
        tables: Sequence of parsed HTML DataFrames.
        search_columns: Column names to search for (e.g. 'Symbol', 'Ticker').

    Returns:
        Sorted, deduplicated list of cleaned ticker symbols.
    """
    normalized_candidates = {col.strip().lower() for col in search_columns}

    for table in tables:
        found_column = None
        for column in table.columns:
            if str(column).strip().lower() in normalized_candidates:
                found_column = column
                break

        if found_column is None:
            continue

        raw_series = table[found_column].dropna().astype(str)
        cleaned_symbols = {
            cleaned
            for raw_val in raw_series
            if (cleaned := clean_symbol(raw_val)) is not None
        }
        if cleaned_symbols:
            return sorted(cleaned_symbols)

    return []


# --- Imperative Shell (Singleton, I/O, Cache & Thread Orchestration) ---


class ExchangeSymbol:
    """Singleton providing equity symbols for major indices fetched from Wikipedia.

    Thread-safe implementation with local JSON caching and background refresh.
    """

    _instance: ClassVar["ExchangeSymbol | None"] = None
    _initialized: ClassVar[bool] = False
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls, *args: object, **kwargs: object) -> "ExchangeSymbol":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        cache_file: Path | None = None,
        *,
        auto_refresh: bool = True,
    ) -> None:
        with ExchangeSymbol._lock:
            if ExchangeSymbol._initialized:
                return

            logger.debug("Initializing ExchangeSymbol singleton...")

            self._cache_file: Path | None = cache_file
            self._sp_500: list[str] = []
            self._sp_100: list[str] = []
            self._nasdaq_100: list[str] = []
            self._dow_30: list[str] = []
            self._russell_1000: list[str] = []
            self._special_symbols: list[str] = list(DEFAULT_SPECIAL_SYMBOLS)

            # 1. Try to load from cache immediately
            self._load_from_cache()

            # 2. Start background thread to refresh data if requested
            if auto_refresh:
                refresh_thread = threading.Thread(
                    target=self._refresh_data, daemon=True
                )
                refresh_thread.start()

            ExchangeSymbol._initialized = True

    @property
    def cache_file(self) -> Path:
        """Resolves active cache file path."""
        return self._cache_file or CACHE_FILE

    def _load_from_cache(self) -> None:
        """Loads symbol lists from local JSON cache if available."""
        target_cache_file = self.cache_file
        if not target_cache_file.exists():
            logger.debug("No symbol cache found at %s", target_cache_file)
            return

        try:
            with target_cache_file.open("r", encoding="utf-8") as file_handle:
                cached_data = json.load(file_handle)

            if not isinstance(cached_data, dict):
                logger.warning("Invalid cache structure at %s", target_cache_file)
                return

            self._sp_500 = list(cached_data.get("sp_500", []))
            self._sp_100 = list(cached_data.get("sp_100", []))
            self._nasdaq_100 = list(cached_data.get("nasdaq_100", []))
            self._dow_30 = list(cached_data.get("dow_30", []))
            self._russell_1000 = list(cached_data.get("russell_1000", []))

            logger.debug(
                "✓ Loaded symbols from cache: SPX=%d, OEX=%d, NDX=%d, DOW=%d, RUI=%d",
                len(self._sp_500),
                len(self._sp_100),
                len(self._nasdaq_100),
                len(self._dow_30),
                len(self._russell_1000),
            )
        except (json.JSONDecodeError, OSError) as error:
            logger.error("Failed to load symbol cache: %s", error)
        except Exception as error:
            logger.error("Unexpected error loading symbol cache: %s", error)

    def _save_to_cache(self) -> None:
        """Saves current symbol lists to local JSON cache atomically."""
        target_cache_file = self.cache_file
        try:
            target_cache_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "sp_500": self._sp_500,
                "sp_100": self._sp_100,
                "nasdaq_100": self._nasdaq_100,
                "dow_30": self._dow_30,
                "russell_1000": self._russell_1000,
            }
            temporary_file = target_cache_file.with_suffix(".tmp")
            with temporary_file.open("w", encoding="utf-8") as file_handle:
                json.dump(payload, file_handle, indent=2)
            temporary_file.replace(target_cache_file)
            logger.debug("✓ Symbol cache saved atomically to %s", target_cache_file)
        except OSError as error:
            logger.error("Failed to save symbol cache: %s", error)
        except Exception as error:
            logger.error("Unexpected error saving symbol cache: %s", error)

    def _refresh_data(self) -> None:
        """Background task to fetch fresh data from Wikipedia."""
        logger.debug("Starting background symbol refresh...")

        try:
            fetched_results: dict[str, list[str]] = {}
            for source in DEFAULT_INDEX_SOURCES:
                symbols = self._fetch_from_wikipedia(
                    url=list(source.urls),
                    search_columns=list(source.search_columns),
                    name=source.name,
                )
                if symbols:
                    fetched_results[source.cache_key] = symbols

            if "sp_500" in fetched_results:
                self._sp_500 = fetched_results["sp_500"]
            if "sp_100" in fetched_results:
                self._sp_100 = fetched_results["sp_100"]
            if "nasdaq_100" in fetched_results:
                self._nasdaq_100 = fetched_results["nasdaq_100"]
            if "dow_30" in fetched_results:
                self._dow_30 = fetched_results["dow_30"]
            if "russell_1000" in fetched_results:
                self._russell_1000 = fetched_results["russell_1000"]

            logger.debug(
                "✓ Symbol refresh complete: S&P 500=%d, S&P 100=%d, NASDAQ-100=%d, Dow 30=%d, Russell 1000=%d",
                len(self._sp_500),
                len(self._sp_100),
                len(self._nasdaq_100),
                len(self._dow_30),
                len(self._russell_1000),
            )

            self._save_to_cache()

        except Exception as error:
            logger.error("Background symbol refresh failed: %s", error)

    def _fetch_from_wikipedia(
        self,
        url: str | Sequence[str],
        search_columns: Sequence[str],
        name: str,
    ) -> list[str]:
        """Fetches all tables from Wikipedia page(s) and identifies the correct one.

        Accepts a single URL or a sequence of fallback URLs.
        """
        urls = [url] if isinstance(url, str) else list(url)

        for target_url in urls:
            try:
                logger.debug("Fetching %s from %s...", name, target_url)

                try:
                    tables = pd.read_html(
                        target_url, storage_options={"User-Agent": "Mozilla/5.0"}
                    )
                except Exception as error:
                    logger.error("Error reading HTML from %s: %s", target_url, error)
                    continue

                symbols = extract_symbols_from_tables(tables, search_columns)
                if symbols:
                    return symbols

                logger.warning(
                    "Could not find matching symbols with columns %s for %s at %s. Found %d tables.",
                    search_columns,
                    name,
                    target_url,
                    len(tables),
                )

            except Exception as error:
                logger.error("Failed to load %s from %s: %s", name, target_url, error)

        return []

    @property
    def sp_500(self) -> list[str]:
        """Returns a copy of S&P 500 symbols."""
        return self._sp_500.copy()

    @property
    def sp_100(self) -> list[str]:
        """Returns a copy of S&P 100 symbols."""
        return self._sp_100.copy()

    @property
    def nasdaq_100(self) -> list[str]:
        """Returns a copy of NASDAQ-100 symbols."""
        return self._nasdaq_100.copy()

    @property
    def dow_30(self) -> list[str]:
        """Returns a copy of Dow Jones 30 symbols."""
        return self._dow_30.copy()

    @property
    def russell_1000(self) -> list[str]:
        """Returns a copy of Russell 1000 symbols."""
        return self._russell_1000.copy()

    @property
    def all(self) -> list[str]:
        """Returns a sorted, deduplicated list of all tracked equity symbols."""
        combined_symbols = set(
            self._dow_30
            + self._nasdaq_100
            + self._sp_500
            + self._sp_100
            + self._russell_1000
            + self._special_symbols
        )
        return sorted(combined_symbols)
