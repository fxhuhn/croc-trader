"""Provides equity symbol lists for major market indices fetched from Wikipedia.

Maintains local cache persistence and thread-safe memory state for S&P 500, S&P 100,
NASDAQ-100, Dow Jones 30, and Russell 1000 indices.
"""

import json
import logging
import threading
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Literal, TypedDict

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

DEFAULT_SPECIAL_SYMBOLS: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "SXRV.DE",
    "DIA",
    "^VIX",
    "TLT",
)


# --- Functional Core (Pure Transformations) ---


class IndexDiff(TypedDict):
    """Constituent differences for an index between iterations."""

    added: list[str]
    removed: list[str]
    total: int


class SymbolRefreshResult(TypedDict):
    """Structured result returned by symbol refresh operations."""

    status: Literal["success", "no_change", "error"]
    changed: bool
    diffs: dict[str, IndexDiff]
    timestamp: str
    error: str | None


def calculate_index_diff(
    old_symbols: Sequence[str],
    new_symbols: Sequence[str],
) -> IndexDiff:
    """Calculates added and removed symbols between two versions of an index list.

    Args:
        old_symbols: Previously known sequence of tickers.
        new_symbols: Newly fetched sequence of tickers.

    Returns:
        IndexDiff: Added and removed symbols sorted, with total count.
    """
    old_set = set(old_symbols)
    new_set = set(new_symbols)
    return {
        "added": sorted(new_set - old_set),
        "removed": sorted(old_set - new_set),
        "total": len(new_set),
    }


def fetch_url_metadata(url: str, timeout: float = 5.0) -> dict[str, str]:
    """Fetches HTTP header metadata (Last-Modified, ETag) via a lightweight HEAD request.

    Args:
        url: Target HTTP/HTTPS URL.
        timeout: Request timeout in seconds.

    Returns:
        dict[str, str]: Dictionary containing 'last_modified' and 'etag' if available.
    """
    if not url.startswith(("http://", "https://")):
        logger.warning("Rejected non-HTTP URL: %s", url)
        return {"last_modified": "", "etag": ""}

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; CrocTrader/1.0)"},
            method="HEAD",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            headers = response.headers
            return {
                "last_modified": headers.get("Last-Modified", ""),
                "etag": headers.get("ETag", "").strip('"'),
            }
    except Exception as error:
        logger.debug("HEAD request failed for %s: %s", url, error)
        return {"last_modified": "", "etag": ""}


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
            self._metadata: dict[str, dict[str, str]] = {}
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

    @property
    def metadata(self) -> dict[str, dict[str, str]]:
        """Returns a copy of the index metadata mapping."""
        return {key: dict(val) for key, val in self._metadata.items()}

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
            raw_metadata = cached_data.get("metadata", {})
            self._metadata = (
                {
                    str(k): dict(v)
                    for k, v in raw_metadata.items()
                    if isinstance(v, dict)
                }
                if isinstance(raw_metadata, dict)
                else {}
            )

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
                "metadata": self._metadata,
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

    def refresh_if_modified(self, *, force: bool = False) -> SymbolRefreshResult:
        """Fetches and updates index symbol lists if Wikipedia pages were modified.

        Performs a lightweight HEAD check on Wikipedia pages. If the Last-Modified header
        is identical to the saved metadata, scraping is skipped unless force=True.

        Args:
            force: When True, bypasses Last-Modified check and forces table re-parsing.

        Returns:
            SymbolRefreshResult: Outcome status, boolean changed indicator, and symbol diffs.
        """
        logger.debug("Starting symbol refresh (force=%s)...", force)
        timestamp = datetime.now(UTC).isoformat()
        diffs: dict[str, IndexDiff] = {}
        any_changed = False

        try:
            with ExchangeSymbol._lock:
                for source in DEFAULT_INDEX_SOURCES:
                    primary_url = source.urls[0]
                    url_meta = (
                        self._fetch_url_metadata(primary_url) if not force else {}
                    )
                    cached_meta = self._metadata.get(source.cache_key, {})

                    # If not forced and Last-Modified matches cached value, skip parsing
                    if (
                        not force
                        and url_meta.get("last_modified")
                        and url_meta.get("last_modified")
                        == cached_meta.get("last_modified")
                    ):
                        logger.debug(
                            "Index %s is unchanged (Last-Modified match).", source.name
                        )
                        continue

                    symbols = self._fetch_from_wikipedia(
                        url=list(source.urls),
                        search_columns=list(source.search_columns),
                        name=source.name,
                    )

                    if not symbols:
                        logger.warning(
                            "Skipping update for %s: No symbols parsed (preserving existing).",
                            source.name,
                        )
                        continue

                    current_symbols = getattr(self, f"_{source.cache_key}", [])
                    diff = calculate_index_diff(current_symbols, symbols)

                    if diff["added"] or diff["removed"]:
                        any_changed = True
                        diffs[source.cache_key] = diff
                        setattr(self, f"_{source.cache_key}", symbols)
                        logger.info(
                            "Index %s updated: +%d added, -%d removed (Total: %d).",
                            source.name,
                            len(diff["added"]),
                            len(diff["removed"]),
                            diff["total"],
                        )

                    if url_meta.get("last_modified") or url_meta.get("etag"):
                        self._metadata[source.cache_key] = url_meta

                if any_changed or force:
                    self._save_to_cache()

            status: Literal["success", "no_change"] = (
                "success" if any_changed else "no_change"
            )
            return {
                "status": status,
                "changed": any_changed,
                "diffs": diffs,
                "timestamp": timestamp,
                "error": None,
            }

        except Exception as error:
            logger.error("Symbol refresh failed: %s", error, exc_info=True)
            return {
                "status": "error",
                "changed": False,
                "diffs": {},
                "timestamp": timestamp,
                "error": str(error),
            }

    def _fetch_url_metadata(self, url: str, timeout: float = 5.0) -> dict[str, str]:
        """Fetches HTTP header metadata via HEAD request."""
        return fetch_url_metadata(url, timeout=timeout)

    def _refresh_data(self, *, force: bool = True) -> SymbolRefreshResult:
        """Legacy refresh method; defaults to force=True for backward compatibility with existing tests."""
        return self.refresh_if_modified(force=force)

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
