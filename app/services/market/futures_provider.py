"""Futures market data provider for TradingView.

Resolves front-month futures contracts via date-based heuristic and fetches
30-minute price bars from TradingView using the tvDatafeed library.

Supported instruments: MNQ (Micro Nasdaq), MES (Micro S&P 500).
"""

import calendar
import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import TypedDict, cast

import pandas as pd
from tvDatafeed import Interval, TvDatafeed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain Models (Functional Core)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FuturesContractSpec:
    """Static configuration for a futures product on TradingView."""

    tv_prefix: str
    exchange: str


FUTURES_REGISTRY: dict[str, FuturesContractSpec] = {
    "MNQ": FuturesContractSpec(tv_prefix="MNQ", exchange="CME_MINI"),
    "MES": FuturesContractSpec(tv_prefix="MES", exchange="CME_MINI"),
}

# Quarterly expiry cycle: March (H), June (M), September (U), December (Z)
QUARTERLY_MONTHS: tuple[int, ...] = (3, 6, 9, 12)
QUARTERLY_MONTH_CODES: dict[int, str] = {3: "H", 6: "M", 9: "U", 12: "Z"}


@dataclass(frozen=True)
class FuturesContract:
    """Resolved futures contract with TradingView symbol."""

    base_symbol: str  # "MNQ"
    tv_symbol: str  # "MNQU2026"
    exchange: str  # "CME_MINI"
    month_code: str  # "U"
    expiry_month: int  # 9
    expiry_year: int  # 2026


class FuturesBarRecord(TypedDict, total=False):
    """Typed representation of a 30-minute futures bar from TradingView."""

    datetime: object
    open: float
    high: float
    low: float
    close: float
    volume: float | int


# ---------------------------------------------------------------------------
# Pure Functions (Functional Core)
# ---------------------------------------------------------------------------


def _find_third_friday(year: int, month: int) -> date:
    """Returns the date of the third Friday in the given month.

    The third Friday is the standard expiry date for CME index futures.
    """
    # Find the first day of the month and its weekday
    first_day_weekday = calendar.weekday(year, month, 1)  # 0=Monday, 4=Friday
    # Days until first Friday
    days_to_first_friday = (4 - first_day_weekday) % 7
    first_friday = 1 + days_to_first_friday
    third_friday = first_friday + 14
    return date(year, month, third_friday)


def resolve_front_month_contract(
    base_symbol: str,
    reference_date: date,
) -> FuturesContract:
    """Determines the active front-month contract via date-based heuristic.

    Uses the 3rd Friday of the expiry month as the rollover boundary.
    Before the 3rd Friday: the current quarterly expiry is active.
    On or after the 3rd Friday: the next quarterly expiry is active.

    Args:
        base_symbol: Internal symbol (e.g. "MNQ" or "MES").
        reference_date: Date for which to resolve the active contract.

    Returns:
        Resolved FuturesContract with TradingView symbol and exchange.

    Raises:
        ValueError: If base_symbol is not in FUTURES_REGISTRY.
    """
    if base_symbol not in FUTURES_REGISTRY:
        raise ValueError(
            f"Unknown futures symbol: {base_symbol}. "
            f"Available: {', '.join(sorted(FUTURES_REGISTRY))}"
        )

    spec = FUTURES_REGISTRY[base_symbol]
    expiry_month, expiry_year = _find_next_expiry(reference_date)
    month_code = QUARTERLY_MONTH_CODES[expiry_month]
    tv_symbol = f"{spec.tv_prefix}{month_code}{expiry_year}"

    return FuturesContract(
        base_symbol=base_symbol,
        tv_symbol=tv_symbol,
        exchange=spec.exchange,
        month_code=month_code,
        expiry_month=expiry_month,
        expiry_year=expiry_year,
    )


def _find_next_expiry(reference_date: date) -> tuple[int, int]:
    """Determines the next active expiry month and year.

    If the reference date is before the 3rd Friday of the current quarterly
    month, that quarter is active. Otherwise, the next quarter.
    """
    year = reference_date.year
    month = reference_date.month

    # Find the current or next quarterly month
    current_quarter_month = _current_or_next_quarterly_month(month)

    # Check if we're past the 3rd Friday of the current quarterly month
    if current_quarter_month >= month:
        third_friday = _find_third_friday(year, current_quarter_month)
        if reference_date < third_friday:
            return current_quarter_month, year
        # Past 3rd Friday: roll to next quarter
        return _advance_quarter(current_quarter_month, year)

    # Quarter month is in a future month already
    return current_quarter_month, year


def _current_or_next_quarterly_month(month: int) -> int:
    """Returns the current or next quarterly expiry month for a given month."""
    for quarterly_month in QUARTERLY_MONTHS:
        if month <= quarterly_month:
            return quarterly_month
    # Past December: wrap to March of next year (handled by caller)
    return QUARTERLY_MONTHS[0]


def _advance_quarter(current_month: int, year: int) -> tuple[int, int]:
    """Advances to the next quarterly expiry month, wrapping year if needed."""
    current_index = list(QUARTERLY_MONTHS).index(current_month)
    if current_index + 1 < len(QUARTERLY_MONTHS):
        return QUARTERLY_MONTHS[current_index + 1], year
    return QUARTERLY_MONTHS[0], year + 1


# ---------------------------------------------------------------------------
# Imperative Shell
# ---------------------------------------------------------------------------


class FuturesDataProvider:
    """Fetches 30-minute futures bars from TradingView via tvDatafeed.

    Handles connection lifecycle, retries, and rate limiting.
    """

    def __init__(
        self,
        max_retries: int = 2,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        """Initializes the provider with retry configuration."""
        self._tv: TvDatafeed | None = None
        self._max_retries: int = max_retries
        self._retry_delay_seconds: float = retry_delay_seconds

    def _get_instance(self) -> TvDatafeed:
        """Lazily instantiates and returns the TvDatafeed client."""
        if self._tv is None:
            self._tv = TvDatafeed()
        return self._tv

    def fetch_history(
        self,
        contract: FuturesContract,
        number_of_bars: int = 500,
    ) -> list[FuturesBarRecord]:
        """Fetches 30-minute price bars for a futures contract from TradingView.

        Args:
            contract: Resolved futures contract with TV symbol and exchange.
            number_of_bars: Number of 30-minute bars to retrieve.

        Returns:
            List of typed bar records ready for FuturesPrice parsing.
        """
        dataframe = self._query_dataframe(contract, number_of_bars)

        if dataframe is None or dataframe.empty:
            logger.warning(
                "TradingView returned empty data for %s on %s",
                contract.tv_symbol,
                contract.exchange,
            )
            return []

        records = self._standardize_records(dataframe)
        time.sleep(0.5)
        return records

    def _query_dataframe(
        self,
        contract: FuturesContract,
        number_of_bars: int,
    ) -> pd.DataFrame | None:
        """Queries TvDatafeed for 30-minute bars with retry logic."""
        for attempt in range(1, self._max_retries + 1):
            try:
                datafeed = self._get_instance()
                dataframe = datafeed.get_hist(
                    symbol=contract.tv_symbol,
                    exchange=contract.exchange,
                    interval=Interval.in_30_minute,
                    n_bars=number_of_bars,
                )
                if dataframe is not None and not dataframe.empty:
                    return dataframe

                self._tv = None
                if attempt < self._max_retries:
                    logger.debug(
                        "TradingView returned no data for %s (attempt %d/%d), retrying...",
                        contract.tv_symbol,
                        attempt,
                        self._max_retries,
                    )
                    time.sleep(self._retry_delay_seconds)
                else:
                    logger.warning(
                        "TradingView returned no data for %s after %d attempts",
                        contract.tv_symbol,
                        self._max_retries,
                    )
            except Exception as error:
                self._tv = None
                if attempt < self._max_retries:
                    logger.warning(
                        "TradingView error for %s (attempt %d/%d): %s. Retrying...",
                        contract.tv_symbol,
                        attempt,
                        self._max_retries,
                        error,
                    )
                    time.sleep(self._retry_delay_seconds)
                else:
                    logger.warning(
                        "TradingView error for %s after %d attempts: %s",
                        contract.tv_symbol,
                        self._max_retries,
                        error,
                    )

        return None

    @staticmethod
    def _standardize_records(dataframe: pd.DataFrame) -> list[FuturesBarRecord]:
        """Transforms raw TvDatafeed DataFrame into standardized records."""
        cleaned = dataframe.copy()
        cleaned.columns = cleaned.columns.str.lower()
        cleaned = cleaned.reset_index()

        rename_map: dict[str, str] = {}
        for column in cleaned.columns:
            if str(column).lower() in ("index",):
                rename_map[str(column)] = "datetime"
        if rename_map:
            cleaned = cleaned.rename(columns=rename_map)

        # Drop TradingView 'symbol' column (contains exchange-prefixed name)
        if "symbol" in cleaned.columns:
            cleaned = cleaned.drop(columns=["symbol"])

        return cast(list[FuturesBarRecord], cleaned.to_dict("records"))
