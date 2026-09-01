"""Cash-session daily bar aggregation from 30-minute futures bars.

Aggregates intraday futures bars into daily OHLCV bars using only the
US equity cash session (09:30–16:00 ET). This allows direct comparison
with cash ETFs such as QQQ and SPY.

With 30-minute bars from TradingView (exchange time / US Eastern):
- Cash Open  = Open of the 09:30 bar (exact match with equity open)
- Cash Close = Close of the 15:30 bar (bar ends at 16:00 = equity close)
- Cash High  = max(high) over all cash-session bars
- Cash Low   = min(low) over all cash-session bars
- Cash Volume = sum(volume) over all cash-session bars

This module contains only pure functions — no I/O, no database, no logging.
"""

import zoneinfo
from dataclasses import dataclass
from datetime import datetime

from app.models import FuturesPrice

# Target Exchange Timezone (US Equity Market is ALWAYS 09:30–16:00 ET)
US_EASTERN_TIMEZONE: zoneinfo.ZoneInfo = zoneinfo.ZoneInfo("America/New_York")
LOCAL_TIMEZONE: zoneinfo.ZoneInfo = zoneinfo.ZoneInfo("Europe/Berlin")

US_CASH_START_HOUR: int = 9
US_CASH_START_MINUTE: int = 30
US_CASH_END_HOUR: int = 15
US_CASH_END_MINUTE: int = 30

# Number of 30-min bars in a full cash session: 13 bars (09:30 ET through 15:30 ET)
FULL_CASH_SESSION_BAR_COUNT: int = 13
# Minimum required bars for a valid partial session
MINIMUM_CASH_SESSION_BAR_COUNT: int = 10


@dataclass(frozen=True)
class CashSessionDailyBar:
    """Daily futures bar aggregated from cash-session hours only.

    Comparable with QQQ/SPY daily bars from Yahoo Finance.
    """

    symbol: str  # "MNQ"
    contract: str  # "MNQU2026"
    date: str  # "2026-08-29" (US Trading Date)
    open: float  # Open of 09:30 ET bar (Kassa-Open)
    high: float  # max(high) over cash session
    low: float  # min(low) over cash session
    close: float  # Close of 15:30 ET bar (ends at 16:00 ET / Kassa-Close)
    volume: int  # sum(volume) over cash session

    def to_db_row(self) -> tuple[object, ...]:
        """Serializes to a tuple for executemany insertion."""
        return (
            self.symbol,
            self.contract,
            self.date,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            "cash",
        )


def aggregate_cash_session_daily_bars(
    hourly_bars: list[FuturesPrice],
) -> list[CashSessionDailyBar]:
    """Aggregates 30-minute futures bars into cash-session daily bars.

    Filters bars to the US equity cash session (09:30–16:00 ET),
    groups by US trading date, and computes OHLCV.

    Days with fewer than MINIMUM_CASH_SESSION_BAR_COUNT bars are excluded
    to avoid producing misleading daily bars from incomplete sessions.

    Args:
        hourly_bars: List of FuturesPrice records.

    Returns:
        List of CashSessionDailyBar, sorted by date ascending.
    """
    cash_bars_by_date = _group_cash_session_bars_by_date(hourly_bars)
    daily_bars: list[CashSessionDailyBar] = []

    for trading_date in sorted(cash_bars_by_date):
        bars = cash_bars_by_date[trading_date]
        if len(bars) < MINIMUM_CASH_SESSION_BAR_COUNT:
            continue

        daily_bar = _build_daily_bar_from_session(trading_date, bars)
        if daily_bar is not None:
            daily_bars.append(daily_bar)

    return daily_bars


def _normalize_to_us_eastern(bar_datetime: datetime) -> datetime:
    """Normalizes a bar datetime from local (Europe/Berlin) to US Eastern time."""
    if bar_datetime.tzinfo is None:
        aware_local = bar_datetime.replace(tzinfo=LOCAL_TIMEZONE)
    else:
        aware_local = bar_datetime
    return aware_local.astimezone(US_EASTERN_TIMEZONE)


def _group_cash_session_bars_by_date(
    bars: list[FuturesPrice],
) -> dict[str, list[FuturesPrice]]:
    """Groups bars by US trading date, filtering to cash-session hours only (09:30–16:00 ET)."""
    grouped: dict[str, list[FuturesPrice]] = {}

    for bar in bars:
        bar_datetime = _parse_bar_time(bar.bar_time)
        if bar_datetime is None:
            continue

        if not _is_within_cash_session(bar_datetime):
            continue

        ny_datetime = _normalize_to_us_eastern(bar_datetime)
        trading_date = ny_datetime.strftime("%Y-%m-%d")
        if trading_date not in grouped:
            grouped[trading_date] = []
        grouped[trading_date].append(bar)

    return grouped


def _is_within_cash_session(bar_datetime: datetime) -> bool:
    """Checks if a bar's timestamp falls within the US equity cash session (09:30–16:00 ET).

    Converts to America/New_York time so that DST transitions (Sommer-/Winterzeit)
    between US and Europe are handled automatically.
    """
    ny_datetime = _normalize_to_us_eastern(bar_datetime)
    bar_minutes = ny_datetime.hour * 60 + ny_datetime.minute
    session_start_minutes = US_CASH_START_HOUR * 60 + US_CASH_START_MINUTE
    session_end_minutes = US_CASH_END_HOUR * 60 + US_CASH_END_MINUTE
    return session_start_minutes <= bar_minutes <= session_end_minutes


def _build_daily_bar_from_session(
    trading_date: str,
    bars: list[FuturesPrice],
) -> CashSessionDailyBar | None:
    """Constructs a CashSessionDailyBar from sorted session bars.

    Returns None if the bars cannot form a valid daily bar
    (e.g. missing the session-open or session-close bar).
    """
    # Sort by bar_time to ensure chronological order
    sorted_bars = sorted(bars, key=lambda bar: bar.bar_time)

    first_bar = sorted_bars[0]
    last_bar = sorted_bars[-1]

    return CashSessionDailyBar(
        symbol=first_bar.symbol,
        contract=first_bar.contract,
        date=trading_date,
        open=first_bar.open,
        high=max(bar.high for bar in sorted_bars),
        low=min(bar.low for bar in sorted_bars),
        close=last_bar.close,
        volume=sum(bar.volume for bar in sorted_bars),
    )


def _parse_bar_time(bar_time: str) -> datetime | None:
    """Parses a bar_time string into a datetime object.

    Handles both 'YYYY-MM-DDTHH:MM:SS' and 'YYYY-MM-DD HH:MM:SS' formats.
    Returns None for unparseable values.
    """
    for date_format in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(bar_time, date_format)
        except ValueError:
            continue
    return None
