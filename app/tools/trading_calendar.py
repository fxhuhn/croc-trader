import datetime

import pandas as pd

from app.tools.market_holidays import MarketHolidayChecker


def get_last_completed_trading_day(
    reference_date: datetime.date,
    holiday_checker: MarketHolidayChecker | None = None,
) -> datetime.date:
    """Calculates the most recent completed market trading day.

    Checks backwards starting from (reference_date - 1 day), since the current
    calendar day is either ongoing or hasn't started yet. Skips weekend days
    (Saturday, Sunday) and official market holidays.

    Args:
        reference_date: The date from which to look backward.
        holiday_checker: Optional holiday checker instance. If None, uses default.

    Returns:
        datetime.date: The date of the most recent completed trading day.
    """
    checker = holiday_checker or MarketHolidayChecker()

    candidate_date = reference_date - datetime.timedelta(days=1)
    while candidate_date.weekday() >= 5 or checker.is_holiday(candidate_date):
        candidate_date -= datetime.timedelta(days=1)

    return candidate_date


def resolve_effective_trading_date(
    available_dates: pd.Index,
    target_date: pd.Timestamp | datetime.date | str,
    max_fallback_days: int = 10,
) -> pd.Timestamp | None:
    """Resolves target trading date against available market price dates.

    Handles:
    1. Exact Match: target_date exists in available_dates index.
    2. Post-Data Case (Data Lag): target_date > latest available date, falls back to latest date.
    3. Gap Case (Holiday/Weekend): target_date not in dates, falls back to latest prior date.

    Args:
        available_dates: Aligned datetime index of available prices.
        target_date: Requested target date (Timestamp, date, or ISO string).
        max_fallback_days: Maximum calendar days lookback for holiday gap (default: 10).

    Returns:
        pd.Timestamp | None: The effective timestamp to run analysis against, or None.
    """
    if available_dates.empty:
        return None

    try:
        target_timestamp = pd.Timestamp(target_date)
    except (ValueError, TypeError):
        return None

    if target_timestamp in available_dates:
        return target_timestamp

    if target_timestamp > available_dates[-1]:
        candidate_date = available_dates[-1]
    else:
        prior_dates = available_dates[available_dates < target_timestamp]
        if prior_dates.empty:
            return None
        candidate_date = prior_dates[-1]

    if (target_timestamp - candidate_date).days > max_fallback_days:
        return None

    return candidate_date
