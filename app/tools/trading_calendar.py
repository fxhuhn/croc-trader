"""Trading calendar utilities for market trading day calculations.

Provides centralized helper functions for calculating previous/next trading days,
evaluating market trading days against weekends and holidays, and calculating
month-end trading windows.
"""

import datetime

import pandas as pd

from app.tools.market_holidays import MarketHolidayChecker

# Monday=0 ... Saturday=5, Sunday=6 per datetime.weekday()
SATURDAY: int = 5
DECEMBER_MONTH: int = 12


def is_trading_day(
    check_date: datetime.date,
    holiday_checker: MarketHolidayChecker | None = None,
) -> bool:
    """Checks whether check_date is an active market trading day.

    A trading day is defined as a weekday (Monday to Friday) that is not an
    official market holiday.

    Args:
        check_date: The date to evaluate.
        holiday_checker: Optional holiday checker instance. If None, uses default.

    Returns:
        bool: True if check_date is a trading day, False otherwise.
    """
    checker = holiday_checker or MarketHolidayChecker()
    return check_date.weekday() < SATURDAY and not checker.is_holiday(check_date)


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
    while not is_trading_day(candidate_date, checker):
        candidate_date -= datetime.timedelta(days=1)

    return candidate_date


def get_next_trading_day(
    reference_date: datetime.date,
    holiday_checker: MarketHolidayChecker | None = None,
) -> datetime.date:
    """Calculates the next market trading day forward from reference_date.

    Checks forwards starting from (reference_date + 1 day). Skips weekend days
    (Saturday, Sunday) and official market holidays.

    Args:
        reference_date: The date from which to look forward.
        holiday_checker: Optional holiday checker instance. If None, uses default.

    Returns:
        datetime.date: The date of the next market trading day.
    """
    checker = holiday_checker or MarketHolidayChecker()

    candidate_date = reference_date + datetime.timedelta(days=1)
    while not is_trading_day(candidate_date, checker):
        candidate_date += datetime.timedelta(days=1)

    return candidate_date


def get_remaining_trading_days_in_month(
    check_date: datetime.date,
    holiday_checker: MarketHolidayChecker | None = None,
) -> int:
    """Calculates remaining trading days in check_date's month including check_date.

    Args:
        check_date: Target date to check.
        holiday_checker: Optional holiday checker instance.

    Returns:
        int: Number of remaining trading days (>= 0).
    """
    checker = holiday_checker or MarketHolidayChecker()

    if check_date.month == DECEMBER_MONTH:
        next_month_start = datetime.date(check_date.year + 1, 1, 1)
    else:
        next_month_start = datetime.date(check_date.year, check_date.month + 1, 1)
    last_day_of_month = next_month_start - datetime.timedelta(days=1)

    trading_day_count = 0
    current = check_date
    while current <= last_day_of_month:
        if is_trading_day(current, checker):
            trading_day_count += 1
        current += datetime.timedelta(days=1)

    return trading_day_count


def is_in_end_of_month_window(
    check_date: datetime.date,
    days_before: int = 4,
    holiday_checker: MarketHolidayChecker | None = None,
) -> bool:
    """Checks whether check_date falls within the End-of-Month window.

    Window is active starting days_before trading days prior to the last
    trading day of the month up to the last trading day.

    Args:
        check_date: Target date.
        days_before: Number of trading days before month end (default: 4).
        holiday_checker: Optional holiday checker.

    Returns:
        bool: True if in month-end window, False otherwise.
    """
    remaining_days = get_remaining_trading_days_in_month(
        check_date, holiday_checker=holiday_checker
    )
    return 1 <= remaining_days <= (days_before + 1)


def is_last_trading_day_of_month(
    check_date: datetime.date,
    holiday_checker: MarketHolidayChecker | None = None,
) -> bool:
    """Checks whether check_date is the last active trading day of its calendar month.

    Args:
        check_date: Date to evaluate.
        holiday_checker: Optional holiday checker instance.

    Returns:
        bool: True if check_date is an active trading day and no further trading days exist in the month.
    """
    checker = holiday_checker or MarketHolidayChecker()
    return (
        is_trading_day(check_date, checker)
        and get_remaining_trading_days_in_month(check_date, checker) == 1
    )


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


__all__ = [
    "DECEMBER_MONTH",
    "SATURDAY",
    "get_last_completed_trading_day",
    "get_next_trading_day",
    "get_remaining_trading_days_in_month",
    "is_in_end_of_month_window",
    "is_last_trading_day_of_month",
    "is_trading_day",
    "resolve_effective_trading_date",
]
