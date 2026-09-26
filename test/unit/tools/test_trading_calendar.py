import datetime
from unittest.mock import MagicMock

import pandas as pd

from app.tools.market_holidays import MarketHolidayChecker
from app.tools.trading_calendar import (
    get_last_completed_trading_day,
    get_next_trading_day,
    get_remaining_trading_days_in_month,
    is_in_end_of_month_window,
    is_last_trading_day_of_month,
    is_trading_day,
    resolve_effective_trading_date,
)


def test_get_last_completed_trading_day_regular_weekday() -> None:
    """Wednesday 2026-07-15 -> Last completed trading day should be Tuesday 2026-07-14."""
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    ref_date = datetime.date(2026, 7, 15)  # Wednesday
    result = get_last_completed_trading_day(ref_date, holiday_checker)
    assert result == datetime.date(2026, 7, 14)  # Tuesday


def test_get_last_completed_trading_day_saturday() -> None:
    """Saturday 2026-07-18 -> Last completed trading day should be Friday 2026-07-17."""
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    ref_date = datetime.date(2026, 7, 18)  # Saturday
    result = get_last_completed_trading_day(ref_date, holiday_checker)
    assert result == datetime.date(2026, 7, 17)  # Friday


def test_get_last_completed_trading_day_monday() -> None:
    """Monday 2026-07-20 -> Last completed trading day should be Friday 2026-07-17."""
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    ref_date = datetime.date(2026, 7, 20)  # Monday
    result = get_last_completed_trading_day(ref_date, holiday_checker)
    assert result == datetime.date(2026, 7, 17)  # Friday


def test_get_last_completed_trading_day_friday_holiday() -> None:
    """Saturday 2026-04-04 when Friday 2026-04-03 is a holiday -> Last completed trading day is Thursday 2026-04-02."""
    holiday_checker = MagicMock(spec=MarketHolidayChecker)

    def is_holiday_side_effect(dt: datetime.date | str) -> bool:
        return dt == datetime.date(2026, 4, 3)

    holiday_checker.is_holiday.side_effect = is_holiday_side_effect

    ref_date = datetime.date(2026, 4, 4)  # Saturday
    result = get_last_completed_trading_day(ref_date, holiday_checker)
    assert result == datetime.date(2026, 4, 2)  # Thursday


def test_resolve_effective_trading_date_exact_match() -> None:
    """Exact match when requested date is in available dates."""
    dates = pd.DatetimeIndex(["2026-06-01", "2026-06-02", "2026-06-03"])
    target = pd.Timestamp("2026-06-02")
    assert resolve_effective_trading_date(dates, target) == target


def test_resolve_effective_trading_date_data_lag() -> None:
    """Data lag fallback when target is beyond available dates."""
    dates = pd.DatetimeIndex(["2026-06-01", "2026-06-02", "2026-06-03"])
    target = pd.Timestamp("2026-06-05")
    assert resolve_effective_trading_date(dates, target) == pd.Timestamp("2026-06-03")


def test_resolve_effective_trading_date_holiday_gap() -> None:
    """Holiday gap fallback when target date falls on missing date."""
    dates = pd.DatetimeIndex(["2026-06-01", "2026-06-02", "2026-06-04"])
    target = pd.Timestamp("2026-06-03")  # Missing date
    assert resolve_effective_trading_date(dates, target) == pd.Timestamp("2026-06-02")


def test_resolve_effective_trading_date_empty_or_too_old() -> None:
    """Returns None when available dates are empty or gap exceeds max fallback."""
    dates = pd.DatetimeIndex([])
    target = pd.Timestamp("2026-06-03")
    assert resolve_effective_trading_date(dates, target) is None

    old_dates = pd.DatetimeIndex(["2026-01-01"])
    assert (
        resolve_effective_trading_date(old_dates, target, max_fallback_days=5) is None
    )


def test_is_trading_day_weekdays_and_weekends() -> None:
    """Verifies that is_trading_day returns True for weekdays and False for weekends."""
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    monday = datetime.date(2026, 7, 13)
    friday = datetime.date(2026, 7, 17)
    saturday = datetime.date(2026, 7, 18)
    sunday = datetime.date(2026, 7, 19)

    assert is_trading_day(monday, holiday_checker) is True
    assert is_trading_day(friday, holiday_checker) is True
    assert is_trading_day(saturday, holiday_checker) is False
    assert is_trading_day(sunday, holiday_checker) is False


def test_is_trading_day_holiday() -> None:
    """Verifies that an official holiday on a weekday is not a trading day."""
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_date = datetime.date(2026, 7, 3)
    holiday_checker.is_holiday.side_effect = lambda d: d == holiday_date

    assert is_trading_day(holiday_date, holiday_checker) is False
    assert is_trading_day(datetime.date(2026, 7, 2), holiday_checker) is True


def test_get_next_trading_day_regular_weekday() -> None:
    """Tuesday 2026-07-14 -> Next trading day should be Wednesday 2026-07-15."""
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    ref_date = datetime.date(2026, 7, 14)
    assert get_next_trading_day(ref_date, holiday_checker) == datetime.date(2026, 7, 15)


def test_get_next_trading_day_friday_to_monday() -> None:
    """Friday 2026-07-17 -> Next trading day should be Monday 2026-07-20."""
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    ref_date = datetime.date(2026, 7, 17)
    assert get_next_trading_day(ref_date, holiday_checker) == datetime.date(2026, 7, 20)


def test_get_next_trading_day_over_holiday() -> None:
    """Thursday 2026-04-02 when Friday 2026-04-03 is a holiday -> Next trading day is Monday 2026-04-06."""
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_date = datetime.date(2026, 4, 3)
    holiday_checker.is_holiday.side_effect = lambda d: d == holiday_date

    ref_date = datetime.date(2026, 4, 2)
    assert get_next_trading_day(ref_date, holiday_checker) == datetime.date(2026, 4, 6)


def test_get_remaining_trading_days_in_month_regular() -> None:
    """July 2026 has 31 days (Friday July 31). Tests remaining count."""
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    # July 27 is Monday, July 31 is Friday -> 5 trading days
    assert (
        get_remaining_trading_days_in_month(datetime.date(2026, 7, 27), holiday_checker)
        == 5
    )
    assert (
        get_remaining_trading_days_in_month(datetime.date(2026, 7, 31), holiday_checker)
        == 1
    )


def test_get_remaining_trading_days_in_month_with_holidays() -> None:
    """July 31 is a holiday -> July 30 is the last trading day."""
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_date = datetime.date(2026, 7, 31)
    holiday_checker.is_holiday.side_effect = lambda d: d == holiday_date

    assert (
        get_remaining_trading_days_in_month(datetime.date(2026, 7, 27), holiday_checker)
        == 4
    )
    assert (
        get_remaining_trading_days_in_month(datetime.date(2026, 7, 30), holiday_checker)
        == 1
    )
    assert (
        get_remaining_trading_days_in_month(datetime.date(2026, 7, 31), holiday_checker)
        == 0
    )


def test_get_remaining_trading_days_in_month_december_and_leap_year() -> None:
    """Tests December rollover and February leap year vs non-leap year."""
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    # Dec 31, 2026 is Thursday
    assert (
        get_remaining_trading_days_in_month(
            datetime.date(2026, 12, 31), holiday_checker
        )
        == 1
    )

    # Leap year Feb 29, 2024 is Thursday
    assert (
        get_remaining_trading_days_in_month(datetime.date(2024, 2, 29), holiday_checker)
        == 1
    )

    # Non-leap year Feb 27, 2026 is Friday (Feb 28 is Saturday)
    assert (
        get_remaining_trading_days_in_month(datetime.date(2026, 2, 27), holiday_checker)
        == 1
    )


def test_is_in_end_of_month_window() -> None:
    """Tests month-end window evaluation with days_before parameter."""
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    # July 2026: last trading day is Friday July 31
    # days_before=4 -> window covers 5 trading days: July 27(Mon) to July 31(Fri)
    assert (
        is_in_end_of_month_window(
            datetime.date(2026, 7, 27), days_before=4, holiday_checker=holiday_checker
        )
        is True
    )
    assert (
        is_in_end_of_month_window(
            datetime.date(2026, 7, 31), days_before=4, holiday_checker=holiday_checker
        )
        is True
    )
    assert (
        is_in_end_of_month_window(
            datetime.date(2026, 7, 24), days_before=4, holiday_checker=holiday_checker
        )
        is False
    )


def test_is_last_trading_day_of_month() -> None:
    """Tests last trading day of month check on trading day, non-last day, and weekend."""
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    # Friday July 31 is the last trading day of July 2026
    assert (
        is_last_trading_day_of_month(datetime.date(2026, 7, 31), holiday_checker)
        is True
    )
    # Thursday July 30 is not the last trading day
    assert (
        is_last_trading_day_of_month(datetime.date(2026, 7, 30), holiday_checker)
        is False
    )
    # Saturday August 1 is not a trading day
    assert (
        is_last_trading_day_of_month(datetime.date(2026, 8, 1), holiday_checker)
        is False
    )

    # If July 31 is holiday, July 30 is the last trading day
    holiday_date = datetime.date(2026, 7, 31)
    holiday_checker.is_holiday.side_effect = lambda d: d == holiday_date
    assert (
        is_last_trading_day_of_month(datetime.date(2026, 7, 30), holiday_checker)
        is True
    )
    assert (
        is_last_trading_day_of_month(datetime.date(2026, 7, 31), holiday_checker)
        is False
    )
