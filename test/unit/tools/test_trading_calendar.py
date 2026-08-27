import datetime
from unittest.mock import MagicMock

import pandas as pd

from app.tools.market_holidays import MarketHolidayChecker
from app.tools.trading_calendar import (
    get_last_completed_trading_day,
    resolve_effective_trading_date,
)


def test_get_last_completed_trading_day_regular_weekday():
    """Wednesday 2026-07-15 -> Last completed trading day should be Tuesday 2026-07-14."""
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    ref_date = datetime.date(2026, 7, 15)  # Wednesday
    result = get_last_completed_trading_day(ref_date, holiday_checker)
    assert result == datetime.date(2026, 7, 14)  # Tuesday


def test_get_last_completed_trading_day_saturday():
    """Saturday 2026-07-18 -> Last completed trading day should be Friday 2026-07-17."""
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    ref_date = datetime.date(2026, 7, 18)  # Saturday
    result = get_last_completed_trading_day(ref_date, holiday_checker)
    assert result == datetime.date(2026, 7, 17)  # Friday


def test_get_last_completed_trading_day_monday():
    """Monday 2026-07-20 -> Last completed trading day should be Friday 2026-07-17."""
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    ref_date = datetime.date(2026, 7, 20)  # Monday
    result = get_last_completed_trading_day(ref_date, holiday_checker)
    assert result == datetime.date(2026, 7, 17)  # Friday


def test_get_last_completed_trading_day_friday_holiday():
    """Saturday 2026-04-04 when Friday 2026-04-03 is a holiday -> Last completed trading day is Thursday 2026-04-02."""
    holiday_checker = MagicMock(spec=MarketHolidayChecker)

    def is_holiday_side_effect(dt: datetime.date) -> bool:
        return dt == datetime.date(2026, 4, 3)

    holiday_checker.is_holiday.side_effect = is_holiday_side_effect

    ref_date = datetime.date(2026, 4, 4)  # Saturday
    result = get_last_completed_trading_day(ref_date, holiday_checker)
    assert result == datetime.date(2026, 4, 2)  # Thursday


def test_resolve_effective_trading_date_exact_match():
    """Exact match when requested date is in available dates."""
    dates = pd.DatetimeIndex(["2026-06-01", "2026-06-02", "2026-06-03"])
    target = pd.Timestamp("2026-06-02")
    assert resolve_effective_trading_date(dates, target) == target


def test_resolve_effective_trading_date_data_lag():
    """Data lag fallback when target is beyond available dates."""
    dates = pd.DatetimeIndex(["2026-06-01", "2026-06-02", "2026-06-03"])
    target = pd.Timestamp("2026-06-05")
    assert resolve_effective_trading_date(dates, target) == pd.Timestamp("2026-06-03")


def test_resolve_effective_trading_date_holiday_gap():
    """Holiday gap fallback when target date falls on missing date."""
    dates = pd.DatetimeIndex(["2026-06-01", "2026-06-02", "2026-06-04"])
    target = pd.Timestamp("2026-06-03")  # Missing date
    assert resolve_effective_trading_date(dates, target) == pd.Timestamp("2026-06-02")


def test_resolve_effective_trading_date_empty_or_too_old():
    """Returns None when available dates are empty or gap exceeds max fallback."""
    dates = pd.DatetimeIndex([])
    target = pd.Timestamp("2026-06-03")
    assert resolve_effective_trading_date(dates, target) is None

    old_dates = pd.DatetimeIndex(["2026-01-01"])
    assert (
        resolve_effective_trading_date(old_dates, target, max_fallback_days=5) is None
    )
