import datetime

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
