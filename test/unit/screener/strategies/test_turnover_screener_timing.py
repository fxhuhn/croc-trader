# filename: test_turnover_screener_timing.py
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import datetime

from app.services.screener.strategies.turnover_timing import TurnoverTimingStrategy
from app.database.repositories.trade import TradeRepository
from app.database.repositories.market_data_provider import MarketDataProvider
from app.tools.market_holidays import MarketHolidayChecker

@pytest.fixture
def mock_trade_repo() -> MagicMock:
    return MagicMock(spec=TradeRepository)

@pytest.fixture
def mock_data_provider() -> MagicMock:
    return MagicMock(spec=MarketDataProvider)

@pytest.fixture
def strategy(mock_trade_repo: MagicMock, mock_data_provider: MagicMock) -> TurnoverTimingStrategy:
    return TurnoverTimingStrategy(trade_repository=mock_trade_repo, data_provider=mock_data_provider)

@pytest.mark.parametrize("test_date, is_friday_holiday, expected_run", [
    ("2026-02-09", False, False), # Monday
    ("2026-02-10", False, False), # Tuesday
    ("2026-02-11", False, False), # Wednesday
    ("2026-02-12", False, False), # Thursday (Normal)
    ("2026-02-13", False, True),  # Friday (Normal)
    ("2026-02-12", True, True),   # Thursday (Friday is Holiday)
])
def test_screener_only_runs_on_correct_days(
    strategy: TurnoverTimingStrategy,
    test_date: str,
    is_friday_holiday: bool,
    expected_run: bool
) -> None:
    """Verifies that the screener only executes on the designated 'End of Week' days."""
    # Arrange
    with patch("app.tools.market_holidays.MarketHolidayChecker.is_holiday") as mock_is_holiday:
        # Mock holiday check for Friday of the week
        def side_effect(date_obj):
            if is_friday_holiday and str(date_obj) == "2026-02-13":
                return True
            return False
        mock_is_holiday.side_effect = side_effect
        
        # Act
        # We only care if it returns 0 (early exit) or proceeds to data loading
        # Mock get_universe_daily_data to return empty to stop execution after timing check
        strategy.data_provider.get_universe_daily_data.return_value = {}
        
        result = strategy.run(analysis_date=test_date)
        
        # Assert
        if expected_run:
            # If it runs, it would attempt to load data (even if it returns 0 eventually)
            strategy.data_provider.get_universe_daily_data.assert_called()
        else:
            assert result == 0
            strategy.data_provider.get_universe_daily_data.assert_not_called()

def test_screener_fails_if_friday_is_holiday_and_checked_on_thursday(
    strategy: TurnoverTimingStrategy
) -> None:
    """Explicitly checks logic for Thursday when Friday is a holiday."""
    # Arrange
    thursday = "2026-02-12"
    friday = datetime.date(2026, 2, 13)
    
    with patch("app.tools.market_holidays.MarketHolidayChecker.is_holiday") as mock_is_holiday:
        mock_is_holiday.return_value = True # Friday is a holiday
        strategy.data_provider.get_universe_daily_data.return_value = {} # Stop there
        
        # Act
        strategy.run(analysis_date=thursday)
        
        # Assert
        strategy.data_provider.get_universe_daily_data.assert_called()
