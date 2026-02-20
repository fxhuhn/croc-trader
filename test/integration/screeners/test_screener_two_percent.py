import pytest
import pandas as pd
from unittest.mock import MagicMock

from app.services.screener.strategies.two_percent_strategy import TwoPercentStrategy
from app.database.repositories.trade import TradeRepository
from app.database.repositories.market_data_provider import MarketDataProvider

# --- FIXTURES ---


@pytest.fixture
def mock_repo():
    return MagicMock(spec=TradeRepository)


@pytest.fixture
def mock_provider():
    return MagicMock(spec=MarketDataProvider)


@pytest.fixture
def strategy(mock_repo, mock_provider):
    return TwoPercentStrategy(trade_repository=mock_repo, data_provider=mock_provider)


# --- TESTS ---

# --- TESTS ---


def test_run_on_normal_friday(strategy, mock_provider, mock_repo):
    """
    Scenario: Today is Friday, and it is NOT a holiday.
    Expectation: Run matches (return 1).
    """
    # Setup
    analysis_date = "2026-01-30"  # A Friday
    today = pd.Timestamp(analysis_date)

    # Mock Holiday Checker on the instance
    strategy.holiday_checker = MagicMock()
    strategy.holiday_checker.is_holiday.return_value = False

    # Mock Data Provider
    df_data = pd.DataFrame({"close": [100.0]}, index=pd.to_datetime([today]))
    mock_provider.get_symbol_history.return_value = df_data

    # Mock Repo (Trade does not exist)
    mock_repo.exists.return_value = False

    # Act
    result = strategy.run(analysis_date=analysis_date)

    # Assert
    assert result == 1
    mock_repo.create_trade.assert_called_once()

    # Verify setup_close and limit_entry calculation
    args, kwargs = mock_repo.create_trade.call_args
    assert kwargs["entry"] == 99.0  # 100 * 0.99


def test_run_on_normal_thursday_skips(strategy, mock_provider, mock_repo):
    """
    Scenario: Today is Thursday, and Friday is NOT a holiday.
    Expectation: Skip (return 0).
    """
    analysis_date = "2026-01-29"  # A Thursday

    strategy.holiday_checker = MagicMock()
    strategy.holiday_checker.is_holiday.return_value = False

    result = strategy.run(analysis_date=analysis_date)

    assert result == 0
    mock_repo.create_trade.assert_not_called()


def test_run_on_thursday_if_friday_is_holiday(strategy, mock_provider, mock_repo):
    """
    Scenario: Today is Thursday, and Friday IS a holiday.
    Expectation: Run matches (return 1).
    """
    analysis_date = "2026-01-29"  # A Thursday
    today = pd.Timestamp(analysis_date)
    friday = pd.Timestamp("2026-01-30")

    strategy.holiday_checker = MagicMock()
    # is_holiday called with friday date should return True
    strategy.holiday_checker.is_holiday.side_effect = lambda d: d == friday.date()

    # Mock Data
    df_data = pd.DataFrame({"close": [100.0]}, index=pd.to_datetime([today]))
    mock_provider.get_symbol_history.return_value = df_data

    mock_repo.exists.return_value = False

    result = strategy.run(analysis_date=analysis_date)

    assert result == 1
    mock_repo.create_trade.assert_called_once()


def test_run_on_friday_if_friday_is_holiday(strategy, mock_provider, mock_repo):
    """
    Scenario: Today is Friday, and it IS a holiday.
    Expectation: Skip.
    """
    analysis_date = "2026-01-30"  # A Friday

    strategy.holiday_checker = MagicMock()
    strategy.holiday_checker.is_holiday.return_value = True

    # Even if data existed
    mock_provider.get_symbol_history.return_value = pd.DataFrame()

    result = strategy.run(analysis_date=analysis_date)

    assert result == 0
    mock_repo.create_trade.assert_not_called()


def test_skip_other_days(strategy, mock_provider, mock_repo):
    """
    Scenario: Monday, Tuesday, Wednesday.
    Expectation: Return 0.
    """
    strategy.holiday_checker = MagicMock()

    # 2026-01-26 is Monday
    result = strategy.run(analysis_date="2026-01-26")
    assert result == 0

    # 2026-01-28 is Wednesday
    result = strategy.run(analysis_date="2026-01-28")
    assert result == 0
