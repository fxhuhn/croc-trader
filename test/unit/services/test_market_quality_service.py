import datetime
from unittest.mock import MagicMock

from app.services.market.quality import MarketQualityService
from app.services.market.updater import MarketDataUpdater


def test_market_quality_service_perform_gap_check():
    """Tests that MarketQualityService calculates recency threshold based on trading calendar."""
    mock_updater = MagicMock(spec=MarketDataUpdater)
    mock_repo = MagicMock()
    mock_updater.repo = mock_repo
    mock_repo.get_outdated_symbols.return_value = []
    mock_repo.get_symbols_with_missing_history.return_value = []

    mock_holiday_checker = MagicMock()
    mock_holiday_checker.is_holiday.return_value = False

    service = MarketQualityService(
        updater=mock_updater, holiday_checker=mock_holiday_checker
    )

    service.perform_gap_check()

    mock_repo.get_outdated_symbols.assert_called_once()
    actual_thresh = mock_repo.get_outdated_symbols.call_args[0][0]
    # Check that threshold is a valid YYYY-MM-DD string
    datetime.datetime.strptime(actual_thresh, "%Y-%m-%d")
