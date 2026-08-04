from unittest.mock import MagicMock

import pytest

from app.const import Strategies
from app.services.screener.view_service import ScreenerViewService


@pytest.fixture
def mock_signal_repository():
    return MagicMock()


def test_get_turnover_candidates_sorting(mock_signal_repository):
    """Verifies that turnover candidates are sorted by Dollar-Volume descending."""
    # Arrange
    service = ScreenerViewService(mock_signal_repository)

    # Sample signals with different Dollar-Volume (setup_turnover_sma)
    # Signal 1: TSLA with high volume (5B)
    # Signal 2: AAPL with medium volume (1B)
    # Signal 3: MSFT with low volume (500M)

    mock_signal_repository.get_trade_candidates.return_value = [
        {
            "symbol": "AAPL",
            "strategy": str(Strategies.TurnOverTiming_05),
            "entry_price": 150.0,
            "signal_context": '{"setup_close": 155.0, "setup_atr": 5.0, "setup_turnover_sma": 1000000000, "date": "2026-02-20"}',
        },
        {
            "symbol": "TSLA",
            "strategy": str(Strategies.TurnOverTiming_05),
            "entry_price": 200.0,
            "signal_context": '{"setup_close": 210.0, "setup_atr": 10.0, "setup_turnover_sma": 5000000000, "date": "2026-02-20"}',
        },
        {
            "symbol": "MSFT",
            "strategy": str(Strategies.TurnOverTiming_05),
            "entry_price": 300.0,
            "signal_context": '{"setup_close": 310.0, "setup_atr": 8.0, "setup_turnover_sma": 500000000, "date": "2026-02-20"}',
        },
    ]

    # Act
    results = service.get_turnover_candidates()

    # Assert
    assert len(results) == 3
    assert results[0]["symbol"] == "TSLA"  # Highest volume
    assert results[1]["symbol"] == "AAPL"  # Medium volume
    assert results[2]["symbol"] == "MSFT"  # Lowest volume
    assert results[0]["dollar_volume"] == 5000000000.0
    assert results[1]["dollar_volume"] == 1000000000.0
    assert results[2]["dollar_volume"] == 500000000.0


def test_get_dip_buyer_candidates_sorting_by_score(mock_signal_repository):
    """Verifies that Dip Buyer candidates are sorted by setup_score descending."""
    # Arrange
    service = ScreenerViewService(mock_signal_repository)

    mock_signal_repository.get_trade_candidates.return_value = [
        {
            "id": 1,
            "symbol": "SPY",
            "strategy": str(Strategies.DipBuyer),
            "entry_price": 400.0,
            "signal_context": '{"setup_close": 405.0, "setup_score": 5, "date": "2026-02-20"}',
        },
        {
            "id": 2,
            "symbol": "QQQ",
            "strategy": str(Strategies.DipBuyer),
            "entry_price": 350.0,
            "signal_context": '{"setup_close": 355.0, "setup_score": 9, "date": "2026-02-20"}',
        },
        {
            "id": 3,
            "symbol": "IWM",
            "strategy": str(Strategies.DipBuyer),
            "entry_price": 200.0,
            "signal_context": '{"setup_close": 205.0, "setup_score": 7, "date": "2026-02-20"}',
        },
    ]

    # Act
    results = service.get_candidates(Strategies.DipBuyer)

    # Assert
    assert len(results) == 3
    assert results[0]["symbol"] == "QQQ"  # Score 9 (highest)
    assert results[1]["symbol"] == "IWM"  # Score 7 (medium)
    assert results[2]["symbol"] == "SPY"  # Score 5 (lowest)
