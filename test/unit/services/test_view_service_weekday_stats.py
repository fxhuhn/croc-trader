# filename: test_view_service_weekday_stats.py
from unittest.mock import MagicMock

import pytest

from app.database.repositories.market import MarketRepository
from app.database.repositories.trade import TradeRepository
from app.services.trade_manager.view_service import TradeViewService


@pytest.fixture
def view_service() -> TradeViewService:
    """Fixture for TradeViewService with mocked repositories."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_market_repo = MagicMock(spec=MarketRepository)
    return TradeViewService(
        trade_repository=mock_trade_repo,
        market_repository=mock_market_repo,
    )


def test_get_weekday_stats_empty(view_service: TradeViewService) -> None:
    """Verify that get_weekday_stats returns empty stats with zero values for all days."""
    stats = view_service.get_weekday_stats([])
    assert len(stats) == 7
    for i in range(7):
        assert stats[i]["count"] == 0
        assert stats[i]["win"] == 0
        assert stats[i]["loss"] == 0
        assert stats[i]["pnl"] == 0.0
        assert stats[i]["average_pnl"] == 0.0


def test_get_weekday_stats_aggregations(view_service: TradeViewService) -> None:
    """Verify that get_weekday_stats correctly aggregates counts and PnL by weekday."""
    # Arrange
    trades = [
        # Monday (2026-06-29) - Winner
        {
            "entry_date": "2026-06-29 09:30:00",
            "realized_pnl": 150.00,
        },
        # Monday (2026-06-29) - Loser
        {
            "entry_date": "2026-06-29",
            "realized_pnl": -50.00,
        },
        # Friday (2026-06-26) - Winner
        {
            "entry_date": "2026-06-26",
            "realized_pnl": 200.00,
        },
        # Missing entry date - ignored
        {
            "entry_date": None,
            "realized_pnl": 100.00,
        },
        # Invalid entry date - logged and ignored
        {
            "entry_date": "invalid-date",
            "realized_pnl": 100.00,
        },
    ]

    # Act
    stats = view_service.get_weekday_stats(trades)

    # Assert
    # Monday is index 0
    assert stats[0]["name"] == "Monday"
    assert stats[0]["count"] == 2
    assert stats[0]["win"] == 1
    assert stats[0]["loss"] == 1
    assert stats[0]["pnl"] == 100.00
    assert stats[0]["average_pnl"] == 50.00

    # Friday is index 4
    assert stats[4]["name"] == "Friday"
    assert stats[4]["count"] == 1
    assert stats[4]["win"] == 1
    assert stats[4]["loss"] == 0
    assert stats[4]["pnl"] == 200.00
    assert stats[4]["average_pnl"] == 200.00

    # Other days should be zeroed
    for i in [1, 2, 3, 5, 6]:
        assert stats[i]["count"] == 0
        assert stats[i]["pnl"] == 0.0
