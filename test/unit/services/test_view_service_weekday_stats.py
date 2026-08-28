# filename: test_view_service_weekday_stats.py
from typing import cast
from unittest.mock import MagicMock

import pytest

from app.database.repositories.market import MarketRepository
from app.database.repositories.trade import TradeRepository
from app.services.trade_manager.view_service import TradeViewData, TradeViewService


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
    trades = cast(
        list[TradeViewData],
        [
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
        ],
    )

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


def test_get_weekday_stats_all_days_and_breakeven_pnl(
    view_service: TradeViewService,
) -> None:
    """Verify that all 7 weekdays and breakeven PnL (0.0) are correctly handled."""
    trades = cast(
        list[TradeViewData],
        [
            # Tuesday (2026-06-30) - Breakeven
            {"entry_date": "2026-06-30", "realized_pnl": 0.0},
            # Wednesday (2026-07-01) - Loser
            {"entry_date": "2026-07-01", "realized_pnl": -120.50},
            # Thursday (2026-07-02) - Winner
            {"entry_date": "2026-07-02", "realized_pnl": 350.25},
            # Saturday (2026-07-04) - Weekend trade
            {"entry_date": "2026-07-04", "realized_pnl": 50.0},
            # Sunday (2026-07-05) - Weekend trade
            {"entry_date": "2026-07-05", "realized_pnl": -30.0},
        ],
    )

    stats = view_service.get_weekday_stats(trades)

    # Tuesday (idx 1) - count 1, win 0, loss 1 (pnl <= 0 is loss), pnl 0.0
    assert stats[1]["name"] == "Tuesday"
    assert stats[1]["count"] == 1
    assert stats[1]["win"] == 0
    assert stats[1]["loss"] == 1
    assert stats[1]["pnl"] == 0.0
    assert stats[1]["average_pnl"] == 0.0

    # Wednesday (idx 2)
    assert stats[2]["name"] == "Wednesday"
    assert stats[2]["count"] == 1
    assert stats[2]["win"] == 0
    assert stats[2]["loss"] == 1
    assert stats[2]["pnl"] == -120.50

    # Thursday (idx 3)
    assert stats[3]["name"] == "Thursday"
    assert stats[3]["count"] == 1
    assert stats[3]["win"] == 1
    assert stats[3]["loss"] == 0
    assert stats[3]["pnl"] == 350.25

    # Saturday (idx 5)
    assert stats[5]["name"] == "Saturday"
    assert stats[5]["count"] == 1
    assert stats[5]["win"] == 1

    # Sunday (idx 6)
    assert stats[6]["name"] == "Sunday"
    assert stats[6]["count"] == 1
    assert stats[6]["loss"] == 1


def test_get_weekday_stats_corrupt_and_none_pnl(
    view_service: TradeViewService,
) -> None:
    """Verify that None PnL and empty entry date strings do not crash calculation."""
    trades = cast(
        list[TradeViewData],
        [
            {"entry_date": "2026-06-29", "realized_pnl": None},
            {"entry_date": "", "realized_pnl": 500.0},
            {"entry_date": "2026-06-29", "realized_pnl": "invalid-float"},
        ],
    )

    stats = view_service.get_weekday_stats(trades)

    # Monday (idx 0) should record the valid entry date with 0.0 PnL fallback
    assert stats[0]["count"] == 2
    assert stats[0]["pnl"] == 0.0
