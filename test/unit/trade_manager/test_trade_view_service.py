"""Unit tests for TradeViewService covering resolve_strategy, index stats, portfolio summary, and broker active trades in app/services/trade_manager/view_service.py."""

from unittest.mock import MagicMock

import pytest

from app.services.trade_manager.view_service import TradeViewService


@pytest.fixture
def mock_trade_repo() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_market_repo() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_broker_repo() -> MagicMock:
    return MagicMock()


@pytest.fixture
def view_service(
    mock_trade_repo: MagicMock,
    mock_market_repo: MagicMock,
    mock_broker_repo: MagicMock,
) -> TradeViewService:
    return TradeViewService(
        trade_repository=mock_trade_repo,
        market_repository=mock_market_repo,
        broker_repository=mock_broker_repo,
    )


def test_resolve_strategy(view_service: TradeViewService) -> None:
    assert view_service.resolve_strategy({"strategy": "dipbuyer"}) == "dipbuyer"
    assert (
        view_service.resolve_strategy({"strategy": "unknown_strat"}) == "unknown_strat"
    )


def test_get_index_stats_empty_and_valid(view_service: TradeViewService) -> None:
    empty_stats = view_service.get_index_stats([])
    assert isinstance(empty_stats, dict)

    trades = [
        {
            "id": 1,
            "status": "CLOSED",
            "realized_pnl": 100.0,
            "signal_context": '{"index": "SPY"}',
        },
        {
            "id": 2,
            "status": "CLOSED",
            "realized_pnl": -50.0,
            "signal_context": '{"index": "SPY"}',
        },
    ]
    stats = view_service.get_index_stats(trades)
    assert "SPY" in stats
    assert stats["SPY"]["win"] == 1
    assert stats["SPY"]["loss"] == 1
    assert stats["SPY"]["pnl"] == 50.0


def test_get_closed_summary(view_service: TradeViewService) -> None:
    trades = [
        {"id": 1, "status": "CLOSED", "realized_pnl": 200.0},
        {"id": 2, "status": "CLOSED", "realized_pnl": -100.0},
    ]
    summary = view_service.get_closed_summary(trades)
    assert summary["count"] == 2
    assert summary["total_pnl"] == 100.0
    assert summary["average_pnl"] == 50.0
    assert summary["win_rate"] == 50.0


def test_prepare_trade_view(view_service: TradeViewService) -> None:
    trade = {
        "id": 10,
        "symbol": "AAPL",
        "strategy": "DipBuyer",
        "status": "ACTIVE",
        "initial_size": 100,
        "current_size": 100,
        "entry_price": 150.0,
        "entry_date": "2026-08-01",
        "signal_context": '{"date": "2026-08-01"}',
    }
    view_service.market_repository.get_latest_price.return_value = 160.0
    view_service.market_repository.get_trading_days_count.return_value = 3

    result = view_service.prepare_trade_view(trade)
    assert result["symbol"] == "AAPL"
    assert result["unrealized_pnl"] == 1000.0
