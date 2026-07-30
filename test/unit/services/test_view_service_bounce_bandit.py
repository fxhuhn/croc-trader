# filename: test_view_service_bounce_bandit.py
"""Unit tests for TradeViewService handling BounceBandit trades and market history prewarming."""

from unittest.mock import MagicMock, create_autospec

import pandas as pd
import pytest

from app.database.repositories.broker import BrokerRepository
from app.database.repositories.market import MarketRepository
from app.database.repositories.trade import TradeRepository
from app.services.trade_manager.view_service import TradeViewService
from app.types import TradeStatus


@pytest.fixture
def mock_market_repo() -> MagicMock:
    """Fixture providing an autospecced MarketRepository to strictly enforce function signatures."""
    return create_autospec(MarketRepository, instance=True)


@pytest.fixture
def mock_trade_repo() -> MagicMock:
    """Fixture providing an autospecced TradeRepository."""
    return create_autospec(TradeRepository, instance=True)


@pytest.fixture
def mock_broker_repo() -> MagicMock:
    """Fixture providing an autospecced BrokerRepository."""
    return create_autospec(BrokerRepository, instance=True)


@pytest.fixture
def view_service(
    mock_trade_repo: MagicMock,
    mock_market_repo: MagicMock,
    mock_broker_repo: MagicMock,
) -> TradeViewService:
    """Fixture providing a TradeViewService instance with strict mocks."""
    return TradeViewService(
        trade_repository=mock_trade_repo,
        market_repository=mock_market_repo,
        broker_repository=mock_broker_repo,
    )


def test_prepare_trade_view_bounce_bandit_queries_symbol_history_with_start_date(
    view_service: TradeViewService,
    mock_market_repo: MagicMock,
) -> None:
    """Verifies that prepare_trade_view for BounceBandit calls get_symbol_history_raw with a valid start_date."""
    # Arrange
    trade = {
        "id": 1,
        "symbol": "AAPL",
        "strategy": "bounce_bandit",
        "status": TradeStatus.ACTIVE.value,
        "entry_price": 150.0,
        "current_price": 155.0,
        "entry_date": "2026-07-01",
        "signal_context": "{}",
    }

    # Generate 10 days of historical candles to satisfy SMA(8) calculation
    dates = pd.date_range("2026-06-15", periods=10, freq="B")
    history_df = pd.DataFrame(
        [
            {
                "date": d,
                "open": 150.0 + i,
                "high": 152.0 + i,
                "low": 149.0 + i,
                "close": 151.0 + i,
                "volume": 1000000,
            }
            for i, d in enumerate(dates)
        ]
    )

    mock_market_repo.get_symbol_history_raw.return_value = history_df
    mock_market_repo.get_latest_price.return_value = 160.0
    mock_market_repo.get_trading_days_count.return_value = 10

    # Act
    result = view_service.prepare_trade_view(trade)

    # Assert
    # Verify get_symbol_history_raw was called with symbol and start_date without raising TypeError
    mock_market_repo.get_symbol_history_raw.assert_called_once()
    call_kwargs = mock_market_repo.get_symbol_history_raw.call_args.kwargs
    call_args = mock_market_repo.get_symbol_history_raw.call_args.args

    symbol_passed = call_kwargs.get("symbol") or (call_args[0] if call_args else None)
    start_date_passed = call_kwargs.get("start_date") or (
        call_args[1] if len(call_args) > 1 else None
    )

    assert symbol_passed == "AAPL"
    assert start_date_passed is not None
    assert isinstance(start_date_passed, str)

    # Verify context was enriched with indicators
    assert result is not None
    assert "context" in result
    assert "sma_8" in result["context"]
    assert "target" in result["context"]


def test_prepare_trade_view_bounce_bandit_handles_empty_history_gracefully(
    view_service: TradeViewService,
    mock_market_repo: MagicMock,
) -> None:
    """Verifies that prepare_trade_view handles empty symbol history without crashing."""
    # Arrange
    trade = {
        "id": 2,
        "symbol": "MSFT",
        "strategy": "bounce_bandit",
        "status": TradeStatus.ACTIVE.value,
        "entry_price": 400.0,
        "current_price": 405.0,
        "signal_context": "{}",
    }

    mock_market_repo.get_symbol_history_raw.return_value = pd.DataFrame()
    mock_market_repo.get_latest_price.return_value = 405.0
    mock_market_repo.get_trading_days_count.return_value = 0

    # Act
    result = view_service.prepare_trade_view(trade)

    # Assert
    assert result["symbol"] == "MSFT"
    assert result["status"] == TradeStatus.ACTIVE.value
