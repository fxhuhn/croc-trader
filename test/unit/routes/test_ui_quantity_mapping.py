# filename: test_ui_quantity_mapping.py
import pytest
from unittest.mock import patch
from app.services.trade_manager.view_service import TradeViewService


@pytest.fixture
def view_service(tmp_path):
    """Fixture for TradeViewService with mocked repositories."""
    test_db = tmp_path / "test.db"
    with (
        patch("app.services.trade_manager.view_service.DatabaseSession"),
        patch("app.services.trade_manager.view_service.TradeRepository"),
        patch(
            "app.services.trade_manager.view_service.MarketRepository"
        ) as mock_market_repo_class,
        patch(
            "app.services.trade_manager.view_service._get_database_path",
            return_value=str(test_db),
        ),
    ):
        service = TradeViewService()
        # Ensure we can return the mock market repo for specific price lookups
        service.market_repository = mock_market_repo_class.return_value
        return service


def test_active_trade_quantity_mapping(view_service: TradeViewService) -> None:
    """Verify that active trades display initial_size correctly."""
    # Arrange
    trade = {
        "status": "ACTIVE",
        "entry_price": 100.0,
        "current_price": 110.0,
        "initial_size": 10,
        "current_size": 10,
        "current_stop_loss": 90.0,
        "symbol": "AAPL",
        "signal_context": "{}",
    }
    view_service.market_repository.get_latest_price.return_value = 110.0
    view_service.market_repository.get_trading_days_count.return_value = 5

    # Act
    view_data = view_service.prepare_trade_view(trade)

    # Assert
    assert view_data["display_size"] == 10
    assert view_data["unrealized_pnl"] == (110.0 - 100.0) * 10
    assert view_data["pnl_pct"] == 10.0


def test_closed_trade_quantity_mapping(view_service: TradeViewService) -> None:
    """Verify that closed trades use initial_size for quantity display even if current_size is 0."""
    # Arrange
    trade = {
        "status": "CLOSED",
        "entry_price": 100.0,
        "exit_price": 110.0,
        "initial_size": 20,
        "current_size": 0,
        "realized_pnl": 0.0,
        "symbol": "MSFT",
        "signal_context": "{}",
    }
    view_service.market_repository.get_trading_days_count.return_value = 10

    # Act
    view_data = view_service.prepare_trade_view(trade)

    # Assert
    expected_pnl = (110.0 - 100.0) * 20
    assert view_data["realized_pnl"] == expected_pnl
    assert view_data["display_size"] == 20
