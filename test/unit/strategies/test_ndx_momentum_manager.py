# filename: test_ndx_momentum_manager.py
import pytest
from unittest.mock import MagicMock
import pandas as pd

from app.services.trade_manager.strategies.ndx_momentum import NDXMomentumTradeStrategy
from app.database.repositories.trade import TradeRepository
from app.types import TradeStatus

@pytest.fixture
def mock_trade_repo() -> MagicMock:
    return MagicMock(spec=TradeRepository)

@pytest.fixture
def strategy() -> NDXMomentumTradeStrategy:
    return NDXMomentumTradeStrategy()

def test_check_entry_bull_regime(strategy, mock_trade_repo):
    """Tests entry when regime is BULL."""
    trade = {
        "id": "N1",
        "symbol": "AAPL",
        "strategy": "ndx_momentum",
        "signal_context": '{"regime": "BULL"}'
    }
    candle = pd.Series({"open": 100.0, "date": pd.Timestamp("2026-02-02")}, name=pd.Timestamp("2026-02-02"))
    
    # Mock no active positions
    mock_trade_repo.get_by_status.return_value = []
    
    result = strategy.check_entry(trade, candle, pd.DataFrame([candle]), mock_trade_repo)
    
    assert result is not None
    assert "FILLED" in result
    mock_trade_repo.update_trade.assert_called_once()
    assert mock_trade_repo.update_trade.call_args[0][1]["status"] == TradeStatus.ACTIVE

def test_check_entry_bear_regime_rejection(strategy, mock_trade_repo):
    """Tests rejection when regime is not BULL."""
    trade = {
        "id": "N1",
        "signal_context": '{"regime": "BEAR"}'
    }
    candle = pd.Series({"date": pd.Timestamp("2026-02-02")}, name=pd.Timestamp("2026-02-02"))
    
    result = strategy.check_entry(trade, candle, pd.DataFrame(), mock_trade_repo)
    
    assert result is not None
    assert "Combined Regime" in result
    assert mock_trade_repo.update_trade.call_args[0][1]["status"] == TradeStatus.INVALID

def test_manage_active_same_month_no_action(strategy, mock_trade_repo):
    """Tests that no rebalancing occurs mid-month."""
    trade = {"symbol": "AAPL"}
    # Feb 02 to Feb 03 (Same month)
    df = pd.DataFrame({
        "date": [pd.Timestamp("2026-02-02"), pd.Timestamp("2026-02-03")],
        "open": [100.0, 101.0]
    })
    
    result = strategy.manage_active_trade(trade, df, mock_trade_repo)
    assert result is None

def test_manage_active_month_switch_dropped_symbol(strategy, mock_trade_repo):
    """Tests that a symbol is closed if it's no longer a leader on month switch."""
    trade = {
        "id": "N1",
        "symbol": "AAPL",
        "signal_context": {"date": "2026-01-30"}
    }
    # Jan 30 (Friday) -> Feb 02 (Monday)
    df = pd.DataFrame({
        "date": [pd.Timestamp("2026-01-30"), pd.Timestamp("2026-02-02")],
        "open": [100.0, 105.0]
    })
    
    # Mock strategy trades: AAPL is from Jan, but new leaders are MSFT and GOOG from Feb 1st
    mock_trade_repo.get_all_by_strategy.return_value = [
        {"symbol": "AAPL", "signal_context": {"date": "2026-01-30"}},
        {"symbol": "MSFT", "signal_context": {"date": "2026-02-01"}},
        {"symbol": "GOOG", "signal_context": {"date": "2026-02-01"}}
    ]
    
    result = strategy.manage_active_trade(trade, df, mock_trade_repo)
    
    assert result is not None
    assert "REBAL_EXIT" in result
    assert mock_trade_repo.update_trade.call_args[0][1]["exit_price"] == 105.0

def test_generate_orders(strategy, mock_trade_repo):
    """Tests order generation for rebalance."""
    trade = {"symbol": "AAPL", "budget": 2000.0}
    df = pd.DataFrame({"close": [100.0]})
    
    orders = strategy.generate_orders(trade, df, 2000.0, mock_trade_repo)
    
    assert orders is not None
    assert orders.quantity == 20
    assert orders.entry.type == "MKT"
    assert orders.entry.tif == "OPG"
