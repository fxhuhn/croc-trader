# filename: test_two_percent_manager.py
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd

from app.services.trade_manager.strategies.two_percent_strategy import TwoPercentStrategy
from app.database.repositories.trade import TradeRepository
from app.types import TradeStatus

@pytest.fixture
def mock_trade_repo() -> MagicMock:
    return MagicMock(spec=TradeRepository)

@pytest.fixture
def strategy() -> TwoPercentStrategy:
    return TwoPercentStrategy()

def test_check_entry_gap_down_fill(strategy, mock_trade_repo):
    """Tests entry when price gaps down below limit."""
    trade = {
        "id": "2P1",
        "entry_price": 100.0,
        "signal_context": '{"date": "2026-02-13"}'
    }
    # Open 98 (below 100)
    candle = pd.Series({"open": 98.0, "low": 97.0, "date": pd.Timestamp("2026-02-16")}, name=pd.Timestamp("2026-02-16"))
    
    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        result = strategy.check_entry(trade, candle, pd.DataFrame([candle]), mock_trade_repo)
        
        assert result is not None
        assert "FILLED" in result
        mock_trade_repo.update_trade.assert_called_once()
        # Verify reason in repo call
        assert "Gap Down" in mock_trade_repo.update_trade.call_args[1]["reason"]
        assert mock_trade_repo.update_trade.call_args[0][1]["entry_price"] == 98.0

def test_check_entry_limit_hit(strategy, mock_trade_repo):
    """Tests normal limit hit (fill at limit)."""
    trade = {
        "id": "2P1",
        "entry_price": 100.0,
        "signal_context": '{"date": "2026-02-13"}'
    }
    # Open 102, Low 99 (hits 100)
    candle = pd.Series({"open": 102.0, "low": 99.0, "date": pd.Timestamp("2026-02-16")}, name=pd.Timestamp("2026-02-16"))
    
    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        result = strategy.check_entry(trade, candle, pd.DataFrame([candle]), mock_trade_repo)
        
        assert result is not None
        assert "FILLED" in result
        # Verify reason in repo call
        assert "Limit Hit" in mock_trade_repo.update_trade.call_args[1]["reason"]
        assert mock_trade_repo.update_trade.call_args[0][1]["entry_price"] == 100.0

def test_check_entry_missed_window_expiration(strategy, mock_trade_repo):
    """Tests expiration if Day 1 ends without fill."""
    trade = {
        "id": "2P1",
        "entry_price": 100.0,
        "signal_context": '{"date": "2026-02-13"}'
    }
    # Low 101. Must provide 'open' as well.
    candle = pd.Series({"open": 105.0, "low": 101.0, "date": pd.Timestamp("2026-02-16")}, name=pd.Timestamp("2026-02-16"))
    
    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        strategy.check_entry(trade, candle, pd.DataFrame([candle]), mock_trade_repo)
        assert mock_trade_repo.update_trade.call_args[0][1]["status"] == TradeStatus.INVALID

def test_manage_active_tp_hit(strategy, mock_trade_repo):
    """Tests Take Profit (2%) hit on Day 2."""
    trade = {
        "id": "2P1",
        "entry_price": 100.0,
        "entry_date": "2026-02-16", # Monday
        "status": "ACTIVE"
    }
    # Tuesday: High 103 (hits 102)
    candle = pd.Series({
        "open": 101.0, 
        "high": 103.0, 
        "close": 102.5, 
        "date": pd.Timestamp("2026-02-17")
    }, name=pd.Timestamp("2026-02-17"))
    
    result = strategy.manage_active_trade(trade, pd.DataFrame([candle]), mock_trade_repo)
    
    assert result is not None
    assert "TARGET_HIT" in result
    assert mock_trade_repo.update_trade.call_args[0][1]["exit_price"] == 102.0

def test_manage_active_friday_time_stop(strategy, mock_trade_repo):
    """Tests Friday EOD exit."""
    trade = {
        "id": "2P1",
        "entry_price": 100.0,
        "entry_date": "2026-02-16",
        "status": "ACTIVE"
    }
    # Friday: No TP hit
    candle = pd.Series({
        "open": 101.0, 
        "high": 101.5, 
        "close": 101.2, 
        "date": pd.Timestamp("2026-02-20")
    }, name=pd.Timestamp("2026-02-20"))
    
    result = strategy.manage_active_trade(trade, pd.DataFrame([candle]), mock_trade_repo)
    
    assert result is not None
    assert "TIME_STOP" in result
    assert mock_trade_repo.update_trade.call_args[0][1]["exit_price"] == 101.2

def test_generate_orders(strategy, mock_trade_repo):
    """Tests order generation logic."""
    trade = {"symbol": "AAPL", "entry_price": 100.0}
    
    order = strategy.generate_orders(trade, pd.DataFrame(), 2000.0, mock_trade_repo)
    assert order is not None
    assert order.mode == "Entry"
    assert order.entry.price == 100.0
    assert order.quantity == 20
