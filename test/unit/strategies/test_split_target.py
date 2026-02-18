# filename: test_split_target.py
import pytest
from unittest.mock import MagicMock
import pandas as pd
import json

from app.services.trade_manager.strategies.split_target import SplitTargetStrategy
from app.database.repositories.trade import TradeRepository
from app.types import TradeStatus, ExitReason

@pytest.fixture
def mock_trade_repo() -> MagicMock:
    return MagicMock(spec=TradeRepository)

@pytest.fixture
def strategy() -> SplitTargetStrategy:
    return SplitTargetStrategy()

def test_manage_active_tp1_hit(strategy, mock_trade_repo):
    """Tests phase 1 -> phase 2 transition when TP1 is hit."""
    trade = {
        "id": "S1",
        "entry_price": 100.0,
        "current_stop_loss": 90.0,
        "status": "ACTIVE",
        "current_size": 20,
        "entry_date": "2026-02-18",
        "signal_context": json.dumps({"tp1": 110.0, "tp3": 130.0, "is_phase_2": False})
    }
    # High 115 (hits 110)
    candle = pd.Series({
        "open": 105.0, 
        "high": 115.0, 
        "low": 104.0, 
        "close": 112.0, 
        "date": pd.Timestamp("2026-02-19")
    }, name=pd.Timestamp("2026-02-19"))
    
    result = strategy.manage_active_trade(trade, pd.DataFrame([candle]), mock_trade_repo)
    
    assert result is not None
    assert "TP1 HIT" in result
    mock_trade_repo.update_trade.assert_called_once()
    payload = mock_trade_repo.update_trade.call_args[0][1]
    assert payload["current_size"] == 10
    assert payload["current_stop_loss"] == 100.0 # Moved to break-even
    assert json.loads(payload["signal_context"])["is_phase_2"] is True

def test_manage_active_tp3_gap_over(strategy, mock_trade_repo):
    """Tests that a gap over TP3 closes the full position immediately even in phase 1."""
    trade = {
        "id": "S1",
        "entry_price": 100.0,
        "current_stop_loss": 90.0,
        "status": "ACTIVE",
        "current_size": 20,
        "entry_date": "2026-02-18",
        "signal_context": json.dumps({"tp1": 110.0, "tp3": 130.0, "is_phase_2": False})
    }
    # Gap up to 135
    candle = pd.Series({
        "open": 135.0, 
        "high": 140.0, 
        "low": 134.0, 
        "close": 138.0, 
        "date": pd.Timestamp("2026-02-19")
    }, name=pd.Timestamp("2026-02-19"))
    
    result = strategy.manage_active_trade(trade, pd.DataFrame([candle]), mock_trade_repo)
    
    assert result is not None
    assert "TARGET_HIT" in result
    mock_trade_repo.update_trade.assert_called_once()
    assert mock_trade_repo.update_trade.call_args[0][1]["exit_price"] == 135.0
    assert mock_trade_repo.update_trade.call_args[0][1]["status"] == TradeStatus.CLOSED

def test_manage_active_phase2_tp3_hit(strategy, mock_trade_repo):
    """Tests final exit in phase 2."""
    trade = {
        "id": "S1",
        "entry_price": 100.0,
        "current_stop_loss": 100.0,
        "status": "ACTIVE",
        "current_size": 10,
        "entry_date": "2026-02-18",
        "signal_context": json.dumps({"tp1": 110.0, "tp3": 130.0, "is_phase_2": True})
    }
    # High 132 (hits 130)
    candle = pd.Series({
        "open": 125.0, 
        "high": 132.0, 
        "low": 124.0, 
        "close": 131.0, 
        "date": pd.Timestamp("2026-02-20")
    }, name=pd.Timestamp("2026-02-20"))
    
    result = strategy.manage_active_trade(trade, pd.DataFrame([candle]), mock_trade_repo)
    
    assert result is not None
    assert mock_trade_repo.update_trade.call_args[0][1]["exit_price"] == 130.0

def test_manage_active_stop_loss(strategy, mock_trade_repo):
    """Tests stop loss hit in phase 2."""
    trade = {
        "id": "S1",
        "entry_price": 100.0,
        "current_stop_loss": 100.0,
        "status": "ACTIVE",
        "current_size": 10,
        "entry_date": "2026-02-18",
        "signal_context": json.dumps({"tp1": 110.0, "tp3": 130.0, "is_phase_2": True})
    }
    # Low 99 (hits 100)
    candle = pd.Series({
        "open": 105.0, 
        "high": 106.0, 
        "low": 99.0, 
        "close": 100.0, 
        "date": pd.Timestamp("2026-02-21")
    }, name=pd.Timestamp("2026-02-21"))
    
    result = strategy.manage_active_trade(trade, pd.DataFrame([candle]), mock_trade_repo)
    
    assert result is not None
    assert "STOP_LOSS" in result
    assert mock_trade_repo.update_trade.call_args[0][1]["exit_reason"] == ExitReason.STOP_LOSS

def test_generate_orders_multi_bracket(strategy, mock_trade_repo):
    """Tests generation of a multi-leg bracket order."""
    trade = {
        "symbol": "AAPL",
        "entry_price": 100.0,
        "current_stop_loss": 90.0,
        "initial_size": 20,
        "signal_context": json.dumps({"tp1": 110.0, "tp3": 130.0})
    }
    
    orders = strategy.generate_orders(trade, pd.DataFrame(), 2000.0, mock_trade_repo)
    
    assert orders is not None
    assert orders.mode == "BRACKET_MULTI"
    assert len(orders.exits) == 3 # SL, TP1, TP3
    
    # Check TP1 quantity (should be half)
    tp1_leg = next(leg for leg in orders.exits if leg.price == 110.0)
    assert tp1_leg.quantity == 10
