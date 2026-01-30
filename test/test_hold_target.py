import pytest
from unittest.mock import MagicMock
import pandas as pd
import json

from app.services.trade_manager.strategies.hold_target import HoldTargetStrategy
from app.types import TradeStatus, ExitReason
from app.database.repositories.trade import TradeRepository

# --- FIXTURES ---

@pytest.fixture
def strategy():
    return HoldTargetStrategy()

@pytest.fixture
def mock_repo():
    return MagicMock(spec=TradeRepository)

@pytest.fixture
def base_trade():
    """Returns a dictionary representing a trade in CREATED status."""
    return {
        'id': 'trade-123',
        'symbol': 'TEST',
        'entry_price': 100.0,
        'current_stop_loss': 90.0,
        'current_target': 130.0,
        'current_size': 0,
        'risk_amount': 100.0,
        'status': 'CREATED'
    }

def create_candle(open_px, high, low, close, date_str='2025-01-02'):
    """Helper to create a 1-row Series representing a daily candle."""
    return pd.Series({
        'open': open_px,
        'high': high,
        'low': low,
        'close': close,
        'date': pd.Timestamp(date_str)
    })

def create_history(open_px, high, low, close, date_str='2025-01-02'):
    """Helper for history dataframe."""
    df = pd.DataFrame([{
        'open': open_px,
        'high': high,
        'low': low,
        'close': close,
        'date': pd.Timestamp(date_str)
    }])
    df['date'] = pd.to_datetime(df['date'])
    return df

# --- ENTRY LOGIC TESTS ---

@pytest.mark.parametrize("open_px, high, low, expected_fill, expected_reason", [
    (100.0, 105.0, 95.0, 100.0, "GAP UP"),     # 1. Gap Up / Touch Open (Open >= Trigger) -> Fill Open
    (102.0, 105.0, 95.0, 102.0, "GAP UP"),     # 2. Real Gap Up (Open > Trigger) -> Fill Open
    (95.0,  105.0, 92.0, 100.0, "BREAKOUT"),   # 3. Intraday Breakout (High > Trigger, Open < Trigger) -> Fill Trigger
    (95.0,  99.0,  92.0, None,  None),         # 4. No Fill (High < Trigger)
])
def test_check_entry_success_scenarios(strategy, mock_repo, base_trade, open_px, high, low, expected_fill, expected_reason):
    """Tests standard entry scenarios (Fills and No-Fills)."""
    
    # Trigger is 100.0 from base_trade
    candle = create_candle(open_px, high, low, 100.0)
    
    result = strategy.check_entry(base_trade, candle, pd.DataFrame(), mock_repo)
    
    if expected_fill:
        assert result is not None
        assert f"FILLED @ {expected_fill:.2f}" in result
        
        # Verify Repo Update
        mock_repo.update_trade.assert_called_once()
        args, kwargs = mock_repo.update_trade.call_args
        data = args[1]
        
        assert data['status'] == TradeStatus.ACTIVE
        assert data['entry_price'] == expected_fill
        assert data['initial_size'] > 0
    else:
        assert result is None
        mock_repo.update_trade.assert_not_called()

def test_check_entry_invalidation_low_below_stop(strategy, mock_repo, base_trade):
    """
    Test the Safety Rule: If Low < Stop, the trade should be INVALIDATED.
    Stop = 90.0. 
    Candle Low = 89.0 (< 90).
    Even if High (105) triggered entry, the Low violation invalidates it.
    """
    candle = create_candle(95.0, 105.0, 89.0, 95.0)
    
    result = strategy.check_entry(base_trade, candle, pd.DataFrame(), mock_repo)
    
    assert result is not None
    assert "INVALID" in result
    
    mock_repo.update_trade.assert_called_once()
    args, kwargs = mock_repo.update_trade.call_args
    data = args[1]
    
    assert data['status'] == TradeStatus.INVALID
    assert data['exit_reason'] == ExitReason.INVALIDATED

# --- RISK CALCULATION TESTS ---

def test_check_entry_risk_calculation(strategy, mock_repo, base_trade):
    """Test position sizing: 100$ Risk / (100 Entry - 90 Stop) = 10 Shares."""
    candle = create_candle(95.0, 101.0, 95.0, 100.0) # Breakout fill at 100.0
    
    strategy.check_entry(base_trade, candle, pd.DataFrame(), mock_repo)
    
    args, _ = mock_repo.update_trade.call_args
    data = args[1]
    
    assert data['initial_size'] == 10  # 100 / 10 = 10

def test_check_entry_risk_div_by_zero_protection(strategy, mock_repo, base_trade):
    """
    Test protection when Stop Loss is 0 (risk calculation fails).
    Should return specific Error string, not fill.
    """
    base_trade['current_stop_loss'] = 0.0
    # Candle triggers entry (High 101 > 100)
    candle = create_candle(95.0, 101.0, 95.0, 100.0)
    
    result = strategy.check_entry(base_trade, candle, pd.DataFrame(), mock_repo)
    
    # Expect Error Message
    assert result == "ERROR: Zero Size"
    
    # Ensure NO Trade Update happens (except maybe logging, but here we check repo calls)
    mock_repo.update_trade.assert_not_called()


# --- EXIT LOGIC TESTS (manage_active_trade) ---

@pytest.mark.parametrize("open_px, high, low, stop, target, expected_exit_px, expected_reason", [
    (100, 110, 89,  90, 120, 90.0, "STOP_LOSS"),   # 1. Normal Stop Hit (Low < Stop)
    (88,  100, 85,  90, 120, 88.0, "STOP_LOSS"),   # 2. Gap Down over Stop (Open < Stop) -> Exit Open
    (100, 121, 95,  90, 120, 120.0, "TARGET_HIT"), # 3. Normal Target Hit (High > Target)
    (125, 130, 100, 90, 120, 125.0, "TARGET_HIT"), # 4. Gap Up over Target (Open > Target) -> Exit Open
    (100, 110, 95,  90, 120, None,  None),         # 5. No Exit (Inside bounds)
])
def test_manage_active_trade_scenarios(strategy, mock_repo, base_trade, open_px, high, low, stop, target, expected_exit_px, expected_reason):
    # Setup Active Trade
    base_trade['status'] = 'ACTIVE'
    base_trade['current_size'] = 10
    base_trade['current_stop_loss'] = stop
    base_trade['current_target'] = target
    
    df = create_history(open_px, high, low, 100)
    
    result = strategy.manage_active_trade(base_trade, df, mock_repo)
    
    if expected_reason:
        assert result is not None
        assert expected_reason in result
        
        mock_repo.update_trade.assert_called_once()
        args, _ = mock_repo.update_trade.call_args
        data = args[1]
        
        assert data['status'] == TradeStatus.CLOSED
        assert data['exit_reason'] == expected_reason
        assert data['exit_price'] == expected_exit_px
    else:
        assert result is None

def test_manage_active_trade_conflict_logic(strategy, mock_repo, base_trade):
    """
    Test conflict case: Low < Stop AND High > Target.
    Priority => STOP LOSS.
    """
    base_trade['status'] = 'ACTIVE'
    base_trade['current_size'] = 10
    base_trade['current_stop_loss'] = 90.0
    base_trade['current_target'] = 110.0
    
    # Candle: Low (85) < Stop (90) AND High (115) > Target (110)
    df = create_history(100, 115, 85, 100)
    
    result = strategy.manage_active_trade(base_trade, df, mock_repo)
    
    # Expectation: STOP LOSS
    args, _ = mock_repo.update_trade.call_args
    data = args[1]
    assert data['exit_reason'] == ExitReason.STOP_LOSS
    assert data['exit_price'] == 90.0

# --- DATE VALIDATION TESTS ---

def test_check_entry_prevents_same_day_execution(strategy, mock_repo, base_trade):
    """Signal Date vs Candle Date Validation."""
    # Signal Date: 2025-01-01
    base_trade['signal_context'] = json.dumps({"date": "2025-01-01"})
    
    # Candle Date: 2025-01-01 (Same Day) -> Should FAIL
    candle = create_candle(105.0, 110.0, 100.0, 105.0, date_str='2025-01-01')
    
    result = strategy.check_entry(base_trade, candle, pd.DataFrame(), mock_repo)
    assert result is None
    
    # Candle Date: 2025-01-02 (Next Day) -> Should PASS
    candle = create_candle(105.0, 110.0, 100.0, 105.0, date_str='2025-01-02')
    result = strategy.check_entry(base_trade, candle, pd.DataFrame(), mock_repo)
    assert result is not None
