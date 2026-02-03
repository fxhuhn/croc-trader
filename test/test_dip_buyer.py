import pytest
import pandas as pd
from unittest.mock import MagicMock
from app.services.trade_manager.strategies.dip_buyer import DipBuyerStrategy
from app.database.repositories.trade import TradeRepository
from app.types import TradeStatus, ExitReason

# --- FIXTURES ---

@pytest.fixture
def strategy():
    return DipBuyerStrategy()

@pytest.fixture
def mock_repo():
    return MagicMock(spec=TradeRepository)

@pytest.fixture
def base_trade():
    """Returns a CREATED trade dict."""
    return {
        'id': 'trade-123',
        'symbol': 'TEST',
        'entry_price': 100.0,
        'current_target': 110.0,
        'current_stop_loss': 90.0, # Should be ignored
        'status': 'CREATED',
        'budget': 2000.0
    }


def create_candle(date_str, open_px, high, low, close):
    """Creates a 1-row Series representing a daily candle."""
    return pd.Series({
        'date': pd.Timestamp(date_str),
        'open': open_px,
        'high': high,
        'low': low,
        'close': close
    })

def create_history(dates, opens, highs, lows, closes):
    """Creates a DataFrame history with datetime objects."""
    df = pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes
    })
    df['date'] = pd.to_datetime(df['date'])
    return df


# --- ENTRY TESTS ---

def test_entry_standard_fill(strategy, mock_repo, base_trade):
    """Happy Path: Low < Limit < Open. Fills at Limit."""
    candle = create_candle("2026-01-02", 102.0, 105.0, 95.0, 100.0)
    df_hist = pd.DataFrame([candle])
    
    # Previous day (Signal Day) - Irrelevant for price check but needed for history context
    prev_candle = create_candle("2026-01-01", 100, 105, 95, 100)
    df_hist = pd.concat([pd.DataFrame([prev_candle]), df_hist], ignore_index=True)

    result = strategy.check_entry(base_trade, candle, df_hist, mock_repo)

    assert result is not None
    assert "FILLED @ 100.00" in result
    
    mock_repo.update_trade.assert_called_once()
    args, _ = mock_repo.update_trade.call_args
    data = args[1]
    assert data['entry_price'] == 100.0
    assert data['status'] == TradeStatus.ACTIVE

def test_entry_gap_down_fill(strategy, mock_repo, base_trade):
    """Gap Down: Open < Limit. Fills at Open (Better Price)."""
    # Limit: 100. Open: 95.
    candle = create_candle("2026-01-02", 95.0, 98.0, 90.0, 92.0)
    df_hist = pd.DataFrame([candle])

    result = strategy.check_entry(base_trade, candle, df_hist, mock_repo)

    assert result is not None
    assert "FILLED @ 95.00" in result  # Fill at Open
    
    mock_repo.update_trade.assert_called_once()
    args, _ = mock_repo.update_trade.call_args
    assert args[1]['entry_price'] == 95.0

def test_entry_no_fill_high_above_limit(strategy, mock_repo, base_trade):
    """Limit not Reached: Low > Limit."""
    # Limit: 100. Low: 101.
    candle = create_candle("2026-01-02", 105.0, 106.0, 101.0, 102.0)
    df_hist = pd.DataFrame([candle])

    result = strategy.check_entry(base_trade, candle, df_hist, mock_repo)

    assert result is not None
    assert "EXPIRED" in result
    
    mock_repo.update_trade.assert_called_once()
    args, _ = mock_repo.update_trade.call_args
    assert args[1]['status'] == TradeStatus.MISSED

# --- LIMIT ON CLOSE (LOC) TESTS ---

def test_exit_loc_triggered_on_active_day(strategy, mock_repo, base_trade):
    """LOC Rule: Close > Previous High -> Exit Market On Close."""
    # Active Trade
    base_trade.update({
        'status': 'ACTIVE', 
        'entry_date': '2026-01-02', 
        'entry_price': 100.0, 
        'current_size': 20
    })

    # Day 1 (Entry): High 105.
    # Day 2 (Current): Close 106 > PrevHigh 105.
    
    dates = ["2026-01-02", "2026-01-03"]
    opens = [100, 100]
    highs = [105, 108] # PrevHigh = 105
    lows  = [95, 95]
    closes= [100, 106] # Close = 106 (> 105)

    df_hist = create_history(dates, opens, highs, lows, closes)
    
    result = strategy.manage_active_trade(base_trade, df_hist, mock_repo)

    # Note: Strategy normally returns a string. If LOC logic is implemented, it should return an exit string.
    assert result is not None
    assert "LOC_HIT" in result or "TARGET" in result # Depending on implementation naming
    
    mock_repo.update_trade.assert_called_once()
    args, _ = mock_repo.update_trade.call_args
    assert args[1]['status'] == TradeStatus.CLOSED
    assert args[1]['exit_price'] == 106.0 # MOC

def test_exit_loc_triggered_on_entry_day_check_entry(strategy, mock_repo, base_trade):
    """
    LOC Rule on Entry Day: 
    We are in `check_entry`. If we fill, we check End-of-Day LOC immediately.
    If Close > PrevHigh, we should setup an IMMEDIATE Exit logic or handle it.
    """
    base_trade.update({
        'status': 'ACTIVE', 
        'entry_date': '2026-01-02', 
        'entry_price': 100.0, 
        'current_size': 20
    })

    # Previous Signal Day High: 104.
    # Entry Day Close: 105 (> 104). -> SHOULD EXIT LOC.
    dates = ["2026-01-01", "2026-01-02"]
    opens = [100, 100]
    highs = [104, 108]
    lows  = [95, 95]
    closes= [100, 105]

    df_hist = create_history(dates, opens, highs, lows, closes)
    
    result = strategy.manage_active_trade(base_trade, df_hist, mock_repo)
    
    assert result is not None
    assert "LOC_HIT" in result
    
    args, _ = mock_repo.update_trade.call_args
    assert args[1]['exit_price'] == 105.0

# --- TAKE PROFIT & DEFERRED LOGIC TESTS ---

def test_tp_blocked_on_entry_day(strategy, mock_repo, base_trade):
    """
    Take Profit (Target) MUST BE IGNORED on Entry Day.
    Only LOC or Stops allowed (but SL is removed, so only LOC).
    """
    base_trade.update({
        'status': 'ACTIVE', 
        'entry_date': '2026-01-02',
        'entry_price': 100.0, 
        'current_target': 110.0,
        'current_size': 20
    })

    # Entry Day: High 115 (> Target 110). 
    # Close 100 (< PrevHigh 105). No LOC.
    dates = ["2026-01-01", "2026-01-02"]
    opens = [100, 100]
    highs = [105, 115] # High > Target
    lows  = [95, 95]
    closes= [100, 100]

    df_hist = create_history(dates, opens, highs, lows, closes)
    
    # Act
    result = strategy.manage_active_trade(base_trade, df_hist, mock_repo)

    # Assert: Should return None (No Exit), because TP is deferred.
    assert result is None
    mock_repo.update_trade.assert_not_called()

def test_tp_active_on_subsequent_day(strategy, mock_repo, base_trade):
    """Take Profit should work from Day 2 onwards."""
    base_trade.update({
        'status': 'ACTIVE', 
        'entry_date': '2026-01-02',
        'entry_price': 100.0, 
        'current_target': 110.0,
        'current_size': 20
    })

    # Day 2: High 112 (> 110).
    dates = ["2026-01-02", "2026-01-03"]
    df_hist = create_history(dates, [100, 100], [105, 112], [95, 95], [100, 100])
    
    result = strategy.manage_active_trade(base_trade, df_hist, mock_repo)
    
    assert result is not None
    assert "TARGET_HIT" in result
    
    args, _ = mock_repo.update_trade.call_args
    assert args[1]['exit_price'] == 110.0

# --- STOP LOSS REMOVAL CHECK ---

def test_stop_loss_ignored(strategy, mock_repo, base_trade):
    """Strategy should IGNORE Stop Loss even if Low < SL."""
    base_trade.update({
        'status': 'ACTIVE', 
        'entry_date': '2026-01-02',
        'entry_price': 100.0, 
        'current_stop_loss': 90.0, # Exists in DB
        'current_size': 20
    })

    # Day 2: Low 80 (< 90). Disaster.
    dates = ["2026-01-02", "2026-01-03"]
    df_hist = create_history(dates, [100, 100], [105, 105], [95, 80], [100, 85])
    
    result = strategy.manage_active_trade(base_trade, df_hist, mock_repo)

    # Assert: NO EXIT (unless TimeStop or LOC triggers, but Close(85) < High(105))
    assert result is None
    mock_repo.update_trade.assert_not_called()

# --- TIME STOP TEST ---

def test_time_stop_triggered(strategy, mock_repo, base_trade):
    """
    Time Stop: Exit if held for >= 10 TRADING days (rows).
    """
    base_trade.update({
        'status': 'ACTIVE', 
        'entry_date': '2026-01-02', 
        'entry_price': 100.0, 
        'current_size': 20
    })

    # Create 11 days of history (Day 1 = Entry day)
    dates = [f"2026-01-{i:02d}" for i in range(2, 13)] # 2nd to 12th = 11 days
    # All prices flat to avoid TP or LOC
    prices = [100.0] * 11
    
    df_hist = create_history(dates, prices, prices, prices, prices)
    
    result = strategy.manage_active_trade(base_trade, df_hist, mock_repo)
    
    assert result is not None
    assert "TIME_STOP" in result
    
    args, _ = mock_repo.update_trade.call_args
    assert args[1]['exit_reason'] == ExitReason.TIME_STOP
