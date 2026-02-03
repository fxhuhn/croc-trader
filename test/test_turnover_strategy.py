import pytest
import pandas as pd
from unittest.mock import MagicMock
from datetime import date

from app.services.trade_manager.strategies.turnover_timing import TurnoverTimingStrategy
from app.database.repositories.trade import TradeRepository
from app.types import TradeStatus, ExitReason

# --- Fixtures ---

@pytest.fixture
def mock_trade_repo():
    return MagicMock(spec=TradeRepository)

@pytest.fixture
def strategy(mock_trade_repo):
    # Strategy in TradeManager is stateless and takes no arguments in __init__
    return TurnoverTimingStrategy()

@pytest.fixture
def sample_trade():
    """Returns a basic created trade dictionary."""
    return {
        "id": 1,
        "symbol": "TEST",
        "status": TradeStatus.CREATED,
        "entry_price": 100.0,
        "current_size": 0,
        # signal_context is JSON string in DB, but strategy parses it.
        # Strategy expects 'signal_context' key.
        "signal_context": '{"setup_date": "2024-01-01"}'
    }

@pytest.fixture
def history_df():
    """Helper to create minimal OHLCV dataframe."""
    def _create(data):
        return pd.DataFrame(data)
    return _create

# --- Entry Tests ---

def test_entry_fill_at_open_gap_down(strategy, mock_trade_repo, sample_trade):
    """
    Scenario: Market opens BELOW limit price.
    Expected: Filled at OPEN price (better price).
    """
    # Arrange
    # Limit is 100.0. Open is 98.0.
    candle = pd.Series({
        "date": "2024-01-02", # Day after setup
        "open": 98.0,
        "high": 105.0,
        "low": 97.0,
        "close": 102.0
    })
    
    # Act
    result = strategy.check_entry(sample_trade, candle, pd.DataFrame(), mock_trade_repo)
    
    # Assert
    assert result == "✅ FILLED @ 98.00"
    mock_trade_repo.update_trade.assert_called_once()
    call_args = mock_trade_repo.update_trade.call_args[0]
    assert call_args[0] == 1 # ID
    payload = call_args[1]
    assert payload["status"] == TradeStatus.ACTIVE
    assert payload["entry_price"] == 98.0
    assert payload["entry_date"] == "2024-01-02"

def test_entry_fill_at_limit_intraday(strategy, mock_trade_repo, sample_trade):
    """
    Scenario: Market opens ABOVE limit, but dips BELOW limit intraday.
    Expected: Filled at LIMIT price.
    """
    # Arrange
    # Limit 100.0. Open 102.0. Low 99.0.
    candle = pd.Series({
        "date": "2024-01-02",
        "open": 102.0,
        "high": 103.0,
        "low": 99.0,
        "close": 101.0
    })
    
    # Act
    result = strategy.check_entry(sample_trade, candle, pd.DataFrame(), mock_trade_repo)
    
    # Assert
    assert result == "✅ FILLED @ 100.00"
    payload = mock_trade_repo.update_trade.call_args[0][1]
    assert payload["entry_price"] == 100.0

def test_entry_missed_expired(strategy, mock_trade_repo, sample_trade):
    """
    Scenario: Price never touches limit.
    Expected: Trade marked as MISSED/EXPIRED (Day Valid).
    """
    # Arrange
    # Limit 100.0. Low 101.0.
    candle = pd.Series({
        "date": "2024-01-02",
        "open": 102.0,
        "high": 105.0,
        "low": 101.0, 
        "close": 104.0
    })
    
    # Act
    result = strategy.check_entry(sample_trade, candle, pd.DataFrame(), mock_trade_repo)
    
    # Assert
    assert "MISSED" in result
    payload = mock_trade_repo.update_trade.call_args[0][1]
    assert payload["status"] == TradeStatus.MISSED
    assert payload["exit_reason"] == ExitReason.EXPIRED

def test_entry_lookahead_protection_ignored(strategy, mock_trade_repo, sample_trade):
    """
    Scenario: Current candle is same day as Signal (or before).
    Expected: No action (None), waiting for next day.
    """
    # Arrange
    # Signal Date is 2024-01-01 (from fixture)
    candle = pd.Series({
        "date": "2024-01-01", # SAME DAY
        "open": 90.0, # Would trigger fill
        "low": 90.0
    })
    
    # Act
    result = strategy.check_entry(sample_trade, candle, pd.DataFrame(), mock_trade_repo)
    
    # Assert
    assert result is None
    mock_trade_repo.update_trade.assert_not_called()

# --- Exit Tests ---

def test_exit_time_stop_friday(strategy, mock_trade_repo, history_df):
    """
    Scenario: Current candle is a Friday.
    Expected: Close trade at Close Price (Time Stop).
    """
    # Arrange
    trade = {
        "id": 2, 
        "symbol": "TEST", 
        "entry_price": 100.0, 
        "current_size": 10,
        "status": TradeStatus.ACTIVE
    }
    # Friday 2024-01-05
    df = history_df({
        "date": ["2024-01-05"],
        "open": [105.0],
        "close": [110.0]
    })
    df["date"] = pd.to_datetime(df["date"])

    # Act
    result = strategy.manage_active_trade(trade, df, mock_trade_repo)
    
    # Assert
    assert "EXIT" in result
    payload = mock_trade_repo.update_trade.call_args[0][1]
    assert payload["status"] == TradeStatus.CLOSED
    assert payload["exit_reason"] == ExitReason.TIME_STOP
    assert payload["exit_price"] == 110.0 # Close price

def test_exit_two_green_candles(strategy, mock_trade_repo, history_df):
    """
    Scenario: Yesterday and Day-Before-Yesterday were Green.
    Expected: Close trade at TODAY's Open.
    """
    # Arrange
    trade = {
        "id": 3, "symbol": "TEST", "entry_price": 100.0, "current_size": 10,
        "entry_date": "2024-01-01", "signal_context": "{}"
    }
    
    # 2024-01-02 (Green), 2024-01-03 (Green), 2024-01-04 (Today)
    df = history_df({
        "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "open": [100.0, 102.0, 105.0],
        "close": [101.0, 104.0, 106.0] 
        # Day 1: 101 > 100 (Green)
        # Day 2: 104 > 102 (Green)
        # Day 3 (Today): We use Open (105.0)
    })
    df["date"] = pd.to_datetime(df["date"])

    # Act
    result = strategy.manage_active_trade(trade, df, mock_trade_repo)
    
    # Assert
    assert "EXIT" in result
    payload = mock_trade_repo.update_trade.call_args[0][1]
    assert payload["exit_reason"] == ExitReason.MANUAL # Currently mapped to MANUAL in code
    assert payload["exit_price"] == 105.0 # Today's Open

def test_exit_holds_mixed_candles(strategy, mock_trade_repo, history_df):
    """
    Scenario: Yesterday Green, but Day-Before RED. Not Friday.
    Expected: No Action (Hold).
    """
    # Arrange
    trade = {
        "id": 4, "symbol": "TEST", "entry_price": 100.0, "current_size": 10,
        "signal_context": "{}"
    }
    
    # Day 1 Red, Day 2 Green, Day 3 Today (Monday)
    df = history_df({
        "date": ["2024-01-08", "2024-01-09", "2024-01-10"],
        "open": [102.0, 100.0, 105.0],
        "close": [101.0, 102.0, 106.0]
        # Day 1: 101 < 102 (Red)
        # Day 2: 102 > 100 (Green)
    })
    df["date"] = pd.to_datetime(df["date"])

    # Act
    result = strategy.manage_active_trade(trade, df, mock_trade_repo)
    
    # Assert
    assert result is None
    mock_trade_repo.update_trade.assert_not_called()
