# filename: test_strategy_hold_target.py
import pytest
from unittest.mock import MagicMock
import pandas as pd
import json

from app.services.trade_manager.strategies.hold_target import HoldTargetStrategy
from app.types import TradeStatus, ExitReason
from app.database.repositories.trade import TradeRepository

# --- FIXTURES ---

@pytest.fixture
def strategy() -> HoldTargetStrategy:
    """Provides a fresh instance of the HoldTargetStrategy."""
    return HoldTargetStrategy()

@pytest.fixture
def mock_repository() -> MagicMock:
    """Provides a mock TradeRepository."""
    return MagicMock(spec=TradeRepository)

@pytest.fixture
def base_trade_data() -> dict:
    """Returns a dictionary representing a trade in CREATED status."""
    return {
        "id": "trade-123",
        "symbol": "TEST",
        "entry_price": 100.0,
        "current_stop_loss": 90.0,
        "current_target": 130.0,
        "current_size": 0,
        "risk_amount": 100.0,
        "status": "CREATED"
    }

# --- HELPERS ---

def create_candle(
    open_price: float, 
    high: float, 
    low: float, 
    close: float, 
    date_str: str = "2025-01-02"
) -> pd.Series:
    """Helper to create a 1-row Series representing a daily candle."""
    return pd.Series({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "date": pd.Timestamp(date_str)
    })

def create_history(
    open_price: float, 
    high: float, 
    low: float, 
    close: float, 
    date_str: str = "2025-01-02"
) -> pd.DataFrame:
    """Helper for history dataframe."""
    df = pd.DataFrame([{
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "date": pd.Timestamp(date_str)
    }])
    df["date"] = pd.to_datetime(df["date"])
    return df

# --- ENTRY LOGIC TESTS ---

@pytest.mark.parametrize("open_price, high, low, expected_fill, expected_reason", [
    (100.0, 105.0, 95.0, 100.0, "GAP UP"),     # 1. Gap Up / Touch Open (Open >= Trigger) -> Fill Open
    (102.0, 105.0, 95.0, 102.0, "GAP UP"),     # 2. Real Gap Up (Open > Trigger) -> Fill Open
    (95.0,  105.0, 92.0, 100.0, "BREAKOUT"),   # 3. Intraday Breakout (High > Trigger, Open < Trigger) -> Fill Trigger
    (95.0,  99.0,  92.0, None,  None),         # 4. No Fill (High < Trigger)
])
def test_check_entry_success_scenarios(
    strategy: HoldTargetStrategy, 
    mock_repository: MagicMock, 
    base_trade_data: dict, 
    open_price: float, 
    high: float, 
    low: float, 
    expected_fill: float | None, 
    expected_reason: str | None
) -> None:
    """Tests standard entry scenarios (Fills and No-Fills)."""
    # Arrange
    candle = create_candle(open_price, high, low, 100.0)
    
    # Act
    result = strategy.check_entry(base_trade_data, candle, pd.DataFrame(), mock_repository)
    
    # Assert
    if expected_fill:
        assert result is not None
        assert f"FILLED @ {expected_fill:.2f}" in result
        
        mock_repository.update_trade.assert_called_once()
        args, _ = mock_repository.update_trade.call_args
        data = args[1]
        
        assert data["status"] == TradeStatus.ACTIVE
        assert data["entry_price"] == expected_fill
    else:
        assert result is None
        mock_repository.update_trade.assert_not_called()

def test_check_entry_day_one_turnaround(
    strategy: HoldTargetStrategy, 
    mock_repository: MagicMock, 
    base_trade_data: dict
) -> None:
    """Verifies that hitting entry and stop on the same day results in a CLOSED trade."""
    # Arrange
    candle = create_candle(95.0, 105.0, 89.0, 95.0)
    
    # Act
    result = strategy.check_entry(base_trade_data, candle, pd.DataFrame(), mock_repository)
    
    # Assert
    assert result is not None
    assert "FILLED" in result
    
    mock_repository.update_trade.assert_called_once()
    args, _ = mock_repository.update_trade.call_args
    data = args[1]
    
    assert data["status"] == TradeStatus.CLOSED
    assert data["exit_reason"] == ExitReason.STOP_LOSS
    assert data["entry_price"] == 100.0
    assert data["exit_price"] == 90.0

def test_check_entry_invalidation_no_trigger(
    strategy: HoldTargetStrategy, 
    mock_repository: MagicMock, 
    base_trade_data: dict
) -> None:
    """Verifies that a setup is invalidated if Low < Stop before entry is triggered."""
    # Arrange
    candle = create_candle(95.0, 99.0, 89.0, 95.0)
    
    # Act
    result = strategy.check_entry(base_trade_data, candle, pd.DataFrame(), mock_repository)
    
    # Assert
    assert result is not None
    assert "INVALID" in result
    
    mock_repository.update_trade.assert_called_once()
    args, _ = mock_repository.update_trade.call_args
    data = args[1]
    
    assert data["status"] == TradeStatus.INVALID
    assert data["exit_reason"] == ExitReason.INVALIDATED

# --- RISK CALCULATION TESTS ---

def test_check_entry_risk_calculation(
    strategy: HoldTargetStrategy, 
    mock_repository: MagicMock, 
    base_trade_data: dict
) -> None:
    """Test position sizing: 100$ Risk / (100 Entry - 90 Stop) = 10 Shares."""
    # Arrange
    candle = create_candle(95.0, 101.0, 95.0, 100.0)
    
    # Act
    strategy.check_entry(base_trade_data, candle, pd.DataFrame(), mock_repository)
    
    # Assert
    args, _ = mock_repository.update_trade.call_args
    data = args[1]
    assert data["initial_size"] == 10

# --- EXIT LOGIC TESTS ---

@pytest.mark.parametrize("open_price, high, low, stop, target, expected_exit_price, expected_reason", [
    (100, 110, 89,  90, 120, 90.0, "STOP_LOSS"),   # 1. Normal Stop Hit
    (88,  100, 85,  90, 120, 88.0, "STOP_LOSS"),   # 2. Gap Down over Stop
    (100, 121, 95,  90, 120, 120.0, "TARGET_HIT"), # 3. Normal Target Hit
    (125, 130, 100, 90, 120, 125.0, "TARGET_HIT"), # 4. Gap Up over Target
])
def test_manage_active_trade_scenarios(
    strategy: HoldTargetStrategy, 
    mock_repository: MagicMock, 
    base_trade_data: dict, 
    open_price: float, 
    high: float, 
    low: float, 
    stop: float, 
    target: float, 
    expected_exit_price: float, 
    expected_reason: str
) -> None:
    """Verifies various exit scenarios for active trades."""
    # Arrange
    base_trade_data.update({
        "status": "ACTIVE",
        "current_size": 10,
        "current_stop_loss": stop,
        "current_target": target
    })
    df_history = create_history(open_price, high, low, 100)
    
    # Act
    result = strategy.manage_active_trade(base_trade_data, df_history, mock_repository)
    
    # Assert
    assert result is not None
    assert expected_reason in result
    
    mock_repository.update_trade.assert_called_once()
    args, _ = mock_repository.update_trade.call_args
    data = args[1]
    
    assert data["status"] == TradeStatus.CLOSED
    assert data["exit_reason"] == expected_reason
    assert data["exit_price"] == expected_exit_price

def test_manage_active_trade_stop_priority(
    strategy: HoldTargetStrategy, 
    mock_repository: MagicMock, 
    base_trade_data: dict
) -> None:
    """Verifies that Stop Loss has priority over Target if both are hit in one day."""
    # Arrange
    base_trade_data.update({
        "status": "ACTIVE",
        "current_size": 10,
        "current_stop_loss": 90.0,
        "current_target": 110.0
    })
    # Low (85) < Stop (90) AND High (115) > Target (110)
    df_history = create_history(100, 115, 85, 100)
    
    # Act
    strategy.manage_active_trade(base_trade_data, df_history, mock_repository)
    
    # Assert
    args, _ = mock_repository.update_trade.call_args
    data = args[1]
    assert data["exit_reason"] == ExitReason.STOP_LOSS
    assert data["exit_price"] == 90.0

def test_check_expiration(
    strategy: HoldTargetStrategy, 
    mock_repository: MagicMock, 
    base_trade_data: dict
) -> None:
    """Verifies that signals expire after 5 days."""
    # Arrange
    base_trade_data["signal_context"] = json.dumps({"date": "2025-01-01"})
    # 6 Days later
    candle = create_candle(95.0, 99.0, 95.0, 95.0, date_str="2025-01-07")
    
    # Act
    result = strategy.check_entry(base_trade_data, candle, pd.DataFrame(), mock_repository)
    
    # Assert
    assert result == "EXPIRED"
    mock_repository.update_trade.assert_called_once()
    args, _ = mock_repository.update_trade.call_args
    assert args[1]["status"] == TradeStatus.CLOSED
    assert args[1]["exit_reason"] == ExitReason.EXPIRED
