import pytest
from unittest.mock import MagicMock, call
from app.services.portfolio.manager import PortfolioManager
from app.services.portfolio.allocator import PortfolioAllocator, AllocationResult

# --- Allocator Tests ---

@pytest.fixture
def allocator():
    return PortfolioAllocator()

@pytest.mark.parametrize("trade_input,expected_size,expected_reason_substr", [
    # Happy Path: DipBuyer (Budget 2000)
    (
        {"strategy": "DipBuyer", "entry_price": 100.0, "symbol": "TEST"}, 
        20, 
        "Fixed Budget"
    ),
    (
        {"strategy": "DipBuyer", "entry_price": 2000.0, "symbol": "TEST"}, 
        1, 
        "Fixed Budget"
    ),
    # Edge Case: DipBuyer Price > Budget
    (
        {"strategy": "DipBuyer", "entry_price": 2001.0, "symbol": "TEST"}, 
        0, 
        "Price > Budget"
    ),
    # Happy Path: HoldTarget (Risk 100)
    (
        {"strategy": "HoldTarget", "entry_price": 50.0, "current_stop_loss": 45.0, "symbol": "TEST"}, 
        20, # Risk/Share = 5. 100/5 = 20.
        "Fixed Risk"
    ),
    # Happy Path: TurnoverTiming (Budget 2000)
    (
        {"strategy": "TurnoverTiming", "entry_price": 50.0, "symbol": "TEST"}, 
        40, 
        "Fixed Budget"
    ),
    # Edge Case: Invalid Entry Price
    (
        {"strategy": "DipBuyer", "entry_price": 0.0, "symbol": "TEST"}, 
        0, 
        "Invalid Entry Price"
    ),
    # Edge Case: Only Strategy Name Match (Case Insensitive)
    (
        {"strategy": "dipbuyer", "entry_price": 100.0, "symbol": "TEST"}, 
        20, 
        "Fixed Budget"
    ),
    # Edge Case: HoldTarget Invalid SL
    (
        {"strategy": "HoldTarget", "entry_price": 50.0, "current_stop_loss": 55.0, "symbol": "TEST"}, 
        0, 
        "Invalid Stop Loss"
    ),
    # Edge Case: HoldTarget Risk/Share > Risk Amount (Size < 1)
    # Risk 100. Entry 1000, SL 500. Risk/Share 500. 100/500 = 0.2 -> 0.
    (
        {"strategy": "HoldTarget", "entry_price": 1000.0, "current_stop_loss": 500.0, "symbol": "TEST"}, 
        0, 
        "Risk/Share > Risk Amount"
    ),
    # Unknown Strategy
    (
        {"strategy": "RandomStrat", "entry_price": 100.0, "symbol": "TEST"}, 
        0, 
        "Unknown Strategy"
    ),
])
def test_allocator_logic(allocator, trade_input, expected_size, expected_reason_substr):
    # Act
    result = allocator.allocate(trade_input)
    
    # Assert
    assert result.size == expected_size
    assert expected_reason_substr in result.reason

# --- Manager Tests ---

@pytest.fixture
def mock_repo():
    return MagicMock()

@pytest.fixture
def manager(mock_repo):
    return PortfolioManager(mock_repo)

def test_manager_processes_trades_successfully(manager, mock_repo):
    # Arrange
    trade_1 = {"id": "1", "symbol": "A", "strategy": "DipBuyer", "entry_price": 100.0, "initial_size": 0}
    trade_2 = {"id": "2", "symbol": "B", "strategy": "HoldTarget", "entry_price": 50.0, "current_stop_loss": 40.0, "initial_size": 0}
    
    # Mock return
    mock_repo.get_by_status.return_value = [trade_1, trade_2]
    
    # Act
    count = manager.process_daily_signals()
    
    # Assert
    assert count == 2
    # Verify updates
    assert mock_repo.update_trade.call_count == 2
    
    # Check Trade 1 Update (DipBuyer: 2000/100 = 20)
    import json
    args_1, kwargs_1 = mock_repo.update_trade.call_args_list[0]
    assert args_1[0] == "1"
    assert args_1[1]["initial_size"] == 20
    
    # Verify Context (Budget)
    context_1 = json.loads(args_1[1]["signal_context"])
    assert context_1["budget"] == 2000.0
    
    # Check Trade 2 Update (HoldTarget: 100/(50-40) = 10)
    args_2, kwargs_2 = mock_repo.update_trade.call_args_list[1]
    assert args_2[0] == "2"
    assert args_2[1]["initial_size"] == 10
    
    # Verify Context (Risk Amount)
    context_2 = json.loads(args_2[1]["signal_context"])
    assert context_2["risk_amount"] == 100.0

def test_manager_skips_trades_with_size(manager, mock_repo):
    # Arrange: Trade already has size
    trade_1 = {"id": "1", "symbol": "A", "strategy": "DipBuyer", "entry_price": 100.0, "initial_size": 10}
    mock_repo.get_by_status.return_value = [trade_1]
    
    # Act
    count = manager.process_daily_signals()
    
    # Assert
    assert count == 0
    mock_repo.update_trade.assert_not_called()

def test_manager_handles_repo_exception_gracefully(manager, mock_repo):
    # Arrange
    mock_repo.get_by_status.side_effect = Exception("DB Connection Failed")
    
    # Act
    count = manager.process_daily_signals()
    
    # Assert
    assert count == 0
    # Should log error but not crash (verified by successful return)

def test_manager_handles_allocation_failure_gracefully(manager, mock_repo):
    # Arrange: Trade that fails allocation (e.g. price > budget)
    trade_1 = {"id": "1", "symbol": "A", "strategy": "DipBuyer", "entry_price": 5000.0, "initial_size": 0}
    mock_repo.get_by_status.return_value = [trade_1]
    
    # Act
    count = manager.process_daily_signals()
    
    # Assert
    assert count == 0
    mock_repo.update_trade.assert_not_called()
