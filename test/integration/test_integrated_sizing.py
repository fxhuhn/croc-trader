import pytest
import pandas as pd
import json
from datetime import date
from unittest.mock import MagicMock

from app.services.backtester.engine import BacktestEngine
from app.services.portfolio.manager import PortfolioManager
from app.services.portfolio.allocator import PortfolioAllocator
from app.database.repositories.trade import TradeRepository
from app.database.repositories.market_data_provider import MarketDataProvider
from app.types import TradeStatus

def test_integrated_sizing_flow():
    """
    Verifies that the PortfolioManager is correctly integrated into the BacktestEngine
    and that it updates trades with size and context metadata.
    """
    # 1. Setup Mocks
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_market = MagicMock(spec=MarketDataProvider)
    
    # 2. Create a Mock Trade (CREATED status, size 0)
    mock_trade = {
        "id": 1,
        "symbol": "TEST",
        "strategy": "DipBuyer",
        "status": TradeStatus.CREATED,
        "entry_price": 100.0,
        "initial_size": 0, # Needs allocation
        "current_size": 0,
        "signal_context": "{}" 
    }
    
    mock_trade_repo.get_by_status.return_value = [mock_trade]
    
    # 3. Setup Engine
    engine = BacktestEngine(
        start_date="2024-01-01", 
        end_date="2024-01-02", 
        market_provider=mock_market, 
        trade_repository=mock_trade_repo
    )
    
    # 4. Manually trigger the sizing step (simulate engine loop)
    # We call process_daily_signals directly to verify logic, 
    # as running the full engine loop requires complex market data setup.
    allocated_count = engine.portfolio_manager.process_daily_signals()
    
    # 5. Assertions
    assert allocated_count == 1
    
    # Verify update_trade was called with correct parameters
    mock_trade_repo.update_trade.assert_called_once()
    call_args = mock_trade_repo.update_trade.call_args
    trade_id, updates = call_args[0]
    
    assert trade_id == 1
    assert updates["initial_size"] > 0
    assert updates["current_size"] > 0
    
    # Verify Context Metadata
    context = json.loads(updates["signal_context"])
    assert "budget" in context
    assert context["budget"] > 0
    # DipBuyer uses budget, risk_amount is 0.0 or not set depending on allocator logic (it returns 0.0)
    assert context.get("risk_amount", 0.0) == 0.0
