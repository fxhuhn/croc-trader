import pytest
from app.routes.views import prepare_view_model

def test_active_trade_quantity_mapping():
    """Verify that active trades display initial_size correctly."""
    trade = {
        "status": "ACTIVE",
        "entry_price": 100.0,
        "current_price": 110.0,
        "initial_size": 10,
        "current_size": 10,
        "current_stop_loss": 90.0,
        "context": {},
        "symbol": "AAPL"
    }
    
    # Mock Market Repository (not used for this test if prices present)
    class MockRepo:
        def get_latest_price(self, symbol): return 110.0
        
    prepare_view_model([trade], MockRepo())
    
    assert trade["display_size"] == 10
    assert trade["unrealized_pnl"] == (110.0 - 100.0) * 10
    assert trade["pnl_pct"] == 10.0

def test_closed_trade_quantity_mapping():
    """Verify that closed trades use initial_size for quantity display even if current_size is 0."""
    trade = {
        "status": "CLOSED",
        "entry_price": 100.0,
        "exit_price": 110.0,
        "initial_size": 20,
        "current_size": 0, # Typical for closed trades
        "realized_pnl": 0.0, # Let view model calculate it
        "context": {},
        "symbol": "MSFT"
    }
    
    class MockRepo:
        def get_latest_price(self, symbol): return 120.0

    prepare_view_model([trade], MockRepo())
    
    # Check fallback PnL calculation
    expected_pnl = (110.0 - 100.0) * 20
    assert trade["realized_pnl"] == expected_pnl
    assert trade["display_size"] == 20
