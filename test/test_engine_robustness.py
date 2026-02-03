
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import logging
import warnings
from app.services.backtester.engine import BacktestEngine

# Suppress Rich's Jupyter warning
pytestmark = pytest.mark.filterwarnings("ignore:install \"ipywidgets\" for Jupyter support")

# --- Fixtures ---

@pytest.fixture
def mock_dependencies():
    market = MagicMock()
    repo = MagicMock()
    console = MagicMock()
    return market, repo, console

# --- Tests ---

def test_engine_runs_with_empty_market_data(mock_dependencies):
    """CRASH+: Engine should handle completely empty market data without crashing."""
    market, repo, console = mock_dependencies
    
    # Arrange: Market returns empty history and empty fallback
    market.get_symbol_history.return_value = pd.DataFrame()
    market.get_available_dates.return_value = []
    
    engine = BacktestEngine("2025-01-01", "2025-01-31", market, repo, console)
    
    # Act
    engine.run()
    
    # Assert
    # Should print CRITICAL error and return
    # We verify it printed "CRITICAL"
    print_calls = [args[0] for args, _ in console.print.call_args_list]
    assert any("CRITICAL" in str(call) for call in print_calls)

def test_engine_fallback_logic_activates(mock_dependencies):
    """Logic: If SPY missing, it should try fallback dates."""
    market, repo, console = mock_dependencies
    
    # Arrange: SPY Empty, But Fallback has dates
    market.get_symbol_history.return_value = pd.DataFrame()
    market.get_available_dates.return_value = ["2025-01-05"]
    
    engine = BacktestEngine("2025-01-01", "2025-01-31", market, repo, console)
    
    # Act
    engine.setup() # Explicit setup call to verify internal state
    
    # Assert
    print_calls = [args[0] for args, _ in console.print.call_args_list]
    assert any("WARNING" in str(call) for call in print_calls)
    assert engine.market_dates == ["2025-01-05"]

def test_engine_skips_processing_if_dates_empty(mock_dependencies):
    """Hardening: Run should do nothing if no dates found."""
    market, repo, console = mock_dependencies
    
    # Arrange
    engine = BacktestEngine("2025-01-01", "2025-01-31", market, repo, console)
    engine.market_dates = [] # Explicitly empty
    
    # Using 'setup' to potentially fill it, but if we mock setup to do nothing?
    # Or just let run() call setup(), which returns empty.
    
    market.get_symbol_history.return_value = pd.DataFrame()
    market.get_available_dates.return_value = []
    
    # Act
    engine.run()
    
    # Assert
    # Progress bar should NOT run (or just init and finish)
    # This is tricky to test with Rich, checking side effects
    assert engine.current_date is None

@patch("app.services.backtester.engine.DipBuyerStrategy")
def test_engine_handles_screener_crash_gracefully(MockScreenerStrats, mock_dependencies):
    """Robustness: One day crashing shouldn't stop the whole backtest?"""
    # Current impl catches generic Exception in _run_screener.
    market, repo, console = mock_dependencies
    
    # Setup Logic
    hist_df = pd.DataFrame({"close": [100]}, index=pd.to_datetime(["2025-01-01"]))
    hist_df.index.name = "date"
    market.get_symbol_history.return_value = hist_df
    
    engine = BacktestEngine("2025-01-01", "2025-01-01", market, repo, console)
    
    # Mock Screener to raise Error
    engine.screener.run.side_effect = Exception("Boom")
    
    # Act
    # Should not raise
    engine.run()
    
    # Assert
    # Verify it continued. Since only 1 day, it finishes.
    # Check logs? We'd need to capture logs.
    assert True # Passed if no raise
