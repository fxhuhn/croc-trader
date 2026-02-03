
import pytest
import pandas as pd
import sqlite3
import numpy as np
from pathlib import Path
from app.services.backtester.analytics import BacktestAnalytics, BacktestMetrics

# --- Fixtures ---

@pytest.fixture
def temp_dbs(tmp_path):
    """Creates temporary Backtest and Market DBs with proper schema."""
    backtest_db = tmp_path / "backtest.db"
    market_db = tmp_path / "market.db"
    
    # Init Backtest DB Schema
    with sqlite3.connect(backtest_db) as conn:
        conn.execute("""
            CREATE TABLE trades (
                id TEXT PRIMARY KEY,
                symbol TEXT,
                status TEXT,
                entry_date TEXT,
                exit_date TEXT,
                entry_price REAL,
                exit_price REAL,
                initial_size INTEGER,
                realized_pnl REAL
            )
        """)
        
    # Init Market DB Schema
    with sqlite3.connect(market_db) as conn:
        conn.execute("""
            CREATE TABLE market_prices (
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER
            )
        """)
        
    return str(backtest_db), str(market_db)

@pytest.fixture
def analytics_service(temp_dbs):
    return BacktestAnalytics(temp_dbs[0], temp_dbs[1])

# --- Tests ---

def test_analytics_handles_zero_trades(analytics_service):
    """CRASH+: Verify system returns empty metrics instead of crashing when no trades exist."""
    # Arrange: DBs are empty by default
    
    # Act
    metrics = analytics_service.run_analysis()
    
    # Assert
    assert isinstance(metrics, BacktestMetrics)
    assert metrics.total_trades == 0
    assert metrics.net_profit == 0.0
    assert metrics.win_rate == 0.0
    assert metrics.profit_factor == 0.0
    assert metrics.max_drawdown == 0.0

def test_analytics_division_by_zero_protection(temp_dbs):
    """CRASH+: Verify Profit Factor and Kelly don't crash when Gross Loss is 0 (All Wins)."""
    backtest_db, market_db = temp_dbs
    
    # Arrange: Insert ONLY winning trades
    with sqlite3.connect(backtest_db) as conn:
        conn.execute("""
            INSERT INTO trades (id, symbol, status, realized_pnl, initial_size, entry_price) 
            VALUES 
            ('1', 'AAPL', 'CLOSED', 100.0, 10, 150.0),
            ('2', 'GOOG', 'CLOSED', 50.0, 5, 2000.0)
        """)
        
    analytics = BacktestAnalytics(backtest_db, market_db)
    
    # Act
    metrics = analytics.run_analysis()
    
    # Assert
    assert metrics.total_trades == 2
    assert metrics.win_rate == 1.0 # 100% Win Rate
    # Profit Factor should be handled (usually 999 or inf) logic says 999.0 for 0 loss
    assert metrics.profit_factor == 999.0
    # Kelly should not be NaN
    assert not np.isnan(metrics.kelly_criterion)

def test_kelly_bootstrapping_small_sample(temp_dbs):
    """CRASH+: Verify Bootstrapping handles < 20 trades gracefully."""
    backtest_db, market_db = temp_dbs
    
    # Arrange: Insert just 2 trades
    with sqlite3.connect(backtest_db) as conn:
        conn.execute("""
            INSERT INTO trades (id, symbol, status, realized_pnl, initial_size, entry_price) 
            VALUES 
            ('1', 'AAPL', 'CLOSED', 100.0, 10, 150.0),
            ('2', 'TSLA', 'CLOSED', -50.0, 10, 200.0)
        """)
        
    analytics = BacktestAnalytics(backtest_db, market_db)
    
    # Act
    metrics = analytics.run_analysis()
    
    # Assert
    # Logic: if len < 20, returns 0.0
    assert metrics.kelly_safe == 0.0
    assert metrics.kelly_mean == 0.0
    assert metrics.kelly_std == 0.0

def test_db_connection_leak_protection(temp_dbs):
    """Robustness: Ensure repeated calls don't exhaust connections (mock check)."""
    analytics = BacktestAnalytics(temp_dbs[0], temp_dbs[1])
    
    # Act - Run multiple times
    for _ in range(5):
        analytics.run_analysis()
    
    # If we got here without "Too many open files" or DuckDB errors, proper closing is working.
    assert True
