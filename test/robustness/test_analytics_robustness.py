# filename: test_analytics_robustness.py
import sqlite3
import pytest
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from app.services.backtester.analytics import BacktestAnalytics, BacktestMetrics

# --- FIXTURES ---

@pytest.fixture
def database_paths(tmp_path: Path) -> tuple[str, str]:
    """Creates temporary Backtest and Market DBs with proper schema."""
    backtest_db = tmp_path / "backtest.db"
    market_db = tmp_path / "market.db"
    
    # Initialize Backtest DB Schema
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
                realized_pnl REAL,
                strategy TEXT,
                exit_reason TEXT,
                current_stop_loss REAL
            )
        """)
        
    # Initialize Market DB Schema
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
def analytics_service(database_paths: tuple[str, str]) -> BacktestAnalytics:
    """Provides a BacktestAnalytics service instance."""
    return BacktestAnalytics(database_paths[0], database_paths[1])

# --- TESTS ---

def test_analytics_handles_zero_trades(analytics_service: BacktestAnalytics) -> None:
    """Verifies that the system returns empty metrics when no trades exist."""
    # Act
    metrics = analytics_service.run_analysis()
    
    # Assert
    assert isinstance(metrics, BacktestMetrics)
    assert metrics.total_trades == 0
    assert metrics.net_profit == 0.0

def test_analytics_division_by_zero_protection(
    database_paths: tuple[str, str]
) -> None:
    """Verifies Profit Factor and Kelly don't crash when Gross Loss is 0."""
    # Arrange
    backtest_db, market_db = database_paths
    with sqlite3.connect(backtest_db) as conn:
        conn.execute("""
            INSERT INTO trades (id, symbol, status, realized_pnl, initial_size, entry_price, strategy) 
            VALUES 
            ('1', 'AAPL', 'CLOSED', 100.0, 10, 150.0, 'TestStrat'),
            ('2', 'GOOG', 'CLOSED', 50.0, 5, 2000.0, 'TestStrat')
        """)
        
    service = BacktestAnalytics(backtest_db, market_db)
    
    # Act
    metrics = service.run_analysis()
    
    # Assert
    assert metrics.total_trades == 2
    assert metrics.win_rate == 1.0
    assert metrics.profit_factor == 999.0
    assert not np.isnan(metrics.kelly_criterion)

def test_kelly_bootstrapping_small_sample(database_paths: tuple[str, str]) -> None:
    """Verifies Bootstrapping handles < 20 trades gracefully."""
    # Arrange
    backtest_db, market_db = database_paths
    with sqlite3.connect(backtest_db) as conn:
        conn.execute("""
            INSERT INTO trades (id, symbol, status, realized_pnl, initial_size, entry_price, strategy) 
            VALUES 
            ('1', 'AAPL', 'CLOSED', 100.0, 10, 150.0, 'TestStrat'),
            ('2', 'TSLA', 'CLOSED', -50.0, 10, 200.0, 'TestStrat')
        """)
        
    service = BacktestAnalytics(backtest_db, market_db)
    
    # Act
    metrics = service.run_analysis()
    
    # Assert
    assert metrics.kelly_safe == 0.0
    assert metrics.kelly_mean == 0.0

@pytest.mark.parametrize("poisoned_pnl", [float('nan'), float('inf'), -float('inf'), 1e20])
def test_analytics_handles_poisoned_data(
    database_paths: tuple[str, str], 
    poisoned_pnl: float
) -> None:
    """Verifies that the system handles NaN or Infinite values in trade data."""
    # Arrange
    backtest_db, market_db = database_paths
    with sqlite3.connect(backtest_db) as conn:
        conn.execute(
            "INSERT INTO trades (id, symbol, status, realized_pnl, initial_size, entry_price, strategy) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ('p1', 'CRASH', 'CLOSED', poisoned_pnl, 10, 100.0, 'TestStrat')
        )

    service = BacktestAnalytics(backtest_db, market_db)
    
    # Act
    metrics = service.run_analysis()
    
    # Assert
    assert isinstance(metrics, BacktestMetrics)

def test_analytics_handling_corrupt_db_gracefully(tmp_path: Path) -> None:
    """Verifies that the system raises a clear error when the database is corrupt."""
    # Arrange
    empty_db = tmp_path / "empty_garbage.db"
    empty_db.write_text("Not a database")
    
    service = BacktestAnalytics(str(empty_db), str(empty_db))
    
    # Act & Assert
    with pytest.raises((sqlite3.DatabaseError, sqlite3.OperationalError, duckdb.IOException)):
        service.run_analysis()
