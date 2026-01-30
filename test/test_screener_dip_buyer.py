import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime

from app.services.screener.strategies.dip_buyer import DipBuyerStrategy, DipBuyerConfig
from app.database.repositories.trade import TradeRepository
from app.database.repositories.market_data_provider import MarketDataProvider

# --- HELPER FUNCTIONS ---

def create_market_data(length=250, price_start=100.0, trend=0.1, symbol="TEST"):
    """Creates a Dict of DataFrames matching the Provider format (Columns=Symbols)."""
    dates = pd.date_range(end=datetime.now(), periods=length, freq='B')
    
    # Generate random walk
    returns = np.random.normal(0, 0.01, length) + (trend / length)
    prices = price_start * np.exp(np.cumsum(returns))
    
    close = prices
    high = close * 1.02
    low = close * 0.98
    open_p = close * 1.01 
    volume = np.random.randint(500_000, 2_000_000, length)
    
    # Structure: Dict[field] -> DataFrame(index=Dates, columns=[Symbol])
    return {
        "close": pd.DataFrame({symbol: close}, index=dates),
        "open": pd.DataFrame({symbol: open_p}, index=dates),
        "high": pd.DataFrame({symbol: high}, index=dates),
        "low": pd.DataFrame({symbol: low}, index=dates),
        "volume": pd.DataFrame({symbol: volume}, index=dates)
    }

# --- FIXTURES ---

@pytest.fixture
def mock_repo():
    return MagicMock(spec=TradeRepository)

@pytest.fixture
def mock_provider():
    return MagicMock(spec=MarketDataProvider)

@pytest.fixture
def strategy(mock_repo, mock_provider):
    # Mock symbols to include our test ticker
    dt_patcher = patch.object(DipBuyerStrategy, '_initialize_symbol_sets', return_value=None)
    dt_patcher.start()
    
    strat = DipBuyerStrategy(trade_repo=mock_repo, data_provider=mock_provider)
    
    # Manually inject symbol sets
    strat._dow_set = {'TEST'}
    strat._sp500_set = {'TEST'}
    strat._ndx_set = set()
    
    yield strat
    dt_patcher.stop()

# --- TESTS ---

def test_compute_indicators_calculation(strategy):
    """Test if indicators (SMA, ATR, IBS) are calculated correctly."""
    data = create_market_data(length=300)
    
    # Run calculation
    indicators = strategy._compute_indicators(
        data["close"], 
        data["high"], 
        data["low"], 
        data["volume"]
    )
    
    assert 'sma200' in indicators
    assert 'atr' in indicators
    assert 'ibs' in indicators
    
    # SMA200 check: Last value should be present
    assert not pd.isna(indicators['sma200'].iloc[-1]).all()
    
    # ATR check: Positive value
    assert (indicators['atr'].iloc[-1] > 0).all()
    
    # IBS Check: Between 0 and 1
    ibs = indicators['ibs'].iloc[-1]
    assert (0 <= ibs).all() and (ibs <= 1.0).all()

def test_entry_filter_logic_success(strategy):
    """Test a scenario that SHOULD pass all filters."""
    # Setup data at Checkpoint
    # 1. Price > SMA200 (Uptrend)
    # 2. Volume > 1M
    # 3. Dip: ATR R3 < -1.0 (Drop > 1 ATR)
    # 4. Volatility > 3%
    # 5. IBS < 0.2 (Close near Low)
    # 6. Red Candle (Close < Open)
    
    idx = pd.Timestamp("2025-01-01")
    
    current = {
        "close": 100.0,
        "open": 102.0,          # Red Candle
        "high": 105.0,
        "low": 99.0,            # Low IBS (100-99)/(105-99) = 1/6 = ~0.16 < 0.2
        "volume": 2_000_000,    # > 1M
        "sma200": 90.0,         # Uptrend (100 > 90)
        "volume_sma": 1_500_000,
        "atr": 5.0,             # Volatility
        "atr_r3": -1.5,         # Large Dip (-1.5 < -1.0)
        "ibs": 0.16,
        "vola_ratio": 0.05      # 5/100 = 5% > 3%
    }
    
    prev = {
        "close": 108.0,
        "open": 110.0           # Red Candle yesterday
    }
    
    # Convert scalar dicts to Series with Index for the method compatibility
    # actually the method takes dict of scalars or series.
    # The Refactored method `_apply_entry_filter` expects DICTS of SCALARS if called per row,
    # OR Dicts of Series if called vectorized.
    # Looking at code: `mask = (current["volume_sma"] > ...)` implies Series operations if `current` has Series.
    # Let's mock it as Series of length 1 to be safe.
    
    def to_series(d):
        return {k: pd.Series([v], index=[idx]) for k, v in d.items()}
    
    curr_s = to_series(current)
    prev_s = to_series(prev)
    
    results = strategy._apply_entry_filter(curr_s, prev_s)
    
    assert not results.empty
    assert len(results) == 1
    assert results.index[0] == idx

def test_entry_filter_fails_downtrend(strategy):
    """Test failure: Price < SMA200."""
    idx = pd.Timestamp("2025-01-01")
    current = {
        "close": 80.0,          # < SMA200
        "open": 82.0,
        "high": 85.0,
        "low": 79.0,
        "volume": 2_000_000,
        "sma200": 90.0,         # Downtrend
        "volume_sma": 1_500_000,
        "atr": 5.0,
        "atr_r3": -1.5,
        "ibs": 0.16,
        "vola_ratio": 0.05
    }
    prev = {"close": 85.0, "open": 87.0}
    
    def to_series(d): return {k: pd.Series([v], index=[idx]) for k, v in d.items()}
    
    results = strategy._apply_entry_filter(to_series(current), to_series(prev))
    
    assert results.empty

def test_process_signals_creates_trade(strategy, mock_repo):
    """Integration test: Signals DF -> Repo Create Call."""
    date = pd.Timestamp("2025-01-10")
    
    signals = pd.DataFrame([{
        "close": 100.0,
        "high": 105.0,
        "volume": 1_000_000,
        "atr": 5.0,
        "sma200": 90.0,
        "atr_r3": -1.5,
        "ibs": 0.1,
        "setup_score": 1.5
    }], index=["TEST"])
    
    strategy._process_signals(signals, date)
    
    mock_repo.create_trade.assert_called_once()
    call_kwargs = mock_repo.create_trade.call_args[1]
    
    assert call_kwargs['symbol'] == "TEST"
    assert call_kwargs['strategy'] == "DipBuyer"
    assert call_kwargs['entry'] == 95.0 # 100 - (5.0 * 1.0)
    assert call_kwargs['target'] == 105.0
