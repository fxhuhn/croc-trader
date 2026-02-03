import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime

from app.services.screener.strategies.turnover_timing import TurnoverTimingStrategy
from app.database.repositories.trade import TradeRepository
from app.database.repositories.market_data_provider import MarketDataProvider

# --- FIXTURES ---

@pytest.fixture
def mock_repo():
    return MagicMock(spec=TradeRepository)

@pytest.fixture
def mock_provider():
    return MagicMock(spec=MarketDataProvider)

@pytest.fixture
def strategy(mock_repo, mock_provider):
    return TurnoverTimingStrategy(trade_repo=mock_repo, data_provider=mock_provider)

# --- HELPER ---

def create_market_data_for_ranking(length=300):
    """
    Creates data for multiple symbols to test ranking.
    - HIGH_T_UP: High Turnover + Uptrend (Ideal Candidate) -> 4 of these
    - HIGH_T_DOWN: High Turnover + Downtrend (Filter out) -> 1 of these
    - MED_T_UP: Medium Turnover + Uptrend (Should be ranked lower) -> 20 of these
    - LOW_T: Low Turnover -> Should be dropped by Top 20 ranking
    """
    dates = pd.date_range(end=pd.Timestamp("2025-01-31"), periods=length, freq='B') # Friday
    
    symbols = []
    
    # Creates dictionary for DataFrames
    closes = {}
    volumes = {}
    highs = {}
    lows = {}
    
    # SMA150 needs 150 points. We generate 300.
    # Uptrend: Close (150) > SMA (100)
    # Downtrend: Close (90) < SMA (100)
    
    # 1. TOP 4 CANDIDATES (High Turnover, Uptrend)
    for i in range(1, 5):
        sym = f"TOP_{i}"
        symbols.append(sym)
        # Price rising from 100 to 200. SMA150 will be < Close.
        closes[sym] = np.linspace(100, 200, length) 
        volumes[sym] = 1_000_000 # High Turnover (200 * 1M = 200M)
        
    # 2. HIGH TURNOVER BUT DOWNTREND (Should be filtered)
    sym = "DOWNTREND"
    symbols.append(sym)
    closes[sym] = np.linspace(200, 100, length) # Falling. SMA > Close.
    volumes[sym] = 1_000_000 
    
    # 3. MEDIUM TURNOVER (Valid Trend, but lower rank)
    # We create 20 of these. Since we take Top 20 by Turnover, 
    # and we have 5 High Turnover stocks above (4 valid, 1 invalid-trend),
    # The Top 20 list will contain: [TOP_1..4, DOWNTREND, MED_1..15].
    # Then Filter Trend: Removes DOWNTREND.
    # List: [TOP_1..4, MED_1..15].
    # Select Top 4: [TOP_1..4].
    # So MED ones should NOT cause trades.
    for i in range(1, 21):
        sym = f"MED_{i}"
        symbols.append(sym)
        closes[sym] = np.linspace(100, 200, length)
        volumes[sym] = 500_000 # Half Turnover
        
    # 4. LOW TURNOVER (Should be dropped immediately)
    sym = "LOW_T"
    symbols.append(sym)
    closes[sym] = np.linspace(100, 200, length)
    volumes[sym] = 100 
    
    # Create DFs
    df_close = pd.DataFrame(closes, index=dates)
    df_volume = pd.DataFrame(volumes, index=dates)
    
    # High/Low for ATR
    df_high = df_close + 5
    df_low = df_close - 5
    df_open = df_close - 2 # Arbitrary Open


    data = {
        "close": df_close,
        "high": df_high,
        "low": df_low,
        "open": df_open,
        "volume": df_volume
    }
    return data, symbols

# --- TESTS ---

@patch('app.services.screener.strategies.turnover_timing.ExchangeSymbol')
def test_ranking_and_selection(MockExchangeSymbol, strategy, mock_provider, mock_repo):
    """
    Test Logic:
    1. Input: 26 Symbols (4 Top, 1 Down, 20 Med, 1 Low).
    2. Bucket: All in 'NASDAQ_100'.
    3. Rank: Sort by SMA20(Turnover).
       - Top 4 have 200M.
       - Downtrend has 100M (using avg price ~100? No, avg price 150 -> 150M).
       - Med have 75M.
       - Low has tiny.
    4. Top 20 Slicing:
       - Should keep TOP_1-4, DOWNTREND, and 15 MEDs.
       - Should drop LOW_T and 5 MEDs.
    5. Trend Filter:
       - Remove DOWNTREND.
    6. Select Top 4:
       - Should be TOP_1, TOP_2, TOP_3, TOP_4.
    """
    data, all_symbols = create_market_data_for_ranking()
    
    # Mock Provider
    mock_provider.get_all_daily_data.return_value = data
    
    # Mock Repository (None exist yet)
    mock_repo.exists.return_value = False
    
    # Mock ExchangeSymbol (Buckets)
    # We put ALL generated symbols into NASDAQ_100 to text the ranking logic within one bucket
    mock_instance = MockExchangeSymbol.return_value 
    mock_instance.nasdaq_100 = all_symbols
    mock_instance.sp_500 = []
    mock_instance.russell_1000 = []
    
    # Run
    count = strategy.run(analysis_date="2025-01-31")
    
    # Expected: 4 Symbols * 2 Factors = 8 Trades
    assert count == 8, f"Expected 8 trades, got {count}"
    
    # Verify exactly WHO was traded
    calls = mock_repo.create_trade.call_args_list
    traded_symbols = set([c[1]['symbol'] for c in calls])
    
    expected_symbols = {"TOP_1", "TOP_2", "TOP_3", "TOP_4"}
    assert traded_symbols == expected_symbols
    
    # Verify Context
    first_call_ctx = calls[0][1]['context']
    assert first_call_ctx['bucket'] == "NASDAQ_100"
    assert "setup_sma150" in first_call_ctx
    assert "setup_turnover_sma20" in first_call_ctx
    
@patch('app.services.screener.strategies.turnover_timing.ExchangeSymbol')
def test_deduplication(MockExchangeSymbol, strategy, mock_provider, mock_repo):
    """Ensure a symbol in multiple buckets triggers only once."""
    data, all_symbols = create_market_data_for_ranking()
    
    # Use just TOP_1
    top1 = "TOP_1"
    
    # Mock Provider
    mock_provider.get_all_daily_data.return_value = data
    mock_repo.exists.return_value = False # First time check
    
    # Put TOP_1 in BOTH NASDAQ and SP500
    mock_instance = MockExchangeSymbol.return_value 
    mock_instance.nasdaq_100 = [top1]
    mock_instance.sp_500 = [top1]
    mock_instance.russell_1000 = []
    
    # Run
    strategy.run(analysis_date="2025-01-31")
    
    # Expected: 2 Trades (Factors) for TOP_1. NOT 4.
    # The deduplication happens BEFORE create_trade loop.
    assert mock_repo.create_trade.call_count == 2
    
    args = mock_repo.create_trade.call_args_list[0][1]
    assert args['symbol'] == top1

def test_run_skips_bad_day(strategy, mock_provider):
    data, _ = create_market_data_for_ranking()
    mock_provider.get_all_daily_data.return_value = data
    
    # Monday
    count = strategy.run(analysis_date="2025-01-27") 
    assert count == 0
