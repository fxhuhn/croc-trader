# filename: test_screener_turnover.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

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
    return TurnoverTimingStrategy(trade_repository=mock_repo, data_provider=mock_provider)

# --- HELPER ---

def create_market_data(
    symbol_list: list[str],
    end_date: str, 
    days: int = 300,
    turnover_val: float = 1_000_000,
    trend: str = "UP"
) -> dict[str, pd.DataFrame]:
    """
    Creates comprehensive market data for testing ranking.
    trend="UP": Close > SMA150
    trend="DOWN": Close < SMA150
    """
    dates = pd.date_range(end=end_date, periods=days, freq='B')
    
    data = {
        "close": pd.DataFrame(index=dates, columns=symbol_list),
        "high": pd.DataFrame(index=dates, columns=symbol_list),
        "low": pd.DataFrame(index=dates, columns=symbol_list),
        "open": pd.DataFrame(index=dates, columns=symbol_list),
        "volume": pd.DataFrame(index=dates, columns=symbol_list)
    }
    
    for symbol in symbol_list:
        if trend == "UP":
            # Rising from 100 to 200. SMA150 (~150) < Close (200)
            prices = np.linspace(100, 200, days)
        else:
            # Falling from 200 to 100. SMA150 (~150) > Close (100)
            prices = np.linspace(200, 100, days)
            
        data["close"][symbol] = prices
        data["high"][symbol] = prices + 5
        data["low"][symbol] = prices - 5
        data["open"][symbol] = prices - 2
        
        # Volume / Turnover
        # Turnover = Price * Vol. 
        # To get fixed turnover, Vol = Turnover / Price
        # But for simplicity, we set constant volume since price is dynamic
        # Logic ranks by SMA20(Turnover). 
        # So we just set volume constant.
        vol = int(turnover_val / prices[-1]) # Approximate
        data["volume"][symbol] = vol

    return data

# --- TESTS ---

@patch('app.services.screener.strategies.turnover_timing.ExchangeSymbol')
@patch('app.services.screener.strategies.turnover_timing.MarketHolidayChecker.is_holiday')
def test_ranking_selects_top_4_highest_turnover_in_uptrend(
    mock_is_holiday, MockExchangeSymbol, strategy, mock_provider, mock_repo
):
    """
    Verifies valid ranking:
    1. Filters Close > SMA150
    2. Ranks by Turnover
    3. Selects Top 4
    """
    mock_is_holiday.return_value = False
    
    # 1. Setup Symbols
    # 4 Mega Caps (High Turnover) - Should Pick
    # 4 Mid Caps (Med Turnover) - Should Ignore
    # 1 Downtrend (High Turnover) - Should Filter Out
    
    mega_caps = ["MEGA_1", "MEGA_2", "MEGA_3", "MEGA_4"]
    mid_caps = ["MID_1", "MID_2", "MID_3", "MID_4"]
    bad_trend = ["DOWN_1"]
    
    all_symbols = mega_caps + mid_caps + bad_trend
    
    # 2. Assign to Index
    mock_ex = MockExchangeSymbol.return_value
    mock_ex.nasdaq_100 = all_symbols
    mock_ex.sp_500 = []
    mock_ex.russell_1000 = []
    
    # 3. Generate Data
    end_date = "2026-01-30" # Friday
    data_mega = create_market_data(mega_caps, end_date, turnover_val=500_000_000) # 500M
    data_mid = create_market_data(mid_caps, end_date, turnover_val=100_000_000)   # 100M
    data_down = create_market_data(bad_trend, end_date, turnover_val=600_000_000, trend="DOWN") # 600M (Highest but Down)
    
    # Merge DataFrames
    merged_data = {}
    for col in ["close", "high", "low", "open", "volume"]:
        merged_data[col] = pd.concat([
            data_mega[col], data_mid[col], data_down[col]
        ], axis=1)
        
    mock_provider.get_universe_daily_data.return_value = merged_data
    mock_repo.exists.return_value = False
    
    # 4. Run
    count = strategy.run(analysis_date=end_date)
    
    # 5. Assert logic
    # DOWN_1 filtered out (Trend).
    # Remaining: Mega (500M) > Mid (100M).
    # Top 4: MEGA_1..4
    
    calls = mock_repo.create_trade.call_args_list
    traded = {c[1]['symbol'] for c in calls}
    
    assert count == 8 # 4 Stocks * 2 Factors
    assert traded == set(mega_caps)
    assert "DOWN_1" not in traded

@patch('app.services.screener.strategies.turnover_timing.ExchangeSymbol')
@patch('app.services.screener.strategies.turnover_timing.MarketHolidayChecker.is_holiday')
def test_execution_skips_if_data_stale(
    mock_is_holiday, MockExchangeSymbol, strategy, mock_provider, mock_repo
):
    """
    CRITICAL: Analyzed date is Friday. Data only available till Thursday.
    Must return 0.
    """
    mock_is_holiday.return_value = False
    
    target_date = "2026-01-30" # Friday
    stale_date = "2026-01-29" # Thursday
    
    data = create_market_data(["TEST"], stale_date) # Ends Thu
    mock_provider.get_universe_daily_data.return_value = data
    
    mock_ex = MockExchangeSymbol.return_value
    mock_ex.nasdaq_100 = ["TEST"]
    mock_ex.sp_500 = []
    mock_ex.russell_1000 = []
    
    count = strategy.run(analysis_date=target_date)
    
    assert count == 0
    mock_repo.create_trade.assert_not_called()

@patch('app.services.screener.strategies.turnover_timing.ExchangeSymbol')
@patch('app.services.screener.strategies.turnover_timing.MarketHolidayChecker.is_holiday')
def test_execution_on_valid_thursday_holiday(
    mock_is_holiday, MockExchangeSymbol, strategy, mock_provider, mock_repo
):
    """
    Run on Thursday if Friday is Holiday.
    """
    # Setup
    thursday = "2026-01-29"
    friday = "2026-01-30"
    
    # Mocks
    def check_holiday(d):
        return str(d) == friday # Friday is holiday
    mock_is_holiday.side_effect = check_holiday
    
    data = create_market_data(["TEST"], thursday)
    mock_provider.get_universe_daily_data.return_value = data
    
    mock_ex = MockExchangeSymbol.return_value
    mock_ex.nasdaq_100 = ["TEST"]
    mock_ex.sp_500 = []
    mock_ex.russell_1000 = []
    
    mock_repo.exists.return_value = False
    
    count = strategy.run(analysis_date=thursday)
    
    assert count == 2 # 1 Sym * 2 Factors
    
@patch('app.services.screener.strategies.turnover_timing.ExchangeSymbol')
@patch('app.services.screener.strategies.turnover_timing.MarketHolidayChecker.is_holiday')
def test_avoids_duplicates(
    mock_is_holiday, MockExchangeSymbol, strategy, mock_provider, mock_repo
):
    """
    Ensure we don't create trade if repo.exists() is True.
    """
    mock_is_holiday.return_value = False
    end_date = "2026-01-30"
    
    data = create_market_data(["TEST"], end_date)
    mock_provider.get_universe_daily_data.return_value = data
    
    mock_ex = MockExchangeSymbol.return_value
    mock_ex.nasdaq_100 = ["TEST"]
    mock_ex.sp_500 = []
    mock_ex.russell_1000 = []
    
    # Simulate trade ALREADY exists
    mock_repo.exists.return_value = True
    
    count = strategy.run(analysis_date=end_date)
    
    assert count == 0
    mock_repo.create_trade.assert_not_called()

@patch('app.services.screener.strategies.turnover_timing.ExchangeSymbol')
@patch('app.services.screener.strategies.turnover_timing.MarketHolidayChecker.is_holiday')
def test_initial_size_set_to_zero(
    mock_is_holiday, MockExchangeSymbol, strategy, mock_provider, mock_repo
):
    """
    Verify that created trades have size=0 (deferring to PortfolioManager).
    """
    mock_is_holiday.return_value = False
    end_date = "2026-01-30"
    
    data = create_market_data(["TEST"], end_date)
    mock_provider.get_universe_daily_data.return_value = data
    
    mock_ex = MockExchangeSymbol.return_value
    mock_ex.nasdaq_100 = ["TEST"]
    mock_ex.sp_500 = []
    mock_ex.russell_1000 = []
    
    mock_repo.exists.return_value = False
    
    strategy.run(analysis_date=end_date)
    
    calls = mock_repo.create_trade.call_args_list
    assert len(calls) > 0
    creation_args = calls[0][1] # kwargs
    
    # Assert Size is 0
    assert creation_args["size"] == 0
    # Assert Entry is calculated (Limit)
    assert creation_args["entry"] > 0
    assert creation_args["entry"] < 205 # Should be below Close (~200)

