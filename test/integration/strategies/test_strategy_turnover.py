# filename: test_strategy_turnover.py
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime, date

from app.services.trade_manager.strategies.turnover_timing import TurnoverTimingStrategy
from app.database.repositories.trade import TradeRepository
from app.types import TradeStatus, ExitReason

# --- CONSTANTS ---
dummy_id = "trade-001"
dummy_symbol = "TEST_SYM"

# --- FIXTURES ---

@pytest.fixture
def strategy() -> TurnoverTimingStrategy:
    """Provides a clean instance of the TurnoverTimingStrategy."""
    return TurnoverTimingStrategy()

@pytest.fixture
def mock_repository() -> MagicMock:
    """Provides a strict mock for the TradeRepository."""
    repo = MagicMock(spec=TradeRepository)
    return repo

@pytest.fixture
def active_trade_data() -> dict:
    """Returns a valid ACTIVE trade dictionary."""
    return {
        "id": dummy_id,
        "symbol": dummy_symbol,
        "status": TradeStatus.ACTIVE,
        "entry_date": "2026-01-02",
        "entry_price": 100.0,
        "initial_size": 10,
        "current_size": 10,
        "signal_context": '{"setup_date": "2025-12-31"}'
    }

@pytest.fixture
def created_trade_data() -> dict:
    """Returns a valid CREATED trade dictionary."""
    return {
        "id": dummy_id,
        "symbol": dummy_symbol,
        "status": TradeStatus.CREATED,
        "entry_price": 100.0,
        "initial_size": 0, # Should be calculated or provided
        "budget": 2000.0,
        "signal_context": '{"setup_date": "2026-01-01"}' # Thursday
    }

# --- HELPERS ---

def create_candle(
    date_str: str, 
    open_price: float, 
    high: float, 
    low: float, 
    close: float
) -> pd.Series:
    """Creates a single price candle series."""
    return pd.Series({
        "date": pd.Timestamp(date_str),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000000
    })

def create_history(data: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    """
    Creates a historical DataFrame from a list of tuples.
    Format: (date, open, high, low, close)
    """
    rows = []
    for d, o, h, l, c in data:
        rows.append(create_candle(d, o, h, l, c))
    
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df

# --- TESTS ---

# 1. ENTRY LOGIC TESTS

def test_check_entry_fills_gap_down_on_next_trading_day(
    strategy: TurnoverTimingStrategy,
    mock_repository: MagicMock,
    created_trade_data: dict
) -> None:
    """
    Verifies that a trade fills at OPEN if the Open is below the Limit Price (Gap Down).
    Strictly on the Next Trading Day.
    """
    # Arrange
    # Signal: Thursday 2026-01-01. Next Trading Day: Friday 2026-01-02.
    # Limit: 100.0. Open: 98.0 (Gap Down).
    created_trade_data["entry_price"] = 100.0
    
    # History must include Signal Date (Jan 1) and Current Date (Jan 2)
    signal_candle = create_candle("2026-01-01", 100.0, 105.0, 95.0, 100.0)
    current_candle = create_candle("2026-01-02", 98.0, 102.0, 95.0, 100.0)
    
    history = pd.DataFrame([signal_candle, current_candle])

    with patch("app.tools.market_holidays.MarketHolidayChecker.is_holiday", return_value=False):
        # Act
        result = strategy.check_entry(created_trade_data, current_candle, history, mock_repository)

    # Assert
    assert result is not None
    assert "FILLED @ 98.00" in result # Filled at Open because Open < Limit
    mock_repository.update_trade.assert_called_once()
    
    # Verify DB update call
    call_args = mock_repository.update_trade.call_args
    assert call_args[0][0] == dummy_id
    payload = call_args[0][1]
    assert payload["status"] == TradeStatus.ACTIVE
    assert payload["entry_price"] == 98.0
    assert "signal_context" in payload

def test_check_entry_fills_intraday_limit(
    strategy: TurnoverTimingStrategy,
    mock_repository: MagicMock,
    created_trade_data: dict
) -> None:
    """Verifies that a trade fills at LIMIT if price drops within the day."""
    # Arrange
    # Signal: Thu 2026-01-01.
    # Limit: 100.0. Open: 102.0. Low: 99.0.
    created_trade_data["entry_price"] = 100.0
    
    # History must include Signal Date (Jan 1) and Current Date (Jan 2)
    signal_candle = create_candle("2026-01-01", 100.0, 105.0, 95.0, 100.0)
    current_candle = create_candle("2026-01-02", 102.0, 105.0, 99.0, 101.0)
    history = pd.DataFrame([signal_candle, current_candle])

    with patch("app.tools.market_holidays.MarketHolidayChecker.is_holiday", return_value=False):
        # Act
        result = strategy.check_entry(created_trade_data, current_candle, history, mock_repository)

    # Assert
    assert result is not None
    assert "FILLED @ 100.00" in result # Filled at Limit
    mock_repository.update_trade.assert_called_once()
    # Payload now includes signal_context
    args = mock_repository.update_trade.call_args[0][1]
    assert args["entry_price"] == 100.0
    assert "signal_context" in args

def test_check_entry_expires_if_missed_next_day(
    strategy: TurnoverTimingStrategy,
    mock_repository: MagicMock,
    created_trade_data: dict
) -> None:
    """Verifies that the trade EXPIRES if the current date is AFTER the valid entry day."""
    # Arrange
    # Signal: Thu 2026-01-01. Valid Entry: Fri 2026-01-02.
    # Current Date: Mon 2026-01-05 (Assume pure trading days for simplicity of test)
    
    # History includes Signal (01), Gap (02), Current (05)
    # Trading Days count = 3 (01, 02, 05) -> > 2 -> Expired
    signal_candle = create_candle("2026-01-01", 100.0, 105.0, 95.0, 100.0)
    next_day = create_candle("2026-01-02", 100.0, 105.0, 95.0, 100.0)
    current_candle = create_candle("2026-01-05", 100.0, 105.0, 95.0, 100.0)
    
    history = pd.DataFrame([signal_candle, next_day, current_candle])

    with patch("app.tools.market_holidays.MarketHolidayChecker.is_holiday", return_value=False):
        # Act
        result = strategy.check_entry(created_trade_data, current_candle, history, mock_repository)

    # Assert
    assert result is not None
    assert "EXPIRED" in result
    mock_repository.update_trade.assert_called_once()
    assert mock_repository.update_trade.call_args[0][1]["status"] == TradeStatus.CLOSED
    assert mock_repository.update_trade.call_args[0][1]["exit_reason"] == ExitReason.EXPIRED

def test_check_entry_ignores_same_day_check(
    strategy: TurnoverTimingStrategy,
    mock_repository: MagicMock,
    created_trade_data: dict
) -> None:
    """Verifies that we DO NOT enter on the Signal Day itself."""
    # Arrange
    # Signal: 2026-01-01. Current: 2026-01-01.
    candle = create_candle("2026-01-01", 90.0, 100.0, 80.0, 95.0) # Low is way below limit
    history = pd.DataFrame([candle])

    with patch("app.tools.market_holidays.MarketHolidayChecker.is_holiday", return_value=False):
        # Act
        result = strategy.check_entry(created_trade_data, candle, history, mock_repository)

    # Assert
    assert result is None # Should skip waiting for next day
    mock_repository.update_trade.assert_not_called()

def test_check_entry_respects_holidays_checks_next_business_day(
    strategy: TurnoverTimingStrategy,
    mock_repository: MagicMock,
    created_trade_data: dict
) -> None:
    """
    Verifies that if the day after signal is a holiday, the entry window moves to the next business day.
    """
    # Arrange
    # Signal: Fri 2026-01-16.
    created_trade_data["signal_context"] = '{"setup_date": "2026-01-16"}'
    
    # Mon 2026-01-19 is MLK Holiday.
    # Tue 2026-01-20 is the Valid Entry Day.
    
    signal_candle = create_candle("2026-01-16", 100.0, 105.0, 95.0, 100.0)
    # Note: 19th is holiday, so usually no candle, but even if it existed, we check dates
    current_candle = create_candle("2026-01-20", 98.0, 102.0, 95.0, 100.0)
    
    # History: 16th, 20th. (Gap of 19th)
    # Trading Days Count:
    # 16th (1)
    # 20th (2) -> Count = 2. valid.
    history = pd.DataFrame([signal_candle, current_candle])

    mock_checker = MagicMock()
    # is_holiday returns True for 2026-01-19 (Mon), False for others
    def mock_is_holiday(d):
        return str(d) == "2026-01-19" # Holiday
    
    with patch("app.tools.market_holidays.MarketHolidayChecker.is_holiday", side_effect=mock_is_holiday):
        # Act
        result = strategy.check_entry(created_trade_data, current_candle, history, mock_repository)

    # Assert
    assert result is not None
    assert "FILLED" in result

# 2. EXIT LOGIC TESTS (CRITICAL)

def test_manage_active_trade_exits_on_green_sequence(
    strategy: TurnoverTimingStrategy,
    mock_repository: MagicMock,
    active_trade_data: dict
) -> None:
    """
    CRITICAL: 'Mu has to be closed on the day after two green candles'
    Scenario:
    Day 1 (Mon): Green
    Day 2 (Tue): Green
    Day 3 (Wed): Current Candle -> Must EXIT at OPEN.
    """
    # Arrange
    history_data = [
        ("2026-01-05", 100.0, 105.0, 99.0, 102.0), # Mon: Green (102 > 100)
        ("2026-01-06", 102.0, 106.0, 101.0, 105.0), # Tue: Green (105 > 102)
        ("2026-01-07", 105.0, 110.0, 104.0, 108.0), # Wed: Current Day
    ]
    df_history = create_history(history_data)
    
    # Day 1 and 2 were Green. Wed morning should exit.
    # We must simulate that the trade context ALREADY has count=2 from previous days.
    active_trade_data["signal_context"] = '{"green_candle_count": 2}'
    
    # Act
    with patch("app.tools.market_holidays.MarketHolidayChecker.is_holiday", return_value=False):
        result = strategy.manage_active_trade(active_trade_data, df_history, mock_repository)

    # Assert
    assert result is not None
    assert "GREEN_SEQUENCE" in result
    assert "105.00" in result # Must exit at Wed OPEN (105.0)

    # Verify DB
    mock_repository.update_trade.assert_called_once()
    args = mock_repository.update_trade.call_args[0][1]
    assert args["status"] == TradeStatus.CLOSED
    assert args["exit_date"] == "2026-01-07 00:00:00"
    assert args["exit_price"] == 105.0

def test_manage_active_trade_holds_if_second_green_is_today(
    strategy: TurnoverTimingStrategy,
    mock_repository: MagicMock,
    active_trade_data: dict
) -> None:
    """
    Ensures we DO NOT exit on the day the second green candle is forming.
    We must wait for the NEXT open.
    """
    # Arrange
    history_data = [
        ("2026-01-05", 100.0, 105.0, 99.0, 102.0), # Mon: Green
        ("2026-01-06", 102.0, 106.0, 101.0, 105.0), # Tue: Green (Today)
    ]
    df_history = create_history(history_data)
    
    with patch("app.tools.market_holidays.MarketHolidayChecker.is_holiday", return_value=False):
        result = strategy.manage_active_trade(active_trade_data, df_history, mock_repository)

    assert result is None # Hold

def test_manage_active_trade_exits_friday_close(
    strategy: TurnoverTimingStrategy,
    mock_repository: MagicMock,
    active_trade_data: dict
) -> None:
    """Verifies Time Stop at End of Week (Friday Close)."""
    # Arrange
    # 2026-01-09 is Friday
    history_data = [
        ("2026-01-08", 100.0, 102.0, 99.0, 101.0),
        ("2026-01-09", 101.0, 103.0, 100.0, 102.0), # Fri: Close at 102
    ]
    df_history = create_history(history_data)
    
    with patch("app.tools.market_holidays.MarketHolidayChecker.is_holiday", return_value=False):
        result = strategy.manage_active_trade(active_trade_data, df_history, mock_repository)

    assert result is not None
    assert "TIME_STOP" in result
    assert "102.00" in result

def test_manage_active_trade_exits_thursday_if_friday_holiday(
    strategy: TurnoverTimingStrategy,
    mock_repository: MagicMock,
    active_trade_data: dict
) -> None:
    """Verifies Time Stop on Thursday if Friday is a Holiday."""
    # Arrange
    # 2026-01-15 is Thursday. 16th is Fri (Mock as Holiday).
    history_data = [
        ("2026-01-14", 98.0, 100.0, 97.0, 99.0), # Wed (Padding)
        ("2026-01-15", 100.0, 102.0, 99.0, 101.0), # Thu
    ]
    df_history = create_history(history_data)

    def mock_is_holiday(d):
        return str(d) == "2026-01-16"
        
    with patch("app.tools.market_holidays.MarketHolidayChecker.is_holiday", side_effect=mock_is_holiday):
        result = strategy.manage_active_trade(active_trade_data, df_history, mock_repository)

    assert result is not None
    assert "TIME_STOP" in result
    assert "101.00" in result # Thu Close

def test_green_sequence_resets_on_red_candle(
    strategy: TurnoverTimingStrategy,
    mock_repository: MagicMock,
    active_trade_data: dict
) -> None:
    """
    Scenario: Green, Red, Green. Should NOT Exit on next day.
    """
    # Arrange
    history_data = [
        ("2026-01-05", 100.0, 105.0, 99.0, 102.0), # Mon: Green
        ("2026-01-06", 102.0, 101.0, 99.0, 100.0), # Tue: RED (Close < Open)
        ("2026-01-07", 100.0, 104.0, 99.0, 103.0), # Wed: Green
        ("2026-01-08", 103.0, 105.0, 100.0, 104.0), # Thu: Current
    ]
    df_history = create_history(history_data)
    
    with patch("app.tools.market_holidays.MarketHolidayChecker.is_holiday", return_value=False):
        result = strategy.manage_active_trade(active_trade_data, df_history, mock_repository)

    assert result is None # 2 consecutive greens required. We have Red, Green.

# 3. EDGE CASES & SAFETY

def test_check_entry_handles_zero_price_data(
    strategy: TurnoverTimingStrategy,
    mock_repository: MagicMock,
    created_trade_data: dict
) -> None:
    """Data Corruption: Zero/Negative prices should not trigger fill."""
    candle = create_candle("2026-01-02", 0.0, 100.0, 0.0, 0.0)
    history = pd.DataFrame([candle])

    with patch("app.tools.market_holidays.MarketHolidayChecker.is_holiday", return_value=False):
        result = strategy.check_entry(created_trade_data, candle, history, mock_repository)

    assert result is None # Should be ignored or safe

def test_manage_active_trade_handles_empty_history(
    strategy: TurnoverTimingStrategy,
    mock_repository: MagicMock,
    active_trade_data: dict
) -> None:
    """Empty history should assume no action."""
    df_history = pd.DataFrame() # Empty
    
    result = strategy.manage_active_trade(active_trade_data, df_history, mock_repository)
    assert result is None
