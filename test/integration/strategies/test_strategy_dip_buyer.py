# filename: test_strategy_dip_buyer.py
import pytest
import pandas as pd
from unittest.mock import MagicMock
from app.services.trade_manager.strategies.dip_buyer import DipBuyerStrategy
from app.database.repositories.trade import TradeRepository
from app.types import TradeStatus, ExitReason

# --- FIXTURES ---


@pytest.fixture
def strategy() -> DipBuyerStrategy:
    """Provides a fresh instance of the DipBuyerStrategy."""
    return DipBuyerStrategy()


@pytest.fixture
def mock_repository() -> MagicMock:
    """Provides a mock TradeRepository."""
    return MagicMock(spec=TradeRepository)


@pytest.fixture
def base_trade_data() -> dict:
    """Returns a CREATED trade dictionary."""
    return {
        "id": "trade-123",
        "symbol": "TEST",
        "entry_price": 100.0,
        "current_target": 110.0,
        "current_stop_loss": 90.0,  # Should be ignored
        "status": "CREATED",
        "budget": 2000.0,
    }


# --- HELPERS ---


def create_candle(
    date_str: str, open_price: float, high: float, low: float, close: float
) -> pd.Series:
    """Creates a 1-row Series representing a daily candle."""
    return pd.Series(
        {
            "date": pd.Timestamp(date_str),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
        }
    )


def create_history(
    dates: list[str],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
) -> pd.DataFrame:
    """Creates a DataFrame history with datetime objects."""
    df = pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


# --- ENTRY TESTS ---


def test_entry_standard_fill(
    strategy: DipBuyerStrategy, mock_repository: MagicMock, base_trade_data: dict
) -> None:
    """Happy Path: Low < Limit < Open. Fills at Limit."""
    # Arrange
    candle = create_candle("2026-01-02", 102.0, 105.0, 95.0, 100.0)
    prev_candle = create_candle("2026-01-01", 100, 105, 95, 100)
    df_history = pd.concat(
        [pd.DataFrame([prev_candle]), pd.DataFrame([candle])], ignore_index=True
    )

    # Act
    # We must ensure signal_date is set in trade data for date validation
    base_trade_data["signal_context"] = '{"setup_date": "2026-01-01"}'
    result = strategy.check_entry(base_trade_data, candle, df_history, mock_repository)

    # Assert
    assert result is not None
    assert "FILLED @ 100.00" in result

    mock_repository.update_trade.assert_called_once()
    args, _ = mock_repository.update_trade.call_args
    data = args[1]
    assert data["entry_price"] == 100.0
    assert data["status"] == TradeStatus.ACTIVE


def test_entry_gap_down_fill(
    strategy: DipBuyerStrategy, mock_repository: MagicMock, base_trade_data: dict
) -> None:
    """Gap Down: Open < Limit. Fills at Open (Better Price)."""
    # Arrange
    candle = create_candle("2026-01-02", 95.0, 98.0, 90.0, 92.0)
    df_history = pd.DataFrame([candle])

    # Act
    base_trade_data["signal_context"] = '{"date": "2026-01-01"}'
    result = strategy.check_entry(base_trade_data, candle, df_history, mock_repository)

    # Assert
    assert result is not None
    assert "FILLED @ 95.00" in result

    mock_repository.update_trade.assert_called_once()
    args, _ = mock_repository.update_trade.call_args
    assert args[1]["entry_price"] == 95.0


def test_entry_no_fill_high_above_limit(
    strategy: DipBuyerStrategy, mock_repository: MagicMock, base_trade_data: dict
) -> None:
    """Limit not Reached: Low > Limit."""
    # Arrange
    candle = create_candle("2026-01-02", 105.0, 106.0, 101.0, 102.0)
    df_history = pd.DataFrame([candle])

    # Act
    base_trade_data["signal_context"] = '{"date": "2026-01-01"}'
    result = strategy.check_entry(base_trade_data, candle, df_history, mock_repository)

    # Assert
    # Logic now invalidates if not hit on the first day
    assert result is not None
    assert "INVALIDATED" in result
    mock_repository.update_trade.assert_called_once()


# --- EXIT LOGIC TESTS (LOC) ---


def test_exit_loc_triggered_on_active_day(
    strategy: DipBuyerStrategy, mock_repository: MagicMock, base_trade_data: dict
) -> None:
    """LOC Rule: Close > Previous High -> Exit Market On Close."""
    # Arrange
    base_trade_data.update(
        {
            "status": "ACTIVE",
            "entry_date": "2026-01-02",
            "entry_price": 100.0,
            "current_size": 20,
        }
    )

    dates = ["2026-01-02", "2026-01-03"]
    opens = [100, 100]
    highs = [105, 108]  # PrevHigh = 105
    lows = [95, 95]
    closes = [100, 106]  # Close = 106 (> 105)

    df_history = create_history(dates, opens, highs, lows, closes)

    # Act
    result = strategy.manage_active_trade(base_trade_data, df_history, mock_repository)

    # Assert
    assert result is not None
    assert "LOC_HIT" in result

    mock_repository.update_trade.assert_called_once()
    args, _ = mock_repository.update_trade.call_args
    assert args[1]["status"] == TradeStatus.CLOSED
    assert args[1]["exit_price"] == 106.0


# --- TAKE PROFIT & PRIORITY TESTS ---


def test_take_profit_priority_over_loc(
    strategy: DipBuyerStrategy, mock_repository: MagicMock
) -> None:
    """Verifies that Take Profit is checked BEFORE Limit On Close."""
    # Arrange
    trade = {
        "id": "TEST_TRADE",
        "symbol": "FANG",
        "entry_price": 140.00,
        "entry_date": "2026-01-07",
        "current_target": 149.50,
        "current_size": 10,
        "budget": 2000.0,
        "status": "ACTIVE",
    }

    dates = ["2026-01-06", "2026-01-07", "2026-01-08"]
    df_history = create_history(
        dates,
        opens=[145.0, 140.0, 145.0],
        highs=[
            148.0,
            142.0,
            150.0,
        ],  # Jan 6 High = 148.0 (LOC), Jan 8 High = 150.0 > TP
        lows=[144.0, 138.0, 144.0],
        closes=[146.0, 140.45, 149.0],  # Jan 8 Close = 149.0 > LOC
    )

    # Act
    result = strategy.manage_active_trade(trade, df_history, mock_repository)

    # Assert
    assert result is not None
    assert "TARGET_HIT" in result

    mock_repository.update_trade.assert_called_once()
    args, _ = mock_repository.update_trade.call_args
    update_data = args[1]
    assert update_data["exit_reason"] == ExitReason.TARGET_HIT
    assert update_data["exit_price"] == 149.50


def test_take_profit_disallowed_on_entry_day(
    strategy: DipBuyerStrategy, mock_repository: MagicMock, base_trade_data: dict
) -> None:
    """Expert Rule: TP can NOT be hit on entry day."""
    # Arrange
    base_trade_data.update(
        {
            "status": "ACTIVE",
            "entry_date": "2026-01-02",
            "entry_price": 100.0,
            "current_target": 110.0,
            "current_size": 20,
        }
    )

    dates = ["2026-01-01", "2026-01-02"]
    df_history = create_history(
        dates,
        opens=[100, 100],
        highs=[105, 115],  # High (115) > Target (110)
        lows=[95, 95],
        closes=[100, 100],
    )

    # Act
    result = strategy.manage_active_trade(base_trade_data, df_history, mock_repository)

    # Assert
    assert result is None
    mock_repository.update_trade.assert_not_called()


def test_loc_allowed_on_entry_day(
    strategy: DipBuyerStrategy, mock_repository: MagicMock, base_trade_data: dict
) -> None:
    """Expert Rule: Only Limit on Close (LOC) is possible for same day."""
    # Arrange
    base_trade_data.update(
        {
            "status": "ACTIVE",
            "entry_date": "2026-01-02",
            "entry_price": 100.0,
            "current_size": 20,
        }
    )

    dates = ["2026-01-01", "2026-01-02"]
    df_history = create_history(
        dates,
        opens=[100, 100],
        highs=[105, 105],  # Jan 1 High = 105
        lows=[95, 95],
        closes=[100, 106],  # Jan 2 Close = 106 (> 105)
    )

    # Act
    result = strategy.manage_active_trade(base_trade_data, df_history, mock_repository)

    # Assert
    assert result is not None
    assert "LOC_HIT" in result
    mock_repository.update_trade.assert_called_once()


def test_stop_loss_ignored(
    strategy: DipBuyerStrategy, mock_repository: MagicMock, base_trade_data: dict
) -> None:
    """Strategy should ignore stop loss even if low is below SL."""
    # Arrange
    base_trade_data.update(
        {
            "status": "ACTIVE",
            "entry_date": "2026-01-02",
            "entry_price": 100.0,
            "current_stop_loss": 90.0,
            "current_size": 20,
        }
    )

    dates = ["2026-01-02", "2026-01-03"]
    df_history = create_history(
        dates, opens=[100, 100], highs=[105, 105], lows=[95, 80], closes=[100, 85]
    )

    # Act
    result = strategy.manage_active_trade(base_trade_data, df_history, mock_repository)

    # Assert
    assert result is None
    mock_repository.update_trade.assert_not_called()


def test_time_stop_triggered_on_day_8(
    strategy: DipBuyerStrategy, mock_repository: MagicMock, base_trade_data: dict
) -> None:
    """Regression: Exit if held for exactly 8 trading days (inclusive)."""
    # Arrange
    base_trade_data.update(
        {
            "status": "ACTIVE",
            "entry_date": "2026-01-02",
            "entry_price": 100.0,
            "current_size": 20,
        }
    )

    # create_history helper expects dates as list
    # Candle 1: Jan 02 (Entry), ..., Candle 8: Jan 11 (assuming consecutive trading days)
    dates = [f"2026-01-{i:02d}" for i in range(2, 10)]  # 8 days: 2,3,4,5,6,7,8,9
    prices = [100.0] * 8
    df_history = create_history(dates, prices, prices, prices, prices)

    # Act
    result = strategy.manage_active_trade(base_trade_data, df_history, mock_repository)

    # Assert
    assert result is not None
    assert "TIME_STOP" in result

    args, _ = mock_repository.update_trade.call_args
    assert args[1]["exit_reason"] == ExitReason.TIME_STOP


def test_entry_rejected_if_missed_on_next_day(
    strategy: DipBuyerStrategy, mock_repository: MagicMock, base_trade_data: dict
) -> None:
    """Regression: If price > limit on the first day after signal, trade is rejected."""
    # Arrange
    # Signal on Jan 01
    base_trade_data["signal_context"] = '{"date": "2026-01-01"}'

    # Check on Jan 02
    candle = create_candle(
        "2026-01-02", 105.0, 110.0, 101.0, 102.0
    )  # Low 101 > Limit 100
    df_history = pd.DataFrame([candle])

    # Act
    result = strategy.check_entry(base_trade_data, candle, df_history, mock_repository)

    # Assert
    assert result is not None
    assert "INVALIDATED" in result  # Base trade strategy uses SETUP INVALIDATED

    mock_repository.update_trade.assert_called_once()
    args, _ = mock_repository.update_trade.call_args
    assert args[1]["status"] == TradeStatus.INVALID


def test_entry_rejected_on_day_plus_two_even_if_price_hit(
    strategy: DipBuyerStrategy, mock_repository: MagicMock, base_trade_data: dict
) -> None:
    """Regression: If price hits on Day +2 but didn't on Day +1, it must be INVALID."""
    # Arrange
    # Signal on Jan 01
    base_trade_data["signal_context"] = '{"date": "2026-01-01"}'

    # History: Jan 01 (Signal), Jan 02 (No Fill), Jan 03 (Price Hits)
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    opens = [100.0, 105.0, 105.0]
    highs = [105.0, 110.0, 110.0]
    lows = [
        95.0,
        101.0,
        95.0,
    ]  # Jan 02 Low (101) > Limit (100); Jan 03 Low (95) < Limit (100)
    closes = [100.0, 102.0, 100.0]
    df_history = create_history(dates, opens, highs, lows, closes)

    # Act
    # Process the Jan 03 candle
    candle = df_history.iloc[-1]
    result = strategy.check_entry(base_trade_data, candle, df_history, mock_repository)

    # Assert
    assert result is not None
    assert "REJECTED" in result  # Should be too late
    assert "Missed Entry Window" in result

    args, _ = mock_repository.update_trade.call_args
    assert args[1]["status"] == TradeStatus.INVALID
