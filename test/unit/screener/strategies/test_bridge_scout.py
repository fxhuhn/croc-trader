"""Unit tests for the Bridge Scout screener and trade execution strategy."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.const import ExitReason, Strategies
from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.repositories.trade import TradeRepository
from app.services.screener.strategies.bridge_scout import (
    BridgeScoutStrategy,
    is_in_end_of_month_window,
)
from app.services.trade_manager.strategies.bridge_scout import BridgeScoutTradeStrategy
from app.tools.market_holidays import MarketHolidayChecker


@pytest.fixture
def mock_trade_repo() -> MagicMock:
    """Fixture providing a mock TradeRepository."""
    repo = MagicMock(spec=TradeRepository)
    repo.exists.return_value = False
    repo.create_trade.return_value = 1
    return repo


@pytest.fixture
def mock_data_provider() -> MagicMock:
    """Fixture providing a mock MarketDataProvider."""
    return MagicMock(spec=MarketDataProvider)


@pytest.fixture
def mock_holiday_checker() -> MagicMock:
    """Fixture providing a mock MarketHolidayChecker."""
    checker = MagicMock(spec=MarketHolidayChecker)
    checker.is_holiday.return_value = False
    return checker


@pytest.fixture
def bridge_scout_screener(
    mock_trade_repo: MagicMock,
    mock_data_provider: MagicMock,
    mock_holiday_checker: MagicMock,
) -> BridgeScoutStrategy:
    """Fixture providing a BridgeScoutStrategy instance."""
    return BridgeScoutStrategy(
        trade_repository=mock_trade_repo,
        data_provider=mock_data_provider,
        holiday_checker=mock_holiday_checker,
    )


# --- WINDOW & HOLIDAY TESTS ---


def test_month_end_window_calculation(mock_holiday_checker: MagicMock) -> None:
    """Tests month-end window calculation for normal month without holidays."""
    # July 2026: 31st is Friday.
    # Trading days: July 27(Mon), 28(Tue), 29(Wed), 30(Thu), 31(Fri) -> 5 days
    assert (
        is_in_end_of_month_window(
            pd.Timestamp("2026-07-27").date(),
            days_before=4,
            holiday_checker=mock_holiday_checker,
        )
        is True
    )
    assert (
        is_in_end_of_month_window(
            pd.Timestamp("2026-07-31").date(),
            days_before=4,
            holiday_checker=mock_holiday_checker,
        )
        is True
    )
    assert (
        is_in_end_of_month_window(
            pd.Timestamp("2026-07-24").date(),
            days_before=4,
            holiday_checker=mock_holiday_checker,
        )
        is False
    )


def test_month_end_window_with_holiday(mock_holiday_checker: MagicMock) -> None:
    """Tests month-end window when a market holiday falls at the end of the month."""

    # Simulate holiday on July 31st
    def is_holiday_side_effect(dt: object) -> bool:
        return str(dt) == "2026-07-31"

    mock_holiday_checker.is_holiday.side_effect = is_holiday_side_effect

    # July 30th is now the last trading day.
    # Last 5 trading days: July 24(Fri), 27(Mon), 28(Tue), 29(Wed), 30(Thu).
    assert (
        is_in_end_of_month_window(
            pd.Timestamp("2026-07-24").date(),
            days_before=4,
            holiday_checker=mock_holiday_checker,
        )
        is True
    )
    assert (
        is_in_end_of_month_window(
            pd.Timestamp("2026-07-31").date(),
            days_before=4,
            holiday_checker=mock_holiday_checker,
        )
        is False
    )


# --- SCREENER STRATEGY TESTS ---


def test_bridge_scout_skips_outside_month_end_window(
    bridge_scout_screener: BridgeScoutStrategy,
    mock_data_provider: MagicMock,
) -> None:
    """Tests that Bridge Scout skips screening outside the month-end window."""
    hits = bridge_scout_screener.run(days=0, analysis_date="2026-07-15")
    assert hits == 0
    mock_data_provider.get_batch_history.assert_not_called()


def test_bridge_scout_generates_signal_on_valid_setup(
    bridge_scout_screener: BridgeScoutStrategy,
    mock_trade_repo: MagicMock,
    mock_data_provider: MagicMock,
) -> None:
    """Tests signal generation when in month-end window and indicators meet thresholds."""
    analysis_date_str = "2026-07-28"
    dates = pd.date_range(end=analysis_date_str, periods=20, freq="B")

    # Create price history with low RSI and low ATR %
    closes = [500.0] * 18 + [495.0, 480.0]  # Drop on final days -> low RSI(2)
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]

    df_history = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100000] * 20,
        }
    )

    mock_data_provider.get_batch_history.return_value = {"QQQ": df_history}

    hits = bridge_scout_screener.run(days=0, analysis_date=analysis_date_str)

    assert hits == 1
    mock_trade_repo.create_trade.assert_called_once()
    call_kwargs = mock_trade_repo.create_trade.call_args.kwargs
    assert call_kwargs["symbol"] == "QQQ"
    assert call_kwargs["strategy"] == Strategies.BridgeScout.value
    assert call_kwargs["entry"] == 480.0


def test_bridge_scout_skips_if_rsi_too_high(
    bridge_scout_screener: BridgeScoutStrategy,
    mock_trade_repo: MagicMock,
    mock_data_provider: MagicMock,
) -> None:
    """Tests setup failure if RSI(2) is >= 40."""
    analysis_date_str = "2026-07-28"
    dates = pd.date_range(end=analysis_date_str, periods=20, freq="B")

    closes = [480.0] * 18 + [490.0, 500.0]  # Rising -> high RSI(2)
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]

    df_history = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100000] * 20,
        }
    )

    mock_data_provider.get_batch_history.return_value = {"QQQ": df_history}

    hits = bridge_scout_screener.run(days=0, analysis_date=analysis_date_str)

    assert hits == 0
    mock_trade_repo.create_trade.assert_not_called()


def test_bridge_scout_skips_if_active_position_exists(
    bridge_scout_screener: BridgeScoutStrategy,
    mock_trade_repo: MagicMock,
    mock_data_provider: MagicMock,
) -> None:
    """Tests strict single position rule (MaxPositions = 1)."""
    analysis_date_str = "2026-07-28"
    dates = pd.date_range(end=analysis_date_str, periods=20, freq="B")

    closes = [500.0] * 18 + [495.0, 480.0]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]

    df_history = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100000] * 20,
        }
    )

    mock_data_provider.get_batch_history.return_value = {"QQQ": df_history}
    mock_trade_repo.exists.return_value = True

    hits = bridge_scout_screener.run(days=0, analysis_date=analysis_date_str)

    assert hits == 0
    mock_trade_repo.create_trade.assert_not_called()


def test_bridge_scout_generates_premarket_setup_when_candle_not_in_history(
    bridge_scout_screener: BridgeScoutStrategy,
    mock_trade_repo: MagicMock,
    mock_data_provider: MagicMock,
) -> None:
    """Tests pre-market setup signal generation when candle for target date is not in DB yet."""
    # Target date is July 28th, but historical data only goes up to July 27th (pre-market)
    analysis_date_str = "2026-07-28"
    dates = pd.date_range(end="2026-07-27", periods=20, freq="B")

    closes = [500.0] * 19 + [495.0]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]

    df_history = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100000] * 20,
        }
    )

    mock_data_provider.get_batch_history.return_value = {"QQQ": df_history}

    hits = bridge_scout_screener.run(days=0, analysis_date=analysis_date_str)

    assert hits == 1
    mock_trade_repo.create_trade.assert_called_once()
    call_kwargs = mock_trade_repo.create_trade.call_args.kwargs
    assert call_kwargs["symbol"] == "QQQ"
    assert call_kwargs["strategy"] == Strategies.BridgeScout.value
    # entry is the calculated req_close_rsi40 threshold
    assert call_kwargs["entry"] > 0
    assert call_kwargs["context"]["req_close_rsi40"] > 0
    assert call_kwargs["context"]["setup_date"] == analysis_date_str


def test_bridge_scout_premarket_skips_when_atr_too_high(
    bridge_scout_screener: BridgeScoutStrategy,
    mock_trade_repo: MagicMock,
    mock_data_provider: MagicMock,
) -> None:
    """Tests that pre-market setup is skipped if ATR% exceeds threshold."""
    analysis_date_str = "2026-07-28"
    dates = pd.date_range(end="2026-07-27", periods=20, freq="B")

    # High volatility swings
    closes = [500.0] * 10 + [
        400.0,
        600.0,
        400.0,
        600.0,
        400.0,
        600.0,
        400.0,
        600.0,
        400.0,
        500.0,
    ]
    highs = [c + 50.0 for c in closes]
    lows = [c - 50.0 for c in closes]

    df_history = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100000] * 20,
        }
    )

    mock_data_provider.get_batch_history.return_value = {"QQQ": df_history}

    hits = bridge_scout_screener.run(days=0, analysis_date=analysis_date_str)

    assert hits == 0
    mock_trade_repo.create_trade.assert_not_called()


# --- EXECUTION STRATEGY TESTS ---


def test_bridge_scout_trade_manager_lifecycle() -> None:
    """Tests execution trade strategy check_entry and active management (month turn exit)."""
    strategy = BridgeScoutTradeStrategy()

    trade_record = {
        "id": 1,
        "symbol": "QQQ",
        "strategy": "bridge_scout",
        "status": "CREATED",
        "entry_price": 480.0,
        "budget": 10000.0,
        "signal_context": '{"setup_date": "2026-07-28", "req_close_rsi40": 480.0}',
    }

    candle_entry = pd.Series({"date": pd.Timestamp("2026-07-28"), "close": 475.0})
    df_empty = pd.DataFrame()

    # 1. Entry Activation (Close 475.0 <= Threshold 480.0 -> ACTIVE)
    transition_entry = strategy.check_entry(trade_record, candle_entry, df_empty)
    assert transition_entry is not None
    assert transition_entry.updates["status"] == "ACTIVE"

    # 2. Manage Active Trade - Same Month -> Hold (None)
    trade_active = {
        "id": 1,
        "symbol": "QQQ",
        "strategy": "bridge_scout",
        "status": "ACTIVE",
        "entry_price": 475.0,
        "entry_date": "2026-07-28",
        "current_size": 100,
    }

    candle_same_month = pd.Series({"date": pd.Timestamp("2026-07-29"), "close": 482.0})
    df_hist_same = pd.DataFrame([candle_entry, candle_same_month])

    transition_hold = strategy.manage_active_trade(trade_active, df_hist_same)
    assert transition_hold is None

    # 3. Manage Active Trade - Next Month (1st trading day of August) -> Exit MOC
    candle_next_month = pd.Series({"date": pd.Timestamp("2026-08-03"), "close": 490.0})
    df_hist_next = pd.DataFrame([candle_entry, candle_same_month, candle_next_month])

    transition_exit = strategy.manage_active_trade(trade_active, df_hist_next)
    assert transition_exit is not None
    assert transition_exit.updates["status"] == "CLOSED"
    assert transition_exit.updates["exit_price"] == 490.0
    assert transition_exit.updates["exit_reason"] == ExitReason.TIME_STOP
