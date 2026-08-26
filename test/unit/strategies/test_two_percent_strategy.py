# filename: test_two_percent_strategy.py
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.repositories.trade import TradeRepository
from app.services.screener.strategies.two_percent_strategy import (
    TwoPercentStrategy as ScreenerStrategy,
)
from app.services.trade_manager.strategies.two_percent_strategy import (
    TwoPercentStrategy as ManagerStrategy,
)
from app.types import ExitReason, TradeStatus

# --- FIXTURES ---


@pytest.fixture
def mock_trade_repository() -> MagicMock:
    """Provides a mock TradeRepository."""
    return MagicMock(spec=TradeRepository)


@pytest.fixture
def mock_data_provider() -> MagicMock:
    """Provides a mock MarketDataProvider."""
    return MagicMock(spec=MarketDataProvider)


@pytest.fixture
def screener_strategy(
    mock_trade_repository: MagicMock, mock_data_provider: MagicMock
) -> ScreenerStrategy:
    """Provides a TwoPercent Screener Strategy instance."""
    return ScreenerStrategy(
        trade_repository=mock_trade_repository, data_provider=mock_data_provider
    )


@pytest.fixture
def manager_strategy() -> ManagerStrategy:
    """Provides a TwoPercent Trade Manager Strategy instance."""
    # MarketHolidayChecker is a singleton, so we need to be careful with patching
    with patch(
        "app.services.trade_manager.strategies.two_percent_strategy.MarketHolidayChecker"
    ) as mock_checker:
        strategy = ManagerStrategy()
        # Attach the mock to the strategy instance for easy configuration in tests
        strategy.holiday_checker = mock_checker.return_value
        strategy.holiday_checker.is_holiday.return_value = False
        return strategy


# --- HELPERS ---


def create_candle(
    date_str: str,
    close: float,
    open_price: float = None,
    high: float = None,
    low: float = None,
) -> pd.Series:
    """Creates a mock candle as a pandas Series."""
    o = open_price if open_price is not None else close
    h = high if high is not None else close
    low_val = low if low is not None else close
    return pd.Series(
        {
            "date": pd.Timestamp(date_str),
            "open": o,
            "high": h,
            "low": low_val,
            "close": close,
        },
        name=pd.Timestamp(date_str),
    )


# --- SCREENER TESTS ---


@pytest.mark.parametrize(
    "test_date", ["2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05"]
)
@patch("pandas.Timestamp.now")
def test_screener_skips_live_runs_before_friday(
    mock_now: MagicMock, screener_strategy: ScreenerStrategy, test_date: str
) -> None:
    """
    Ensures that the screener does not generate signals mid-week (Mon-Thu) if run live.
    """
    # Arrange
    mock_now.return_value = pd.Timestamp(test_date)
    candle = create_candle(test_date, 100.0)
    screener_strategy.data_provider.get_symbol_history.return_value = pd.DataFrame(
        [candle]
    )

    # Act
    result = screener_strategy.run(analysis_date=test_date)

    # Assert
    assert result == 0
    screener_strategy.trade_repository.create_trade.assert_not_called()


@patch("pandas.Timestamp.now")
def test_screener_rejects_midweek_in_backtest(
    mock_now: MagicMock, screener_strategy: ScreenerStrategy
) -> None:
    """
    Ensures that in a backtest, a midweek day is rejected because future days
    for that week exist in the full database history.
    """
    # Arrange
    # Backtest is running in the future
    mock_now.return_value = pd.Timestamp("2026-05-01")

    wed_date = "2026-02-04"
    thu_date = "2026-02-05"
    fri_date = "2026-02-06"

    history = pd.DataFrame(
        [
            create_candle(wed_date, 100.0),
            create_candle(thu_date, 101.0),
            create_candle(fri_date, 102.0),
        ]
    )
    screener_strategy.data_provider.get_symbol_history.return_value = history

    # Act
    # Simulating backtester calling run() for Wednesday
    result = screener_strategy.run(analysis_date=wed_date)

    # Assert
    assert result == 0
    screener_strategy.trade_repository.create_trade.assert_not_called()


@patch("pandas.Timestamp.now")
def test_screener_signals_on_friday(
    mock_now: MagicMock, screener_strategy: ScreenerStrategy
) -> None:
    """
    Ensures that the screener generates a signal on Friday when the week has concluded.
    """
    # Arrange
    friday_date = "2026-02-06"
    # Simulate running on the following Monday
    mock_now.return_value = pd.Timestamp("2026-02-09")
    candle = create_candle(friday_date, 1000.0)
    screener_strategy.data_provider.get_symbol_history.return_value = pd.DataFrame(
        [candle]
    )
    screener_strategy.trade_repository.exists.return_value = False

    # Act
    result = screener_strategy.run(analysis_date=friday_date)

    # Assert
    assert result == 1
    screener_strategy.trade_repository.create_trade.assert_called_once()
    args, kwargs = screener_strategy.trade_repository.create_trade.call_args
    assert kwargs["entry"] == 990.0
    from app.const import Strategies

    assert kwargs.get("strategy") == Strategies.TwoPercent


@pytest.mark.parametrize("run_date", ["2026-02-07", "2026-02-08"])  # Saturday, Sunday
@patch("pandas.Timestamp.now")
def test_screener_signals_on_friday_when_run_on_weekend(
    mock_now: MagicMock, screener_strategy: ScreenerStrategy, run_date: str
) -> None:
    """
    Ensures that the screener successfully generates a signal on Friday when run over the weekend.
    """
    # Arrange
    friday_date = "2026-02-06"
    # Simulate running on the weekend
    mock_now.return_value = pd.Timestamp(run_date)
    candle = create_candle(friday_date, 1000.0)
    screener_strategy.data_provider.get_symbol_history.return_value = pd.DataFrame(
        [candle]
    )
    screener_strategy.trade_repository.exists.return_value = False

    # Act
    result = screener_strategy.run(analysis_date=friday_date)

    # Assert
    assert result == 1
    screener_strategy.trade_repository.create_trade.assert_called_once()


@patch("pandas.Timestamp.now")
def test_screener_signals_on_thursday_if_friday_missing(
    mock_now: MagicMock, screener_strategy: ScreenerStrategy
) -> None:
    """
    Ensures that the screener falls back to Thursday if Friday data is missing
    (e.g. because Friday was a market holiday), assuming the week has concluded.
    """
    # Arrange
    thursday_date = "2026-04-02"  # Week 14
    # Simulate running on the following Monday
    mock_now.return_value = pd.Timestamp("2026-04-06")
    screener_strategy._get_real_today = MagicMock(
        return_value=pd.Timestamp("2026-04-06").date()
    )

    candle = create_candle(thursday_date, 1000.0)
    screener_strategy.data_provider.get_symbol_history.return_value = pd.DataFrame(
        [candle]
    )
    screener_strategy.trade_repository.exists.return_value = False

    # Act
    result = screener_strategy.run(analysis_date=thursday_date)

    # Assert
    assert result == 1
    screener_strategy.trade_repository.create_trade.assert_called_once()


@patch("pandas.Timestamp.now")
def test_screener_signals_on_wednesday_if_thursday_and_friday_missing(
    mock_now: MagicMock, screener_strategy: ScreenerStrategy
) -> None:
    """
    Ensures that the screener falls back to Wednesday if Thursday and Friday data are missing.
    """
    # Arrange
    wednesday_date = "2026-04-01"  # Week 14
    # Simulate running on the following Monday
    mock_now.return_value = pd.Timestamp("2026-04-06")
    screener_strategy._get_real_today = MagicMock(
        return_value=pd.Timestamp("2026-04-06").date()
    )

    candle = create_candle(wednesday_date, 1000.0)
    screener_strategy.data_provider.get_symbol_history.return_value = pd.DataFrame(
        [candle]
    )
    screener_strategy.trade_repository.exists.return_value = False

    # Act
    result = screener_strategy.run(analysis_date=wednesday_date)

    # Assert
    assert result == 1
    screener_strategy.trade_repository.create_trade.assert_called_once()


@patch("pandas.Timestamp.now")
def test_screener_skips_thursday_fallback_when_run_on_friday_morning(
    mock_now: MagicMock, screener_strategy: ScreenerStrategy
) -> None:
    """
    Ensures that running on Friday morning evaluating Thursday does NOT
    falsely trigger the Thursday fallback end-of-week condition.
    """
    thursday_date = "2026-06-11"
    friday_date = "2026-06-12"

    # Simulate clock is Friday morning (e.g. 06:30)
    mock_now.return_value = pd.Timestamp(f"{friday_date} 06:30:00")
    screener_strategy._get_real_today = MagicMock(
        return_value=pd.Timestamp(friday_date).date()
    )

    # Database only has history up to Thursday (Friday hasn't traded yet)
    candle = create_candle(thursday_date, 1000.0)
    screener_strategy.data_provider.get_symbol_history.return_value = pd.DataFrame(
        [candle]
    )
    screener_strategy.trade_repository.exists.return_value = False

    # Act
    # Simulating the screener running for Thursday date
    result = screener_strategy.run(analysis_date=thursday_date)

    # Assert: Should not generate a signal
    assert result == 0
    screener_strategy.trade_repository.create_trade.assert_not_called()


# --- TRADE MANAGER ENTRY TESTS ---


def test_entry_not_allowed_on_setup_day(
    manager_strategy: ManagerStrategy, mock_trade_repository: MagicMock
) -> None:
    """
    Ensures that a trade cannot enter on the same day the signal was generated.
    """
    # Arrange
    setup_date = "2026-02-06"
    trade = {
        "id": "TRADE_SETUP_DAY",
        "symbol": "SXRV.DE",
        "strategy": "TwoPercent",
        "entry_price": 990.0,
        "signal_context": '{"date": "2026-02-06"}',
    }
    candle = create_candle(setup_date, 980.0)  # Price hit on Friday

    # Act
    result = manager_strategy.check_entry(
        trade, candle, pd.DataFrame([candle]), mock_trade_repository
    )

    # Assert
    assert result is None
    mock_trade_repository.update_trade.assert_not_called()


def test_entry_on_monday_fill_at_limit(
    manager_strategy: ManagerStrategy, mock_trade_repository: MagicMock
) -> None:
    """
    Standard Fill: Entry on Monday if price reaches the limit.
    """
    # Arrange
    monday_date = "2026-02-09"
    trade = {
        "id": "TRADE_MONDAY_FILL",
        "symbol": "SXRV.DE",
        "strategy": "TwoPercent",
        "entry_price": 990.0,
        "signal_context": '{"date": "2026-02-06"}',
    }
    # Monday: Open 1000, Low 985 (hits 990), Close 1010
    candle = create_candle(
        monday_date, 1010.0, open_price=1000.0, high=1015.0, low=985.0
    )

    # Act
    result = manager_strategy.check_entry(
        trade, candle, pd.DataFrame([candle]), mock_trade_repository
    )

    # Assert
    assert result is not None
    assert "FILLED" in result
    mock_trade_repository.update_trade.assert_called_once()
    call_args = mock_trade_repository.update_trade.call_args[0]
    assert call_args[1]["entry_price"] == 990.0


def test_entry_gap_down_fill_at_open(
    manager_strategy: ManagerStrategy, mock_trade_repository: MagicMock
) -> None:
    """
    Gap Down Fill: If Monday opens below the limit, entry is at the open price.
    """
    # Arrange
    monday_date = "2026-02-09"
    trade = {
        "id": "TRADE_ENTRY_GAP",
        "symbol": "SXRV.DE",
        "strategy": "TwoPercent",
        "entry_price": 990.0,
        "signal_context": '{"date": "2026-02-06"}',
    }
    # Monday: Open 980 (below 990)
    candle = create_candle(monday_date, 990.0, open_price=980.0, high=995.0, low=975.0)

    # Act
    result = manager_strategy.check_entry(
        trade, candle, pd.DataFrame([candle]), mock_trade_repository
    )

    # Assert
    assert result is not None
    assert "FILLED" in result
    call_args = mock_trade_repository.update_trade.call_args[0]
    assert call_args[1]["entry_price"] == 980.0


def test_entry_invalidated_if_no_fill_on_day_one(
    manager_strategy: ManagerStrategy, mock_trade_repository: MagicMock
) -> None:
    """
    One-Shot Rule: If price doesn't hit the limit on the first day after the signal, the trade is invalidated.
    """
    # Arrange
    monday_date = "2026-02-09"
    trade = {
        "id": "TRADE_NO_FILL",
        "symbol": "SXRV.DE",
        "strategy": "TwoPercent",
        "entry_price": 990.0,
        "signal_context": '{"date": "2026-02-06"}',
    }
    # Monday: Open 1000, Low 995 (Limit 990 NOT hit), Close 1005
    candle = create_candle(
        monday_date, 1005.0, open_price=1000.0, high=1010.0, low=995.0
    )

    # Act
    result = manager_strategy.check_entry(
        trade, candle, pd.DataFrame([candle]), mock_trade_repository
    )

    # Assert
    assert result is not None
    assert "REJECTED" in result or "INVALIDATED" in result
    mock_trade_repository.update_trade.assert_called_once()
    call_args = mock_trade_repository.update_trade.call_args[0]
    assert call_args[1]["status"] == TradeStatus.INVALID


# --- TRADE MANAGER EXIT TESTS ---


def test_exit_on_friday_time_stop(
    manager_strategy: ManagerStrategy, mock_trade_repository: MagicMock
) -> None:
    """
    Time Stop: Trade must close on Friday.
    """
    # Arrange
    trade = {
        "id": "TRADE_FRIDAY_STOP",
        "symbol": "SXRV.DE",
        "entry_price": 990.0,
        "entry_date": "2026-02-09",
        "status": "ACTIVE",
        "current_size": 1,
    }
    friday_date = "2026-02-13"  # Next Friday
    candle = create_candle(friday_date, 1005.0)
    history = pd.DataFrame([candle])

    # Act
    result = manager_strategy.manage_active_trade(trade, history, mock_trade_repository)

    # Assert
    assert result is not None
    assert "TIME_STOP" in result
    call_args = mock_trade_repository.update_trade.call_args[0]
    assert call_args[1]["exit_reason"] == ExitReason.TIME_STOP
    assert call_args[1]["exit_price"] == 1005.0


def test_exit_on_thursday_if_friday_holiday(
    manager_strategy: ManagerStrategy, mock_trade_repository: MagicMock
) -> None:
    """
    Time Stop (Holiday): If Friday is a holiday, trade must close on Thursday.
    """
    # Arrange
    thursday_date = "2026-04-02"
    friday_date = "2026-04-03"

    manager_strategy.holiday_checker.is_holiday.side_effect = lambda d: (
        str(d) == friday_date
    )

    trade = {
        "id": "TRADE_HOLIDAY_STOP",
        "symbol": "SXRV.DE",
        "entry_price": 990.0,
        "entry_date": "2026-03-30",
        "status": "ACTIVE",
        "current_size": 1,
    }
    candle = create_candle(thursday_date, 1005.0)
    history = pd.DataFrame([candle])

    # Act
    result = manager_strategy.manage_active_trade(trade, history, mock_trade_repository)

    # Assert
    assert result is not None
    assert "TIME_STOP" in result
    call_args = mock_trade_repository.update_trade.call_args[0]
    assert call_args[1]["exit_reason"] == ExitReason.TIME_STOP


def test_target_hit_only_allowed_from_day_plus_one(
    manager_strategy: ManagerStrategy, mock_trade_repository: MagicMock
) -> None:
    """
    Ensures Take Profit is not triggered on the same day as entry.
    """
    # Arrange
    entry_date = "2026-02-09"  # Monday
    trade = {
        "id": "TRADE_TP_SAME_DAY",
        "entry_price": 1000.0,
        "entry_date": entry_date,
        "status": "ACTIVE",
    }
    # Target is 1020 (1000 * 1.02)
    # Price hits 1025 on entry day
    candle = create_candle(entry_date, 1010.0, high=1025.0)
    history = pd.DataFrame([candle])

    # Act
    result = manager_strategy.manage_active_trade(trade, history, mock_trade_repository)

    # Assert
    assert result is None
    mock_trade_repository.update_trade.assert_not_called()


def test_target_hit_on_day_plus_one_success(
    manager_strategy: ManagerStrategy, mock_trade_repository: MagicMock
) -> None:
    """
    Take Profit: Successful exit on Tuesday if target hit.
    """
    # Arrange
    entry_date = "2026-02-09"
    tuesday_date = "2026-02-10"
    trade = {
        "id": "TRADE_TP_SUCCESS",
        "entry_price": 1000.0,
        "entry_date": entry_date,
        "status": "ACTIVE",
        "current_size": 1,
    }
    # Target is 1020
    candle = create_candle(tuesday_date, 1015.0, high=1025.0)
    history = pd.DataFrame([candle])

    # Act
    result = manager_strategy.manage_active_trade(trade, history, mock_trade_repository)

    # Assert
    assert result is not None
    assert "TARGET_HIT" in result
    call_args = mock_trade_repository.update_trade.call_args[0]
    assert call_args[1]["exit_price"] == 1020.0


def test_regression_feb_2026_entry_exit(
    manager_strategy: ManagerStrategy, mock_trade_repository: MagicMock
) -> None:
    """
    Regression Test for Week of Feb 9th 2026:
    - Signal: Fri Feb 6 (Screener runs).
    - Entry: Mon Feb 9 (Day 1). Price dips to limit -> Fill.
    - Active: Tue Feb 10 (Day 2). No exit.
    - Exit: Wed Feb 11 (Day 3). Price hits target.
    """
    # 1. Setup Trade (Simulating it was created by Screener on Fri Feb 6)
    # Fri Feb 6 Close was X. Limit = 0.99 * X.
    # Let's say Close = 100.0, Limit = 99.0.
    trade_id = "TRADE_REGRESSION_FEB_2026"
    trade_entry_price = 99.0
    signal_date = "2026-02-06"

    trade = {
        "id": trade_id,
        "symbol": "SXRV.DE",
        "strategy": "TwoPercent",
        "entry_price": trade_entry_price,
        "signal_context": f'{{"date": "{signal_date}"}}',
        "entry_date": None,  # Not entered yet
        "status": TradeStatus.CREATED,
    }

    # 2. Monday Feb 9: Check Entry
    # Mock History so _get_trading_days_post_signal calculates correctly.
    # We need Fri Feb 6 and Mon Feb 9 in history.
    candle_fri = create_candle("2026-02-06", 100.0)
    candle_mon = create_candle(
        "2026-02-09", 101.0, open_price=100.0, low=98.5
    )  # Low 98.5 < 99.0 Limit -> FILL

    history_mon = pd.DataFrame([candle_fri, candle_mon])

    # Act: Check Entry
    # We must patch _get_trading_days_post_signal because it relies on complex trading day logic
    # that might rely on external data or bigger history.
    # OR better: we trust the Abstract Strategy logic if we provide enough history.
    # Let's assume BaseTradeStrategy uses self.data_provider or the dataframe passed in.
    # If it uses dataframe passed in, history_mon is sufficient.

    # NOTE: Since we can't easily rely on base class logic without looking at it,
    # we will assume the standard "next trading day" logic works if dates are contiguous weekdays.

    with patch.object(
        manager_strategy, "_get_trading_days_post_signal", return_value=1
    ):
        result_entry = manager_strategy.check_entry(
            trade, candle_mon, history_mon, mock_trade_repository
        )

    assert result_entry is not None
    assert "FILLED" in result_entry

    # 3. Simulate Entry Update in Trade Repo
    trade["status"] = TradeStatus.ACTIVE
    trade["entry_date"] = "2026-02-09"
    trade["entry_price"] = trade_entry_price
    trade["current_size"] = 10

    # 4. Tuesday Feb 10: Manage Active (No Exit)
    # Price moves but doesn't hit target (Target = 99 * 1.02 = 100.98)
    # Let's say High is 100.5
    candle_tue = create_candle("2026-02-10", 100.0, high=100.5)
    history_tue = pd.DataFrame([candle_fri, candle_mon, candle_tue])

    result_manage_tue = manager_strategy.manage_active_trade(
        trade, history_tue, mock_trade_repository
    )
    assert result_manage_tue is None  # No exit

    # 5. Wednesday Feb 11: Manage Active (Target Hit)
    # Price spikes to 102.0 (Target 100.98)
    candle_wed = create_candle("2026-02-11", 101.5, high=102.0)
    history_wed = pd.DataFrame([candle_fri, candle_mon, candle_tue, candle_wed])

    result_manage_wed = manager_strategy.manage_active_trade(
        trade, history_wed, mock_trade_repository
    )

    assert result_manage_wed is not None
    assert "TARGET_HIT" in result_manage_wed

    # Verify Exit Price logic (max of Open/Target)
    # If Open was 100.0, Target 100.98. Exit at Target.
    call_args = mock_trade_repository.update_trade.call_args[0]
    # The last call should be the exit
    assert call_args[1]["exit_reason"] == ExitReason.TARGET_HIT


def test_monday_holiday_check_but_no_kill(
    manager_strategy: ManagerStrategy, mock_trade_repository: MagicMock
) -> None:
    """
    Scenario: Monday is a holiday, but we have data. No fill occurs.
    Rule: Don't invalidate, wait for Tuesday.
    """
    # Arrange
    monday_date = "2026-02-16"  # Presumed Holiday
    manager_strategy.holiday_checker.is_holiday.side_effect = lambda d: (
        str(d) == monday_date
    )

    trade = {
        "id": "TRADE_MONDAY_HOLIDAY",
        "symbol": "SXRV.DE",
        "strategy": "TwoPercent",
        "entry_price": 990.0,
        "signal_context": '{"date": "2026-02-13"}',  # Friday
    }
    # Monday: Open 1000, Low 995 (Limit 990 NOT hit)
    candle = create_candle(monday_date, 1005.0, open_price=1000.0, low=995.0)
    # History must include signal day (Fri)
    history = pd.DataFrame([create_candle("2026-02-13", 1000.0), candle])

    # Act
    with patch.object(
        manager_strategy, "_get_trading_days_post_signal", return_value=1
    ):
        result = manager_strategy.check_entry(
            trade, candle, history, mock_trade_repository
        )

    # Assert
    assert result is None  # Still CREATED, no rejection
    mock_trade_repository.update_trade.assert_not_called()


def test_tuesday_fill_after_monday_holiday(
    manager_strategy: ManagerStrategy, mock_trade_repository: MagicMock
) -> None:
    """
    Scenario: Monday was a holiday. Tuesday fills normally (Limit Hit).
    """
    # Arrange
    monday_date = "2026-02-16"
    tuesday_date = "2026-02-17"
    manager_strategy.holiday_checker.is_holiday.side_effect = lambda d: {
        "2026-02-16": True,
        "2026-02-17": False,
    }.get(str(d), False)

    trade = {
        "id": "TRADE_TUESDAY_FILL",
        "symbol": "SXRV.DE",
        "strategy": "TwoPercent",
        "entry_price": 990.0,
        "signal_context": '{"date": "2026-02-13"}',
    }

    candle_mon = create_candle(monday_date, 1000.0)
    candle_tue = create_candle(
        tuesday_date, 1010.0, open_price=1000.0, low=985.0
    )  # Low 985 < 990 -> FILL
    history = pd.DataFrame(
        [
            create_candle("2026-02-13", 1000.0),
            candle_mon,
            candle_tue,
        ]
    )

    # Act
    with patch.object(
        manager_strategy, "_get_trading_days_post_signal", return_value=2
    ):
        result = manager_strategy.check_entry(
            trade, candle_tue, history, mock_trade_repository
        )

    # Assert
    assert result is not None
    assert "FILLED" in result

    # Verify reason in Repository call
    call_args = mock_trade_repository.update_trade.call_args
    assert "Tuesday-after-Holiday" in call_args[1]["reason"]
    assert call_args[0][1]["entry_price"] == 990.0


def test_tuesday_gap_down_after_monday_holiday(
    manager_strategy: ManagerStrategy, mock_trade_repository: MagicMock
) -> None:
    """
    Scenario: Monday was a holiday. Tuesday gaps down below limit.
    """
    # Arrange
    monday_date = "2026-02-16"
    tuesday_date = "2026-02-17"
    manager_strategy.holiday_checker.is_holiday.side_effect = lambda d: {
        "2026-02-16": True,
        "2026-02-17": False,
    }.get(str(d), False)

    trade = {
        "id": "TRADE_TUESDAY_GAP",
        "symbol": "SXRV.DE",
        "strategy": "TwoPercent",
        "entry_price": 990.0,
        "signal_context": '{"date": "2026-02-13"}',
    }

    candle_mon = create_candle(monday_date, 1000.0)
    candle_tue = create_candle(
        tuesday_date, 985.0, open_price=980.0
    )  # Open 980 < 990 -> GAP FILL
    history = pd.DataFrame(
        [
            create_candle("2026-02-13", 1000.0),
            candle_mon,
            candle_tue,
        ]
    )

    # Act
    with patch.object(
        manager_strategy, "_get_trading_days_post_signal", return_value=2
    ):
        result = manager_strategy.check_entry(
            trade, candle_tue, history, mock_trade_repository
        )

    # Assert
    assert result is not None
    assert "FILLED" in result

    # Verify reason in Repository call
    call_args = mock_trade_repository.update_trade.call_args
    assert "Gap Down" in call_args[1]["reason"]
    assert call_args[0][1]["entry_price"] == 980.0


def test_monday_no_fill_no_holiday_invalidation(
    manager_strategy: ManagerStrategy, mock_trade_repository: MagicMock
) -> None:
    """
    Scenario: Monday is NOT a holiday. No fill occurs.
    Rule: Invalidate immediately.
    """
    # Arrange
    monday_date = "2026-02-09"
    manager_strategy.holiday_checker.is_holiday.return_value = False

    trade = {
        "id": "TRADE_MONDAY_REGULAR_FAIL",
        "symbol": "SXRV.DE",
        "strategy": "TwoPercent",
        "entry_price": 990.0,
        "signal_context": '{"date": "2026-02-06"}',
    }
    # Monday: Open 1000, Low 995 (No Fill)
    candle = create_candle(monday_date, 1005.0, open_price=1000.0, low=995.0)
    history = pd.DataFrame([create_candle("2026-02-06", 1000.0), candle])

    # Act
    with patch.object(
        manager_strategy, "_get_trading_days_post_signal", return_value=1
    ):
        result = manager_strategy.check_entry(
            trade, candle, history, mock_trade_repository
        )

    # Assert
    # Assert
    assert result is not None
    assert (
        "REJECTED" in result or "INVALIDATED" in result
    )  # Base Strategy might return REJECTED
    call_args = mock_trade_repository.update_trade.call_args[0]
    assert call_args[1]["status"] == TradeStatus.INVALID


def test_two_percent_calculate_target_price_zero(
    manager_strategy: ManagerStrategy,
) -> None:
    """Tests _calculate_target_price returns 0.0 for entry_price <= 0."""
    from decimal import Decimal

    assert manager_strategy._calculate_target_price(Decimal("0.0")) == Decimal("0.0")


def test_two_percent_generate_entry_order_small_budget(
    manager_strategy: ManagerStrategy,
) -> None:
    """Tests _generate_entry_order returns None when quantity < 1."""
    trade = {"symbol": "AAPL", "entry_price": 500.0}
    assert (
        manager_strategy._generate_entry_order(trade, pd.DataFrame(), budget=10.0)
        is None
    )


def test_two_percent_generate_exit_order_date_bounds(
    manager_strategy: ManagerStrategy,
) -> None:
    """Tests _generate_exit_order returns None when next_day <= entry_date or target <= 0."""
    trade = {"symbol": "AAPL", "current_size": 10, "entry_date": "2026-02-10"}
    df_history = pd.DataFrame([{"date": "2026-02-09"}])
    assert (
        manager_strategy._generate_exit_order(trade, df_history, budget=1000.0) is None
    )

    trade_no_target = {
        "symbol": "AAPL",
        "current_size": 10,
        "entry_date": "2026-02-01",
        "entry_price": 0.0,
        "current_target": 0.0,
    }
    df_history_2 = pd.DataFrame([{"date": "2026-02-09"}])
    assert (
        manager_strategy._generate_exit_order(
            trade_no_target, df_history_2, budget=1000.0
        )
        is None
    )


def test_two_percent_check_entry_missing_signal_date_or_stale(
    manager_strategy: ManagerStrategy,
    mock_trade_repository: MagicMock,
) -> None:
    """Tests check_entry returns None when signal_date missing, or invalidates when stale."""
    trade_no_signal = {"symbol": "AAPL", "entry_price": 100.0, "signal_context": "{}"}
    candle = create_candle("2026-02-09", 100.0)
    assert (
        manager_strategy.check_entry(
            trade_no_signal, candle, pd.DataFrame([candle]), mock_trade_repository
        )
        is None
    )

    trade_stale = {
        "id": "STALE_1",
        "symbol": "AAPL",
        "entry_price": 100.0,
        "signal_context": '{"date": "2026-01-30"}',
    }
    with patch.object(
        manager_strategy, "_get_trading_days_post_signal", return_value=3
    ):
        result = manager_strategy.check_entry(
            trade_stale, candle, pd.DataFrame([candle]), mock_trade_repository
        )
        assert result is not None
        assert "REJECTED" in result or "INVALIDATED" in result


def test_two_percent_process_day_two_entry_no_fill(
    manager_strategy: ManagerStrategy,
) -> None:
    """Tests _process_day_two_entry rejects setup when low > limit."""
    trade = {"id": "D2_FAIL", "symbol": "AAPL"}
    transition = manager_strategy._process_day_two_entry(
        trade,
        open_price=105.0,
        low_price=102.0,
        limit_price=100.0,
        date_string="2026-02-10",
    )
    assert transition is not None
    assert transition.updates["status"] == TradeStatus.INVALID.value


def test_two_percent_do_manage_active_trade_no_entry_date(
    manager_strategy: ManagerStrategy,
) -> None:
    """Tests _do_manage_active_trade returns None when entry_date is missing."""
    trade_no_date = {"symbol": "AAPL", "status": "ACTIVE"}
    candle = create_candle("2026-02-10", 105.0)
    assert (
        manager_strategy._do_manage_active_trade(
            trade_no_date, candle, "2026-02-10", pd.DataFrame([candle])
        )
        is None
    )


def test_two_percent_send_signal_report(
    mock_trade_repository: MagicMock, mock_data_provider: MagicMock
) -> None:
    """Tests _send_signal_report correctly formats telegram dataframe with Entry and Action."""
    mock_telegram = MagicMock()
    strategy = ScreenerStrategy(
        trade_repository=mock_trade_repository,
        data_provider=mock_data_provider,
        telegram_bot=mock_telegram,
    )

    strategy._send_signal_report("2026-02-20", close=600.0, entry=594.0)

    assert mock_telegram.send_dataframe.called
    sent_df, kwargs = (
        mock_telegram.send_dataframe.call_args[0],
        mock_telegram.send_dataframe.call_args[1],
    )
    df = (
        sent_df
        if isinstance(sent_df, pd.DataFrame)
        else mock_telegram.send_dataframe.call_args[0][0]
    )
    assert df.iloc[0]["Symbol"] == "SXRV.DE"
    assert df.iloc[0]["Action"] == "BUY LMT"
    assert df.iloc[0]["Entry"] == "594.00"
    assert "two_percent Entries" in kwargs.get(
        "title", mock_telegram.send_dataframe.call_args[1].get("title", "")
    )
