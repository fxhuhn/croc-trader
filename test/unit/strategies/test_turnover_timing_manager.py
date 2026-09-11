# filename: test_turnover_timing_manager.py
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.database.repositories.trade import TradeRepository
from app.services.trade_manager.strategies.turnover_timing import (
    TurnoverTimingStrategy,
    calculate_consecutive_green_candles,
)
from app.types import TradeStatus


@pytest.fixture
def mock_trade_repo() -> MagicMock:
    return MagicMock(spec=TradeRepository)


@pytest.fixture
def strategy() -> TurnoverTimingStrategy:
    return TurnoverTimingStrategy()


def test_check_entry_limit_hit_and_green_count(strategy, mock_trade_repo):
    """Tests entry when limit is hit and updates green candle count if day is green."""
    trade = {
        "id": "T1",
        "symbol": "AAPL",
        "entry_price": 100.0,
        "signal_context": json.dumps({"date": "2026-02-13", "green_candle_count": 0}),
    }
    # Current day: Feb 16th (Monday). Open 99, Low 98 (hits 100), Close 102 (Green).
    candle = pd.Series(
        {"open": 99.0, "low": 98.0, "close": 102.0, "date": pd.Timestamp("2026-02-16")},
        name=pd.Timestamp("2026-02-16"),
    )

    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        result = strategy.check_entry(
            trade, candle, pd.DataFrame([candle]), mock_trade_repo
        )

        assert result is not None
        assert "FILLED" in result
        mock_trade_repo.update_trade.assert_called_once()
        payload = mock_trade_repo.update_trade.call_args[0][1]
        assert json.loads(payload["signal_context"])["green_candle_count"] == 1


def test_check_entry_limit_hit_inherits_setup_green_state(strategy, mock_trade_repo):
    """Tests entry when limit is hit and setup day was also green."""
    trade = {
        "id": "T1",
        "symbol": "AAPL",
        "entry_price": 100.0,
        "signal_context": json.dumps(
            {
                "date": "2026-02-13",
                "setup_candle_green": True,
                "green_candle_count": 0,
            }
        ),
    }
    # Current day: Feb 16th (Monday). Open 99, Low 98 (hits 100), Close 102 (Green).
    candle = pd.Series(
        {"open": 99.0, "low": 98.0, "close": 102.0, "date": pd.Timestamp("2026-02-16")},
        name=pd.Timestamp("2026-02-16"),
    )

    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        result = strategy.check_entry(
            trade, candle, pd.DataFrame([candle]), mock_trade_repo
        )

        assert result is not None
        assert "FILLED" in result
        mock_trade_repo.update_trade.assert_called_once()
        payload = mock_trade_repo.update_trade.call_args[0][1]
        assert json.loads(payload["signal_context"])["green_candle_count"] == 2


def test_check_entry_expired(strategy, mock_trade_repo):
    """Tests immediate expiration if limit not hit on Day 1."""
    trade = {
        "id": "T1",
        "entry_price": 100.0,
        "signal_context": json.dumps({"date": "2026-02-13"}),
    }
    # Low 101 (above 100). Must provide 'open' as well.
    candle = pd.Series(
        {"open": 105.0, "low": 101.0, "date": pd.Timestamp("2026-02-16")},
        name=pd.Timestamp("2026-02-16"),
    )

    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        result = strategy.check_entry(
            trade, candle, pd.DataFrame([candle]), mock_trade_repo
        )
        assert "EXPIRED" in result
        assert (
            mock_trade_repo.update_trade.call_args[0][1]["status"]
            == TradeStatus.INVALID
        )


def test_manage_active_green_sequence_exit(strategy, mock_trade_repo):
    """Tests exit at OPEN after 2 green candles."""
    trade = {
        "id": "T1",
        "symbol": "AAPL",
        "entry_price": 100.0,
        "status": "ACTIVE",
        "signal_context": json.dumps({"green_candle_count": 2}),
    }
    candle = pd.Series(
        {"open": 110.0, "date": pd.Timestamp("2026-02-18")},
        name=pd.Timestamp("2026-02-18"),
    )

    result = strategy.manage_active_trade(
        trade, pd.DataFrame([candle]), mock_trade_repo
    )

    assert result is not None
    assert "GREEN_SEQUENCE" in result
    assert mock_trade_repo.update_trade.call_args[0][1]["exit_price"] == 110.0


def test_manage_active_friday_time_stop(strategy, mock_trade_repo):
    """Tests exit on Friday Close."""
    trade = {
        "id": "T1",
        "status": "ACTIVE",
        "signal_context": json.dumps({"green_candle_count": 0}),
    }
    # Feb 20th is Friday
    candle = pd.Series(
        {"open": 105.0, "close": 102.0, "date": pd.Timestamp("2026-02-20")},
        name=pd.Timestamp("2026-02-20"),
    )

    result = strategy.manage_active_trade(
        trade, pd.DataFrame([candle]), mock_trade_repo
    )

    assert result is not None
    assert "TIME_STOP" in result
    assert mock_trade_repo.update_trade.call_args[0][1]["exit_price"] == 102.0


def test_generate_orders_entry(strategy, mock_trade_repo):
    """Tests entry order generation."""
    trade = {"symbol": "AAPL", "status": "CREATED", "entry_price": 100.0}

    order = strategy.generate_orders(trade, pd.DataFrame(), 2000.0, mock_trade_repo)
    assert order is not None
    assert order.mode == "Entry"
    assert order.entry.price == 100.0
    assert order.quantity == 20


def test_generate_orders_active_friday_time_stop(strategy, mock_trade_repo):
    """Tests exit order generation for active trades prior to weekend/time stop."""
    trade = {
        "symbol": "AAPL",
        "status": "ACTIVE",
        "current_size": 20,
        "entry_price": 100.0,
    }
    df_hist = pd.DataFrame([{"date": pd.Timestamp("2026-02-19")}])

    with patch.object(strategy, "_is_end_of_trading_week", return_value=True):
        order = strategy.generate_orders(trade, df_hist, 2000.0, mock_trade_repo)
        assert order is not None
        assert order.mode == "Exit"
        assert len(order.exits) == 1
        assert order.exits[0].action == "SELL"
        assert order.exits[0].type == "MOC"
        assert order.exits[0].quantity == 20


def test_calculate_consecutive_green_candles_pure() -> None:
    """Verifies pure calculation function across sequences, red resets, and setup anchor."""
    dates = pd.to_datetime(["2026-09-04", "2026-09-08", "2026-09-09", "2026-09-10"])
    # 2026-09-04 (Setup): Red (Close 311 < Open 315)
    # 2026-09-08 (Entry): Red (Close 311 < Open 312)
    # 2026-09-09: Red (Close 315.34 < Open 315.49)
    # 2026-09-10: Green (Close 325.80 > Open 316.79)
    df_history = pd.DataFrame(
        {
            "date": dates,
            "open": [315.0, 312.0, 315.49, 316.79],
            "close": [311.0, 311.0, 315.34, 325.80],
        }
    )

    count = calculate_consecutive_green_candles(
        df_history, start_date="2026-09-04", initial_setup_candle_green=False
    )
    assert count == 1

    # Now simulate two consecutive greens (e.g. Wednesday was also green)
    df_history_two_greens = pd.DataFrame(
        {
            "date": dates,
            "open": [315.0, 312.0, 315.0, 316.79],
            "close": [311.0, 311.0, 318.0, 325.80],
        }
    )
    count_two = calculate_consecutive_green_candles(
        df_history_two_greens, start_date="2026-09-04", initial_setup_candle_green=False
    )
    assert count_two == 2


def test_turnover_timing_self_healing_after_historical_price_correction(
    strategy: TurnoverTimingStrategy,
) -> None:
    """Verifies that historical price revisions in stocks.db self-heal the green candle count.

    Recreates the exact AAPL defect scenario:
    1. Preliminary run had a green candle on 2026-09-09, leaving green_candle_count=2 in context.
    2. Authoritative sync corrected 2026-09-09 to red.
    3. get_daily_updates and _generate_exit_order must recalculate statelessly and NOT generate
       an errant MKT OPG exit order for 2026-09-11.
    """
    trade = {
        "id": 1449,
        "symbol": "AAPL",
        "strategy": "turnover_timing_0.5",
        "status": "ACTIVE",
        "current_size": 9,
        "entry_price": 315.69,
        "signal_context": json.dumps(
            {
                "setup_date": "2026-09-04",
                "setup_candle_green": False,
                # Stale unadjusted accumulator value from previous day's run
                "green_candle_count": 2,
                "last_processed_date": "2026-09-10 00:00:00",
            }
        ),
    }

    # Authoritative corrected historical data from stocks.db
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-09-04", "2026-09-08", "2026-09-09", "2026-09-10"]
            ),
            "open": [328.31, 317.10, 315.49, 316.79],
            "close": [319.97, 316.22, 315.34, 325.80],  # 09-09 is RED (315.34 < 315.49)
            "high": [328.93, 320.70, 319.15, 326.68],
            "low": [317.86, 314.90, 309.90, 316.57],
        }
    )

    # 1. get_daily_updates must self-heal green_candle_count to 1
    daily_updates = strategy.get_daily_updates(trade, history)
    assert daily_updates["green_candle_count"] == 1

    # 2. _generate_exit_order must NOT generate a MKT OPG order
    mock_repo = MagicMock()
    with patch.object(strategy, "_is_end_of_trading_week", return_value=False):
        order = strategy.generate_orders(trade, history, 2000.0, mock_repo)
        assert order is None
