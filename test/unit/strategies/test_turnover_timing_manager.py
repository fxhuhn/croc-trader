# filename: test_turnover_timing_manager.py
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.database.repositories.trade import TradeRepository
from app.services.trade_manager.strategies.turnover_timing import TurnoverTimingStrategy
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
