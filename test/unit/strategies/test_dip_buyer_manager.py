# filename: test_dip_buyer_manager.py
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd

from app.services.trade_manager.strategies.dip_buyer import DipBuyerStrategy
from app.database.repositories.trade import TradeRepository
from app.types import TradeStatus, ExitReason


@pytest.fixture
def mock_trade_repo() -> MagicMock:
    return MagicMock(spec=TradeRepository)


@pytest.fixture
def strategy() -> DipBuyerStrategy:
    return DipBuyerStrategy()


def test_check_entry_limit_hit(strategy, mock_trade_repo):
    """Tests that a trade enters when the limit price is reached."""
    # Arrange
    trade = {
        "id": "T1",
        "symbol": "AAPL",
        "entry_price": 100.0,
        "signal_context": '{"date": "2026-02-17"}',
        "current_size": 10,
    }
    # Current day: Feb 18th. Open 105, Low 99 (hits 100), Close 102.
    candle = pd.Series(
        {
            "open": 105.0,
            "low": 99.0,
            "close": 102.0,
            "date": pd.Timestamp("2026-02-18"),
        },
        name=pd.Timestamp("2026-02-18"),
    )

    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        # Act
        result = strategy.check_entry(
            trade, candle, pd.DataFrame([candle]), mock_trade_repo
        )

        # Assert
        assert result is not None
        assert "FILLED" in result
        mock_trade_repo.update_trade.assert_called_once()
        args, kwargs = mock_trade_repo.update_trade.call_args
        assert args[1]["status"] == TradeStatus.ACTIVE
        assert args[1]["entry_price"] == 100.0


def test_check_entry_gap_down_fill(strategy, mock_trade_repo):
    """Tests that a trade fills at the open price if there's a gap down below the limit."""
    trade = {
        "id": "T1",
        "symbol": "AAPL",
        "entry_price": 100.0,
        "signal_context": '{"date": "2026-02-17"}',
    }
    # Open 95 (below 100)
    candle = pd.Series(
        {"open": 95.0, "low": 94.0, "close": 98.0, "date": pd.Timestamp("2026-02-18")},
        name=pd.Timestamp("2026-02-18"),
    )

    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        result = strategy.check_entry(
            trade, candle, pd.DataFrame([candle]), mock_trade_repo
        )

        assert result is not None
        assert mock_trade_repo.update_trade.call_args[0][1]["entry_price"] == 95.0


def test_check_entry_expired(strategy, mock_trade_repo):
    """Tests that a trade expires if the limit is not hit on the first trading day after signal."""
    trade = {
        "id": "T1",
        "symbol": "AAPL",
        "entry_price": 100.0,
        "signal_context": '{"date": "2026-02-17"}',
    }
    # Low 101 (never hits 100)
    candle = pd.Series(
        {
            "open": 105.0,
            "low": 101.0,
            "close": 102.0,
            "date": pd.Timestamp("2026-02-18"),
        },
        name=pd.Timestamp("2026-02-18"),
    )

    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        result = strategy.check_entry(
            trade, candle, pd.DataFrame([candle]), mock_trade_repo
        )

        assert result is not None
        assert "INVALIDATED" in result
        assert (
            mock_trade_repo.update_trade.call_args[0][1]["status"]
            == TradeStatus.INVALID
        )


def test_manage_active_target_hit(strategy, mock_trade_repo):
    """Tests exit when the take profit target is hit."""
    trade = {
        "id": "T1",
        "symbol": "AAPL",
        "entry_price": 100.0,
        "entry_date": "2026-02-18",
        "status": "ACTIVE",
        "current_target": 110.0,
    }
    # High 115 (hits 110)
    candle = pd.Series(
        {
            "open": 105.0,
            "high": 115.0,
            "low": 104.0,
            "close": 112.0,
            "date": pd.Timestamp("2026-02-19"),
        },
        name=pd.Timestamp("2026-02-19"),
    )

    result = strategy.manage_active_trade(
        trade, pd.DataFrame([candle]), mock_trade_repo
    )

    assert result is not None
    assert "TARGET_HIT" in result
    assert mock_trade_repo.update_trade.call_args[0][1]["exit_price"] == 110.0


def test_manage_active_time_stop(strategy, mock_trade_repo):
    """Tests exit on the 8th day (time stop)."""
    trade = {"id": "T1", "entry_date": "2026-02-18", "status": "ACTIVE"}
    # 8 days later
    dates = pd.to_datetime(["2026-02-" + str(i + 18) for i in range(8)])
    dataframe = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * 8,
            "high": [105.0] * 8,
            "low": [95.0] * 8,
            "close": [102.0] * 8,
        }
    )

    result = strategy.manage_active_trade(trade, dataframe, mock_trade_repo)

    assert result is not None
    assert "TIME_STOP" in result
    assert (
        mock_trade_repo.update_trade.call_args[0][1]["exit_reason"]
        == ExitReason.TIME_STOP
    )


def test_generate_orders(strategy, mock_trade_repo):
    """Tests the order generation logic."""
    trade = {"symbol": "AAPL", "entry_price": 100.0, "initial_size": 10}

    dataframe_history = pd.DataFrame([pd.Series({"high": 98.49})])
    orders = strategy.generate_orders(trade, dataframe_history, 2000.0, mock_trade_repo)

    assert orders is not None
    assert orders.symbol == "AAPL"
    assert orders.quantity == 10
    assert orders.entry.price == 100.0


def test_generate_orders_active_trade(strategy, mock_trade_repo):
    """Tests the order generation logic for an active trade."""
    trade = {
        "symbol": "AAPL",
        "entry_price": 100.0,
        "initial_size": 10,
        "status": "ACTIVE",
        "current_target": 105.0,
        "signal_context": '{"threshold_loc": 98.5}',
    }

    dataframe_history = pd.DataFrame([pd.Series({"high": 98.49})])
    orders = strategy.generate_orders(trade, dataframe_history, 2000.0, mock_trade_repo)

    assert orders is not None
    assert orders.symbol == "AAPL"
    assert orders.quantity == 10
    assert orders.entry is None
    assert len(orders.exits) == 2
    # Exit 1: TP Sell LMT at 105
    assert orders.exits[0].action == "SELL"
    assert orders.exits[0].type == "LMT"
    assert orders.exits[0].price == 105.0
    # Exit 2: Threshold Sell LOC at 98.5
    assert orders.exits[1].action == "SELL"
    assert orders.exits[1].type == "LOC"
    assert orders.exits[1].price == 98.5
