# filename: test_ndx_momentum_manager.py
import pytest
from unittest.mock import MagicMock
import pandas as pd

from app.services.trade_manager.strategies.ndx_momentum import NDXMomentumTradeStrategy
from app.database.repositories.trade import TradeRepository
from app.types import TradeStatus


@pytest.fixture
def mock_trade_repo() -> MagicMock:
    return MagicMock(spec=TradeRepository)


@pytest.fixture
def strategy() -> NDXMomentumTradeStrategy:
    return NDXMomentumTradeStrategy()


def test_check_entry_bull_regime(strategy, mock_trade_repo):
    """Tests entry when regime is BULL and at least one day has passed."""
    signal_date = pd.Timestamp("2026-01-30")
    trade = {
        "id": "N1",
        "symbol": "AAPL",
        "strategy": "ndx_momentum",
        "signal_context": '{"qqq_regime": "BULL", "regime": "BULL", "date": "2026-01-30"}',
    }
    # Signal Friday (30th) -> Candle Monday (Feb 2nd)
    candle = pd.Series(
        {"open": 100.0, "date": pd.Timestamp("2026-02-02")},
        name=pd.Timestamp("2026-02-02"),
    )
    history = pd.DataFrame(
        [
            {"date": signal_date, "open": 99.0, "close": 100.0},
            {"date": pd.Timestamp("2026-02-02"), "open": 100.0, "close": 101.0},
        ]
    )

    # Mock no active positions
    mock_trade_repo.get_by_status.return_value = []

    result = strategy.check_entry(trade, candle, history, mock_trade_repo)

    assert result is not None
    assert "FILLED" in result
    mock_trade_repo.update_trade.assert_called_once()
    assert mock_trade_repo.update_trade.call_args[0][1]["status"] == TradeStatus.ACTIVE


def test_check_entry_lookahead_rejection(strategy, mock_trade_repo):
    """Tests that entry is rejected on the same day as the signal."""
    signal_date = pd.Timestamp("2026-02-02")
    trade = {
        "id": "N1",
        "symbol": "AAPL",
        "strategy": "ndx_momentum",
        "signal_context": '{"qqq_regime": "BULL", "date": "2026-02-02"}',
    }
    candle = pd.Series({"open": 100.0, "date": signal_date}, name=signal_date)
    history = pd.DataFrame([candle])

    result = strategy.check_entry(trade, candle, history, mock_trade_repo)

    assert result is None


def test_check_entry_bear_regime_rejection(strategy, mock_trade_repo):
    """Tests rejection when regime is not BULL."""
    trade = {
        "id": "N1",
        "signal_context": '{"qqq_regime": "BEAR", "regime": "BULL", "date": "2026-01-30"}',
    }
    candle = pd.Series(
        {"date": pd.Timestamp("2026-02-02")}, name=pd.Timestamp("2026-02-02")
    )
    history = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-30")},
            {"date": pd.Timestamp("2026-02-02")},
        ]
    )

    result = strategy.check_entry(trade, candle, history, mock_trade_repo)

    assert result is not None
    assert "QQQ Regime" in result
    assert mock_trade_repo.update_trade.call_args[0][1]["status"] == TradeStatus.INVALID


def test_manage_active_same_month_no_action(strategy, mock_trade_repo):
    """Tests that no rebalancing occurs mid-month."""
    trade = {"symbol": "AAPL"}
    # Feb 02 to Feb 03 (Same month)
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-02-02"), pd.Timestamp("2026-02-03")],
            "open": [100.0, 101.0],
        }
    )

    result = strategy.manage_active_trade(trade, df, mock_trade_repo)
    assert result is None


def test_manage_active_month_switch_dropped_symbol(strategy, mock_trade_repo):
    """Tests that a symbol is closed if it's no longer a leader on month switch."""
    trade = {"id": "N1", "symbol": "AAPL", "signal_context": '{"date": "2026-01-30"}'}
    # Jan 30 (Friday) -> Feb 02 (Monday)
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-30"), pd.Timestamp("2026-02-02")],
            "open": [100.0, 105.0],
        }
    )

    # Mock strategy trades: AAPL is from Jan, but new leaders are MSFT and GOOG from Feb 1st
    mock_trade_repo.get_all_by_strategy.return_value = [
        {"symbol": "AAPL", "signal_context": '{"date": "2026-01-30"}'},
        {"symbol": "MSFT", "signal_context": '{"date": "2026-02-01"}'},
        {"symbol": "GOOG", "signal_context": '{"date": "2026-02-01"}'},
    ]

    result = strategy.manage_active_trade(trade, df, mock_trade_repo)

    assert result is not None
    assert "REBALANCE_EXIT" in result
    assert mock_trade_repo.update_trade.call_args[0][1]["exit_price"] == 105.0


def test_manage_active_month_switch_keep_leader(strategy, mock_trade_repo):
    """Tests that a symbol is KEPT if it remains a leader across months."""
    trade = {"id": "N1", "symbol": "AAPL", "signal_context": '{"date": "2026-01-30"}'}
    # Jan 30 (Friday) -> Feb 02 (Monday)
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-30"), pd.Timestamp("2026-02-02")],
            "open": [100.0, 105.0],
        }
    )

    # Mock strategy trades: AAPL is from Jan, but ALSO a leader in Feb
    mock_trade_repo.get_all_by_strategy.return_value = [
        {"symbol": "AAPL", "signal_context": '{"date": "2026-01-30"}'},  # Old
        {
            "symbol": "AAPL",
            "signal_context": '{"date": "2026-02-01"}',
        },  # New leader entry
        {
            "symbol": "MSFT",
            "signal_context": '{"date": "2026-02-01"}',
        },  # New leader entry
    ]

    result = strategy.manage_active_trade(trade, df, mock_trade_repo)

    assert result is None  # Should NOT exit


def test_generate_orders(strategy, mock_trade_repo):
    """Tests order generation for rebalance."""
    trade = {"symbol": "AAPL", "budget": 2000.0}
    df = pd.DataFrame({"close": [100.0]})

    orders = strategy.generate_orders(trade, df, 2000.0, mock_trade_repo)

    assert orders is not None
    assert orders.quantity == 20
    assert orders.entry.type == "MKT"
    assert orders.entry.time_in_force == "OPG"


def test_generate_orders_exit_on_month_switch(strategy):
    """Tests that an exit order is generated on month switch if not in leaders."""
    trade = {
        "status": "ACTIVE",
        "symbol": "AAPL",
        "current_size": 50,
        "strategy": "ndx_momentum",
    }
    # History ends in June
    df = pd.DataFrame(
        {"date": [pd.Timestamp("2026-06-29"), pd.Timestamp("2026-06-30")]}
    )

    # Reference date is in July
    orders = strategy.generate_orders(
        trade=trade,
        dataframe_history=df,
        budget=10000.0,
        created_symbols={"MSFT", "GOOG"},  # AAPL is not in new leaders
        reference_date="2026-07-01",
    )

    assert orders is not None
    assert orders.symbol == "AAPL"
    assert orders.quantity == 50
    assert orders.mode == "Exit"
    assert orders.exits[0].action == "SELL"
    assert orders.exits[0].type == "MKT"
    assert orders.exits[0].time_in_force == "OPG"


def test_generate_orders_no_exit_same_month(strategy):
    """Tests that no exit order is generated when reference_date is in the same month."""
    trade = {
        "status": "ACTIVE",
        "symbol": "AAPL",
        "current_size": 50,
        "strategy": "ndx_momentum",
    }
    # History ends in June
    df = pd.DataFrame(
        {"date": [pd.Timestamp("2026-06-29"), pd.Timestamp("2026-06-30")]}
    )

    # Reference date is also in June
    orders = strategy.generate_orders(
        trade=trade,
        dataframe_history=df,
        budget=10000.0,
        created_symbols={"MSFT", "GOOG"},
        reference_date="2026-06-30",
    )

    assert orders is None
