# filename: test_hold_target.py
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.database.repositories.trade import TradeRepository
from app.services.trade_manager.strategies.hold_target import HoldTargetStrategy
from app.types import ExitReason, TradeStatus


@pytest.fixture
def mock_trade_repo() -> MagicMock:
    return MagicMock(spec=TradeRepository)


@pytest.fixture
def strategy() -> HoldTargetStrategy:
    return HoldTargetStrategy()


def test_check_entry_breakout_fill(strategy, mock_trade_repo):
    """Tests standard breakout entry (STOP BUY)."""
    trade = {
        "id": "H1",
        "symbol": "AAPL",
        "entry_price": 150.0,
        "signal_context": '{"date": "2026-02-17"}',
        "current_size": 10,
    }
    # High 155 (hits 150 breakout level)
    candle = pd.Series(
        {
            "open": 149.0,
            "high": 155.0,
            "low": 148.0,
            "close": 152.0,
            "date": pd.Timestamp("2026-02-18"),
        },
        name=pd.Timestamp("2026-02-18"),
    )

    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        result = strategy.check_entry(
            trade, candle, pd.DataFrame([candle]), mock_trade_repo
        )

        assert result is not None
        assert "BREAKOUT" in result or "FILLED" in result
        mock_trade_repo.update_trade.assert_called_once()
        assert mock_trade_repo.update_trade.call_args[0][1]["entry_price"] == 150.0


def test_check_entry_day_one_turnaround(strategy, mock_trade_repo):
    """Tests entry on Day 1 when price closes above setup high after opening below."""
    trade = {
        "id": "H1",
        "entry_price": 150.0,
        "signal_context": '{"date": "2026-02-17"}',
        "current_size": 10,
    }
    # Open 145, High 152 (hits 150), Close 151
    candle = pd.Series(
        {
            "open": 145.0,
            "high": 152.0,
            "low": 144.0,
            "close": 151.0,
            "date": pd.Timestamp("2026-02-18"),
        },
        name=pd.Timestamp("2026-02-18"),
    )

    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        result = strategy.check_entry(
            trade, candle, pd.DataFrame([candle]), mock_trade_repo
        )
        assert result is not None
        assert mock_trade_repo.update_trade.call_args[0][1]["entry_price"] == 150.0


def test_check_entry_expiration(strategy, mock_trade_repo):
    """Tests that a trade expires after 5 days without fill."""
    trade = {
        "id": "H1",
        "entry_price": 150.0,
        "signal_context": '{"date": "2026-02-10"}',
    }
    candle = pd.Series(
        {"high": 149.0, "date": pd.Timestamp("2026-02-18")},
        name=pd.Timestamp("2026-02-18"),
    )

    with patch.object(strategy, "_get_trading_days_post_signal", return_value=6):
        result = strategy.check_entry(
            trade, candle, pd.DataFrame([candle]), mock_trade_repo
        )

        assert result is not None
        assert "EXPIRED" in result
        mock_trade_repo.update_trade.assert_called_once()
        assert (
            mock_trade_repo.update_trade.call_args[0][1]["status"]
            == TradeStatus.INVALID
        )
        assert (
            mock_trade_repo.update_trade.call_args[0][1]["exit_reason"]
            == ExitReason.INVALIDATED
        )


def test_manage_active_stop_loss_hit(strategy, mock_trade_repo):
    """Tests exit when stop loss is hit."""
    trade = {
        "id": "H1",
        "entry_price": 150.0,
        "current_stop_loss": 140.0,  # Correct key
        "status": "ACTIVE",
        "entry_date": "2026-02-18",
    }
    # Low 135 (hits 140)
    candle = pd.Series(
        {
            "open": 145.0,
            "high": 146.0,
            "low": 135.0,
            "close": 138.0,
            "date": pd.Timestamp("2026-02-19"),
        },
        name=pd.Timestamp("2026-02-19"),
    )

    result = strategy.manage_active_trade(
        trade, pd.DataFrame([candle]), mock_trade_repo
    )

    assert result is not None
    assert "STOP_LOSS" in result
    assert mock_trade_repo.update_trade.call_args[0][1]["exit_price"] == 140.0


def test_manage_active_target_hit(strategy, mock_trade_repo):
    """Tests exit when final target (Hold) is hit."""
    trade = {
        "id": "H1",
        "entry_price": 100.0,
        "current_target": 130.0,
        "status": "ACTIVE",
        "entry_date": "2026-02-18",
    }
    # High 135 (hits 130)
    candle = pd.Series(
        {
            "open": 105.0,
            "high": 135.0,
            "low": 104.0,
            "close": 132.0,
            "date": pd.Timestamp("2026-02-20"),
        },
        name=pd.Timestamp("2026-02-20"),
    )

    result = strategy.manage_active_trade(
        trade, pd.DataFrame([candle]), mock_trade_repo
    )

    assert result is not None
    assert "TARGET_HIT" in result
    assert mock_trade_repo.update_trade.call_args[0][1]["exit_price"] == 130.0


def test_generate_orders(strategy, mock_trade_repo):
    """Tests the order generation logic for breakout."""
    trade = {
        "symbol": "AAPL",
        "entry_price": 150.0,
        "current_stop_loss": 140.0,  # Correct key
        "current_size": 10,
    }

    orders = strategy.generate_orders(trade, pd.DataFrame(), 2000.0, mock_trade_repo)

    assert orders is not None
    assert orders.entry.type == "STP"
    assert orders.entry.price == 150.0


def test_hold_target_no_breakeven(strategy, mock_trade_repo):
    """Tests that HoldTarget Strategy does NOT move stop to breakeven."""
    trade = {
        "id": "H1",
        "entry_price": 100.0,
        "current_target": 130.0,
        "current_stop_loss": 90.0,
        "status": "ACTIVE",
        "entry_date": "2026-02-18",
    }
    # High 125, hasn't hit target
    candle = pd.Series(
        {
            "open": 105.0,
            "high": 125.0,
            "low": 104.0,
            "close": 120.0,
            "date": pd.Timestamp("2026-02-20"),
        },
        name=pd.Timestamp("2026-02-20"),
    )

    result = strategy.manage_active_trade(
        trade, pd.DataFrame([candle]), mock_trade_repo
    )

    assert result is None  # Should return None, no breakeven trigger
    mock_trade_repo.update_trade.assert_not_called()


def test_hold_target_check_entry_zero_price_or_early_date(
    strategy, mock_trade_repo
) -> None:
    """Tests check_entry returns None for zero entry price or candle date <= signal date."""
    candle = pd.Series(
        {"date": pd.Timestamp("2026-02-18"), "open": 100.0, "high": 105.0, "low": 95.0}
    )
    assert (
        strategy.check_entry(
            {"entry_price": 0.0}, candle, pd.DataFrame([candle]), mock_trade_repo
        )
        is None
    )

    trade_early = {
        "entry_price": 100.0,
        "signal_context": '{"date": "2026-02-18"}',
    }
    assert (
        strategy.check_entry(
            trade_early, candle, pd.DataFrame([candle]), mock_trade_repo
        )
        is None
    )


def test_hold_target_day_one_turnaround(strategy, mock_trade_repo) -> None:
    """Tests day 1 turnaround when filled and stopped on the same day."""
    trade = {
        "id": "HT_TURN",
        "entry_price": 100.0,
        "current_stop_loss": 90.0,
        "initial_size": 10,
        "current_size": 10,
        "signal_context": '{"date": "2026-02-17"}',
    }
    candle = pd.Series(
        {"date": pd.Timestamp("2026-02-18"), "open": 102.0, "high": 105.0, "low": 85.0},
        name=pd.Timestamp("2026-02-18"),
    )
    result = strategy.check_entry(
        trade, candle, pd.DataFrame([candle]), mock_trade_repo
    )
    assert result is not None
    assert "STOPPED" in result


def test_hold_target_manage_active_gap_down_stop_and_gap_up_target(
    strategy, mock_trade_repo
) -> None:
    """Tests manage_active_trade with gap down stop loss and gap up target."""
    # Gap down stop loss
    trade_stop = {
        "id": "H1",
        "entry_price": 150.0,
        "current_stop_loss": 140.0,
        "status": "ACTIVE",
        "entry_date": "2026-02-18",
    }
    gap_down_candle = pd.Series(
        {
            "open": 130.0,
            "high": 132.0,
            "low": 125.0,
            "close": 128.0,
            "date": pd.Timestamp("2026-02-19"),
        },
        name=pd.Timestamp("2026-02-19"),
    )
    strategy.manage_active_trade(
        trade_stop, pd.DataFrame([gap_down_candle]), mock_trade_repo
    )
    assert mock_trade_repo.update_trade.call_args[0][1]["exit_price"] == 130.0

    # Gap up target
    trade_target = {
        "id": "H2",
        "entry_price": 100.0,
        "current_target": 130.0,
        "status": "ACTIVE",
        "entry_date": "2026-02-18",
    }
    gap_up_candle = pd.Series(
        {
            "open": 135.0,
            "high": 140.0,
            "low": 130.0,
            "close": 138.0,
            "date": pd.Timestamp("2026-02-20"),
        },
        name=pd.Timestamp("2026-02-20"),
    )
    strategy.manage_active_trade(
        trade_target, pd.DataFrame([gap_up_candle]), mock_trade_repo
    )
    assert mock_trade_repo.update_trade.call_args[0][1]["exit_price"] == 135.0


def test_generate_orders_with_target(strategy, mock_trade_repo) -> None:
    """Tests generate_orders creates 2 exit legs when target is specified."""
    trade = {
        "symbol": "AAPL",
        "entry_price": 150.0,
        "current_stop_loss": 140.0,
        "current_target": 170.0,
        "current_size": 10,
    }
    order = strategy._generate_entry_order(trade, pd.DataFrame(), budget=2000.0)
    assert order is not None
    assert len(order.exits) == 2
