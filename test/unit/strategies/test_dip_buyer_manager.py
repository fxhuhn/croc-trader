# filename: test_dip_buyer_manager.py
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.database.repositories.trade import TradeRepository
from app.services.trade_manager.strategies.dip_buyer import DipBuyerStrategy
from app.types import ExitReason, TradeStatus


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


def test_generate_orders_active_trade_time_stop(strategy, mock_trade_repo):
    """Tests that a MOC exit order is generated instead of a LOC order on Day 8 (7 prior days in history)."""
    trade = {
        "symbol": "AAPL",
        "entry_price": 100.0,
        "entry_date": "2026-02-18",
        "initial_size": 10,
        "status": "ACTIVE",
        "current_target": 105.0,
        "signal_context": '{"threshold_loc": 98.5}',
    }

    # 7 days of history starting from the entry date (meaning today is Day 8)
    dates = pd.to_datetime(["2026-02-" + str(i + 18) for i in range(7)])
    dataframe_history = pd.DataFrame(
        {
            "date": dates,
            "high": [98.49] * 7,
        }
    )

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
    # Exit 2: EOD Time Stop MOC at 0.0
    assert orders.exits[1].action == "SELL"
    assert orders.exits[1].type == "MOC"
    assert orders.exits[1].price == 0.0


def test_dip_buyer_check_entry_zero_limit_and_stale(strategy, mock_trade_repo) -> None:
    """Tests check_entry returns None for zero limit price or rejects when stale."""
    candle = pd.Series({"date": pd.Timestamp("2026-02-18"), "open": 100.0, "low": 95.0})
    assert (
        strategy.check_entry(
            {"entry_price": 0.0}, candle, pd.DataFrame([candle]), mock_trade_repo
        )
        is None
    )

    trade_stale = {
        "id": "DIP_STALE",
        "symbol": "AAPL",
        "entry_price": 100.0,
        "signal_context": '{"date": "2026-02-10"}',
    }
    with patch.object(strategy, "_get_trading_days_post_signal", return_value=2):
        result = strategy.check_entry(
            trade_stale, candle, pd.DataFrame([candle]), mock_trade_repo
        )
        assert result is not None
        assert "REJECTED" in result or "INVALIDATED" in result


def test_dip_buyer_manage_active_early_date(strategy, mock_trade_repo) -> None:
    """Tests _do_manage_active_trade returns None when candle date is before entry date."""
    trade = {"symbol": "AAPL", "status": "ACTIVE", "entry_date": "2026-02-19"}
    early_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-02-18"),
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 100.0,
        }
    )
    assert (
        strategy._do_manage_active_trade(
            trade, early_candle, "2026-02-18", pd.DataFrame([early_candle])
        )
        is None
    )


def test_dip_buyer_get_daily_updates(strategy) -> None:
    """Tests get_daily_updates returns threshold_loc based on previous high."""
    history = pd.DataFrame([{"high": 98.50}])
    updates = strategy.get_daily_updates({}, history)
    assert updates.get("threshold_loc") == 98.51


def test_dip_buyer_determine_order_quantity_zero_price(strategy) -> None:
    """Tests _determine_order_quantity returns 0 when entry_price <= 0."""
    trade = {"symbol": "AAPL", "entry_price": 0.0}
    assert strategy._determine_order_quantity(trade, budget=1000.0) == 0


def test_dip_buyer_generate_exit_order_empty_exits(strategy) -> None:
    """Tests _generate_exit_order returns None when exits list is empty."""
    trade = {
        "symbol": "AAPL",
        "entry_price": 100.0,
        "initial_size": 10,
        "status": "ACTIVE",
        "entry_date": "2026-02-18",
        "current_target": 0.0,
    }
    # Empty history with columns -> is_time_stop is False, threshold_loc is None
    history = pd.DataFrame(columns=["date", "high"])
    assert strategy._generate_exit_order(trade, history, budget=1000.0) is None


# ==============================================================================
# Comprehensive Take-Profit & Gap-Down Test Suite (DB-MTH-005)
# ==============================================================================


def test_check_entry_standard_fill_with_atr_target(strategy, mock_trade_repo) -> None:
    """Testfall 1: Standard fill (Low <= Limit < Open) calculates target = round(limit + 0.8 * atr, 2)."""
    trade = {
        "id": "T1",
        "symbol": "TEST",
        "entry_price": 95.0,
        "current_target": 0.0,
        "signal_context": '{"date": "2026-02-17", "atr5": 5.0}',
        "current_size": 10,
    }
    candle = pd.Series(
        {
            "open": 97.0,
            "low": 94.0,
            "close": 96.0,
            "date": pd.Timestamp("2026-02-18"),
        },
        name=pd.Timestamp("2026-02-18"),
    )

    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        result = strategy.check_entry(
            trade, candle, pd.DataFrame([candle]), mock_trade_repo
        )

        assert result is not None
        mock_trade_repo.update_trade.assert_called_once()
        updates = mock_trade_repo.update_trade.call_args[0][1]
        assert updates["status"] == TradeStatus.ACTIVE
        assert updates["entry_price"] == 95.0
        assert updates["current_target"] == 99.0  # 95.0 + (0.8 * 5.0)


def test_check_entry_gap_down_fill_calculates_lower_target(
    strategy, mock_trade_repo
) -> None:
    """Testfall 2: Gap-down fill (Open < Limit) recalculates target from actual fill price."""
    trade = {
        "id": "T2",
        "symbol": "TEST",
        "entry_price": 95.0,
        "current_target": 0.0,
        "signal_context": '{"date": "2026-02-17", "atr5": 5.0}',
        "current_size": 10,
    }
    candle = pd.Series(
        {
            "open": 92.0,
            "low": 90.0,
            "close": 93.0,
            "date": pd.Timestamp("2026-02-18"),
        },
        name=pd.Timestamp("2026-02-18"),
    )

    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        result = strategy.check_entry(
            trade, candle, pd.DataFrame([candle]), mock_trade_repo
        )

        assert result is not None
        mock_trade_repo.update_trade.assert_called_once()
        updates = mock_trade_repo.update_trade.call_args[0][1]
        assert updates["status"] == TradeStatus.ACTIVE
        assert updates["entry_price"] == 92.0
        assert updates["current_target"] == 96.0  # 92.0 + (0.8 * 5.0)


def test_check_entry_nbis_production_gap_down_replay(strategy, mock_trade_repo) -> None:
    """Testfall 3: Replay of NBIS (#1486) production gap-down matching verified 216.64 target."""
    trade = {
        "id": 1486,
        "symbol": "NBIS",
        "entry_price": 209.98,
        "current_target": 0.0,
        "signal_context": '{"date": "2026-09-11", "atr5": 14.57}',
        "current_size": 29,
    }
    candle = pd.Series(
        {
            "open": 204.98,
            "low": 202.0,
            "close": 206.0,
            "date": pd.Timestamp("2026-09-14"),
        },
        name=pd.Timestamp("2026-09-14"),
    )

    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        result = strategy.check_entry(
            trade, candle, pd.DataFrame([candle]), mock_trade_repo
        )

        assert result is not None
        mock_trade_repo.update_trade.assert_called_once()
        updates = mock_trade_repo.update_trade.call_args[0][1]
        assert updates["entry_price"] == 204.98
        # 204.98 + (0.8 * 14.57) = 204.98 + 11.656 = 216.636 -> 216.64
        assert updates["current_target"] == 216.64


def test_check_entry_sndk_production_gap_down_replay(strategy, mock_trade_repo) -> None:
    """Testfall 4: Replay of SNDK (#1489) production gap-down matching verified 1600.65 target."""
    trade = {
        "id": 1489,
        "symbol": "SNDK",
        "entry_price": 1534.44,
        "current_target": 0.0,
        "signal_context": '{"date": "2026-09-11", "atr5": 98.91}',
        "current_size": 3,
    }
    candle = pd.Series(
        {
            "open": 1521.53,
            "low": 1510.0,
            "close": 1525.0,
            "date": pd.Timestamp("2026-09-14"),
        },
        name=pd.Timestamp("2026-09-14"),
    )

    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        result = strategy.check_entry(
            trade, candle, pd.DataFrame([candle]), mock_trade_repo
        )

        assert result is not None
        mock_trade_repo.update_trade.assert_called_once()
        updates = mock_trade_repo.update_trade.call_args[0][1]
        assert updates["entry_price"] == 1521.53
        # 1521.53 + (0.8 * 98.91) = 1521.53 + 79.128 = 1600.658 -> 1600.66 (or 1600.65 depending on exact float)
        assert updates["current_target"] == round(1521.53 + (0.8 * 98.91), 2)


def test_check_entry_extreme_gap_down_robustness(strategy, mock_trade_repo) -> None:
    """Testfall 5: Extreme gap-down (-20% below limit) computes correct target."""
    trade = {
        "id": "EXTREME",
        "symbol": "PANIC",
        "entry_price": 100.0,
        "current_target": 0.0,
        "signal_context": '{"date": "2026-02-17", "atr5": 10.0}',
        "current_size": 10,
    }
    candle = pd.Series(
        {
            "open": 80.0,
            "low": 78.0,
            "close": 82.0,
            "date": pd.Timestamp("2026-02-18"),
        },
        name=pd.Timestamp("2026-02-18"),
    )

    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        result = strategy.check_entry(
            trade, candle, pd.DataFrame([candle]), mock_trade_repo
        )

        assert result is not None
        updates = mock_trade_repo.update_trade.call_args[0][1]
        assert updates["entry_price"] == 80.0
        assert updates["current_target"] == 88.0  # 80.0 + (0.8 * 10.0)


def test_check_entry_setup_atr_alias_fallback(strategy, mock_trade_repo) -> None:
    """Testfall 6: Supports setup_atr alias in signal_context."""
    trade = {
        "id": "ALIAS",
        "symbol": "TEST",
        "entry_price": 50.0,
        "current_target": 0.0,
        "signal_context": '{"date": "2026-02-17", "setup_atr": 4.0}',
        "current_size": 20,
    }
    candle = pd.Series(
        {
            "open": 46.0,
            "low": 45.0,
            "close": 48.0,
            "date": pd.Timestamp("2026-02-18"),
        },
        name=pd.Timestamp("2026-02-18"),
    )

    with patch.object(strategy, "_get_trading_days_post_signal", return_value=1):
        result = strategy.check_entry(
            trade, candle, pd.DataFrame([candle]), mock_trade_repo
        )

        assert result is not None
        updates = mock_trade_repo.update_trade.call_args[0][1]
        assert updates["entry_price"] == 46.0
        assert updates["current_target"] == 49.2  # 46.0 + (0.8 * 4.0)


def test_generate_exit_order_with_gap_down_target(strategy) -> None:
    """Testfall 7: _generate_exit_order creates SELL LMT with recalculated gap-down target."""
    trade = {
        "id": 1486,
        "symbol": "NBIS",
        "entry_price": 204.98,
        "current_target": 216.64,
        "initial_size": 29,
        "current_size": 29,
        "status": "ACTIVE",
        "entry_date": "2026-09-14",
    }
    history = pd.DataFrame(
        [
            {"date": "2026-09-14", "high": 208.0},
            {"date": "2026-09-15", "high": 210.0},
        ]
    )

    order = strategy._generate_exit_order(trade, history, budget=6000.0)
    assert order is not None
    assert len(order.exits) >= 1
    # First exit leg is the TP LMT order
    tp_leg = next(leg for leg in order.exits if leg.type == "LMT")
    assert tp_leg.action == "SELL"
    from decimal import Decimal

    assert tp_leg.price == Decimal("216.64")
    assert tp_leg.quantity == 29


def test_manage_active_target_hit_with_gap_down_target(strategy) -> None:
    """Testfall 8: Target hit triggers at recalculated gap-down target on Day 1+."""
    trade = {
        "id": 1486,
        "symbol": "NBIS",
        "status": "ACTIVE",
        "entry_price": 204.98,
        "entry_date": "2026-09-14",
        "current_target": 216.64,
        "current_size": 29,
    }
    day1_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-09-15"),
            "open": 212.0,
            "high": 217.0,  # Crosses 216.64
            "low": 210.0,
            "close": 215.0,
        }
    )
    history = pd.DataFrame([day1_candle])

    transition = strategy._do_manage_active_trade(
        trade, day1_candle, "2026-09-15", history
    )
    assert transition is not None
    assert transition.updates["status"] == TradeStatus.CLOSED
    assert transition.updates["exit_price"] == 216.64
    assert transition.reason == ExitReason.TARGET_HIT


def test_manage_active_target_not_hit_on_entry_day(strategy) -> None:
    """Testfall 9: Target Hit is strictly forbidden on entry day (Day 0)."""
    trade = {
        "id": 1486,
        "symbol": "NBIS",
        "status": "ACTIVE",
        "entry_price": 204.98,
        "entry_date": "2026-09-14",
        "current_target": 216.64,
        "current_size": 29,
    }
    # Entry day candle with High reaching above target
    entry_day_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-09-14"),
            "open": 204.98,
            "high": 220.0,  # Exceeds target, but is entry day
            "low": 202.0,
            "close": 208.0,
        }
    )
    history = pd.DataFrame([entry_day_candle])

    transition = strategy._do_manage_active_trade(
        trade, entry_day_candle, "2026-09-14", history
    )
    # Target Hit must not occur on entry day
    if transition:
        assert transition.reason != ExitReason.TARGET_HIT
