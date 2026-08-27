"""Unit tests for edge cases across trade manager strategies to maximize branch coverage."""

import json
from decimal import Decimal

import pandas as pd
import pytest

from app.models import Order
from app.services.trade_manager.strategies.abstract import BaseTradeStrategy
from app.services.trade_manager.strategies.dip_buyer import DipBuyerStrategy
from app.services.trade_manager.strategies.hold_target import HoldTargetStrategy
from app.services.trade_manager.strategies.ndx_momentum import NDXMomentumTradeStrategy
from app.services.trade_manager.strategies.turnover_timing import TurnoverTimingStrategy
from app.services.trade_manager.strategies.two_percent_strategy import (
    TwoPercentStrategy,
)
from app.services.trade_manager.types import TradeTransition
from app.types import ExitReason, TradeStatus


class DummyConcreteStrategy(BaseTradeStrategy):
    """Concrete strategy for testing BaseTradeStrategy abstract methods."""

    name = "dummy"

    def get_current_parameters(self, trade, dataframe_history=None):
        return None

    def check_entry(self, trade, candle, dataframe_history, active_symbols=None):
        return None

    def _do_manage_active_trade(
        self, trade, current_candle, date_string, dataframe_history, latest_leaders=None
    ):
        return None

    def _generate_entry_order(
        self,
        trade,
        dataframe_history,
        budget,
        created_symbols=None,
        reference_date=None,
    ):
        return None

    def _generate_exit_order(
        self,
        trade,
        dataframe_history,
        budget,
        created_symbols=None,
        reference_date=None,
    ):
        return None


@pytest.fixture
def dummy_strategy() -> DummyConcreteStrategy:
    """Fixture for DummyConcreteStrategy."""
    return DummyConcreteStrategy()


def test_base_trade_strategy_create_entry_order(
    dummy_strategy: DummyConcreteStrategy,
) -> None:
    """Tests _create_entry_order constructs a valid Order object."""
    from app.services.trade_manager.strategies.abstract import OrderOptions

    order = dummy_strategy._create_entry_order(
        symbol="AAPL",
        quantity=10,
        entry_price=Decimal("150.0"),
        options=OrderOptions(order_type="LMT", time_in_force="GTC"),
    )
    assert isinstance(order, Order)
    assert order.symbol == "AAPL"
    assert order.quantity == 10
    assert order.entry is not None
    assert order.entry.type == "LMT"
    assert order.entry.price == Decimal("150.0")


def test_base_trade_strategy_create_exit_order(
    dummy_strategy: DummyConcreteStrategy,
) -> None:
    """Tests _create_exit_order constructs a valid exit Order object."""
    from app.services.trade_manager.strategies.abstract import OrderOptions

    order = dummy_strategy._create_exit_order(
        symbol="AAPL",
        quantity=10,
        price=Decimal("160.0"),
        options=OrderOptions(order_type="LMT", time_in_force="GTC"),
    )
    assert isinstance(order, Order)
    assert order.symbol == "AAPL"
    assert order.quantity == 10
    assert len(order.exits) > 0
    assert order.exits[0].type == "LMT"
    assert order.exits[0].action == "SELL"


def test_base_trade_strategy_execute_activation(
    dummy_strategy: DummyConcreteStrategy,
) -> None:
    """Tests _execute_activation creates an ACTIVE TradeTransition."""
    trade = {
        "id": 1,
        "symbol": "AAPL",
        "entry_price": 150.0,
        "initial_size": 10,
        "current_size": 10,
    }
    transition = dummy_strategy._execute_activation(
        trade, 152.0, "Limit Hit", "2026-07-20"
    )

    assert isinstance(transition, TradeTransition)
    assert transition.updates["status"] == TradeStatus.ACTIVE.value
    assert transition.updates["entry_price"] == 152.0
    assert transition.updates["entry_date"] == "2026-07-20"


def test_base_trade_strategy_close_trade(
    dummy_strategy: DummyConcreteStrategy,
) -> None:
    """Tests _close_trade creates a CLOSED TradeTransition with PnL."""
    trade = {"id": 1, "symbol": "AAPL", "entry_price": 150.0, "current_size": 10}
    transition = dummy_strategy._close_trade(
        trade, 160.0, ExitReason.TAKE_PROFIT.value, "2026-07-25"
    )
    assert isinstance(transition, TradeTransition)
    assert transition.updates["status"] == TradeStatus.CLOSED.value
    assert transition.updates["exit_reason"] == ExitReason.TAKE_PROFIT.value
    assert transition.updates["realized_pnl"] == 100.0


def test_base_trade_strategy_invalidate_trade(
    dummy_strategy: DummyConcreteStrategy,
) -> None:
    """Tests _invalidate_trade creates an INVALID TradeTransition."""
    trade = {"id": 1, "symbol": "AAPL"}
    transition = dummy_strategy._invalidate_trade(
        trade, low_price=90.0, stop_loss=95.0, date_string="2026-07-20"
    )
    assert isinstance(transition, TradeTransition)
    assert transition.updates["status"] == TradeStatus.INVALID.value
    assert transition.updates["exit_reason"] == ExitReason.INVALIDATED.value


def test_base_trade_strategy_generate_orders_invalid_status(
    dummy_strategy: DummyConcreteStrategy,
) -> None:
    """Tests generate_orders returns None when status is CLOSED or INVALIDATED."""
    trade_closed = {"symbol": "AAPL", "status": "CLOSED"}
    assert (
        dummy_strategy.generate_orders(trade_closed, pd.DataFrame(), budget=1000.0)
        is None
    )

    trade_invalid = {"symbol": "AAPL", "status": TradeStatus.INVALID}
    assert (
        dummy_strategy.generate_orders(trade_invalid, pd.DataFrame(), budget=1000.0)
        is None
    )


def test_base_trade_strategy_get_daily_updates_default(
    dummy_strategy: DummyConcreteStrategy,
) -> None:
    """Tests get_daily_updates default implementation returns empty dict."""
    assert dummy_strategy.get_daily_updates({}, pd.DataFrame()) == {}


def test_base_trade_strategy_get_strategy_budget(
    dummy_strategy: DummyConcreteStrategy,
) -> None:
    """Tests _get_strategy_budget uses trade budget, override budget, or config default."""
    trade_with_budget = {"budget": 5000.0}
    assert dummy_strategy._get_strategy_budget(trade_with_budget) == 5000.0

    trade_no_budget = {}
    assert (
        dummy_strategy._get_strategy_budget(trade_no_budget, override_budget=3000.0)
        == 3000.0
    )


def test_turnover_timing_strategy_edge_cases() -> None:
    """Tests TurnoverTimingStrategy parameters and order generation edge cases."""
    strategy = TurnoverTimingStrategy()

    # get_current_parameters
    trade = {"entry_price": 100.0, "current_size": 50}
    params = strategy.get_current_parameters(trade)
    assert params is not None
    assert params.extras["current_size"] == 50.0

    # _generate_entry_order edge cases
    df_empty = pd.DataFrame()
    assert (
        strategy._generate_entry_order({"entry_price": 0.0}, df_empty, budget=1000.0)
        is None
    )
    assert (
        strategy._generate_entry_order({"entry_price": 100.0}, df_empty, budget=10.0)
        is None
    )

    # _generate_exit_order edge cases
    assert (
        strategy._generate_exit_order({"current_size": 0}, df_empty, budget=1000.0)
        is None
    )
    assert (
        strategy._generate_exit_order({"current_size": 10}, df_empty, budget=1000.0)
        is None
    )


def test_hold_target_strategy_edge_cases() -> None:
    """Tests HoldTargetStrategy parameters and order generation edge cases."""
    strategy = HoldTargetStrategy()

    # get_current_parameters
    trade = {"entry_price": 100.0, "current_stop_loss": 90.0, "current_target": 110.0}
    params = strategy.get_current_parameters(trade)
    assert params is not None
    assert params.stop_loss == 90.0
    assert params.take_profit_1 == 110.0

    # _generate_entry_order edge cases
    df_empty = pd.DataFrame()
    assert (
        strategy._generate_entry_order({"entry_price": 0.0}, df_empty, budget=1000.0)
        is None
    )
    assert (
        strategy._generate_entry_order({"entry_price": 100.0}, df_empty, budget=10.0)
        is None
    )

    # _generate_exit_order edge cases
    assert (
        strategy._generate_exit_order({"current_size": 0}, df_empty, budget=1000.0)
        is None
    )


def test_dip_buyer_strategy_edge_cases() -> None:
    """Tests DipBuyerStrategy parameters and order generation edge cases."""
    strategy = DipBuyerStrategy()

    # get_current_parameters
    trade = {"entry_price": 100.0, "current_stop_loss": 90.0, "current_target": 110.0}
    params = strategy.get_current_parameters(trade)
    assert params is not None

    # _generate_entry_order edge cases
    df_empty = pd.DataFrame()
    assert (
        strategy._generate_entry_order({"entry_price": 0.0}, df_empty, budget=1000.0)
        is None
    )
    assert (
        strategy._generate_entry_order({"entry_price": 100.0}, df_empty, budget=10.0)
        is None
    )

    # _generate_exit_order edge cases
    assert (
        strategy._generate_exit_order({"current_size": 0}, df_empty, budget=1000.0)
        is None
    )


def test_two_percent_strategy_edge_cases() -> None:
    """Tests TwoPercentStrategy parameters and order generation edge cases."""
    strategy = TwoPercentStrategy()

    # get_current_parameters
    trade = {"entry_price": 100.0, "current_target": 110.0}
    params = strategy.get_current_parameters(trade)
    assert params is not None

    # _generate_entry_order edge cases
    df_empty = pd.DataFrame()
    assert (
        strategy._generate_entry_order({"entry_price": 0.0}, df_empty, budget=1000.0)
        is None
    )

    # _generate_exit_order edge cases
    assert (
        strategy._generate_exit_order({"current_size": 0}, df_empty, budget=1000.0)
        is None
    )


def test_ndx_momentum_strategy_edge_cases() -> None:
    """Tests NDXMomentumTradeStrategy parameters and order generation edge cases."""
    strategy = NDXMomentumTradeStrategy()

    # get_current_parameters
    trade = {"entry_price": 100.0, "current_stop_loss": 90.0, "current_target": 110.0}
    params = strategy.get_current_parameters(trade)
    assert params is not None

    # Duplicate active symbol check in check_entry
    dup_trade = {
        "id": 1,
        "symbol": "AAPL",
        "signal_context": json.dumps({"qqq_regime": "BULL"}),
    }
    candle = pd.Series({"date": "2026-07-20", "open": 150.0})
    transition = strategy.check_entry(
        dup_trade, candle, pd.DataFrame([candle]), active_symbols={"AAPL"}
    )
    assert transition is not None
    assert transition.updates["status"] == TradeStatus.INVALID.value

    # Month switch order error handling with bad dates
    bad_df = pd.DataFrame([{"date": "invalid_date"}])
    assert not strategy._is_month_switch_order(bad_df, reference_date="invalid_ref")

    # Extract latest leaders with empty symbol
    trades_with_empty_symbol = [
        {"symbol": "", "signal_context": json.dumps({"date": "2026-07-20"})}
    ]
    assert strategy.extract_latest_leaders(trades_with_empty_symbol) == set()

    # _generate_entry_order edge cases (empty history or zero closing price)
    df_empty = pd.DataFrame()
    assert (
        strategy._generate_entry_order({"entry_price": 0.0}, df_empty, budget=1000.0)
        is None
    )
    zero_price_df = pd.DataFrame([{"close": 0.0}])
    assert (
        strategy._generate_entry_order({"symbol": "AAPL"}, zero_price_df, budget=1000.0)
        is None
    )

    # _generate_exit_order edge cases
    assert (
        strategy._generate_exit_order({"current_size": 0}, df_empty, budget=1000.0)
        is None
    )
    valid_df = pd.DataFrame([{"date": "2026-06-30"}, {"date": "2026-07-01"}])
    assert (
        strategy._generate_exit_order(
            {"symbol": "AAPL", "current_size": 0}, valid_df, budget=1000.0
        )
        is None
    )


def test_base_trade_strategy_execute_immediate_loss(
    dummy_strategy: DummyConcreteStrategy,
) -> None:
    """Tests _execute_immediate_loss creates a CLOSED TradeTransition with STOP_LOSS reason."""
    trade = {
        "id": 1,
        "symbol": "AAPL",
        "entry_price": 150.0,
        "current_stop_loss": 140.0,
        "initial_size": 10,
        "current_size": 10,
    }
    transition = dummy_strategy._execute_immediate_loss(
        trade,
        fill_price=150.0,
        reason="Gap Down",
        stop_loss=140.0,
        date_string="2026-07-20",
    )
    assert isinstance(transition, TradeTransition)
    assert transition.updates["status"] == TradeStatus.CLOSED.value
    assert transition.updates["exit_reason"] == ExitReason.STOP_LOSS.value
    assert transition.updates["realized_pnl"] == -100.0


def test_base_trade_strategy_manage_active_trade_non_active(
    dummy_strategy: DummyConcreteStrategy,
) -> None:
    """Tests manage_active_trade returns None when trade status is CREATED or CLOSED."""
    created_trade = {"id": 1, "symbol": "AAPL", "status": TradeStatus.CREATED.value}
    assert dummy_strategy.manage_active_trade(created_trade, pd.DataFrame()) is None

    closed_trade = {"id": 1, "symbol": "AAPL", "status": TradeStatus.CLOSED.value}
    assert dummy_strategy.manage_active_trade(closed_trade, pd.DataFrame()) is None


def test_turnover_timing_active_trade_green_candles() -> None:
    """Tests TurnoverTimingStrategy exits after 2 consecutive green candles."""
    strategy = TurnoverTimingStrategy()
    trade = {
        "id": 1,
        "symbol": "QQQ",
        "status": TradeStatus.ACTIVE.value,
        "entry_price": 500.0,
        "entry_date": "2026-07-20",
        "current_size": 10,
        "signal_context": json.dumps({"green_candle_count": 2}),
    }
    day = {
        "date": pd.Timestamp("2026-07-21"),
        "open": 504.0,
        "high": 510.0,
        "low": 503.0,
        "close": 508.0,
    }
    df_history = pd.DataFrame([day])

    transition = strategy.manage_active_trade(trade, df_history)
    assert transition is not None
    assert transition.updates["status"] == TradeStatus.CLOSED.value
    assert transition.updates["exit_reason"] == ExitReason.GREEN_SEQUENCE.value


def test_turnover_timing_get_daily_updates() -> None:
    """Tests TurnoverTimingStrategy get_daily_updates increments green candle count."""
    strategy = TurnoverTimingStrategy()
    trade = {"id": 1, "symbol": "QQQ"}
    green_day = pd.Series(
        {
            "date": "2026-07-21",
            "open": 500.0,
            "high": 510.0,
            "low": 499.0,
            "close": 508.0,
        }
    )
    df_history = pd.DataFrame([green_day])

    updates = strategy.get_daily_updates(trade, df_history)
    assert updates.get("green_candle_count") == 1
    assert updates.get("last_processed_date") == "2026-07-21"
