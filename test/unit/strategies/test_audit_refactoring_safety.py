# filename: test_audit_refactoring_safety.py
"""
Pre-Refactoring Safety Net Test Suite for BaseTradeStrategy and TwoPercentStrategy.
Validates exact financial target calculation, exception logging, and strategy contract stability.
"""

import json
import logging
from unittest.mock import MagicMock

import pandas as pd

from app.services.trade_manager.strategies.abstract import BaseTradeStrategy
from app.services.trade_manager.strategies.two_percent_strategy import (
    TwoPercentStrategy,
)
from app.types import ExitReason, TradeStatus


class DummyStrategy(BaseTradeStrategy):
    """Minimal implementation of BaseTradeStrategy for testing base helpers."""

    name = "Dummy"

    def check_entry(
        self,
        trade: dict,
        candle: pd.Series,
        dataframe_history: pd.DataFrame,
        active_symbols: set[str] | None = None,
    ):
        return None

    def _do_manage_active_trade(
        self,
        trade: dict,
        current_candle: pd.Series,
        date_string: str,
        dataframe_history: pd.DataFrame,
        latest_leaders: set[str] | None = None,
    ):
        return None

    def _generate_entry_order(
        self,
        trade: dict,
        dataframe_history: pd.DataFrame,
        budget: float,
        created_symbols: set[str] | None = None,
        reference_date: str | None = None,
    ):
        return None

    def _generate_exit_order(
        self,
        trade: dict,
        dataframe_history: pd.DataFrame,
        budget: float,
        created_symbols: set[str] | None = None,
        reference_date: str | None = None,
    ):
        return None

    def get_current_parameters(
        self, trade: dict, dataframe_history: pd.DataFrame | None = None
    ):
        return None


# --- TESTS FOR BASE TRADE STRATEGY (abstract.py) ---


def test_base_strategy_get_context_value_logs_warning_on_invalid_json(caplog):
    """Ensures corrupt signal_context logs a warning instead of swallowing silently."""
    strategy = DummyStrategy()
    trade = {"id": 42, "signal_context": "{invalid_json}"}

    with caplog.at_level(logging.WARNING):
        val = strategy._get_context_value(trade, "date")

    assert val is None
    assert (
        "Failed to parse signal_context" in caplog.text
        or "42" in caplog.text
        or len(caplog.records) > 0
    )


def test_base_strategy_close_trade_pnl_long_and_short():
    """Validates exact PnL computation for long and short trade exits."""
    strategy = DummyStrategy()

    long_trade = {
        "id": 1,
        "entry_price": 100.0,
        "current_size": 10,
        "realized_pnl": 0.0,
        "signal_context": json.dumps({"direction": "long"}),
    }
    transition_long = strategy._close_trade(
        long_trade,
        exit_price=105.0,
        reason=ExitReason.TARGET_HIT,
        date_string="2026-02-02",
    )
    assert transition_long.updates["realized_pnl"] == 50.0

    short_trade = {
        "id": 2,
        "entry_price": 100.0,
        "current_size": 10,
        "realized_pnl": 0.0,
        "signal_context": json.dumps({"direction": "short"}),
    }
    transition_short = strategy._close_trade(
        short_trade,
        exit_price=95.0,
        reason=ExitReason.TARGET_HIT,
        date_string="2026-02-02",
    )
    assert transition_short.updates["realized_pnl"] == 50.0


def test_base_strategy_is_end_of_trading_week_holiday_protocol():
    """Validates end-of-week detection with a mock holiday checker implementing Protocol."""
    strategy = DummyStrategy()
    friday = pd.Timestamp("2026-02-06")  # Friday
    thursday = pd.Timestamp("2026-02-05")  # Thursday

    mock_checker = MagicMock()
    mock_checker.is_holiday.side_effect = lambda d: (
        d.strftime("%Y-%m-%d") == "2026-02-06"
    )

    assert strategy._is_end_of_trading_week(friday, mock_checker) is True
    assert strategy._is_end_of_trading_week(thursday, mock_checker) is True


# --- TESTS FOR TWO PERCENT STRATEGY (two_percent_strategy.py) ---


def test_two_percent_target_price_precision():
    """Validates exact target price calculation for fractional entry prices."""
    strategy = TwoPercentStrategy()

    trade = {
        "id": 10,
        "entry_price": 10.33,
        "current_target": 0.0,
        "current_size": 100,
    }
    params = strategy.get_current_parameters(trade)
    # 10.33 * 1.02 = 10.5366 -> round to 2 decimal places = 10.54
    assert params.take_profit_1 == 10.54


def test_two_percent_generate_exit_order_early_returns():
    """Ensures empty dataframe or missing entry date results in None without crashing."""
    strategy = TwoPercentStrategy()
    trade = {
        "symbol": "AAPL",
        "current_size": 100,
        "entry_price": 100.0,
        "current_target": 102.0,
    }

    # Empty dataframe
    exit_order = strategy._generate_exit_order(trade, pd.DataFrame(), budget=2000.0)
    assert exit_order is None

    # Missing entry date
    df_history = pd.DataFrame([{"date": "2026-02-02"}])
    exit_order_no_date = strategy._generate_exit_order(trade, df_history, budget=2000.0)
    assert exit_order_no_date is None


def test_two_percent_day_one_entry_target_calculation():
    """Verifies target price calculation during Day 1 entry activation."""
    strategy = TwoPercentStrategy()
    trade = {
        "id": 20,
        "symbol": "MSFT",
        "entry_price": 200.0,
        "current_size": 0,
        "budget": 2000.0,
        "signal_context": json.dumps({"date": "2026-02-01"}),
    }

    # Open below limit -> Limit set to Open price
    transition = strategy._process_day_one_entry(
        trade=trade,
        open_price=195.0,
        low_price=194.0,
        limit_price=200.0,
        date_string="2026-02-02",
        is_today_holiday=False,
    )
    assert transition is not None
    assert transition.updates["status"] == TradeStatus.ACTIVE
    assert transition.updates["entry_price"] == 195.0
    # Target = 195.0 * 1.02 = 198.9
    assert transition.updates["current_target"] == 198.9
