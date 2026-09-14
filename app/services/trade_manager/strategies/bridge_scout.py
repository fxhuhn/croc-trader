"""Bridge Scout Trade Manager Execution Strategy.

Execution Rules:
1. Entry (CREATED -> ACTIVE):
   - Limit On Close (LOC) on the setup day where Close <= req_close_rsi40.
   - Position size calculated from portfolio budget allocation.
2. Exits (ACTIVE -> CLOSED):
   - Exit on 1st trading day of new calendar month (DateYear(BarDate) != DateYear(EntryDate)
     or DateMonth(BarDate) != DateMonth(EntryDate)).
   - Market On Close (MOC) exit on that day.
"""

import logging
from typing import final, override

import pandas as pd

from ....const import ExitReason, Strategies
from ....models import Order, TradeParams
from ....types import TradeData
from ..types import TradeTransition
from .abstract import BaseTradeStrategy, OrderOptions

logger = logging.getLogger(__name__)


@final
class BridgeScoutTradeStrategy(BaseTradeStrategy):
    """Manages execution and exit lifecycle for the 'Bridge Scout' strategy.

    Rules:
    1. Entry: LOC entry executed on month-end setup close if Close <= req_close_rsi40.
    2. Exit: MOC exit executed on the 1st trading day of the new calendar month.
    """

    STRATEGY_IDENTIFIER = Strategies.BridgeScout
    name = Strategies.BridgeScout

    @override
    def get_current_parameters(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame | None = None,
    ) -> TradeParams | None:
        """Calculates current strategy parameters for display."""
        entry_price = float(trade.get("entry_price") or 0.0)

        return TradeParams(
            stop_loss=0.0,
            take_profit_1=0.0,
            extras={
                "entry_price": entry_price,
                "current_size": float(trade.get("current_size") or 0.0),
            },
        )

    @override
    def _generate_entry_order(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        budget: float,
        created_symbols: set[str] | None = None,
        reference_date: str | None = None,
    ) -> Order | None:
        """Generates LOC entry order for CREATED trades with threshold price."""
        threshold_price = self._extract_entry_price(trade)
        req_close_val = self._get_context_value(trade, "req_close_rsi40")
        if req_close_val is not None:
            try:
                parsed_threshold = float(req_close_val)
                if parsed_threshold > 0:
                    threshold_price = parsed_threshold
            except (ValueError, TypeError):
                pass

        if threshold_price <= 0:
            return None

        return self._generate_budget_entry_order(
            trade=trade,
            budget=budget,
            options=OrderOptions(
                order_type="LOC",
                time_in_force="DAY",
                price_override=threshold_price,
            ),
        )

    @override
    def _generate_exit_order(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        budget: float,
        created_symbols: set[str] | None = None,
        reference_date: str | None = None,
    ) -> Order | None:
        """Generates MOC exit orders for ACTIVE trades."""
        return self._generate_standard_exit_order(
            trade=trade,
            dataframe_history=dataframe_history,
            options=OrderOptions(order_type="MOC", time_in_force="DAY"),
        )

    @override
    def check_entry(
        self,
        trade: TradeData,
        candle: pd.Series,
        dataframe_history: pd.DataFrame,
        active_symbols: set[str] | None = None,
    ) -> TradeTransition | None:
        """Activates entry on month-end setup date MOC if Close <= req_close_rsi40.

        Invalidates setup if entry window is missed or condition fails.
        """
        raw_entry_price = trade.get("entry_price") or 0.0
        threshold_price = float(raw_entry_price)
        if threshold_price <= 0.0:
            return None

        current_close = float(candle["close"])
        candle_date = pd.Timestamp(candle["date"]).date()
        date_string = candle_date.strftime("%Y-%m-%d")

        setup_date = self._get_setup_date(trade)

        req_close_val = self._get_context_value(trade, "req_close_rsi40")
        if req_close_val is not None:
            try:
                threshold_price = float(req_close_val)
            except (ValueError, TypeError):
                pass

        if setup_date:
            if candle_date < setup_date:
                return None
            if candle_date > setup_date:
                return self._reject_setup(
                    trade,
                    date_string,
                    "Missed Entry Window (Bridge Scout MOC)",
                )

        if current_close <= threshold_price:
            return self._execute_activation(
                trade,
                current_close,
                "Bridge Scout MOC Entry",
                date_string,
            )

        return self._reject_setup(
            trade,
            date_string,
            f"Bridge Scout condition failed: Close {current_close:.2f} > Threshold {threshold_price:.2f}",
        )

    @override
    def _do_manage_active_trade(
        self,
        trade: TradeData,
        current_candle: pd.Series,
        date_string: str,
        dataframe_history: pd.DataFrame,
        latest_leaders: set[str] | None = None,
    ) -> TradeTransition | None:
        """Manages Exits for Bridge Scout:

        Exits MOC when current candle month/year differs from entry date month/year.
        """
        entry_date_string = trade.get("entry_date")
        if not entry_date_string or dataframe_history.empty:
            return None

        entry_date = pd.Timestamp(entry_date_string).date()
        current_date = pd.Timestamp(current_candle["date"]).date()

        if (
            current_date.year != entry_date.year
            or current_date.month != entry_date.month
        ):
            current_close = float(current_candle["close"])
            return self._close_trade(
                trade,
                current_close,
                ExitReason.TIME_STOP,
                date_string,
            )

        return None
