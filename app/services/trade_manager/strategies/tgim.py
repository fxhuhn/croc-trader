"""TGIM (Thank God It's Monday) Trade Manager Execution Strategy.

Execution Rules:
1. Entry (CREATED -> ACTIVE):
   - Market On Close (MOC) on Monday close (Bar 0).
   - Position size calculated from portfolio budget allocation.
2. Exits (ACTIVE -> CLOSED):
   - Bar 1 (Tuesday): c1exit if Tuesday Close > Monday Close (MOC exit, ExitReason.TAKE_PROFIT).
   - Bar 2 (Wednesday): c1exit if Wednesday Close > Tuesday Close (MOC exit, ExitReason.TAKE_PROFIT),
     otherwise TE (Time Exit) at Wednesday Close (MOC exit, ExitReason.TIME_STOP).
"""

import logging
from decimal import Decimal
from typing import final, override

import pandas as pd

from ....const import Strategies
from ....models import Order, TradeParams
from ....types import ExitReason, TradeData
from ..types import TradeTransition
from .abstract import BaseTradeStrategy

logger = logging.getLogger(__name__)


@final
class TGIMTradeStrategy(BaseTradeStrategy):
    """Manages execution and exit lifecycle for the 'TGIM' strategy.

    Rules:
    1. Entry: MOC entry executed on Monday setup close.
    2. Exits:
       - Bar 1 (Tuesday): c1exit if Tuesday Close > Monday Close.
       - Bar 2 (Wednesday): c1exit if Wednesday Close > Tuesday Close,
         otherwise Time Exit (TE) at Wednesday Close.
    """

    STRATEGY_IDENTIFIER = Strategies.TGIM
    name = Strategies.TGIM

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
        """Generates MOC entry order for CREATED trades."""
        entry_price = float(trade.get("entry_price") or 0.0)
        trade_budget = self._get_strategy_budget(trade, budget)

        if entry_price <= 0 or trade_budget <= 0:
            return None

        quantity = int(trade_budget / entry_price)
        if quantity < 1:
            return None

        return self._create_entry_order(
            symbol=trade["symbol"],
            quantity=quantity,
            entry_price=Decimal(str(entry_price)),
            order_type="MKT",
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
        """Generates exit orders for ACTIVE trades."""
        quantity = int(trade.get("current_size") or 0)
        if quantity <= 0 or dataframe_history.empty:
            return None

        last_candle = dataframe_history.iloc[-1]
        close_price = float(last_candle["close"])

        return self._create_exit_order(
            symbol=trade["symbol"],
            quantity=quantity,
            price=Decimal(str(close_price)),
            order_type="MKT",
            time_in_force="DAY",
        )

    @override
    def check_entry(
        self,
        trade: TradeData,
        candle: pd.Series,
        dataframe_history: pd.DataFrame,
        active_symbols: set[str] | None = None,
    ) -> TradeTransition | None:
        """Activates entry on Monday close (Bar 0) if Monday Close < threshold."""
        threshold_price = float(trade.get("entry_price") or 0.0)
        if threshold_price <= 0:
            return None

        current_close = float(candle["close"])
        date_string = str(candle["date"])

        # Check if Monday close meets the setup condition (at or below threshold)
        if current_close <= threshold_price:
            return self._execute_activation(
                trade,
                current_close,
                "Monday MOC Entry",
                date_string,
            )

        return None

    @override
    def _do_manage_active_trade(
        self,
        trade: TradeData,
        current_candle: pd.Series,
        date_string: str,
        dataframe_history: pd.DataFrame,
        latest_leaders: set[str] | None = None,
    ) -> TradeTransition | None:
        """Manages Exits for TGIM:

        - Bar 1 (Tuesday): Exit if Close > Monday Close (c1exit).
        - Bar 2 (Wednesday): Exit if Close > Tuesday Close (c1exit) OR Time Exit (TE).
        """
        entry_date_string = trade.get("entry_date")
        if not entry_date_string or dataframe_history.empty:
            return None

        entry_date = pd.Timestamp(entry_date_string).date()

        dates = pd.to_datetime(dataframe_history["date"]).dt.date
        history_from_entry = dataframe_history[dates >= entry_date]

        bars_held = len(history_from_entry) - 1

        if bars_held < 1:
            return None

        current_close = float(current_candle["close"])
        prev_candle = dataframe_history.iloc[-2]
        prev_close = float(prev_candle["close"])

        if current_close > prev_close:
            return self._close_trade(
                trade,
                current_close,
                ExitReason.TAKE_PROFIT,
                date_string,
            )

        if bars_held >= 2:
            return self._close_trade(
                trade,
                current_close,
                ExitReason.TIME_STOP,
                date_string,
            )

        return None
