"""Bridge Scout Trade Manager Execution Strategy.

Execution Rules:
1. Entry (CREATED -> ACTIVE):
   - Market On Close (MOC) on the setup day where entry conditions were met.
   - Position size calculated from portfolio budget allocation.
2. Exits (ACTIVE -> CLOSED):
   - Exit on 1st trading day of new calendar month (DateYear(BarDate) != DateYear(EntryDate)
     or DateMonth(BarDate) != DateMonth(EntryDate)).
   - Market On Close (MOC) exit on that day.
"""

import json
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
class BridgeScoutTradeStrategy(BaseTradeStrategy):
    """Manages execution and exit lifecycle for the 'Bridge Scout' strategy.

    Rules:
    1. Entry: MOC entry executed on month-end setup close.
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
        """Activates entry on setup date MOC."""
        entry_price = float(trade.get("entry_price") or 0.0)
        if entry_price <= 0:
            return None

        date_string = str(candle["date"])
        try:
            raw_ctx = trade.get("signal_context")
            ctx = json.loads(raw_ctx) if isinstance(raw_ctx, str) else (raw_ctx or {})
            if isinstance(ctx, dict) and ctx.get("date"):
                date_string = str(ctx["date"])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        return self._execute_activation(
            trade,
            entry_price,
            "Bridge Scout MOC Entry",
            date_string,
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
