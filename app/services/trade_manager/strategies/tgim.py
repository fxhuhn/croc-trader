"""TGIM (Thank God It's Monday) Trade Manager Execution Strategy.

Execution Rules:
1. Entry (CREATED -> ACTIVE):
   - Market On Close (MOC) on Monday close (Bar 0).
   - Position size calculated from portfolio budget allocation.
2. Exits (ACTIVE -> CLOSED):
   - Bar 1 (Tuesday): Exit if Tuesday Close > Monday Close (MOC exit, ExitReason.TAKE_PROFIT).
   - Bar 2 (Wednesday): Exit if Wednesday Close > Tuesday Close (MOC exit, ExitReason.TAKE_PROFIT),
     otherwise Time Exit at Wednesday Close (MOC exit, ExitReason.TIME_STOP).
"""

import logging
from decimal import ROUND_FLOOR, Decimal
from typing import final, override

import pandas as pd

from ....const import ExitReason, Strategies
from ....models import Order, TradeParams
from ....types import TradeData
from ..types import TradeTransition
from .abstract import BaseTradeStrategy

logger = logging.getLogger(__name__)


def evaluate_tgim_exit(
    bars_held: int,
    current_close: Decimal,
    previous_close: Decimal,
) -> ExitReason | None:
    """Pure calculation: Evaluates TGIM exit logic without side effects.

    - Bar 1 (Tuesday): Take profit if current_close > previous_close.
    - Bar 2 (Wednesday): Take profit if current_close > previous_close, else Time Stop exit.
    """
    if bars_held < 1:
        return None

    if current_close > previous_close:
        return ExitReason.TAKE_PROFIT

    if bars_held >= 2:
        return ExitReason.TIME_STOP

    return None


def calculate_tgim_position_quantity(
    allocated_budget: Decimal,
    entry_price: Decimal,
) -> int:
    """Calculates integer share quantity using strict Decimal floor rounding."""
    if entry_price <= Decimal("0") or allocated_budget <= Decimal("0"):
        return 0

    raw_quantity = (allocated_budget / entry_price).quantize(
        Decimal("1"), rounding=ROUND_FLOOR
    )
    return int(raw_quantity)


@final
class TGIMTradeStrategy(BaseTradeStrategy):
    """Manages execution and exit lifecycle for the 'TGIM' strategy.

    Rules:
    1. Entry: MOC entry executed on Monday setup close.
    2. Exits:
       - Bar 1 (Tuesday): Exit if Tuesday Close > Monday Close.
       - Bar 2 (Wednesday): Exit if Wednesday Close > Tuesday Close,
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
        entry_price_val = trade.get("entry_price") or 0.0
        trade_budget_val = self._get_strategy_budget(trade, budget)

        entry_price = Decimal(str(entry_price_val))
        trade_budget = Decimal(str(trade_budget_val))

        quantity = calculate_tgim_position_quantity(trade_budget, entry_price)
        if quantity < 1:
            return None

        return self._create_entry_order(
            symbol=trade["symbol"],
            quantity=quantity,
            entry_price=entry_price,
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
        return self._generate_standard_exit_order(
            trade=trade,
            dataframe_history=dataframe_history,
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
        """Activates entry on Monday close (Bar 0) if Monday Close <= threshold.

        Invalidates setup if entry window is missed or condition fails.
        """
        threshold_price = Decimal(str(trade.get("entry_price") or 0.0))
        if threshold_price <= Decimal("0"):
            return None

        current_close = Decimal(str(candle["close"]))
        candle_date = pd.Timestamp(candle["date"]).date()
        date_string = candle_date.strftime("%Y-%m-%d")

        setup_date = self._get_setup_date(trade)
        if setup_date:
            if candle_date < setup_date:
                return None
            if candle_date > setup_date:
                return self._reject_setup(
                    trade,
                    date_string,
                    "Missed Entry Window (Monday Close)",
                )

        if current_close <= threshold_price:
            return self._execute_activation(
                trade,
                float(current_close),
                "Monday MOC Entry",
                date_string,
            )

        return self._reject_setup(
            trade,
            date_string,
            "Missed Entry Window (Monday Close)",
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
        """Manages Exits for TGIM:

        - Bar 1 (Tuesday): Exit if Close > Monday Close (ExitReason.TAKE_PROFIT).
        - Bar 2 (Wednesday): Exit if Close > Tuesday Close (ExitReason.TAKE_PROFIT) OR Time Exit (ExitReason.TIME_STOP).
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

        current_close = Decimal(str(current_candle["close"]))
        prev_candle = dataframe_history.iloc[-2]
        prev_close = Decimal(str(prev_candle["close"]))

        exit_reason = evaluate_tgim_exit(
            bars_held=bars_held,
            current_close=current_close,
            previous_close=prev_close,
        )

        if exit_reason is not None:
            return self._close_trade(
                trade,
                float(current_close),
                exit_reason,
                date_string,
            )

        return None
