import logging
import json
import uuid
from typing import TypedDict, final, override

import pandas as pd

from ....database.repositories.trade import TradeRepository
from ....models import Order, OrderLeg, TradeParams
from ....tools.market_holidays import MarketHolidayChecker
from ....types import ExitReason, TradeData
from ....const import Strategies
from .abstract import BaseTradeStrategy

logger = logging.getLogger(__name__)


class TurnoverContext(TypedDict, total=False):
    """Context structure for Turnover Strategy signal data."""

    green_candle_count: int


@final
class TurnoverTimingStrategy(BaseTradeStrategy):
    """
    Manages execution for 'TurnoverTiming' Strategy.

    Rules:
    1. Entry: Limit Buy at specific level from signal.
    2. Exit:
       - Early: Two consecutive green candles (+ logic).
       - Time: Friday Close.
    """

    name: str = Strategies.TurnOverTiming
    """Unique identifier for the strategy."""

    DEFAULT_SLIPPAGE: float = 0.0
    """Default slippage applied to executions."""

    def __init__(self, strategy_name: str | None = None):
        if strategy_name:
            self.name = strategy_name

    def _is_green_candle(self, open_price: float, close_price: float) -> bool:
        """Determines if a candle is green (Close > Open)."""
        return close_price > open_price

    @override
    def get_current_params(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame | None = None,
        repository: TradeRepository | None = None,
    ) -> TradeParams:
        """Standardizes TradeParams for turnover strategy."""
        return TradeParams(
            stop_loss=0.0,
            take_profit_1=0.0,
            extras={
                "variant": trade.get("strategy", "STD"),
                "current_size": float(trade.get("current_size") or 0.0),
            },
        )

    @override
    def generate_orders(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        budget: float,
        repository: TradeRepository,
    ) -> Order | None:
        """Generates Entry or Exit Orders based on state."""
        status = trade.get("status")

        # 1. Entry Order (CREATED)
        if status == "CREATED":
            entry_price = float(trade.get("entry_price") or 0.0)
            if entry_price <= 0 or budget <= 0:
                return None

            quantity = int(budget / entry_price)
            if quantity < 1:
                return None

            return Order(
                id=str(uuid.uuid4()),
                symbol=trade["symbol"],
                quantity=quantity,
                mode="Entry",
                entry=OrderLeg(
                    action="BUY", type="LMT", price=entry_price, quantity=quantity
                ),
                exits=[],
            )

        # 2. Exit Order (ACTIVE)
        if status == "ACTIVE":
            raw_context = trade.get("signal_context") or "{}"
            context: TurnoverContext = json.loads(raw_context)
            green_candle_count = context.get("green_candle_count", 0)
            quantity = int(trade.get("current_size") or 0)

            if quantity <= 0:
                return None

            # a) Green Sequence Exit (TRIGGERED)
            if green_candle_count >= 2:
                return Order(
                    id=str(uuid.uuid4()),
                    symbol=trade["symbol"],
                    quantity=quantity,
                    mode="Exit",
                    entry=None,
                    exits=[
                        OrderLeg(
                            action="SELL", type="MKT", price=0.0, quantity=quantity
                        )
                    ],
                )

            # b) Friday Time Stop
            if not dataframe_history.empty:
                last_date = pd.Timestamp(dataframe_history.iloc[-1]["date"])
                day_of_week = last_date.dayofweek

                # Logic Clean-up:
                # We want to exit on Friday.
                # If we are generating orders based on THURSDAY data (day=3),
                # we generate an exit for Friday.
                # Note: Real execution will be Friday Open (MKT) or Close (MOC).
                # Assuming MKT for now as strict "Close" requires MOC support.
                if day_of_week == 3:  # Thursday
                    return Order(
                        id=str(uuid.uuid4()),
                        symbol=trade["symbol"],
                        quantity=quantity,
                        mode="Exit",
                        entry=None,
                        exits=[
                            OrderLeg(
                                action="SELL", type="MKT", price=0.0, quantity=quantity
                            )
                        ],
                    )

        return None

    @override
    def check_entry(
        self,
        trade: TradeData,
        candle: pd.Series,
        dataframe_history: pd.DataFrame,
        repository: TradeRepository,
    ) -> str | None:
        """Checks if the limit entry was reached on the NEXT trading day only."""
        limit_price = float(trade.get("entry_price") or 0.0)

        if limit_price <= 0:
            return None
        
        # Determine Timestamps
        current_date_ts = pd.Timestamp(candle["date"])
        current_date = str(current_date_ts.date())
        signal_date_ts = self._get_signal_date(trade)
        
        if not signal_date_ts:
            return None

        # Count trading days strictly after signal
        days_post_signal = self._get_trading_days_post_signal(trade, dataframe_history)
        
        # 1. Too early (Same day) -> count = 0
        if days_post_signal < 1:
            return None
            
        # 2. Too late (Expired) -> count > 1
        if days_post_signal > 1:
             return self._expire_trade(trade, repository, current_date)

        # 3. Check Fill on the next trading day (valid) -> count == 2
        low_price = float(candle["low"])
        open_price = float(candle["open"])
        
        if low_price <= limit_price:
            fill_price = (
                min(open_price, limit_price)
                if open_price < limit_price
                else limit_price
            )
            # Ensure fill_price is valid
            if fill_price <= 0:
                return None

            # Update green candle count if the entry day itself is green
            close_price = float(candle["close"])
            raw_context = trade.get("signal_context") or "{}"
            try:
                context: TurnoverContext = json.loads(raw_context)
            except json.JSONDecodeError:
                 context = {"green_candle_count": 0}
            
            # Use strict helper
            if self._is_green_candle(open_price, close_price):
                 count = context.get("green_candle_count", 0) + 1
            else:
                 count = 0

            # Update context correctly
            context["green_candle_count"] = count

            return self._execute_activation(
                trade,
                repository,
                fill_price,
                "LIMIT",
                current_date,
                extra_updates={"signal_context": json.dumps(context, default=str)},
            )

        return None

    @override
    def manage_active_trade(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        repository: TradeRepository,
    ) -> str | None:
        """Manages Exits: Multi-Day Green sequence (Next Open) or Time Stop (EOD)."""
        if dataframe_history.empty:
            return None

        current_candle = dataframe_history.iloc[-1]
        date_string = str(current_candle["date"])

        # 1. Signal-Specific Exit (State-Based Green Candles sequence)
        # Rule: If count >= 2, exit at current candle OPEN (Next Open Rule).
        # Otherwise, update count based on current candle color.

        raw_context = trade.get("signal_context") or "{}"
        try:
             context: TurnoverContext = json.loads(raw_context)
        except json.JSONDecodeError:
             context = {"green_candle_count": 0}

        green_candle_count = context.get("green_candle_count", 0)

        # Check for Exit Trigger (Next Open)
        if green_candle_count >= 2:
            return self._close_trade(
                trade,
                repository,
                float(current_candle["open"]),
                ExitReason.GREEN_SEQUENCE,
                date_string,
            )

        # Update Count for the NEXT day
        # Use strict helper
        if self._is_green_candle(float(current_candle["open"]), float(current_candle["close"])):
            green_candle_count += 1
        else:
            green_candle_count = 0

        # Persist Count back to context
        context["green_candle_count"] = green_candle_count
        repository.update_trade(
            trade["id"],
            {"signal_context": json.dumps(context, default=str)},
            reason=f"Update Green Candle Count: {green_candle_count}",
        )

        # 2. End of Week Time Stop (Friday or Holiday-Thursday Close)
        current_date_timestamp = pd.Timestamp(current_candle["date"])
        day_of_week = current_date_timestamp.dayofweek
        holiday_checker = MarketHolidayChecker()

        is_end_of_week = False
        if day_of_week == 4:  # Friday
            is_end_of_week = True
        elif day_of_week == 3:  # Thursday
            # If Friday is a holiday, Thursday is the end of the week
            tomorrow = current_date_timestamp + pd.Timedelta(days=1)
            # Use explicit date() for checker
            if holiday_checker.is_holiday(tomorrow.date()):
                is_end_of_week = True

        if is_end_of_week:
            return self._close_trade(
                trade,
                repository,
                float(current_candle["close"]),
                ExitReason.TIME_STOP,
                date_string,
            )

        return None