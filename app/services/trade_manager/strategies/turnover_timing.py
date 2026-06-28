import json
import logging
from decimal import Decimal
from typing import TypedDict, final, override

import pandas as pd

from ..types import TradeTransition
from ....models import Order, TradeParams
from ....tools.market_holidays import MarketHolidayChecker
from ....types import ExitReason, TradeData
from ....const import Strategies
from .abstract import BaseTradeStrategy

logger = logging.getLogger(__name__)


class TurnoverContext(TypedDict, total=False):
    """Context structure for Turnover Strategy signal data."""

    green_candle_count: int
    last_processed_date: str


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

    name = Strategies.TurnOverTiming
    """Unique identifier for the strategy."""

    DEFAULT_SLIPPAGE: float = 0.0
    """Default slippage applied to executions."""

    def __init__(self, strategy_name: str | None = None) -> None:
        """Initializes the strategy, optionally overriding the default name.

        Instantiates the holiday checker once to avoid per-call allocation
        on every manage_active_trade invocation.

        Args:
            strategy_name: Optional override for the strategy registry key.
        """
        super().__init__()
        if strategy_name:
            self.name = strategy_name
        self._holiday_checker = MarketHolidayChecker()

    def _is_green_candle(self, open_price: float, close_price: float) -> bool:
        """Determines if a candle is green (Close > Open).

        Args:
            open_price: The opening price of the candle.
            close_price: The closing price of the candle.

        Returns:
            bool: True if the closing price is greater than the opening price.
        """
        return close_price > open_price

    @override
    def get_current_parameters(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame | None = None,
    ) -> TradeParams:
        """Standardizes TradeParams for turnover strategy.

        Args:
            trade: The trade data dictionary.
            dataframe_history: Optional historical market data.

        Returns:
            TradeParams: Object containing strategy parameters.
        """
        return TradeParams(
            stop_loss=0.0,
            take_profit_1=0.0,
            extras={
                "variant": trade.get("strategy", "Standard"),
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
    ) -> Order | None:
        entry_price = self._extract_entry_price(trade)
        trade_budget = self._get_strategy_budget(trade, budget)
        if entry_price <= 0 or trade_budget <= 0:
            return None

        quantity = int(trade_budget / entry_price)
        if quantity < 1:
            return None

        return self._create_entry_order(
            trade["symbol"], quantity, Decimal(str(entry_price))
        )

    @override
    def _generate_exit_order(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        budget: float,
        created_symbols: set[str] | None = None,
    ) -> Order | None:
        context = self._get_full_context(trade)
        green_candle_count = context.get("green_candle_count", 0)
        quantity = int(trade.get("current_size") or 0)

        if quantity <= 0:
            return None

        # a) Green Sequence Exit (TRIGGERED)
        if green_candle_count >= 2:
            return self._create_exit_order(
                trade["symbol"],
                quantity,
                order_type="MKT",
                time_in_force="OPG",
            )

        # b) Friday Time Stop
        if not dataframe_history.empty:
            last_date = pd.Timestamp(dataframe_history.iloc[-1]["date"])
            day_of_week = last_date.dayofweek

            # Logic Clean-up:
            # We want to exit on Friday.
            # If we are generating orders based on THURSDAY data (day=3),
            # we generate an exit for Friday.
            # Note: Real execution will be Friday Close (MOC).
            if day_of_week == self.THURSDAY_INDEX:  # Thursday
                return self._create_exit_order(
                    trade["symbol"],
                    quantity,
                    order_type="MOC",
                    time_in_force="DAY",
                )

        return None

    @override
    def check_entry(
        self,
        trade: TradeData,
        candle: pd.Series,
        dataframe_history: pd.DataFrame,
        active_symbols: set[str] | None = None,
    ) -> TradeTransition | None:
        """Checks if the limit entry was reached on the NEXT trading day only.

        Args:
            trade: The trade data.
            candle: The current market candle.
            dataframe_history: Historical market data.
            active_symbols: Currently active symbols set.

        Returns:
            TradeTransition | None: Result transition or None if no entry occurred.
        """
        limit_price = self._extract_entry_price(trade)

        if limit_price <= 0:
            return None

        # Determine Timestamps
        current_date_timestamp = pd.Timestamp(candle["date"])
        current_date = str(current_date_timestamp.date())
        signal_date_timestamp = self._get_signal_date(trade)

        if not signal_date_timestamp:
            return None

        # Count trading days strictly after signal
        days_post_signal = self._get_trading_days_post_signal(trade, dataframe_history)

        # 1. Too early (Same day) -> count = 0
        if days_post_signal < 1:
            return None

        # 2. Evaluation for the first trading day after setup (Day 1)
        if days_post_signal == 1:
            return self._evaluate_day_one_entry(
                trade, candle, limit_price, current_date
            )

        # 3. Too late (Expired) -> count > 1
        return self._expire_trade(trade, current_date)

    def _evaluate_day_one_entry(
        self,
        trade: TradeData,
        candle: pd.Series,
        limit_price: float,
        current_date: str,
    ) -> TradeTransition | None:
        """Evaluates entry triggers strictly on the first day after the signal."""
        low_price = float(candle["low"])
        open_price = float(candle["open"])

        if low_price > limit_price:
            # Day 1 finished and price was never below limit -> Expire immediately!
            return self._expire_trade(trade, current_date)

        fill_price = (
            min(open_price, limit_price) if open_price < limit_price else limit_price
        )

        if fill_price <= 0:
            return None

        # Calculate initial green candle context
        close_price = float(candle["close"])
        context = self._get_full_context(trade)

        entry_is_green = self._is_green_candle(open_price, close_price)
        setup_was_green = context.get("setup_candle_green", False)

        context["green_candle_count"] = (
            2 if (entry_is_green and setup_was_green) else (1 if entry_is_green else 0)
        )
        context["last_processed_date"] = current_date

        return self._execute_activation(
            trade,
            fill_price,
            "LIMIT",
            current_date,
            extra_updates={
                "signal_context": json.dumps(context, default=str, ensure_ascii=False)
            },
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
        """Manages Exits: Multi-Day Green sequence (Next Open) or Time Stop (EOD)."""

        # 1. Signal-Specific Exit (State-Based Green Candles sequence)
        # Rule: If count >= 2, exit at current candle OPEN (Next Open Rule).
        context = self._get_full_context(trade)

        # Idempotency check: Do not process the same candle twice
        last_processed_date = context.get("last_processed_date")
        if last_processed_date == date_string:
            return None

        green_candle_count = context.get("green_candle_count", 0)

        # Check for Exit Trigger (Next Open)
        if green_candle_count >= 2:
            return self._close_trade(
                trade,
                float(current_candle["open"]),
                ExitReason.GREEN_SEQUENCE,
                date_string,
            )

        # 2. End of Week Time Stop (Friday or Holiday-Thursday Close)
        current_date_timestamp = pd.Timestamp(current_candle["date"])

        if self._is_end_of_trading_week(current_date_timestamp, self._holiday_checker):
            return self._close_trade(
                trade,
                float(current_candle["close"]),
                ExitReason.TIME_STOP,
                date_string,
            )

        # No exit occurred; get_daily_updates handles green candle count tracking.
        return None

    @override
    def get_daily_updates(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
    ) -> dict[str, object]:
        """Provides daily updates to the trade context, specifically tracking the green candle count."""
        if dataframe_history.empty:
            return {}

        current_candle = dataframe_history.iloc[-1]
        date_string = str(current_candle["date"])

        context = self._get_full_context(trade)

        # Avoid duplicate updates for the same day
        last_processed_date = context.get("last_processed_date")
        if last_processed_date == date_string:
            return {}

        green_candle_count = context.get("green_candle_count", 0)

        # Track consecutive green candles
        if self._is_green_candle(
            float(current_candle["open"]), float(current_candle["close"])
        ):
            green_candle_count += 1
        else:
            green_candle_count = 0

        return {
            "green_candle_count": green_candle_count,
            "last_processed_date": date_string,
        }
