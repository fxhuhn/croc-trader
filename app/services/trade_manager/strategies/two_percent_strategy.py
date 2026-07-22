import logging
from decimal import Decimal
from typing import final, override

import pandas as pd

from ....const import Strategies
from ....models import Order, TradeParams
from ....tools.market_holidays import MarketHolidayChecker
from ....types import ExitReason, TradeData
from ..types import TradeTransition
from .abstract import BaseTradeStrategy, HolidayCheckerProtocol

logger = logging.getLogger(__name__)


@final
class TwoPercentStrategy(BaseTradeStrategy):
    """
    Manages execution for 'TwoPercent' strategy.

    Rules:
    1. Entry: Limit Buy at Signal Price (Setup Close * 0.99).
       - Special Case: If Monday Open < Limit, Entry = Open.
    2. Exit:
       - Take Profit: Entry + 2%.
       - Timing: Take Profit ONLY active from Day + 1 (Tuesday).
       - Time Stop: End of Week (Friday Close) -> Market Exit.
    """

    STRATEGY_IDENTIFIER = Strategies.TwoPercent
    name = Strategies.TwoPercent
    REWARD_TARGET_MULTIPLIER = 1.02

    def __init__(self, holiday_checker: HolidayCheckerProtocol | None = None) -> None:
        """Initializes the strategy with optional holiday checking support.

        Args:
            holiday_checker: Optional holiday checking protocol instance.
        """
        super().__init__()
        self.holiday_checker = holiday_checker or MarketHolidayChecker()

    def _calculate_target_price(self, entry_price: Decimal) -> Decimal:
        """Calculates exact 2% take profit target using Decimal precision and banker's rounding."""
        if entry_price <= Decimal("0"):
            return Decimal("0.0")
        multiplier = Decimal(str(self.REWARD_TARGET_MULTIPLIER))
        return (entry_price * multiplier).quantize(Decimal("0.01"))

    @override
    def get_current_parameters(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame | None = None,
    ) -> TradeParams | None:
        """Calculates current strategy parameters for display."""
        entry_price = float(trade.get("entry_price") or 0.0)
        target_exit_price = (
            float(self._calculate_target_price(Decimal(str(entry_price))))
            if entry_price > 0
            else float(trade.get("current_target") or 0.0)
        )

        return TradeParams(
            stop_loss=0.0,
            take_profit_1=target_exit_price,
            extras={
                "entry_limit": entry_price,
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
        quantity = int(trade.get("current_size") or 0)
        if quantity <= 0 or dataframe_history.empty:
            return None

        time_stop_order = self._generate_time_stop_exit_order(
            trade, dataframe_history, self.holiday_checker
        )
        if time_stop_order is not None:
            return time_stop_order

        entry_date_str = trade.get("entry_date")
        if not entry_date_str:
            return None

        last_date = pd.Timestamp(dataframe_history.iloc[-1]["date"])
        next_day = last_date + pd.Timedelta(days=1)
        entry_date = pd.Timestamp(entry_date_str)

        if next_day.date() <= entry_date.date():
            return None

        entry_price = float(trade.get("entry_price") or 0.0)
        target_price = float(trade.get("current_target") or 0.0)
        if target_price <= 0 and entry_price > 0:
            target_price = float(
                self._calculate_target_price(Decimal(str(entry_price)))
            )

        if target_price <= 0:
            return None

        return self._create_exit_order(
            symbol=trade["symbol"],
            quantity=quantity,
            price=Decimal(str(target_price)),
            order_type="LMT",
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
        """
        Checks if the limit entry was filled on the current 'candle'.

        Args:
            trade: The trade record.
            candle: The current price candle.
            dataframe_history: Price history.
            active_symbols: Set of symbols with currently active positions.

        Returns:
            TradeTransition: Activation transition if filled, else None.
        """
        limit_price = float(trade.get("entry_price") or 0.0)
        if limit_price <= 0:
            return None

        # Candle Data
        open_price = float(candle["open"])
        low_price = float(candle["low"])
        date_string = str(candle["date"])
        current_date_obj = pd.Timestamp(candle["date"]).date()

        # 0. Session & Holiday Validation
        days_passed = self._get_trading_days_post_signal(trade, dataframe_history)
        if days_passed == 0:
            return None

        signal_date = self._get_signal_date(trade)
        if not signal_date:
            return None

        # Determine "Calendar Day 1" (Expected Monday)
        calendar_day_1 = (signal_date + pd.Timedelta(days=1)).date()
        if calendar_day_1.weekday() >= 5:  # Skip Weekend
            calendar_day_1 = (signal_date + pd.Timedelta(days=3)).date()

        is_today_holiday = self.holiday_checker.is_holiday(current_date_obj)
        was_day_1_holiday = self.holiday_checker.is_holiday(calendar_day_1)

        # 1. Day 1 Processing (Strict Entry Window)
        if days_passed == 1:
            return self._process_day_one_entry(
                trade, open_price, low_price, limit_price, date_string, is_today_holiday
            )

        # 2. Day 2 Processing (Only allowed if Day 1 was a holiday)
        if days_passed == 2 and was_day_1_holiday:
            return self._process_day_two_entry(
                trade, open_price, low_price, limit_price, date_string
            )

        # 3. Everything else: Too late
        return self._reject_setup(
            trade, date_string, "Missed Entry Window (Stale Signal)"
        )

    def _process_day_one_entry(
        self,
        trade: TradeData,
        open_price: float,
        low_price: float,
        limit_price: float,
        date_string: str,
        is_today_holiday: bool,
    ) -> TradeTransition | None:
        """Handles entry logic for the primary entry window (Day 1)."""
        if open_price < limit_price:
            target_price = float(self._calculate_target_price(Decimal(str(open_price))))
            return self._execute_activation(
                trade,
                open_price,
                "Gap Down (Open < Limit)",
                date_string,
                extra_updates={"current_target": target_price},
            )

        if low_price <= limit_price:
            target_price = float(
                self._calculate_target_price(Decimal(str(limit_price)))
            )
            return self._execute_activation(
                trade,
                limit_price,
                "Limit Hit",
                date_string,
                extra_updates={"current_target": target_price},
            )

        if is_today_holiday:
            return None

        return self._reject_setup(trade, date_string, "Missed Entry Window (Day 1)")

    def _process_day_two_entry(
        self,
        trade: TradeData,
        open_price: float,
        low_price: float,
        limit_price: float,
        date_string: str,
    ) -> TradeTransition | None:
        """Handles fallback entry logic if Day 1 was a holiday."""
        if open_price < limit_price:
            target_price = float(self._calculate_target_price(Decimal(str(open_price))))
            return self._execute_activation(
                trade,
                open_price,
                "Gap Down (Tuesday-after-Holiday)",
                date_string,
                extra_updates={"current_target": target_price},
            )

        if low_price <= limit_price:
            target_price = float(
                self._calculate_target_price(Decimal(str(limit_price)))
            )
            return self._execute_activation(
                trade,
                limit_price,
                "Limit Hit (Tuesday-after-Holiday)",
                date_string,
                extra_updates={"current_target": target_price},
            )

        return self._reject_setup(trade, date_string, "Missed Entry Window (Day 2)")

    @override
    def _do_manage_active_trade(
        self,
        trade: TradeData,
        current_candle: pd.Series,
        date_string: str,
        dataframe_history: pd.DataFrame,
        latest_leaders: set[str] | None = None,
    ) -> TradeTransition | None:
        """Manages Exits: Take Profit and Time Stop."""
        entry_price = float(trade.get("entry_price") or 0.0)
        entry_date_string = trade.get("entry_date")
        if not entry_date_string:
            return None

        entry_date_timestamp = pd.Timestamp(entry_date_string)
        current_date_timestamp = pd.Timestamp(current_candle["date"])

        # Target Calculation via Decimal precision
        target_exit_price = float(
            self._calculate_target_price(Decimal(str(entry_price)))
        )

        # Current Stats
        high_price = float(current_candle["high"])
        close_price = float(current_candle["close"])
        open_price = float(current_candle["open"])

        # 1. Take Profit Check (Only from Day + 1)
        # Using .days check for difference in calendar days.
        # Example: Entry Monday (Day 1), Target active from Tuesday (Day 2).
        total_days_since_entry = (
            current_date_timestamp.date() - entry_date_timestamp.date()
        ).days

        if total_days_since_entry >= 1:
            if high_price >= target_exit_price:
                # Benefit from gap ups above target
                exit_execution_price = max(open_price, target_exit_price)
                return self._close_trade(
                    trade,
                    exit_execution_price,
                    ExitReason.TARGET_HIT,
                    date_string,
                )

        # 2. Time Stop (Friday Close or Thursday if Friday is a holiday)
        if self._is_end_of_trading_week(current_date_timestamp, self.holiday_checker):
            return self._close_trade(
                trade, close_price, ExitReason.TIME_STOP, date_string
            )

        return None
