import logging
import uuid
import pandas as pd
from typing import override, final, Any

from ....database.repositories.trade import TradeRepository
from ....types import ExitReason, TradeData
from ....models import Order, TradeParams, OrderLeg
from ....tools.market_holidays import MarketHolidayChecker
from .abstract import BaseTradeStrategy
from ....const import Strategies

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
    REWARD_TARGET_MULTIPLIER = 1.02

    def __init__(self) -> None:
        """Initializes the strategy with holiday checking support."""
        self.holiday_checker = MarketHolidayChecker()

    @override
    def get_current_parameters(
        self, 
        trade: TradeData, 
        dataframe_history: pd.DataFrame | None = None, 
        repository: TradeRepository | None = None
    ) -> TradeParams | None:
        """
        Calculates current strategy parameters for display.

        Args:
            trade: The current trade record.
            dataframe_history: Optional historical price data.
            repository: Optional trade repository.

        Returns:
            TradeParams: Object containing stop loss and take profit levels.
        """
        entry_price = float(trade.get('entry_price') or 0.0)
        target_exit_price = round(
            entry_price * self.REWARD_TARGET_MULTIPLIER, 2
        ) if entry_price > 0 else 0.0
        
        return TradeParams(
            stop_loss=0.0,
            take_profit_1=target_exit_price,
            extras={
                "entry_limit": entry_price,
                "current_size": float(trade.get('current_size') or 0.0)
            }
        )

    @override
    def generate_orders(
        self, 
        trade: dict[str, Any], 
        dataframe_history: pd.DataFrame, 
        budget: float, 
        repository: TradeRepository
    ) -> Order | None:
        """
        Generates a Limit Buy Order for the entry setup.

        Args:
            trade: The current trade record.
            dataframe_history: Historical price data.
            budget: Allocated capital for this trade.
            repository: The trade repository.

        Returns:
            Order: The generated order shell, or None.
        """
        entry_price = float(trade.get('entry_price') or 0.0)
        if entry_price <= 0:
            return None
            
        quantity = int(budget / entry_price)
        if quantity < 1: 
            return None

        # Create Entry Leg
        entry_leg = OrderLeg(
            action="BUY",
            type="LMT",
            price=entry_price,
            quantity=quantity
        )
        
        # Create Order Shell
        return Order(
            id=str(uuid.uuid4()),
            symbol=trade['symbol'],
            quantity=quantity,
            mode="Entry",
            entry=entry_leg,
            exits=[]
        )

    @override
    def check_entry(
        self,
        trade: dict[str, Any],
        candle: pd.Series,
        dataframe_history: pd.DataFrame,
        repository: TradeRepository,
    ) -> str | None:
        """
        Checks if the limit entry was filled on the current 'candle'.

        Args:
            trade: The trade record.
            candle: The current price candle.
            dataframe_history: Price history.
            repository: The trade repository.

        Returns:
            str: Activation message if filled, else None.
        """
        limit_price = float(trade.get('entry_price') or 0.0)
        if limit_price <= 0:
            return None
        
        # Candle Data
        open_price = float(candle['open'])
        low_price = float(candle['low'])
        date_string = str(candle['date'])
        current_date_obj = pd.Timestamp(candle['date']).date()

        # 0. Session & Holiday Validation
        days_passed = self._get_trading_days_post_signal(trade, dataframe_history)
        if days_passed == 0:
            return None
        
        signal_date = self._get_signal_date(trade)
        if not signal_date:
            return None

        # Determine "Calendar Day 1" (Expected Monday)
        calendar_day_1 = (signal_date + pd.Timedelta(days=1)).date()
        if calendar_day_1.weekday() >= 5: # Skip Weekend
            calendar_day_1 = (signal_date + pd.Timedelta(days=3)).date()
            
        is_today_holiday = self.holiday_checker.is_holiday(current_date_obj)
        was_day_1_holiday = self.holiday_checker.is_holiday(calendar_day_1)

        # 1. Day 1 Processing (Strict Entry Window)
        if days_passed == 1:
            # Check Fill (Gap Down)
            if open_price < limit_price:
                return self._execute_activation(
                    trade, repository, open_price, "Gap Down (Open < Limit)", date_string
                )
            
            # Check Fill (Limit Hit)
            if low_price <= limit_price:
                return self._execute_activation(
                    trade, repository, limit_price, "Limit Hit", date_string
                )
            
            # No Fill on Day 1:
            # If it's a holiday, keep alive for Tuesday.
            if is_today_holiday:
                return None
                
            # Not a holiday and no fill -> Invalidate
            return self._reject_setup(
                trade, repository, date_string, "Missed Entry Window (Day 1)"
            )

        # 2. Day 2 Processing (Only allowed if Day 1 was a holiday)
        if days_passed == 2 and was_day_1_holiday:
            # Check Fill (Gap Down)
            if open_price < limit_price:
                return self._execute_activation(
                    trade, 
                    repository, 
                    open_price, 
                    "Gap Down (Tuesday-after-Holiday)", 
                    date_string
                )
            
            # Check Fill (Limit Hit)
            if low_price <= limit_price:
                return self._execute_activation(
                    trade, 
                    repository, 
                    limit_price, 
                    "Limit Hit (Tuesday-after-Holiday)", 
                    date_string
                )
            
            # No fill on Tuesday after Monday holiday -> Invalidate
            return self._reject_setup(
                trade, repository, date_string, "Missed Entry Window (Day 2)"
            )

        # 3. Everything else: Too late
        return self._reject_setup(
            trade, repository, date_string, "Missed Entry Window (Stale Signal)"
        )

    @override
    def manage_active_trade(
        self, 
        trade: dict[str, Any], 
        dataframe_history: pd.DataFrame, 
        repository: TradeRepository
    ) -> str | None:
        """
        Manages Exits: Take Profit and Time Stop.

        Args:
            trade: The active trade record.
            dataframe_history: Historical price data.
            repository: The trade repository.

        Returns:
            str: Close message if exit triggered, else None.
        """
        entry_price = float(trade.get('entry_price') or 0.0)
        entry_date_string = trade.get('entry_date')
        if not entry_date_string:
            return None
            
        entry_date_timestamp = pd.Timestamp(entry_date_string)
        current_candle = dataframe_history.iloc[-1]
        current_date_timestamp = pd.Timestamp(current_candle['date'])
        date_string = str(current_candle['date'])
        
        # Target Calculation
        target_exit_price = round(entry_price * self.REWARD_TARGET_MULTIPLIER, 2)
        
        # Current Stats
        high_price = float(current_candle['high'])
        close_price = float(current_candle['close'])
        open_price = float(current_candle['open'])
        
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
                    repository, 
                    exit_execution_price, 
                    ExitReason.TARGET_HIT, 
                    date_string
                )
        
        # 2. Time Stop (Friday Close or Thursday if Friday is a holiday)
        is_end_of_week = False
        if current_date_timestamp.dayofweek == 4: # Friday
            is_end_of_week = True
        elif current_date_timestamp.dayofweek == 3: # Thursday
            next_day = current_date_timestamp + pd.Timedelta(days=1)
            if self.holiday_checker.is_holiday(next_day.date()):
                 is_end_of_week = True
                 
        if is_end_of_week:
            return self._close_trade(
                trade, 
                repository, 
                close_price, 
                ExitReason.TIME_STOP, 
                date_string
            )
            
        return None
