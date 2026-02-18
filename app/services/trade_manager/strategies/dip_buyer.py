import logging
from typing import override, final
import pandas as pd

from ....types import ExitReason, TradeData
from ....const import Strategies
from ....models import TradeParams, Order, OrderLeg
from ....database.repositories.trade import TradeRepository
from .abstract import BaseTradeStrategy

logger = logging.getLogger(__name__)

@final
class DipBuyerStrategy(BaseTradeStrategy):
    """Dip Buyer Strategy: Enters on weakness via Limit Order.
    
    Attributes:
        name: Strategy identifier.
        DEFAULT_BUDGET: Default capital allocation for the trade.
        TIME_STOP_DAYS: Maximum holding period in trading days.
    """
    
    name = Strategies.DipBuyer
    DEFAULT_BUDGET: float = 2000.0
    TIME_STOP_DAYS: int = 8
    
    @override
    def get_current_params(
        self, 
        trade: TradeData, 
        dataframe_history: pd.DataFrame | None = None, 
        repository: TradeRepository | None = None
    ) -> TradeParams:
        """Extracts current strategy parameters for display.

        Args:
            trade: Current trade data.
            dataframe_history: Historical price data.
            repository: Trade database repository.

        Returns:
            TradeParams: Extracted parameters for UI display.
        """
        return TradeParams(
            stop_loss=0.0, # No stop loss
            take_profit_1=float(trade.get("current_target") or 0.0),
            extras={
                "entry_limit": float(trade.get("entry_price") or 0.0),
                "current_size": float(trade.get("current_size") or 0.0)
            }
        )

    @override
    def check_entry(
        self, 
        trade: TradeData, 
        candle: pd.Series, 
        dataframe_history: pd.DataFrame, 
        repository: TradeRepository
    ) -> str | None:
        """Checks if the limit entry was reached.

        Rules:
        1. Skip signal day (Day 0).
        2. Entry on Next Day (Day 1) via Limit Order.
        3. Invalidate if Day > 1.

        Args:
            trade: Current trade record.
            candle: Latest market candle.
            dataframe_history: Historical data for context.
            repository: Trade repository for updates.

        Returns:
            str | None: Description of transition if triggered.
        """
        limit_price = float(trade.get("entry_price") or 0.0)
        if limit_price <= 0:
            return None

        # 1. Date/Session Validation
        # Rules: Skip signal day (Day 0), Entry on Next Day (Day 1), Invalidate if Day > 1
        days_passed = self._get_trading_days_post_signal(trade, dataframe_history)
        
        if days_passed == 0:
            # Too early (Signal Day)
            return None
        
        date_string = str(candle["date"])
        if days_passed > 1:
            # Too late: Missed the entry window
            return self._reject_setup(
                trade, repository, date_string, "Missed Entry Window"
            )

        # 2. Check Fill (Day 1)
        low_price = float(candle["low"])
        open_price = float(candle["open"])

        if low_price > limit_price:
            # Missed entry on the target day -> Invalidate the setup
            return self._invalidate_trade(
                trade, repository, low_price, limit_price, date_string
            )

        # 3. Execution (with Gap Down benefit)
        fill_price = (
            min(open_price, limit_price) if open_price < limit_price else limit_price
        )

        return self._execute_activation(
            trade, repository, fill_price, "LIMIT", date_string
        )

    @override
    def manage_active_trade(
        self, 
        trade: TradeData, 
        dataframe_history: pd.DataFrame, 
        repository: TradeRepository
    ) -> str | None:
        """Manages exits: LOC, Target, and Time Stop.

        Exits:
        1. LOC (Limit On Close) - If Close > Previous Day High.
        2. Target (Take Profit) - Predefined target hit.
        3. Time Stop - Closed at end of day 8.

        Args:
            trade: Current active trade.
            dataframe_history: Historical price sequence.
            repository: Trade repository for updates.

        Returns:
            str | None: Description of transition if triggered.
        """
        if dataframe_history.empty:
            return None
        
        candle = dataframe_history.iloc[-1]
        date_string = str(candle['date'])
        current_date_obj = pd.Timestamp(candle['date'])
        
        # 1. Day Check
        entry_date_str = trade.get("entry_date")
        is_entry_day = False
        if entry_date_str:
            entry_date = pd.Timestamp(entry_date_str).date()
            if current_date_obj.date() < entry_date:
                return None
            is_entry_day = (current_date_obj.date() == entry_date)

        # 2. Target Logic (Take Profit)
        # Rule: Target can NOT be hit on entry day.
        target_price = float(trade.get('current_target') or 0.0)
        high_price = float(candle['high'])
        open_price = float(candle['open'])
        
        if not is_entry_day and target_price > 0 and high_price >= target_price:
            exit_price = max(open_price, target_price)
            return self._close_trade(
                trade, repository, exit_price, ExitReason.TARGET_HIT, date_string
            )

        # 3. LOC (Limit On Close) Logic
        # Rule: Only Limit on Close is possible for same day.
        if len(dataframe_history) >= 2:
            prev_candle = dataframe_history.iloc[-2]
            prev_high = float(prev_candle["high"])
            close_price = float(candle["close"])
            
            if close_price > prev_high:
                return self._close_trade(
                    trade, repository, close_price, "LOC_HIT", date_string
                )

        entry_date_str = trade.get("entry_date")
        if entry_date_str:
            # Count trading days since entry
            # User Request: Close AT the 8th trading day
            trading_days_held = len(
                dataframe_history[dataframe_history["date"] >= entry_date_str]
            )
            if trading_days_held >= self.TIME_STOP_DAYS:
                close_price = float(candle["close"])
                return self._close_trade(
                    trade,
                    repository,
                    close_price,
                    ExitReason.TIME_STOP,
                    date_string,
                )

        return None

    @override
    def generate_orders(
        self, 
        trade: TradeData, 
        dataframe_history: pd.DataFrame, 
        budget: float, 
        repository: TradeRepository
    ) -> Order | None:
        """Generates an Order object for IBKR export.

        Args:
            trade: Current trade data.
            dataframe_history: Historical price sequence.
            budget: Strategy allocation budget.
            repository: Trade database.

        Returns:
            Order | None: Order descriptor if valid.
        """
        symbol = trade.get('symbol', 'UNKNOWN')
        entry_price = float(trade.get('entry_price') or 0.0)
        
        if entry_price <= 0:
            return None

        # 1. Quantity Calculation
        db_size = float(trade.get('initial_size') or 0.0)
        if db_size > 0:
            quantity = int(db_size)
        else:
            trade_budget = float(trade.get('budget') or budget or self.DEFAULT_BUDGET)
            quantity = int(trade_budget / entry_price)
        
        if quantity <= 0:
            return None

        # 2. Construct Legs
        entry_leg = OrderLeg(
            action="BUY",
            type="LMT",
            price=entry_price,
            quantity=quantity,
            tif="DAY"
        )
        
        exits = []
        target_price = float(trade.get('current_target') or 0.0)
        if target_price > 0:
            exits.append(OrderLeg(
                action="SELL",
                type="LOC",
                price=target_price,
                quantity=quantity,
                tif="DAY"
            ))

        return Order(
            id=f"{symbol}_{self.name}",
            symbol=symbol,
            quantity=quantity,
            mode="BRACKET",
            entry=entry_leg,
            exits=exits,
            last_status="CREATED"
        )