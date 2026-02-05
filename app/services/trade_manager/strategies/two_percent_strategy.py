import logging
import pandas as pd
from typing import override
from dataclasses import dataclass
import uuid

from ....database.repositories.trade import TradeRepository
from ....types import Order, TradeParams, OrderLeg, OrderAction, OrderType, TradeStatus
from .abstract import BaseTradeStrategy

logger = logging.getLogger(__name__)

class TwoPercentStrategyManager(BaseTradeStrategy):
    """
    Manages execution for 'TwoPercentStrategy'.
    
    Rules:
    1. Entry: Limit Buy at Signal Price (Friday Close * 0.99).
       - Special Case: If Monday Open < Limit, Entry = Open.
    2. Exit:
       - Take Profit: Entry + 2%.
       - Timing: TP ONLY active from Day + 1 (Tuesday).
       - Time Stop: End of Week (Friday Close) -> Market Exit.
    """

    @override
    def generate_orders(
        self, 
        trade: dict, 
        df_history: pd.DataFrame, 
        budget: float, 
        repo: TradeRepository
    ) -> Order | None:
        # Simple Limit Buy Order for next day
        entry_price = trade.get('entry_price')
        if not entry_price or entry_price <= 0:
            return None
            
        qty = int(budget / entry_price)
        if qty < 1: return None

        # Create Entry Leg
        entry_leg = OrderLeg(
            action="BUY",
            type="LMT",
            price=entry_price,
            qty=qty
        )
        
        # Create Order Shell
        return Order(
            id=str(uuid.uuid4()),
            symbol=trade['symbol'],
            qty=qty,
            mode="Entry",
            entry=entry_leg,
            exits=[]
        )

    @override
    def check_entry(
        self,
        trade: dict,
        candle: pd.Series,
        df_history: pd.DataFrame,
        repo: TradeRepository,
    ) -> str | None:
        """
        Checks if the limit entry was filled on the current 'candle'.
        """
        limit_price = trade['entry_price']
        
        # Candle Data
        open_price = candle['open']
        low_price = candle['low']
        date_of_candle = candle.name # DateTime Index assumed from df_history
        
        # Check Fill
        filled_price = None
        
        # 1. Gap Down Check: Open below Limit -> Fill at Open
        if open_price < limit_price:
            filled_price = open_price
            reason = "Gap Down (Open < Limit)"
        
        # 2. Normal Check: Low below Limit -> Fill at Limit
        elif low_price <= limit_price:
            filled_price = limit_price
            reason = "Limit Hit"
            
        if filled_price:
            # Update Trade to ACTIVE
            repo.update_trade(trade['id'], {
                "status": TradeStatus.ACTIVE,
                "entry_price": filled_price, # Update valid fill price
                "entry_date": str(date_of_candle),
                "current_price": candle['close']
            }, reason=reason)
            return f"{reason} @ {filled_price}"
            
        return None

    @override
    def manage_active_trade(
        self, 
        trade: dict, 
        df_history: pd.DataFrame, 
        repo: TradeRepository
    ) -> str | None:
        """
        Manages Exits:
        - TP: Entry + 2% (Active from Entry Day + 1)
        - Time Exit: End of Week (Friday)
        """
        entry_price = trade['entry_price']
        entry_date = pd.Timestamp(trade['entry_date'])
        current_candle = df_history.iloc[-1]
        current_date = current_candle.name # Timestamp
        
        # Constants
        TARGET_PCT = 1.02
        target_price = round(entry_price * TARGET_PCT, 2)
        
        # Current Stats
        high_price = current_candle['high']
        close_price = current_candle['close']
        
        exit_price = None
        exit_reason = None
        
        # 1. Time Limit Check (End of Week)
        # If today is Friday (4) AND it's the end of the day (assuming we run this post-market or close to check)
        # OR if we just check "Is it Friday?" -> Trigger Market Close Exit
        # Logic says: "Take Profit ... or at the end of the week (Market on Close)"
        
        is_friday = (current_date.dayofweek == 4)
        days_since_entry = (current_date - entry_date).days
        
        # 2. Take Profit Check (Only from Day + 1)
        # "Take profit is set at the day after the entry"
        if days_since_entry >= 1:
            if high_price >= target_price:
                # Assuming we had a Limit Sell order sitting there
                exit_price = target_price
                exit_reason = "Take Profit (+2%)"
        
        # 3. Time Stop (Friday Close)
        # If NOT filled by TP (even on Friday, TP could hit first), close at Market
        if not exit_price and is_friday:
            exit_price = close_price
            exit_reason = "Time Stop (End of Week)"

        if exit_price:
            repo.update_trade(trade['id'], {
                "status": TradeStatus.CLOSED,
                "exit_price": exit_price,
                "exit_date": str(current_date),
                "exit_reason": exit_reason,
                "realized_pnl": (exit_price - entry_price) * trade['current_size']
            }, reason=exit_reason)
            return f"EXIT: {exit_reason} @ {exit_price}"
            
        return None

    @override
    def get_current_params(
        self, 
        trade: dict, 
        df_history: pd.DataFrame, 
        repo: TradeRepository
    ) -> TradeParams | None:
        return None # Not strictly required yet
