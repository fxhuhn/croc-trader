import json
import logging
from typing import override, final, Optional

import pandas as pd

from ....types import TradeStatus, TradeParams, EntryReason, ExitReason, Order, OrderLeg, TradeData
from ....database.repositories.trade import TradeRepository
from .abstract import BaseTradeStrategy

logger = logging.getLogger(__name__)

@final
class HoldTargetStrategy(BaseTradeStrategy):
    """
    Manager für Croc Breakouts (Hold/TP3).
    Ziel: Große Trends reiten (3R oder mehr) mit Breakout-Entry.
    """
    name = "HoldTarget"

    @override
    def get_current_params(
        self,
        trade: TradeData,
        dataframe_history: Optional[pd.DataFrame] = None,
        repository: Optional[TradeRepository] = None
    ) -> TradeParams:
        dict_trade = dict(trade)
        return TradeParams(
            stop_loss=float(dict_trade.get('current_stop_loss') or 0.0),
            tp_1=float(dict_trade.get('current_target') or 0.0),
            extras={
                "entry_limit": float(dict_trade.get('entry_price') or 0.0),
                "current_size": float(dict_trade.get('current_size') or 0.0)
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
        """
        Prüft auf Breakout-Entry (Stop Buy) und validiert vorher den Stop Loss.
        """
        dict_trade = dict(trade)
        symbol = dict_trade.get('symbol', 'UNKNOWN')
        entry_price = float(dict_trade.get('entry_price') or 0.0)
        stop_loss = float(dict_trade.get('current_stop_loss') or 0.0)
        
        # 1. Pre-Flight Checks
        if entry_price <= 0:
            return None

        current_date_obj = pd.Timestamp(candle['date'])
        
        # Entry darf erst NACH dem Signal-Datum erfolgen
        signal_date = self._get_signal_date(dict_trade)
        if signal_date and current_date_obj.date() <= signal_date.date():
            return None

        # 2. Market Data
        high_price = float(candle['high'])
        low_price = float(candle['low'])
        open_price = float(candle['open'])
        date_str = str(candle['date'])

        # 3. Setup Validation (Invalidation check before entry)
        # Rule: If Low <= Stop Loss, the setup is invalidated (too much volatility)
        if stop_loss > 0 and low_price <= stop_loss:
            return self._invalidate_trade(dict_trade, repository, low_price, stop_loss, date_str)

        # 4. Entry Logic (Stop Buy)
        # Check Gap Up or Intraday Breakout
        filled, fill_price, reason = False, 0.0, ""

        if open_price >= entry_price:
            # Gapped over Entry
            filled, fill_price, reason = True, open_price, EntryReason.GAP_UP
        elif high_price >= entry_price:
            # Intraday touched Entry
            filled, fill_price, reason = True, entry_price, EntryReason.BREAKOUT

        if not filled:
            return None

        # 5. Execution
        return self._execute_activation(dict_trade, repository, fill_price, reason, date_str)

    @override
    def manage_active_trade(
        self, 
        trade: TradeData, 
        dataframe_history: pd.DataFrame, 
        repository: TradeRepository
    ) -> str | None:
        """
        Prüft Exits für aktive Trades: Stop Loss vs. Target.
        """
        if dataframe_history.empty:
            return None

        dict_trade = dict(trade)
        candle = dataframe_history.iloc[-1]
        
        current_date_obj = pd.Timestamp(candle['date'])
        
        # Sanity Check: Exit cannot be before Entry
        entry_date_str = dict_trade.get('entry_date')
        if entry_date_str:
            entry_date = pd.Timestamp(entry_date_str)
            if current_date_obj.date() < entry_date.date():
                return None

        # values
        stop_loss = float(dict_trade.get('current_stop_loss') or 0.0)
        target = float(dict_trade.get('current_target') or 0.0)
        
        # Market Data
        low_price = float(candle['low'])
        high_price = float(candle['high'])
        open_price = float(candle['open'])
        date_str = str(candle['date'])

        # 1. Stop Loss Logic (Pessimistic: Check first)
        # Handle Gap Down below Stop -> Execute at Open
        if stop_loss > 0 and low_price <= stop_loss:
            exit_price = stop_loss
            if open_price < stop_loss:
                exit_price = open_price # Gap execution
            
            return self._close_trade(dict_trade, repository, exit_price, ExitReason.STOP_LOSS, date_str)

        # 2. Target Logic
        # Handle Gap Up above Target -> Execute at Open
        if target > 0 and high_price >= target:
            exit_price = target
            if open_price > target:
                exit_price = open_price # Gap execution (Benefit)

            return self._close_trade(dict_trade, repository, exit_price, ExitReason.TARGET_HIT, date_str)

        return None

    @override
    def generate_orders(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        budget: float,
        repository: TradeRepository
    ) -> Order | None:
        """
        Erstellt ein Order-Objekt für den IBKR-Export.
        Logik: Stop Buy (Breakout) + Bracket (Stop Loss, optional Target).
        """
        symbol = trade.get('symbol', 'UNKNOWN')
        entry_price = float(trade.get('entry_price') or 0.0)
        stop_loss = float(trade.get('current_stop_loss') or 0.0)
        
        if entry_price <= 0:
            logger.warning(f"[{symbol}] Cannot generate order: Invalid Entry Price {entry_price}")
            return None

        # 1. Quantity Calculation
        # Check explicit size first
        db_size = float(trade.get('initial_size') or 0.0)
        
        if db_size > 0:
            qty = int(db_size)
        elif stop_loss > 0 and entry_price > stop_loss:
            # Risk Based Calculation
            risk_amount = float(trade.get('risk_amount') or 100.0)
            risk_per_share = entry_price - stop_loss
            qty = int(risk_amount / risk_per_share)
            logger.info(f"[{symbol}] Calculated Qty via Risk ({risk_amount}): {qty} (Risk/Share: {risk_per_share:.2f})")
        else:
            logger.warning(f"[{symbol}] Cannot allow Order Gen: Unknown Size and invalid SL/Risk setup.")
            return None

        if qty <= 0:
            logger.warning(f"[{symbol}] Calculated Quantity is 0.")
            return None

        # 2. Entry: STOP BUY
        entry_leg = OrderLeg(
            action="BUY",
            type="STP",
            price=entry_price,
            # qty handled by parent
            tif="DAY"
        )
        
        exits = []
        
        # Exit 1: Stop Loss (Mandatory for this strategy)
        if stop_loss > 0:
            exits.append(OrderLeg(
                action="SELL",
                type="STP",
                price=stop_loss,
                qty=qty,
                tif="GTC"
            ))
        else:
            logger.warning(f"[{symbol}] Order Gen without Stop Loss? Unsafe for HoldTarget.")
            # We proceed ONLY if Target is there? No, Strategy requires SL.
            return None

        # Exit 2: Target (Optional)
        target_price = float(trade.get('current_target') or 0.0)
        if target_price > 0:
            exits.append(OrderLeg(
                action="SELL",
                type="LMT",
                price=target_price,
                qty=qty,
                tif="GTC"
            ))

        # 3. Assemble
        order_id = f"{symbol}_{self.name}"
        
        return Order(
            id=order_id,
            symbol=symbol,
            qty=qty,
            mode="BRACKET",
            entry=entry_leg,
            exits=exits,
            last_status="CREATED"
        )

    # --- Private Execution Helpers ---

    def _get_signal_date(self, trade: dict) -> pd.Timestamp | None:
        """Extracts the signal date from the JSON context."""
        try:
            ctx_str = trade.get('signal_context')
            if not ctx_str:
                return None
            context = json.loads(ctx_str)
            date_val = context.get('date') or context.get('setup_date')
            return pd.Timestamp(date_val) if date_val else None
        except (ValueError, TypeError, json.JSONDecodeError):
            return None

    def _invalidate_trade(
        self,
        trade: dict,
        repository: TradeRepository,
        low_price: float,
        stop_loss: float,
        date_str: str
    ) -> str:
        outcome_reason = f"SETUP INVALIDATED: Low {low_price:.2f} <= SL {stop_loss:.2f}"
        repository.update_trade(trade['id'], {
            "status": TradeStatus.INVALID,
            "exit_reason": ExitReason.INVALIDATED,
            "exit_date": date_str,
            "realized_pnl": 0.0
        }, reason=outcome_reason)
        return outcome_reason

    def _execute_activation(
        self,
        trade: dict,
        repository: TradeRepository,
        fill_price: float,
        reason: str,
        date_str: str
    ) -> str:
        # Calculate Position Size based on Risk
        size = float(trade.get('current_size') or 0.0)
        stop_loss = float(trade.get('current_stop_loss') or 0.0)
        risk_amount = float(trade.get('risk_amount') or 100.0) # Default Risk if missing?
        
        # If size is not pre-calculated, calculate based on risk
        if size == 0 and stop_loss > 0 and fill_price > stop_loss:
            risk_per_share = fill_price - stop_loss
            size = int(risk_amount / risk_per_share)
        
        if size <= 0:
            logger.warning(f"[{trade.get('symbol')}] Zero size calculated. Ignoring entry.")
            return "ERROR: Zero Size" 

        repository.update_trade(trade['id'], {
            "status": TradeStatus.ACTIVE,
            "entry_date": date_str,
            "entry_price": fill_price,
            "initial_size": size,
            "current_size": size
        }, reason=f"{reason} FILLED @ {fill_price:.2f}")

        return f"FILLED @ {fill_price:.2f} ({int(size)} Shares)"

    def _close_trade(
        self,
        trade: dict,
        repository: TradeRepository,
        exit_price: float,
        reason: str,
        date_str: str
    ) -> str:
        entry_price = float(trade.get('entry_price') or 0.0)
        size = float(trade.get('current_size') or 0.0)
        pnl = (exit_price - entry_price) * size
        
        repository.update_trade(trade['id'], {
            "status": TradeStatus.CLOSED,
            "exit_reason": reason,
            "exit_price": exit_price,
            "exit_date": date_str,
            "realized_pnl": pnl
        })
        return f"{reason} @ {exit_price:.2f} (PnL: {pnl:.2f})"