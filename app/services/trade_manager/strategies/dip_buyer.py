import logging
import json
from typing import override, final, Optional, Any
from datetime import date

import pandas as pd

from ....types import TradeStatus, TradeParams, Order, ExitReason, TradeData
from ....database.repositories.trade import TradeRepository
from .abstract import BaseTradeStrategy

logger = logging.getLogger(__name__)

@final
class DipBuyerStrategy(BaseTradeStrategy):
    """
    Dip Buyer Strategie: Versucht Anteile mittels Limit-Order in die Schwäche zu kaufen.

    Regeln:
    - Entry: Limit Buy unter aktuellem Marktpreis.
    - Exit 1: LOC (Limit On Close) - Wenn Close > High(Vorheriger Tag).
    - Exit 2: Target (Take Profit) - Aus dem Signal übernommen (ignoriert am Entry-Tag).
    - Exit 3: TimeStop nach N Handelstagen (nicht Kalendertage!).
    """
    
    name = "DipBuyer"
    
    # Configuration Constants
    DEFAULT_BUDGET: float = 2000.0
    TIME_STOP_DAYS: int = 10
    
    @override
    def get_current_params(
        self, 
        trade: TradeData, 
        df_history: Optional[pd.DataFrame] = None, 
        repository: Optional[TradeRepository] = None
    ) -> TradeParams:
        """Extrahiert die aktuellen Trade-Parameter für das Frontend/Logging."""
        # TypedDict access - use .get() for safety if key is optional
        return TradeParams(
            stop_loss=0.0, # Strategy has no Stop Loss
            tp_1=float(trade.get('current_target') or 0.0),
            extras={
                "entry_limit": float(trade.get('entry_price') or 0.0),
                "current_size": float(trade.get('current_size') or 0.0)
            }
        )

    @override
    def check_entry(
        self, 
        trade: TradeData, 
        candle: pd.Series, 
        df_history: pd.DataFrame, 
        repository: TradeRepository
    ) -> str | None:
        """
        Prüft, ob das Limit für den Entry erreicht wurde.
        
        Logik:
        1. Validierung des Handelstages (nicht am Signal-Tag handeln).
        2. Prüfung ob Low <= Limit Preis.
        3. Ausführung (ggf. Gap-Down Schutz).
        4. Verfall (Expire), falls Limit nicht erreicht (Day Order).
        """
        symbol = trade.get('symbol', 'UNKNOWN')
        
        try:
            dict_trade = dict(trade) # Cast for safety if needed, though TypedDict behaves like dict
            
            limit_price = float(dict_trade.get('entry_price') or 0.0)
            if limit_price <= 0:
                logger.warning("Trade %s hat keinen validen Limit-Preis (<= 0).", symbol)
                return None

            # 1. Datum Validierung (Early Return)
            if not self._is_valid_entry_date(dict_trade, candle):
                return None

            low_price = float(candle['low'])
            
            # 2. Check: Limit erreicht?
            if low_price > limit_price:
                # Limit nicht erreicht -> Day Order expired
                self._handle_expired_order(dict_trade, candle, repository)
                return "EXPIRED (Limit not reached)"

            # 3. Execution Logic
            # Gap Schutz: Wenn Open < Limit, bekommen wir den Open Price (besserer Einstieg)
            open_price = float(candle['open'])
            fill_price = min(open_price, limit_price) if open_price < limit_price else limit_price
            
            return self._execute_trade(dict_trade, fill_price, str(candle['date']), repository)

        except Exception:
            logger.exception("Kritischer Fehler beim Entry-Check für %s", symbol)
            return None

    @override
    def manage_active_trade(
        self, 
        trade: TradeData, 
        df_history: pd.DataFrame, 
        repository: TradeRepository
    ) -> str | None:
        """
        Verwaltet offene Trades (LOC, TP, TimeStop).
        """
        if df_history.empty:
            return None
        
        try:
            # Cast safe copy
            dict_trade = dict(trade)
            candle = df_history.iloc[-1]
            return self._check_exit_conditions(dict_trade, candle, df_history, repository)
        except Exception:
            logger.exception("Fehler beim Management von Trade %s", trade.get('symbol'))
            return None

    @override
    def generate_orders(
        self, 
        trade: TradeData, 
        df_history: pd.DataFrame, 
        budget: float, 
        repository: TradeRepository
    ) -> Order | None:
        # Nicht implementiert für diese Strategie
        return None

    # --- Private Helper Methods (Clean Code) ---

    def _is_valid_entry_date(self, trade: dict, candle: pd.Series) -> bool:
        """
        Prüft, ob der aktuelle Tag nach dem Signal-Tag liegt.
        """
        try:
            context_str = trade.get('signal_context')
            if not context_str:
                return True 
                
            context = json.loads(context_str)
            signal_date_str = context.get('date') or context.get('setup_date')
            
            if not signal_date_str:
                return True

            signal_date = pd.Timestamp(signal_date_str).date()
            current_date = pd.Timestamp(candle['date']).date()
            
            if current_date <= signal_date:
                return False
                
            return True
        except (ValueError, TypeError, json.JSONDecodeError):
            logger.warning("Konnte Signal-Datum für Trade nicht parsen. Erlaube Entry.")
            return True

    def _handle_expired_order(self, trade: dict, candle: pd.Series, repository: TradeRepository) -> None:
        """Setzt den Trade auf CLOSED/EXPIRED."""
        repository.update_trade(trade['id'], {
            "status": TradeStatus.INVALID,
            "exit_reason": ExitReason.EXPIRED,
            "exit_date": str(candle['date']),
            "realized_pnl": 0.0
        })

    def _execute_trade(self, trade: dict, fill_price: float, date_str: str, repository: TradeRepository) -> str:
        """Führt den Trade aus."""
        if fill_price <= 0:
            logger.error("Versuchter Fill mit Preis <= 0 für Trade %s", trade.get('id'))
            return "ERROR: Fill Price <= 0"

        budget = float(trade.get('budget') or self.DEFAULT_BUDGET)
        current_size = int(budget / fill_price)
        
        repository.update_trade(trade['id'], {
            "status": TradeStatus.ACTIVE,
            "entry_date": date_str,
            "entry_price": fill_price,
            "initial_size": current_size,
            "current_size": current_size
        }, reason=f"LIMIT FILLED @ {fill_price:.2f} | Budget: {budget}$")
        
        return f"FILLED @ {fill_price:.2f} ({current_size} Stk)"

    def _check_exit_conditions(
        self, 
        trade: dict, 
        candle: pd.Series,
        df_history: pd.DataFrame, 
        repository: TradeRepository
    ) -> str | None:
        """Prüft LOC, Target und TimeStop Logik."""
        current_daily_high = float(candle['high'])
        current_daily_close = float(candle['close'])
        current_daily_open = float(candle['open'])
        
        entry_price = float(trade['entry_price'])
        size = float(trade['current_size'])
        target_price = float(trade.get('current_target') or 0.0)

        # Days Held Calculation: Trading Days (Rows) since Entry
        entry_date_str = trade.get('entry_date')
        if entry_date_str:
             # Count rows where date >= entry_date
             trading_days_held = len(df_history[df_history['date'] >= entry_date_str])
        else:
             trading_days_held = 0

        # Note: If today is entry day, trading_days_held is 1.

        exit_reason: str | None = None
        exit_price = 0.0

        # --- Rule 1: Limit On Close (LOC) ---
        # Exit if Close > Previous Trading Day High
        if len(df_history) >= 2:
            prev_candle = df_history.iloc[-2]
            prev_high = float(prev_candle['high'])
            
            if current_daily_close > prev_high:
                return self._execute_exit(trade, "LOC_HIT", current_daily_close, size, entry_price, str(candle['date']), repository)

        # --- Rule 2: Take Profit (TP) ---
        # Block TP on Entry Day. 
        if trading_days_held > 1 and target_price > 0: 
            if current_daily_high >= target_price:
                exit_reason = ExitReason.TARGET_HIT
                # Best Execution bei Gap Up
                exit_price = max(current_daily_open, target_price) if current_daily_open > target_price else target_price

        # --- Rule 3: Time Stop ---
        # If no exit reason yet, Check Time Stop
        if not exit_reason and trading_days_held > self.TIME_STOP_DAYS:
            exit_reason = ExitReason.TIME_STOP
            exit_price = current_daily_close

        if exit_reason:
            return self._execute_exit(trade, exit_reason, exit_price, size, entry_price, str(candle['date']), repository)
            
        return None

    def _execute_exit(
        self, 
        trade: dict, 
        reason: str, 
        exit_price: float, 
        size: float, 
        entry_price: float, 
        date_str: str, 
        repository: TradeRepository
    ) -> str:
        """Führt den Exit aus und berechnet PnL."""
        pnl = (exit_price - entry_price) * size
        
        repository.update_trade(trade['id'], {
            "status": TradeStatus.CLOSED,
            "exit_reason": reason,
            "exit_price": exit_price,
            "exit_date": date_str,
            "realized_pnl": pnl
        })
        
        return f"{reason} @ {exit_price:.2f} (PnL: {pnl:.2f})"