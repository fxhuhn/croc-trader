
import json
import logging
import calendar
from typing import Any, override
import pandas as pd

from ....types import TradeStatus, TradeParams, ExitReason, TradeData
from ....database.repositories.trade import TradeRepository
from .abstract import BaseTradeStrategy

logger = logging.getLogger(__name__)

class TurnoverTimingStrategy(BaseTradeStrategy):
    """
    Manager für Turnover Timing Trades.
    Entry: Limit Order (Low <= Limit).
    Exit: 2 Green Candles in Folge ODER Freitag Close.
    """
    name = "TurnoverTiming"
    
    # -------------------------------------------------------------------------
    # 1. Parameter Extraction
    # -------------------------------------------------------------------------
    @override
    def get_current_params(
        self, 
        trade: TradeData, 
        df_history: pd.DataFrame | None = None, 
        repository: TradeRepository | None = None
    ) -> TradeParams:
        dict_trade = dict(trade)
        return TradeParams(
            symbol=dict_trade['symbol'],
            entry=float(dict_trade.get('entry_price') or 0),
            size=float(dict_trade.get('current_size') or 0),
            stop=0.0, 
            target=0.0
        )

    # -------------------------------------------------------------------------
    # 2. Entry Logic
    # -------------------------------------------------------------------------
    @override
    def check_entry(
        self, 
        trade: TradeData, 
        candle: pd.Series, 
        df_history: pd.DataFrame, 
        repository: TradeRepository
    ) -> str | None:
        """
        Prüft Limit-Entry.
        Regel: Low <= Limit Preis.
        Gültigkeit: Nur am ersten Tag nach dem Signal (Day Valid).
        """
        dict_trade = dict(trade)
        context = self._get_context(dict_trade)
        
        signal_date_str = context.get("setup_date") or context.get("date")
        if not signal_date_str: 
            return None
        
        signal_date = pd.Timestamp(signal_date_str).date()
        current_date = pd.Timestamp(candle['date']).date()
        
        # 1. Backtest-Schutz: Nicht am Signaltag handeln (Look-Ahead Bias verhindern)
        if current_date <= signal_date:
            return None
            
        limit_price = float(dict_trade['entry_price'])
        low = float(candle['low'])
        open_price = float(candle['open'])
        
        # 2. Limit Check
        # Gap-Handling: Wenn Open < Limit, kaufen wir zum Open (besserer Preis)
        # Wenn Low <= Limit, kaufen wir zum Limit
        fill_price = 0.0
        filled = False
        
        if open_price <= limit_price:
            fill_price = open_price
            filled = True
        elif low <= limit_price:
            fill_price = limit_price
            filled = True
            
        if filled:
            # Size berechnen (falls noch 0, z.B. 2000$ Budget)
            current_size = float(dict_trade.get('current_size') or 0)
            if current_size == 0:
                budget = 2000.0 # Standard Backtest Budget
                current_size = int(budget / fill_price) if fill_price > 0 else 0
            
            repository.update_trade(dict_trade['id'], {
                "status": TradeStatus.ACTIVE,
                "entry_date": str(candle['date']),
                "entry_price": fill_price,
                "initial_size": current_size,
                "current_size": current_size
            }, reason=f"LIMIT FILLED @ {fill_price:.2f}")
            return f"✅ FILLED @ {fill_price:.2f}"
        else:
            # Order expired (Day Valid) -> Trade wird ungültig
            repository.update_trade(dict_trade['id'], {
                "status": TradeStatus.MISSED,
                "exit_reason": ExitReason.EXPIRED 
            })
            return f"❌ MISSED (Low {low} > Limit {limit_price})"

    # -------------------------------------------------------------------------
    # 3. Active Trade Management (Exits)
    # -------------------------------------------------------------------------
    @override
    def manage_active_trade(
        self, 
        trade: TradeData, 
        df_history: pd.DataFrame, 
        repository: TradeRepository
    ) -> str | None:
        """
        Exit Logik:
        1. 2 Grüne Kerzen in Folge -> Exit Market Open.
        2. Ende der Woche (Freitag) -> Exit Market Close.
        """
        if df_history.empty: 
            return None
        
        dict_trade = dict(trade)
        current_candle = df_history.iloc[-1]
        current_date = pd.Timestamp(current_candle['date'])
        current_close = float(current_candle['close'])
        
        context = self._get_context(dict_trade)
        
        # --- A) Zeit-Stopp (Freitag) ---
        # Wenn heute Freitag ist, steigen wir zum Close aus.
        if current_date.weekday() == calendar.FRIDAY:
            return self._close_trade(repository, dict_trade, current_close, ExitReason.TIME_STOP, current_date)
            
        # --- B) 2 Grüne Kerzen ---
        # Wir prüfen die Historie (Gestern und Vorgestern).
        # Wenn beide grün waren, steigen wir HEUTE zum Open aus.
        
        yesterday_green = False
        before_yesterday_green = False
        
        # Helper: Ist Kerze an Position idx grün?
        def is_green(idx):
            if len(df_history) >= abs(idx):
                c = df_history.iloc[idx]
                return c['close'] > c['open']
            return False

        # Gestern (iloc[-2])
        if len(df_history) >= 2:
            yesterday_green = is_green(-2)
        else:
            # Fallback auf Setup-Kerze (vom Screener Context)
            yesterday_green = bool(context.get("setup_candle_green", False))

        # Vorgestern (iloc[-3])
        if len(df_history) >= 3:
            before_yesterday_green = is_green(-3)
        else:
            # Wenn History zu kurz, nehmen wir an, Vorgestern war die Setup-Kerze
            before_yesterday_green = bool(context.get("setup_candle_green", False))

        # TRIGGER
        if yesterday_green and before_yesterday_green:
            # Exit zum Open von HEUTE
            exit_price = float(current_candle['open'])
            # Logic: Exit was caused by technical signal "2 Green Candles"
            return self._close_trade(repository, dict_trade, exit_price, ExitReason.TAKE_PROFIT, current_date)

        return None

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _get_context(self, trade: dict) -> dict:
        raw = trade.get("signal_context")
        try:
            return json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            return {}

    def _close_trade(self, repository, trade, price, reason, date_obj):
        entry_price = float(trade['entry_price'])
        size = float(trade['current_size'])
        pnl = (price - entry_price) * size
        
        repository.update_trade(trade['id'], {
            "status": TradeStatus.CLOSED,
            "exit_reason": reason,
            "exit_price": price,
            "exit_date": date_obj.strftime("%Y-%m-%d"),
            "realized_pnl": pnl
        })
        return f"EXIT {trade['symbol']}: {reason} @ {price:.2f}"

    @override
    def generate_orders(self, trade, df_history, budget, repo):
        # Für Live-Trading Order-Generierung (optional)
        return None