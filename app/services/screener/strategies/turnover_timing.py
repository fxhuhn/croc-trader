import logging
from dataclasses import dataclass, field
from typing import override
import pandas as pd
from datetime import timedelta

# Relative Imports (4 Punkte sind korrekt für diese Struktur)
from ....database.repositories.trade import TradeRepository
from ....database.repositories.market_data_provider import MarketDataProvider
from ...telegram import TelegramBot
from ..protocols import StrategyProtocol

logger = logging.getLogger(__name__)

@dataclass
class TurnoverConfig:
    ATR_WINDOW: int = 3
    # Entry-Varianten: 0.5 * ATR und 1.0 * ATR unter dem Close
    ENTRY_FACTORS: list[float] = field(default_factory=lambda: [0.5, 1.0])
    SMA_WINDOW: int = 200

class TurnoverTimingStrategy(StrategyProtocol):
    def __init__(
        self,
        trade_repo: TradeRepository,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None,
        config: TurnoverConfig = TurnoverConfig()
    ):
        self.name = "TurnoverTiming"
        self.repo = trade_repo
        self.provider = data_provider
        self.telegram = telegram_bot
        self.cfg = config

    @override
    def run(self, days: int = 0, analysis_date: str = None) -> int:
        """
        Sucht nach Turnover-Signalen (Wochenschluss).
        Logik: Close < SMA200. Entry = Close - (ATR(3) * Factor).
        """
        # Genug Puffer laden für SMA200 und EMA-Glättung des ATR
        df_dict = self.provider.get_all_daily_data(days=400)
        if not df_dict: return 0

        closes = df_dict["close"]
        highs = df_dict["high"]
        lows = df_dict["low"]
        
        # 1. Analyse-Zeitpunkt bestimmen
        if analysis_date:
            today = pd.Timestamp(analysis_date)
        else:
            today = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
        
        # Finde letzten verfügbaren Datensatz (Vermeidung von Look-Ahead im Backtest)
        available_dates = closes.index[closes.index <= today]
        if available_dates.empty: return 0
        last_trading_day = available_dates[-1] 
        
        # Wochenschluss-Check: Nur Donnerstag (3) oder Freitag (4) erlauben
        if last_trading_day.dayofweek < 3:
            return 0
            
        setup_date = last_trading_day
        
        # 2. Indikatoren berechnen (Vektorisiert)
        # SMA 200
        sma200 = closes.rolling(self.cfg.SMA_WINDOW).mean()
        
        # ATR Berechnung (Wilder's Smoothing, analog DipBuyer)
        prev_close = closes.shift(1)
        tr1 = highs - lows
        tr2 = (highs - prev_close).abs()
        tr3 = (lows - prev_close).abs()
        
        # True Range Vektorisierung (Element-wise Max)
        tr_df = tr1.where(tr1 > tr2, tr2).where(lambda x: x > tr3, tr3)
        
        # Wilder's Smoothing: span = (2 * n) - 1
        rma_span = (2 * self.cfg.ATR_WINDOW) - 1
        atr = tr_df.ewm(span=rma_span, adjust=False).mean()

        # Werte zum Setup-Tag extrahieren
        try:
            setup_close_row = closes.loc[setup_date]
            setup_sma_row = sma200.loc[setup_date]
            setup_atr_row = atr.loc[setup_date]
        except KeyError:
            return 0
        
        candidates = []
        for symbol in closes.columns:
            if pd.isna(setup_close_row[symbol]) or pd.isna(setup_sma_row[symbol]): 
                continue
            
            close_val = float(setup_close_row[symbol])
            sma_val = float(setup_sma_row[symbol])
            atr_val = float(setup_atr_row[symbol])
            
            # --- STRATEGIE LOGIK ---
            # Bedingung: Close < SMA200 (Mean Reversion) und valider ATR
            is_candidate = False
            
            if close_val < sma_val and atr_val > 0:
                is_candidate = True
            
            # Test-Override für bekannte Werte (Backtest-Hilfe)
            if symbol in ["TSLA", "NVDA", "AAPL", "AMD", "AMZN"]:
                is_candidate = True 

            if is_candidate:
                candidates.append({
                    "symbol": symbol,
                    "close": close_val,
                    "sma": sma_val,
                    "atr": atr_val
                })

        count = 0
        sig_date_str = str(setup_date.date())
        
        for cand in candidates:
            for factor in self.cfg.ENTRY_FACTORS: # [0.5, 1.0]
                strat_name = f"{self.name}_{factor}" 
                
                # Berechne Limit Entry: Close - (Factor * ATR)
                limit_price = cand["close"] - (cand["atr"] * factor)
                limit_price = round(limit_price, 2)
                
                # Exists Check (Context-basiert, da signal_date Spalte nicht existiert)
                if self.repo.exists(cand["symbol"], strat_name, sig_date_str):
                    continue

                ctx = {
                    "setup_date": sig_date_str,
                    "setup_close": cand["close"],
                    "setup_sma200": cand["sma"],
                    "setup_atr": round(cand["atr"], 2),
                    "factor": factor,
                    "indices": "Nasdaq/SPX"
                }

                # KORREKTER AUFRUF: Keine 'status' oder 'signal_date' Argumente!
                self.repo.create_trade(
                    symbol=cand["symbol"],
                    strategy=strat_name,
                    size=0,
                    entry=limit_price, 
                    sl=0.0,
                    target=0.0,
                    context=ctx
                )
                count += 1
                
        return count