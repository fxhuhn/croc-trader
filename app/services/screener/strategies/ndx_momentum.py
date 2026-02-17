import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import ClassVar, override

from ....database.repositories.market_data_provider import MarketDataProvider
from ....database.repositories.trade import TradeRepository
from ....services.telegram import TelegramBot
from ....tools.symbol_lists import ExchangeSymbol
from ....tools import indicators
from .base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NDXMomentumConfig:
    """Configuration for the NDX Momentum strategy."""
    
    # Universe
    INDEX_TICKER: str = "QQQ"
    MAX_TICKER: int = 5
    
    # Regime Parameters
    INDEX_SMA_WINDOW: int = 200
    BREADTH_SMA_WINDOW: int = 200
    BREADTH_FAST_WINDOW: int = 21   # EMA 21 of Breadth
    BREADTH_SLOW_WINDOW: int = 63   # EMA 63 of Breadth
    
    # Momentum Parameters (Daily approximations of 1, 3, 6, 12 months)
    MOMENTUM_WINDOWS: ClassVar[list[int]] = [21, 63, 126, 252]
    
    # Data Fetching
    LOOKBACK_DAYS: int = 400  # Need enough for 252d ROC + 200d SMA


class NDXMomentumScreener(BaseStrategy):
    """
    NDX Momentum Screener.
    
    Logic:
    1. Regime Filter: 
       - Index (QQQ) > SMA(200)
       - Breadth (Stocks > SMA(200)) -> Fast EMA > Slow EMA
    2. Momentum Score:
       - Sum of ROC(1m, 3m, 6m, 12m)
    3. Selection:
       - Bull Regime: Select Top 5 Momentum stocks.
       - Bear Regime: Select intersection of Top 5 and currently held stocks (no new entries).
    """

    name: str = "NDXMomentum"

    def __init__(
        self,
        trade_repository: TradeRepository,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None,
        config: NDXMomentumConfig | None = None,
    ) -> None:
        super().__init__(data_provider, telegram_bot)
        self.trade_repository = trade_repository
        self.config = config or NDXMomentumConfig()

    @override
    def run(
        self,
        days: int = 0,  # Unused, defined by config
        analysis_date: str | None = None,
        specific_symbols: list[str] | None = None,
    ) -> int:
        """Executes the screener."""
        
        # 1. Setup Universe
        exchange = ExchangeSymbol()
        ndx_symbols = exchange.nasdaq_100
        universe = list(set(ndx_symbols + [self.config.INDEX_TICKER]))
        
        # 2. Load Data
        lookback = self.config.LOOKBACK_DAYS
        data = self.data_provider.get_universe_daily_data(universe, days=lookback)
        
        if not data or "close" not in data or data["close"].empty:
            logger.warning(f"[{self.name}] No data found for universe.")
            return 0
            
        # 3. Resolve Target Date
        target_date = self._resolve_target_date(data["close"], analysis_date)
        if target_date is None:
            return 0
            
        # 4. Calculate Regime & Signals
        return self._execute_strategy(data, target_date, ndx_symbols)

    def _resolve_target_date(
        self, closes: pd.DataFrame, analysis_date: str | None
    ) -> pd.Timestamp | None:
        """Helper to determine the valid analysis date."""
        if analysis_date:
            try:
                t_date = pd.Timestamp(analysis_date)
                if t_date in closes.index:
                    return t_date
                # Fallback logic could go here similar to DipBuyer
                logger.warning(f"[{self.name}] Date {analysis_date} not in data.")
                return None
            except Exception:
                return None
        return closes.index[-1]

    def _execute_strategy(
        self, 
        data: dict[str, pd.DataFrame], 
        target_date: pd.Timestamp,
        ndx_universe: list[str]
    ) -> int:
        closes = data["close"]
        
        # --- A. Regime Calculation ---
        
        # 1. Index Regime (QQQ)
        qqq_close = closes[self.config.INDEX_TICKER]
        qqq_sma = indicators.calculate_sma(qqq_close, self.config.INDEX_SMA_WINDOW)
        
        # Regime Condition 1: Index Price > Index SMA
        index_bullish = qqq_close > qqq_sma
        
        # 2. Breadth Regime
        # Breadth = Number of NDX stocks > Their SMA(200)
        # We calculate SMA(200) for ALL NDX stocks
        valid_ndx = [s for s in ndx_universe if s in closes.columns]
        ndx_closes = closes[valid_ndx]
        
        # Calculate SMA200 for the whole dataframe
        ndx_sma200 = ndx_closes.rolling(window=self.config.BREADTH_SMA_WINDOW).mean()
        
        # Boolean DF: True where Close > SMA
        above_sma = ndx_closes > ndx_sma200
        
        # Count TRUES per day (Breadth Raw)
        breadth_raw = above_sma.sum(axis=1)
        
        # Calculate EMAs of Breadth
        breadth_fast = indicators.calculate_ema(breadth_raw, self.config.BREADTH_FAST_WINDOW)
        breadth_slow = indicators.calculate_ema(breadth_raw, self.config.BREADTH_SLOW_WINDOW)
        
        # Regime Condition 2: Fast > Slow
        breadth_bullish = breadth_fast > breadth_slow
        
        # --- Get Values for Target Date ---
        try:
            current_idx_bull = index_bullish.loc[target_date]
            current_breadth_bull = breadth_bullish.loc[target_date]
            
            # Values for logging/context
            val_qqq = qqq_close.loc[target_date]
            val_qqq_sma = qqq_sma.loc[target_date]
            val_breadth_fast = breadth_fast.loc[target_date]
            val_breadth_slow = breadth_slow.loc[target_date]
            
        except KeyError:
            logger.error(f"[{self.name}] Data missing for {target_date.date()}")
            return 0

        is_bull_regime = current_idx_bull and current_breadth_bull
        
        logger.info(
            f"[{self.name}] Mode: {'BULL' if is_bull_regime else 'BEAR'} | "
            f"Index: {current_idx_bull} ({val_qqq:.2f}/{val_qqq_sma:.2f}) | "
            f"Breadth: {current_breadth_bull} ({val_breadth_fast:.1f}/{val_breadth_slow:.1f})"
        )

        # --- B. Momentum Calculation ---
        
        # Calculate ROC Sum for all NDX stocks
        # ROC = Sum(ROC_21, ROC_63, ROC_126, ROC_252)
        combined_momentum = pd.DataFrame(0.0, index=ndx_closes.index, columns=ndx_closes.columns)
        
        for window in self.config.MOMENTUM_WINDOWS:
            roc_w = indicators.calculate_roc(ndx_closes, window)
            combined_momentum = combined_momentum.add(roc_w, fill_value=0)
            
        # Get momentum for target date
        try:
            day_momentum = combined_momentum.loc[target_date].dropna()
        except KeyError:
            return 0
            
        # Select Top N
        top_candidates = day_momentum.nlargest(self.config.MAX_TICKER * 2) # Get more for safety
        top_n_symbols = top_candidates.nlargest(self.config.MAX_TICKER).index.tolist()
        
        # --- C. Selection Logic ---
        
        final_selection = []
        
        if is_bull_regime:
            # Bull Market: Take top N
            final_selection = top_n_symbols
        else:
            # Bear Market: Only keep existing if they are in Top N
            # 1. Fetch currently active trades for this strategy
            active_trades = self.trade_repository.get_active_trades() # This gets CREATED/ACTIVE
            # Filter for this strategy strictly
            my_active_symbols = [
                t["symbol"] for t in active_trades 
                if t.get("strategy") == self.name
            ]
            
            # Intersection
            final_selection = list(set(my_active_symbols) & set(top_n_symbols))
            
            logger.info(
                f"[{self.name}] Bear Regime. Holding intersection: {final_selection} "
                f"(Active: {len(my_active_symbols)}, Top: {len(top_n_symbols)})"
            )

        # --- D. Process Signals ---
        return self._create_trades(final_selection, day_momentum, target_date, data, 
                                   regime_info={
                                       "bull": is_bull_regime, 
                                       "qqq": val_qqq, 
                                       "b_fast": val_breadth_fast, 
                                       "b_slow": val_breadth_slow
                                   })

    def _create_trades(
        self, 
        symbols: list[str], 
        momentum_map: pd.Series, 
        date: pd.Timestamp, 
        data: dict[str, pd.DataFrame],
        regime_info: dict
    ) -> int:
        saved_count = 0
        date_str = date.strftime("%Y-%m-%d")
        
        created_report = []

        for symbol in symbols:
            try:
                # Basic Data
                close_price = data["close"].at[date, symbol]
                score = momentum_map.at[symbol]
                
                # Context used for UI and debugging
                context = {
                    "source": "screener",
                    "date": date_str,
                    "momentum_score": round(score, 2),
                    "regime": "BULL" if regime_info["bull"] else "BEAR",
                    "info": f"BF:{regime_info['b_fast']:.1f}/BS:{regime_info['b_slow']:.1f}"
                }

                # Create Trade (Entry Signal)
                # Note: No specific Stop/Target defined in user request. 
                # Setting 0 means "Managed elsewhere" or "Market Order".
                self.trade_repository.create_trade(
                    symbol=symbol,
                    strategy=self.name,
                    size=0,
                    entry=round(close_price, 2),
                    stop_loss=0.0,
                    target=0.0,
                    context=context
                )
                saved_count += 1
                created_report.append({
                    "Symbol": symbol,
                    "Score": round(score, 2),
                    "Price": round(close_price, 2)
                })
                
            except Exception as e:
                logger.error(f"[{self.name}] Error creating trade for {symbol}: {e}")

        # Telegram Reporting
        if self.telegram_bot and created_report:
            df_rep = pd.DataFrame(created_report)
            title = f"{self.name} ({'BULL' if regime_info['bull'] else 'BEAR'})"
            self._send_telegram_report(title, date_str, df_rep)
            
        return saved_count
