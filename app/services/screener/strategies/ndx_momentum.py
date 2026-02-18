import logging
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import override

from ....database.repositories.market_data_provider import MarketDataProvider
from ....database.repositories.trade import TradeRepository
from ....services.telegram import TelegramBot
from ....tools.symbol_lists import ExchangeSymbol
from ....tools.market_holidays import MarketHolidayChecker
from ....const import Strategies
from .base import BaseStrategy

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class NDXMomentumConfiguration:
    """Configuration settings for the NDX Momentum strategy."""
    maximum_ticker_count: int = 5
    maximum_lookback_period: int = 252

class NDXMomentumScreener(BaseStrategy):
    """
    Screener for the NASDAQ-100 Momentum strategy.
    
    This strategy identifies top momentum stocks in the NASDAQ-100 index
    based on a combined Rate of Change (ROC) score across multiple windows.
    It executes rebalances exclusively on the last trading day of the month.
    """
    name: str = str(Strategies.NDXMomentum)

    def __init__(
        self,
        trade_repository: TradeRepository,
        market_data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None,
        configuration: NDXMomentumConfiguration | None = None,
    ) -> None:
        """
        Initializes the NDX Momentum screener.
        
        Args:
            trade_repository: Repository for trade persistence.
            market_data_provider: Provider for historical price data.
            telegram_bot: Optional bot for notifications.
            configuration: Specific strategy configuration parameters.
        """
        super().__init__(market_data_provider, telegram_bot)
        self.trade_repository = trade_repository
        self.configuration = configuration or NDXMomentumConfiguration()
        self.holiday_checker = MarketHolidayChecker()

    @override
    def run(
        self, 
        days: int = 0, 
        analysis_date: str | None = None, 
        specific_symbols: list[str] | None = None
    ) -> int:
        """
        Executes the monthly momentum screening process.
        
        This method checks if the analysis date is the last trading day of the
        month. If so, it calculates momentum scores, determines the market
        regime, and creates trade entries for the top performers.
        
        Args:
            days: Ignored for this strategy (uses fixed lookback).
            analysis_date: The date to perform analysis for.
            specific_symbols: Optional override for symbols to screen.
            
        Returns:
            The number of trades successfully created.
        """
        if not analysis_date:
            analysis_date = datetime.now().strftime("%Y-%m-%d")
        
        target_date = pd.Timestamp(analysis_date)
        if not self._is_last_trading_day(target_date):
            return 0

        logger.info("[%s] Monthly rebalance triggered for %s", self.name, analysis_date)

        # 1. Fetch Universe (NDX)
        exchange_symbol_provider = ExchangeSymbol()
        nasdaq_100_symbols = exchange_symbol_provider.nasdaq_100
        if not nasdaq_100_symbols:
             logger.error("[%s] NASDAQ-100 symbols list is empty.", self.name)
             return 0
             
        universe_symbols = list(set(nasdaq_100_symbols + ["QQQ"]))
        
        # Use get_batch_history with 450 calendar days to safely cover 252 trading days
        full_history_map = self.data_provider.get_batch_history(
            universe_symbols, 
            days=450, 
            end_date=analysis_date
        )
        
        if not full_history_map:
            logger.warning("[%s] No data found for universe.", self.name)
            return 0

        # Data Pre-processing: Concatenate and Pivot
        history_dataframes = []
        for symbol, dataframe in full_history_map.items():
            dataframe["symbol"] = symbol
            history_dataframes.append(dataframe)
            
        universe_dataframe = pd.concat(history_dataframes, ignore_index=True)
        
        pivoted_data = {
            "close": universe_dataframe.pivot(index="date", columns="symbol", values="close"),
            "high": universe_dataframe.pivot(index="date", columns="symbol", values="high"),
            "low": universe_dataframe.pivot(index="date", columns="symbol", values="low")
        }
        
        if "QQQ" not in pivoted_data["close"].columns:
             logger.warning("[%s] QQQ data missing.", self.name)
             return 0
        
        try:
            available_date = pivoted_data["close"].index.asof(target_date)
            if pd.isna(available_date) or available_date != target_date:
                 logger.warning("[%s] %s not in price data. Skip.", self.name, analysis_date)
                 return 0
        except Exception as exception:
            logger.error("[%s] Date resolution error: %s", self.name, exception)
            return 0

        # 2. Calculation Pipeline
        qqq_close_series = pivoted_data["close"]["QQQ"]
        current_qqq_price = qqq_close_series.at[target_date]
        index_moving_average = qqq_close_series.tail(200).mean()
        
        valid_nasdaq_symbols = [s for s in nasdaq_100_symbols if s in pivoted_data["close"].columns]
        nasdaq_closes = pivoted_data["close"][valid_nasdaq_symbols]
        
        # Breadth
        sma100_matrix = nasdaq_closes.rolling(window=100).mean()
        percentage_above_sma100 = (nasdaq_closes > sma100_matrix).mean(axis=1) * 100
        breadth_fast_average = percentage_above_sma100.rolling(window=10).mean().at[target_date]
        breadth_slow_average = percentage_above_sma100.rolling(window=50).mean().at[target_date]
        
        is_bull_regime = (current_qqq_price > index_moving_average) and (breadth_fast_average > breadth_slow_average)
        
        # Momentum
        rolling_roc_results = {}
        for window in [21, 63, 126, 252]:
            rolling_roc_results[window] = nasdaq_closes.pct_change(periods=window) * 100
        
        combined_momentum_matrix = (
            rolling_roc_results[21] + 
            rolling_roc_results[63] + 
            rolling_roc_results[126] + 
            rolling_roc_results[252]
        )
            
        target_date_momentum_sum = combined_momentum_matrix.loc[target_date].dropna()
        if target_date_momentum_sum.empty:
            logger.warning("[%s] No momentum scores for %s.", self.name, analysis_date)
            return 0
            
        selected_leaders = target_date_momentum_sum.nlargest(self.configuration.maximum_ticker_count).index.tolist()
        
        # 3. Create Trades
        return self._create_trades_direct(
            selected_leaders, 
            target_date_momentum_sum, 
            rolling_roc_results,
            target_date,
            pivoted_data,
            regime_indicators={
                "bull": is_bull_regime, 
                "qqq": current_qqq_price, 
                "qqq_sma": index_moving_average,
                "breadth_fast": breadth_fast_average, 
                "breadth_slow": breadth_slow_average
            }
        )

    def _is_last_trading_day(self, date: pd.Timestamp) -> bool:
        """Checks if the date represents the last trading day of its month."""
        current_month = date.month
        lookahead_date = date + pd.Timedelta(days=1)
        for _ in range(5):
             if lookahead_date.month != current_month:
                  return True
             if lookahead_date.dayofweek < 5 and not self.holiday_checker.is_holiday(lookahead_date):
                  return False
             lookahead_date += pd.Timedelta(days=1)
        return True

    def _create_trades_direct(
        self, 
        symbols: list[str], 
        momentum_scores: pd.Series, 
        roc_matrices: dict[int, pd.DataFrame],
        analysis_date: pd.Timestamp, 
        price_data: dict[str, pd.DataFrame],
        regime_indicators: dict
    ) -> int:
        """Writes the selected leaders to the trades table as CREATED status."""
        date_iso_string = analysis_date.strftime("%Y-%m-%d")
        self.trade_repository.execute(
            "DELETE FROM trades WHERE strategy = ? AND status = 'CREATED'", 
            (self.name,)
        )
        
        created_count = 0
        for symbol in symbols:
            try:
                total_momentum_score = float(momentum_scores.at[symbol])
                closing_price = float(price_data["close"].at[analysis_date, symbol])
                
                trade_context = {
                    "source": "screener",
                    "date": date_iso_string,
                    "roc_1": round(float(roc_matrices[21].at[analysis_date, symbol]), 2),
                    "roc_3": round(float(roc_matrices[63].at[analysis_date, symbol]), 2),
                    "roc_6": round(float(roc_matrices[126].at[analysis_date, symbol]), 2),
                    "roc_12": round(float(roc_matrices[252].at[analysis_date, symbol]), 2),
                    "momentum_score": round(total_momentum_score, 2),
                    "qqq_regime": "BULL" if (regime_indicators["qqq"] > regime_indicators["qqq_sma"]) else "BEAR",
                    "regime": "BULL" if regime_indicators["bull"] else "BEAR"
                }

                self.trade_repository.create_trade(
                    symbol=symbol,
                    strategy=self.name,
                    size=0,
                    entry=round(closing_price, 2),
                    stop_loss=0.0,
                    target=0.0,
                    context=trade_context
                )
                created_count += 1
            except Exception as exception:
                logger.error("[%s] Error creating trade for %s: %s", self.name, symbol, exception)

        logger.info("[%s] Created %d CREATED trades.", self.name, created_count)
        return created_count
