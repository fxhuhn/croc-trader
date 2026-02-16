import logging
import pandas as pd
from typing import override, TypedDict

from ....database.repositories.trade import TradeRepository
from ....database.repositories.market_data_provider import MarketDataProvider
from ...telegram import TelegramBot
from .base import BaseStrategy
from ....tools.market_holidays import MarketHolidayChecker
from ....const import Strategies

logger = logging.getLogger(__name__)

class TwoPercentStrategyContext(TypedDict):
    """Context data for the TwoPercent signal."""
    date: str
    setup_close: float
    limit_entry: float
    day: str
    source: str

class TwoPercentStrategy(BaseStrategy):
    """
    TwoPercent Strategy:
    - Runs on Fridays at market close.
    - EXCEPTION: If Friday is a holiday, runs on Thursday close.
    - Entry: Limit at 99% of Setup Close.
    """
    STRATEGY_IDENTIFIER = Strategies.TwoPercent
    ENTRY_LIMIT_DISCOUNT = 0.99
    
    def __init__(
        self,
        trade_repository: TradeRepository,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None
    ):
        super().__init__(data_provider, telegram_bot)
        self.name = self.STRATEGY_IDENTIFIER
        self.trade_repository = trade_repository
        self.symbol = "SXRV.DE"
        self.holiday_checker = MarketHolidayChecker()

    @override
    def run(self, days: int = 0, analysis_date: str | None = None) -> int:
        """
        Executes the strategy logic to find setup candles.

        Args:
            days: Lookback periods (unused here).
            analysis_date: The date to run the analysis FOR.
                           If None, uses current system date.

        Returns:
            int: 1 if signal was generated and persisted, 0 otherwise.
        """
        # 1. Determine "Today" (Analysis Date)
        if analysis_date:
            current_date_timestamp = pd.Timestamp(analysis_date)
        else:
            current_date_timestamp = pd.Timestamp.now().normalize()

        # 2. Get Data for SXRV.DE
        lookback_days = 20
        if analysis_date:
            days_since_now = (pd.Timestamp.now() - current_date_timestamp).days
            lookback_days = max(20, days_since_now + 20)

        price_history = self.data_provider.get_symbol_history(
            self.symbol, days=lookback_days
        )
        
        if price_history.empty:
            logger.warning(
                "[%s] No data found for %s (Lookback: %s days)",
                self.name, self.symbol, lookback_days
            )
            return 0

        # 3. Filter/Slice Data up to analysis_date
        if 'date' in price_history.columns:
            price_history['date'] = pd.to_datetime(price_history['date'])
            date_mask = price_history['date'] <= current_date_timestamp
            dataframe_slice = price_history.loc[date_mask]
        else:
            if not isinstance(price_history.index, pd.DatetimeIndex):
                logger.error(
                    "[%s] DataFrame index is not DatetimeIndex and 'date' column missing.",
                    self.name
                )
                return 0
            dataframe_slice = price_history.loc[price_history.index <= current_date_timestamp]

        if dataframe_slice.empty:
            logger.debug("[%s] No data found up to %s", self.name, current_date_timestamp)
            return 0

        # 4. Get the "Last Candle"
        last_candle = dataframe_slice.iloc[-1]
        
        if 'date' in dataframe_slice.columns:
            last_candle_timestamp = pd.Timestamp(last_candle['date'])
        else:
            last_candle_timestamp = pd.Timestamp(last_candle.name)
            
        last_candle_date = last_candle_timestamp.date()

        # Only proceed if we have data for the requested analysis period
        if last_candle_date != current_date_timestamp.date():
            return 0
        
        # 5. Validate Setup Day (Friday or Thursday holiday exception)
        is_valid_setup_day = False
        weekday = last_candle_date.weekday()
        
        if weekday == 4: # Friday
            is_valid_setup_day = True
        elif weekday == 3: # Thursday
            next_day_timestamp = last_candle_timestamp + pd.Timedelta(days=1)
            if self.holiday_checker.is_holiday(next_day_timestamp.date()):
                logger.info(
                    "[%s] Thursday Candle (%s) accepted because Friday is holiday.",
                    self.name, last_candle_date
                )
                is_valid_setup_day = True
        
        if not is_valid_setup_day:
            return 0

        # 6. Extract Price
        try:
            close_price = float(last_candle['close'])
            if pd.isna(close_price):
                 logger.warning(
                     "[%s] Close price for %s on %s is NaN.",
                     self.name, self.symbol, last_candle_date
                 )
                 return 0
        except (KeyError, ValueError, TypeError) as error:
            logger.error("[%s] Error reading close price: %s", self.name, error)
            return 0

        # 7. Calculate Limit Entry
        limit_entry = round(close_price * self.ENTRY_LIMIT_DISCOUNT, 2)
        
        # 8. Check if trade already exists
        signal_date_string = str(last_candle_date)
        if self.trade_repository.exists(self.symbol, self.STRATEGY_IDENTIFIER, signal_date_string):
            return 0

        # 9. Create Trade Proposal
        context: TwoPercentStrategyContext = {
            "date": signal_date_string,
            "setup_close": close_price,
            "limit_entry": limit_entry,
            "day": "Friday" if weekday == 4 else "Thursday (Holiday Exception)",
            "source": "screener",
        }

        self.trade_repository.create_trade(
            symbol=self.symbol,
            strategy=self.STRATEGY_IDENTIFIER,
            size=0,
            entry=limit_entry,
            stop_loss=0.0,
            target=0.0,
            context=dict(context) # type: ignore
        )
        
        # 10. Telegram Report
        if self.telegram_bot:
           report_rows = [{
               "Symbol": self.symbol,
               "Setup Close": f"{close_price:.2f}",
               "Limit Entry": f"{limit_entry:.2f}"
           }]
           report_dataframe = pd.DataFrame(report_rows)
           self._send_telegram_report(
               f"{self.STRATEGY_IDENTIFIER} Entries", 
               signal_date_string, 
               report_dataframe
           )

        return 1
