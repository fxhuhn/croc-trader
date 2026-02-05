import logging
import pandas as pd
from typing import override

from ....database.repositories.trade import TradeRepository
from ....database.repositories.market_data_provider import MarketDataProvider
from ...telegram import TelegramBot
from .base import BaseStrategy

logger = logging.getLogger(__name__)

class TwoPercentStrategy(BaseStrategy):
    """
    SXRV.DE Strategy:
    - Runs on Fridays.
    - Entry: Limit at 99% of Friday Close.
    """
    def __init__(
        self,
        trade_repo: TradeRepository,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None
    ):
        super().__init__(data_provider, telegram_bot)
        self.name = "TwoPercentStrategy"
        self.trade_repo = trade_repo
        self.symbol = "SXRV.DE"

    @override
    def run(self, days: int = 0, analysis_date: str | None = None) -> int:
        # Determine Analysis Date
        if analysis_date:
            today = pd.Timestamp(analysis_date).normalize()
        else:
            today = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)

        # 1. Check if Friday (weekday 4)
        if today.dayofweek != 4:
            # logger.info(f"[{self.name}] Skipped: {today.date()} is not a Friday.")
            return 0

        # 2. Get Data for SXRV.DE
        # We need the close of 'today'
        # Fetch a bit of history to be safe
        df = self.data_provider.get_symbol_history(self.symbol, days=10)
        
        if df.empty:
            logger.warning(f"[{self.name}] No data found for {self.symbol}")
            return 0

        # Ensure we have data for 'today'
        # df index is datetime
        try:
            # Locate the specific date
            # We must normalize the index to compare dates safely
            df_dates = df.index.normalize()
            if today not in df_dates:
               # Try to find the last available date if today is not in DB yet (e.g. running late Friday)
               # But strategy demands "Friday Close". If data is missing for Friday, we can't run.
               logger.warning(f"[{self.name}] No data for analysis date {today.date()}")
               return 0
            
            # Get the row
            row = df.loc[df_dates == today].iloc[0]
            close_price = float(row['close'])
            
        except Exception as e:
            logger.error(f"[{self.name}] Error reading data: {e}")
            return 0

        # 3. Calculate Limit Entry (1% discount)
        limit_entry = round(close_price * 0.99, 2)
        
        # 4. Check if trade already exists
        strat_key = "two_percent_strategy"
        signal_date_str = str(today.date())
        
        # Check existence using trade_repo
        if self.trade_repo.exists(self.symbol, strat_key, signal_date_str):
            return 0

        # 5. Create Trade
        context = {
            "date": signal_date_str,
            "setup_close": close_price,
            "limit_entry": limit_entry,
            "day": "Friday"
        }

        self.trade_repo.create_trade(
            symbol=self.symbol,
            strategy=strat_key,
            size=0, # Budget managed later
            entry=limit_entry,
            sl=0.0,
            target=0.0,
            context=context
        )
        
        # 6. Telegram Report
        if self.telegram_bot:
           report_rows = [{
               "Symbol": self.symbol,
               "Freitag Close": f"{close_price:.2f}",
               "Entry": f"{limit_entry:.2f}"
           }]
           df_report = pd.DataFrame(report_rows)
           self._send_telegram_report(f"Two Percent Entries", signal_date_str, df_report)

        return 1
