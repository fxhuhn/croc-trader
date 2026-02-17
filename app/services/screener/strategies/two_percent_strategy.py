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
    Implementation of the TwoPercent trading strategy.
    
    Strategy Logic:
    - Execution: Runs on Fridays at market close.
    - Exception: If Friday is a holiday, it runs on Thursday close.
    - Entry: Limit order at 99% of the 'Setup Close' (Friday/Thursday close).
    """
    
    STRATEGY_IDENTIFIER = Strategies.TwoPercent
    ENTRY_LIMIT_DISCOUNT = 0.99
    DEFAULT_LOOKBACK_PERIOD = 20
    SYMBOL = "SXRV.DE"

    def __init__(
        self,
        trade_repository: TradeRepository,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None
    ) -> None:
        """
        Initializes the TwoPercent strategy with required dependencies.
        
        Args:
           trade_repository: Repository for trade persistence.
           data_provider: Provider for market historical data.
           telegram_bot: Optional bot for reporting signals.
        """
        super().__init__(data_provider, telegram_bot)
        self.name = self.STRATEGY_IDENTIFIER
        self.trade_repository = trade_repository
        self.holiday_checker = MarketHolidayChecker()

    @override
    def run(self, days: int = 0, analysis_date: str | None = None) -> int:
        """
        Orchestrates the strategy execution for a given analysis date.
        
        This follows the Step-down Rule by delegating logic to specialized methods.
        
        Returns:
           int: 1 if a signal was generated and saved, 0 otherwise.
        """
        analysis_timestamp = self._get_analysis_timestamp(analysis_date)
        
        price_history = self._fetch_price_history(analysis_timestamp)
        if price_history.empty:
            return 0

        last_candle = self._get_last_valid_candle(
            price_history, analysis_timestamp
        )
        if last_candle is None:
            return 0

        last_candle_timestamp = self._extract_timestamp(last_candle)
        if not self._is_valid_setup_day(last_candle_timestamp):
            return 0

        close_price = self._extract_close_price(last_candle)
        if close_price is None:
            return 0

        entry_price = self._calculate_entry_price(close_price)
        signal_date_str = str(last_candle_timestamp.date())

        if self._trade_exists(signal_date_str):
            return 0

        day_label = self._get_day_label(last_candle_timestamp)
        self._create_trade_proposal(
            signal_date_str, close_price, entry_price, day_label
        )
        
        self._send_signal_report(signal_date_str, close_price, entry_price)

        return 1

    def _get_analysis_timestamp(self, analysis_date: str | None) -> pd.Timestamp:
        """Determines the effective timestamp for analysis."""
        if analysis_date:
            return pd.Timestamp(analysis_date)
        return pd.Timestamp.now().normalize()

    def _fetch_price_history(self, analysis_timestamp: pd.Timestamp) -> pd.DataFrame:
        """Fetches historical price data with sufficient lookback."""
        lookback = self.DEFAULT_LOOKBACK_PERIOD
        if analysis_timestamp < pd.Timestamp.now().normalize():
            days_ago = (pd.Timestamp.now() - analysis_timestamp).days
            lookback = max(self.DEFAULT_LOOKBACK_PERIOD, days_ago + 20)

        history = self.data_provider.get_symbol_history(
            self.SYMBOL, days=lookback
        )
        
        if history.empty:
            logger.warning(
                "[%s] No data for %s (Lookback: %d days)",
                self.name, self.SYMBOL, lookback
            )
        
        return history

    def _get_last_valid_candle(
        self, history: pd.DataFrame, analysis_timestamp: pd.Timestamp
    ) -> pd.Series | None:
        """Filters history up to the analysis date and returns the latest candle."""
        if "date" in history.columns:
            history["date"] = pd.to_datetime(history["date"])
            mask = history["date"] <= analysis_timestamp
            filtered = history.loc[mask]
        else:
            if not isinstance(history.index, pd.DatetimeIndex):
                logger.error("[%s] Missing DatetimeIndex or 'date' column.", self.name)
                return None
            filtered = history.loc[history.index <= analysis_timestamp]

        if filtered.empty:
            logger.debug("[%s] No data up to %s", self.name, analysis_timestamp)
            return None

        last_candle = filtered.iloc[-1]
        candle_date = self._extract_timestamp(last_candle).date()
        
        if candle_date != analysis_timestamp.date():
            return None
            
        return last_candle

    def _extract_timestamp(self, candle: pd.Series) -> pd.Timestamp:
        """Extracts the timestamp from a candle series."""
        if "date" in candle:
            return pd.Timestamp(candle["date"])
        return pd.Timestamp(candle.name)

    def _is_valid_setup_day(self, timestamp: pd.Timestamp) -> bool:
        """Checks if the given timestamp is a valid strategy execution day."""
        weekday = timestamp.weekday()
        
        if weekday == 4:  # Friday
            return True
            
        if weekday == 3:  # Thursday
            next_day = (timestamp + pd.Timedelta(days=1)).date()
            if self.holiday_checker.is_holiday(next_day):
                logger.info(
                    "[%s] Thursday (%s) accepted (Friday holiday).",
                    self.name, timestamp.date()
                )
                return True
                
        return False

    def _extract_close_price(self, candle: pd.Series) -> float | None:
        """Safely extracts the close price from a candle."""
        try:
            close = float(candle["close"])
            if pd.isna(close):
                logger.warning("[%s] Close price is NaN.", self.name)
                return None
            return close
        except (KeyError, ValueError, TypeError) as error:
            logger.error("[%s] Error reading close price: %s", self.name, error)
            return None

    def _calculate_entry_price(self, close_price: float) -> float:
        """Calculates the limit entry price based on the discount."""
        return round(close_price * self.ENTRY_LIMIT_DISCOUNT, 2)

    def _trade_exists(self, date_str: str) -> bool:
        """Checks if a trade for this strategy and date already exists."""
        return self.trade_repository.exists(
            self.SYMBOL, self.STRATEGY_IDENTIFIER, date_str
        )

    def _get_day_label(self, timestamp: pd.Timestamp) -> str:
        """Returns a human-readable label for the execution day."""
        if timestamp.weekday() == 4:
            return "Friday"
        return "Thursday (Holiday Exception)"

    def _create_trade_proposal(
        self, date_str: str, close: float, entry: float, day_label: str
    ) -> None:
        """Persists the generated signal as a trade proposal."""
        context: TwoPercentStrategyContext = {
            "date": date_str,
            "setup_close": close,
            "limit_entry": entry,
            "day": day_label,
            "source": "screener",
        }

        self.trade_repository.create_trade(
            symbol=self.SYMBOL,
            strategy=self.STRATEGY_IDENTIFIER,
            size=0,
            entry=entry,
            stop_loss=0.0,
            target=0.0,
            context=dict(context)  # type: ignore
        )

    def _send_signal_report(self, date_str: str, close: float, entry: float) -> None:
        """Sends a notification report via Telegram if available."""
        if not self.telegram_bot:
            return

        report_data = pd.DataFrame([{
            "Symbol": self.SYMBOL,
            "Setup Close": f"{close:.2f}",
            "Limit Entry": f"{entry:.2f}"
        }])
        
        self._send_telegram_report(
            f"{self.STRATEGY_IDENTIFIER} Entries",
            date_str,
            report_data
        )
