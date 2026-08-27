import datetime
import logging
from typing import TypedDict, override

import pandas as pd

from ....const import Strategies
from ....database.repositories.market_data_provider import MarketDataProvider
from ....database.repositories.trade import TradeRepository
from ...telegram import TelegramBot
from ..models import SignalReportItem
from .base import BaseStrategy

logger = logging.getLogger(__name__)

WEDNESDAY_WEEKDAY: int = 2
THURSDAY_WEEKDAY: int = 3
FRIDAY_WEEKDAY: int = 4


class TwoPercentStrategyContext(TypedDict):
    """Context data for the TwoPercent signal."""

    date: str
    setup_close: float
    limit_entry: float
    day: str
    source: str


class TwoPercentStrategy(BaseStrategy[int]):
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
        telegram_bot: TelegramBot | None = None,
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

        last_candle = self._get_last_valid_candle(price_history, analysis_timestamp)
        if last_candle is None:
            return 0

        last_candle_timestamp = self._extract_timestamp(last_candle)

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

    def _get_real_today(self) -> datetime.date:
        """Returns the current date in real time.

        This boundary helper isolates the side-effect of querying the system clock.
        """
        return datetime.date.today()

    def _fetch_price_history(self, analysis_timestamp: pd.Timestamp) -> pd.DataFrame:
        """Fetches historical price data with sufficient lookback."""
        lookback = self.DEFAULT_LOOKBACK_PERIOD
        if analysis_timestamp < pd.Timestamp.now().normalize():
            days_ago = (pd.Timestamp.now() - analysis_timestamp).days
            lookback = max(self.DEFAULT_LOOKBACK_PERIOD, days_ago + 20)

        history = self.data_provider.get_symbol_history(self.SYMBOL, days=lookback)

        if history.empty:
            logger.warning(
                "[%s] No data for %s (Lookback: %d days)",
                self.name,
                self.SYMBOL,
                lookback,
            )

        return history

    def _get_last_valid_candle(
        self, history: pd.DataFrame, analysis_timestamp: pd.Timestamp
    ) -> pd.Series | None:
        """Filters history up to analysis_timestamp and returns candle if end of week."""
        if history.empty:
            logger.warning("History is empty")
            return None

        date_column = self._extract_date_series(history)
        if date_column is None:
            return None

        analysis_date = analysis_timestamp.date()
        filtered = history.loc[date_column <= analysis_date]
        if filtered.empty:
            logger.debug("[%s] No data up to %s", self.name, analysis_timestamp)
            return None

        last_candle = filtered.iloc[-1]
        candle_date = self._extract_timestamp(last_candle).date()
        if candle_date != analysis_date:
            return None

        if self._is_end_of_week_candle(candle_date, date_column, analysis_date):
            return last_candle

        return None

    def _is_end_of_week_candle(
        self,
        candle_date: datetime.date,
        date_column: pd.Series,
        analysis_date: datetime.date,
    ) -> bool:
        """Checks if the candle represents Friday or a valid fallback weekday."""
        weekday = candle_date.weekday()
        if weekday == FRIDAY_WEEKDAY:
            return True
        if weekday < FRIDAY_WEEKDAY:
            return self._is_fallback_weekday_end_of_week(
                candle_date, weekday, set(date_column), reference_date=analysis_date
            )
        return False

    def _extract_date_series(self, history: pd.DataFrame) -> pd.Series | None:
        """Extracts date column as pd.Series of datetime.date objects."""
        if "date" in history.columns:
            return pd.to_datetime(history["date"]).dt.date
        if not isinstance(history.index, pd.DatetimeIndex):
            logger.error("[%s] Missing DatetimeIndex or 'date' column.", self.name)
            return None
        return history.index.to_series().dt.date

    def _is_fallback_weekday_end_of_week(
        self,
        candle_date: datetime.date,
        weekday: int,
        existing_dates: set[datetime.date],
        reference_date: datetime.date | None = None,
    ) -> bool:
        """Helper to determine if a non-Friday weekday is the weekly EOD due to holiday.

        Args:
            candle_date: Date of the current candle.
            weekday: Weekday index (0=Monday, etc.).
            existing_dates: Set of all trading dates in the history.
            reference_date: Optional evaluation reference date (for historical backtesting).

        Returns:
            bool: True if this is the last trading day of the week.
        """
        friday = candle_date + datetime.timedelta(days=FRIDAY_WEEKDAY - weekday)

        # Check for any future days in the same week using standard python set membership
        future_days = [
            candle_date + datetime.timedelta(days=i)
            for i in range(1, FRIDAY_WEEKDAY - weekday + 1)
        ]
        for future_day in future_days:
            if future_day in existing_dates:
                # A subsequent day in this week exists in the DB.
                # Therefore, this candle is NOT the end of the week.
                return False

        # Ensure those missing days have actually happened in real time
        real_today = self._get_real_today()
        return friday < real_today

    def _extract_timestamp(self, candle: pd.Series) -> pd.Timestamp:
        """Extracts the timestamp from a candle series."""
        if "date" in candle:
            return pd.Timestamp(candle["date"])
        return pd.Timestamp(candle.name)

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
        weekday = timestamp.weekday()
        if weekday == FRIDAY_WEEKDAY:
            return "Friday"
        if weekday == THURSDAY_WEEKDAY:
            return "Thursday (Fallback)"
        if weekday == WEDNESDAY_WEEKDAY:
            return "Wednesday (Fallback)"
        return "Fallback"

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
            context=dict(context),
        )

    def _send_signal_report(self, date_str: str, close: float, entry: float) -> None:
        """Sends a notification report via Telegram if available."""
        if not self.telegram_bot:
            return

        report_items = [
            SignalReportItem(
                symbol=self.SYMBOL,
                action="BUY LMT",
                entry_price=round(entry, 2),
                details={"Setup Close": round(close, 2)},
            )
        ]

        self._send_telegram_report(
            f"{self.STRATEGY_IDENTIFIER} Entries", report_items, date_str
        )
