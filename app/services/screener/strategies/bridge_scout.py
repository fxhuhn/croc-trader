"""Bridge Scout Screener Strategy.

End-of-Month Mean-Reversion strategy for QQQ that buys short-term dips
in the final trading days of the calendar month and exits on the first
trading day of the next month.
"""

import datetime
import logging
from typing import TypedDict, override

import pandas as pd

from ....const import Strategies
from ....database.repositories.market_data_provider import MarketDataProvider
from ....database.repositories.trade import TradeRepository
from ....tools.indicators import (
    calculate_atr,
    calculate_max_close_for_rsi,
    calculate_rsi,
)
from ....tools.market_holidays import MarketHolidayChecker
from ....types import TradeStatus
from ...telegram import TelegramBot
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class BridgeScoutStrategyContext(TypedDict):
    """Context payload stored with Bridge Scout signals."""

    date: str
    setup_close: float
    rsi_2: float
    atr_pct: float
    req_close_rsi40: float
    source: str


def get_remaining_trading_days_in_month(
    check_date: datetime.date,
    holiday_checker: MarketHolidayChecker | None = None,
) -> int:
    """Calculates remaining trading days in check_date's month including check_date.

    Args:
        check_date: Target date to check.
        holiday_checker: Optional holiday checker instance.

    Returns:
        int: Number of remaining trading days (>= 1).
    """
    checker = holiday_checker or MarketHolidayChecker()

    # Find month boundary (last calendar day of the month)
    if check_date.month == 12:
        next_month_start = datetime.date(check_date.year + 1, 1, 1)
    else:
        next_month_start = datetime.date(check_date.year, check_date.month + 1, 1)
    last_day_of_month = next_month_start - datetime.timedelta(days=1)

    trading_day_count = 0
    current = check_date
    while current <= last_day_of_month:
        if current.weekday() < 5 and not checker.is_holiday(current):
            trading_day_count += 1
        current += datetime.timedelta(days=1)

    return trading_day_count


def is_in_end_of_month_window(
    check_date: datetime.date,
    days_before: int = 4,
    holiday_checker: MarketHolidayChecker | None = None,
) -> bool:
    """Checks whether check_date falls within the End-of-Month window.

    Window is active starting days_before trading days prior to the last
    trading day of the month up to the last trading day.

    Args:
        check_date: Target date.
        days_before: Number of trading days before month end (default: 4).
        holiday_checker: Optional holiday checker.

    Returns:
        bool: True if in month-end window, False otherwise.
    """
    remaining_days = get_remaining_trading_days_in_month(
        check_date, holiday_checker=holiday_checker
    )
    return 1 <= remaining_days <= (days_before + 1)


class BridgeScoutStrategy(BaseStrategy[int]):
    """Implementation of the Bridge Scout trading strategy.

    Strategy Logic:
    - Asset: QQQ exclusively.
    - Timing: End-of-Month window (last 5 trading days of the calendar month).
    - Setup: RSI(2) < 40.0 AND (ATR(10) / Close) * 100 < 3.5%.
    - Entry: Market On Close (MOC) at setup close.
    - Exit: Market On Close (MOC) on 1st trading day of new calendar month.
    - Guard: MaxPositions = 1 (strictly single position allowed).
    """

    STRATEGY_IDENTIFIER = Strategies.BridgeScout
    TARGET_SYMBOL = "QQQ"
    DEFAULT_ENTRY_DAYS_BEFORE = 4
    DEFAULT_RSI_THRESHOLD = 40.0
    DEFAULT_ATR_LEN = 10
    DEFAULT_MAX_ATR_PCT = 3.5
    DEFAULT_LOOKBACK_PERIOD = 60

    def __init__(
        self,
        trade_repository: TradeRepository,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None,
        holiday_checker: MarketHolidayChecker | None = None,
    ) -> None:
        """Initializes Bridge Scout screener strategy with required dependencies."""
        super().__init__(data_provider=data_provider, telegram_bot=telegram_bot)
        self.trade_repository = trade_repository
        self.holiday_checker = holiday_checker or MarketHolidayChecker()

    @override
    def run(self, days: int = 0, analysis_date: str | None = None) -> int:
        """Executes Bridge Scout screening for specified date."""
        target_date = self._resolve_target_date(days, analysis_date)
        target_date_str = target_date.strftime("%Y-%m-%d")

        if not is_in_end_of_month_window(
            target_date,
            days_before=self.DEFAULT_ENTRY_DAYS_BEFORE,
            holiday_checker=self.holiday_checker,
        ):
            logger.debug(
                "Skipping Bridge Scout screening for %s (outside month-end window).",
                target_date_str,
            )
            return 0

        history_map = self.data_provider.get_batch_history(
            symbols=[self.TARGET_SYMBOL],
            days=self.DEFAULT_LOOKBACK_PERIOD,
            end_date=target_date_str,
        )
        price_history = history_map.get(self.TARGET_SYMBOL, pd.DataFrame())

        if price_history.empty or len(price_history) < (self.DEFAULT_ATR_LEN + 1):
            logger.warning(
                "Insufficient price history for %s on %s.",
                self.TARGET_SYMBOL,
                target_date_str,
            )
            return 0

        latest_candle = price_history.iloc[-1]
        raw_date = latest_candle["date"]
        candle_date = (
            pd.Timestamp(raw_date).date()
            if isinstance(raw_date, str)
            else raw_date.date()
        )

        if candle_date != target_date:
            logger.debug(
                "Market data date %s does not match target date %s (likely holiday).",
                candle_date,
                target_date,
            )
            return 0

        # Calculate indicators
        close_series = price_history["close"].astype(float)
        high_series = price_history["high"].astype(float)
        low_series = price_history["low"].astype(float)

        rsi_series = calculate_rsi(close_series, window=2)
        atr_series = calculate_atr(
            high_series, low_series, close_series, window=self.DEFAULT_ATR_LEN
        )

        current_close = float(close_series.iloc[-1])
        current_rsi = float(rsi_series.iloc[-1])
        current_atr = float(atr_series.iloc[-1])
        atr_pct = (current_atr / current_close) * 100.0

        if current_rsi >= self.DEFAULT_RSI_THRESHOLD:
            logger.debug(
                "Bridge Scout setup failed for QQQ on %s: RSI(2)=%.2f >= %.2f.",
                target_date_str,
                current_rsi,
                self.DEFAULT_RSI_THRESHOLD,
            )
            return 0

        if atr_pct >= self.DEFAULT_MAX_ATR_PCT:
            logger.debug(
                "Bridge Scout setup failed for QQQ on %s: ATR%%=%.2f%% >= %.2f%%.",
                target_date_str,
                atr_pct,
                self.DEFAULT_MAX_ATR_PCT,
            )
            return 0

        # Strict single position check (MaxPositions = 1)
        active_or_created = self.trade_repository.get_by_status(
            [TradeStatus.CREATED, TradeStatus.ACTIVE]
        )
        if any(
            t.get("symbol") == self.TARGET_SYMBOL
            and "bridge_scout" in str(t.get("strategy")).lower()
            for t in active_or_created
        ) or self.trade_repository.exists(
            self.TARGET_SYMBOL, self.STRATEGY_IDENTIFIER, target_date_str
        ):
            logger.info(
                "Bridge Scout trade or active position already exists for %s on %s.",
                self.TARGET_SYMBOL,
                target_date_str,
            )
            return 0

        req_close_rsi40 = calculate_max_close_for_rsi(
            close_series.iloc[:-1],
            window=2,
            rsi_target=self.DEFAULT_RSI_THRESHOLD,
        )

        context: BridgeScoutStrategyContext = {
            "date": target_date_str,
            "setup_close": current_close,
            "rsi_2": round(current_rsi, 2),
            "atr_pct": round(atr_pct, 2),
            "req_close_rsi40": round(req_close_rsi40, 2),
            "source": "ScreenerEngine",
        }

        trade_id = self.trade_repository.create_trade(
            symbol=self.TARGET_SYMBOL,
            strategy=self.STRATEGY_IDENTIFIER.value,
            size=0.0,
            entry=current_close,
            stop_loss=0.0,
            target=0.0,
            context=context,
        )

        logger.info(
            "Generated Bridge Scout signal for %s on %s (Trade ID: %s).",
            self.TARGET_SYMBOL,
            target_date_str,
            trade_id,
        )
        return 1

    def _resolve_target_date(
        self, days: int, analysis_date: str | None
    ) -> datetime.date:
        """Resolves target analysis date as datetime.date object."""
        if analysis_date:
            return datetime.datetime.strptime(analysis_date, "%Y-%m-%d").date()
        target_datetime = datetime.datetime.now() - datetime.timedelta(days=days)
        return target_datetime.date()
