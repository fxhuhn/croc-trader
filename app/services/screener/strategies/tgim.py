"""TGIM (Thank God It's Monday) Screener Strategy.

Identifies Monday setup bars for SPY where Monday's close is strictly
lower than both Friday's close and Thursday's close.
"""

import datetime
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import TypedDict, override

import pandas as pd

from ....const import Strategies
from ....database.repositories.market_data_provider import MarketDataProvider
from ....database.repositories.trade import TradeRepository
from ....tools.market_holidays import MarketHolidayChecker
from ....tools.trading_calendar import (
    MONDAY,
    is_trading_day,
    roll_weekend_to_monday,
)
from ...telegram import TelegramBot
from ..models import SignalReportItem
from .base import BaseStrategy

logger = logging.getLogger(__name__)

MIN_PREMARKET_HISTORY_BARS: int = 2
MIN_POSTMARKET_HISTORY_BARS: int = 3


@dataclass(frozen=True)
class TGIMSetupResult:
    """Immutable result of pure TGIM setup evaluation."""

    is_signal: bool
    setup_close: Decimal
    threshold_price: Decimal
    friday_close: Decimal
    thursday_close: Decimal


class TGIMStrategyContext(TypedDict, total=False):
    """Context data payload for the TGIM signal."""

    date: str
    setup_date: str
    setup_close: float
    threshold_price: float
    friday_close: float
    thursday_close: float
    day: str
    max_holding_bars: int
    source: str


def evaluate_tgim_setup(
    current_close: Decimal,
    friday_close: Decimal,
    thursday_close: Decimal,
) -> TGIMSetupResult:
    """Pure calculation: Evaluates TGIM setup condition without side effects.

    Setup Condition: Monday Close < min(Friday Close, Thursday Close).
    """
    threshold_price = min(friday_close, thursday_close)
    is_signal = current_close < threshold_price

    return TGIMSetupResult(
        is_signal=is_signal,
        setup_close=current_close,
        threshold_price=threshold_price,
        friday_close=friday_close,
        thursday_close=thursday_close,
    )


class TGIMStrategy(BaseStrategy[int]):
    """Implementation of the TGIM trading strategy.

    Strategy Logic:
    - Asset: SPY exclusively.
    - Timing: Calendar Monday only (date.weekday() == 0). Skipped on holidays.
    - Setup Threshold: min(Friday Close, Thursday Close).
    - Setup Trigger: Monday Close < min(Friday Close, Thursday Close).
    - Pre-Market Screening: Prepares SPY setup candidate on Monday using Friday and Thursday closes.
    - Entry: Market On Close (MOC) at Monday's close if condition is met.
    - Exits: c1exit (Close > Close[1]) or TE (Time Exit at Bar 2 / Wednesday close).
    """

    STRATEGY_IDENTIFIER = Strategies.TGIM
    name = Strategies.TGIM
    TARGET_SYMBOL = "SPY"
    DEFAULT_MAX_HOLDING_BARS = 2
    DEFAULT_LOOKBACK_PERIOD = 30

    def __init__(
        self,
        trade_repository: TradeRepository,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None,
        *,
        holiday_checker: MarketHolidayChecker | None = None,
    ) -> None:
        """Initializes the TGIM screener strategy with required dependencies."""
        super().__init__(data_provider=data_provider, telegram_bot=telegram_bot)
        self.trade_repository = trade_repository
        self.holiday_checker = holiday_checker or MarketHolidayChecker()

    @override
    def run(self, days: int = 0, analysis_date: str | None = None) -> int:
        """Executes the TGIM screening logic for the specified date."""
        target_date = self._resolve_target_date(days, analysis_date)
        if not self._is_eligible_monday(target_date):
            return 0

        target_date_str = target_date.strftime("%Y-%m-%d")

        history_map = self.data_provider.get_batch_history(
            symbols=[self.TARGET_SYMBOL],
            days=self.DEFAULT_LOOKBACK_PERIOD,
            end_date=target_date_str,
        )
        price_history = history_map.get(self.TARGET_SYMBOL, pd.DataFrame())

        if price_history.empty or len(price_history) < MIN_PREMARKET_HISTORY_BARS:
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

        if candle_date == target_date:
            # Monday candle is present in history (post-market or historical backtest)
            if len(price_history) < MIN_POSTMARKET_HISTORY_BARS:
                return 0
            current_close = Decimal(str(latest_candle["close"]))
            friday_close = Decimal(str(price_history.iloc[-2]["close"]))
            thursday_close = Decimal(str(price_history.iloc[-3]["close"]))

            setup_result = evaluate_tgim_setup(
                current_close=current_close,
                friday_close=friday_close,
                thursday_close=thursday_close,
            )

            if not setup_result.is_signal:
                logger.debug(
                    "TGIM setup condition failed for SPY on %s: close=%s not < min(%s, %s).",
                    target_date_str,
                    current_close,
                    friday_close,
                    thursday_close,
                )
                return 0

            threshold_price = setup_result.threshold_price
            setup_close = float(current_close)
        else:
            # Pre-market live screening (Monday candle not in DB yet)
            friday_close = Decimal(str(latest_candle["close"]))
            thursday_close = Decimal(str(price_history.iloc[-2]["close"]))
            threshold_price = min(friday_close, thursday_close)
            setup_close = 0.0

        threshold_price_float = float(threshold_price)
        context_data: TGIMStrategyContext = {
            "date": target_date_str,
            "setup_date": target_date_str,
            "setup_close": setup_close,
            "threshold_price": threshold_price_float,
            "friday_close": float(friday_close),
            "thursday_close": float(thursday_close),
            "day": "Monday",
            "max_holding_bars": self.DEFAULT_MAX_HOLDING_BARS,
            "source": "ScreenerEngine",
        }

        if self.trade_repository.exists(
            self.TARGET_SYMBOL, self.STRATEGY_IDENTIFIER, target_date_str
        ):
            logger.info(
                "TGIM trade already exists for %s on %s.",
                self.TARGET_SYMBOL,
                target_date_str,
            )
            return 0

        self.trade_repository.create_trade(
            symbol=self.TARGET_SYMBOL,
            strategy=self.STRATEGY_IDENTIFIER,
            size=0.0,
            entry=threshold_price_float,
            stop_loss=0.0,
            target=0.0,
            context=dict(context_data),
        )
        logger.info(
            "[%s] CREATED trade recorded for %s @ %s (threshold: %s).",
            self.name,
            self.TARGET_SYMBOL,
            threshold_price_float,
            threshold_price_float,
        )

        if self.telegram_bot:
            self._send_telegram_report(
                "TGIM",
                [
                    SignalReportItem(
                        symbol=self.TARGET_SYMBOL,
                        action="BUY MOC",
                        entry_price=round(threshold_price_float, 2),
                    )
                ],
                target_date_str,
            )

        return 1

    def _resolve_target_date(
        self, days: int, analysis_date: str | None
    ) -> datetime.date:
        """Resolves the target analysis date as a datetime.date object.

        If the analysis date falls on Friday (weekend setup bar), Saturday, or Sunday,
        it automatically rolls forward to the target Monday for TGIM screening.
        """
        resolved_date = self._resolve_analysis_date(days, analysis_date)
        return roll_weekend_to_monday(resolved_date)

    def _is_eligible_monday(self, target_date: datetime.date) -> bool:
        """Validates that target_date is a non-holiday Monday."""
        if target_date.weekday() != MONDAY:
            logger.debug("Skipping TGIM screening for %s (not a Monday).", target_date)
            return False

        if not is_trading_day(target_date, self.holiday_checker):
            logger.info(
                "Skipping TGIM screening for %s (market holiday: %s).",
                target_date,
                self.holiday_checker.get_holiday_name(target_date) or "Holiday",
            )
            return False

        return True


__all__ = [
    "MIN_POSTMARKET_HISTORY_BARS",
    "MIN_PREMARKET_HISTORY_BARS",
    "TGIMSetupResult",
    "TGIMStrategy",
    "TGIMStrategyContext",
    "evaluate_tgim_setup",
]
