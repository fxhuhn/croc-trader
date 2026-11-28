"""TGIM (Thank God It's Monday) Screener Strategy.

Identifies Monday setup bars for SPY where Monday's close is strictly below
both Friday's close and Thursday's close (RealTest: C < min(C[1], C[2])).
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
    get_last_completed_trading_day,
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

    Setup Condition: Monday Close < min(C[1], C[2]) (strict inequality matching
    RealTest reference logic; mapped to BUY LOC at threshold_price - tick_size).
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


@dataclass(frozen=True)
class TGIMConfiguration:
    """Central configuration parameters for the TGIM (Thank God It's Monday) trading strategy."""

    target_symbol: str = "SPY"
    max_holding_bars: int = 2
    lookback_period: int = 30
    min_premarket_history_bars: int = 2
    min_postmarket_history_bars: int = 3
    tick_size: Decimal = Decimal("0.01")


class TGIMStrategy(BaseStrategy[int]):
    """Implementation of the TGIM trading strategy.

    Strategy Logic:
    - Asset: SPY exclusively.
    - Timing: Calendar Monday only (date.weekday() == 0). Skipped on holidays.
    - Setup Threshold: min(C[1], C[2]) (last two completed trading closes).
    - Setup Trigger: Monday Close < min(C[1], C[2]) (strict RealTest inequality).
    - Pre-Market Screening: Prepares SPY setup candidate on Monday using C[1] and C[2] closes.
    - Entry: Limit On Close (LOC) at Monday's close with limit = threshold_price - tick_size.
    - Exits: Bar 1 SELL LOC at C[1] + tick_size (Close > Close[1]) or Bar 2 MOC Time Exit.
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
        configuration: TGIMConfiguration | None = None,
    ) -> None:
        """Initializes the TGIM screener strategy with required dependencies."""
        super().__init__(data_provider=data_provider, telegram_bot=telegram_bot)
        self.trade_repository = trade_repository
        self.holiday_checker = holiday_checker or MarketHolidayChecker()
        self.configuration = configuration or TGIMConfiguration()

    @override
    def run(self, days: int = 0, analysis_date: str | None = None) -> int:
        """Executes the TGIM screening logic for the specified date."""
        resolved_date = self._resolve_analysis_date(days, analysis_date)
        target_date = roll_weekend_to_monday(resolved_date)
        if not self._is_eligible_monday(target_date):
            return 0

        target_date_str = target_date.strftime("%Y-%m-%d")
        history_end_date_str = resolved_date.strftime("%Y-%m-%d")

        history_map = self.data_provider.get_batch_history(
            symbols=[self.configuration.target_symbol],
            days=self.configuration.lookback_period,
            end_date=history_end_date_str,
        )
        price_history = history_map.get(
            self.configuration.target_symbol, pd.DataFrame()
        )

        setup_prices = self._resolve_setup_prices(
            price_history, target_date, target_date_str
        )
        if setup_prices is None:
            return 0

        threshold_price, setup_close, friday_close, thursday_close = setup_prices
        threshold_price_float = float(threshold_price)

        if self._has_existing_trade_or_position(
            self.trade_repository,
            self.configuration.target_symbol,
            self.STRATEGY_IDENTIFIER,
            target_date_str,
        ):
            logger.info(
                "TGIM trade or active position already exists for %s on %s.",
                self.configuration.target_symbol,
                target_date_str,
            )
            return 0

        context_data: TGIMStrategyContext = {
            "date": target_date_str,
            "setup_date": target_date_str,
            "setup_close": setup_close,
            "threshold_price": threshold_price_float,
            "friday_close": float(friday_close),
            "thursday_close": float(thursday_close),
            "day": "Monday",
            "max_holding_bars": self.configuration.max_holding_bars,
            "source": "ScreenerEngine",
        }

        self.trade_repository.create_trade(
            symbol=self.configuration.target_symbol,
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
            self.configuration.target_symbol,
            threshold_price_float,
            threshold_price_float,
        )

        if self.telegram_bot:
            self._send_telegram_report(
                "TGIM",
                [
                    SignalReportItem(
                        symbol=self.configuration.target_symbol,
                        action="BUY LOC",
                        entry_price=round(threshold_price_float, 2),
                    )
                ],
                target_date_str,
            )

        return 1

    def _resolve_setup_prices(
        self,
        price_history: pd.DataFrame,
        target_date: datetime.date,
        target_date_str: str,
    ) -> tuple[Decimal, float, Decimal, Decimal] | None:
        """Resolves (threshold_price, setup_close, friday_close, thursday_close) with freshness guards."""
        if (
            price_history.empty
            or len(price_history) < self.configuration.min_premarket_history_bars
        ):
            logger.warning(
                "Insufficient price history for %s on %s.",
                self.configuration.target_symbol,
                target_date_str,
            )
            return None

        latest_candle = price_history.iloc[-1]
        candle_date = self._extract_candle_date(latest_candle["date"])
        expected_prev_trading_day = get_last_completed_trading_day(
            target_date, self.holiday_checker
        )

        if candle_date == target_date:
            return self._resolve_postmarket_setup_prices(
                price_history, target_date_str, expected_prev_trading_day
            )

        if candle_date != expected_prev_trading_day:
            logger.warning(
                "Stale pre-market history for %s targeting %s: expected latest candle %s, got %s.",
                self.configuration.target_symbol,
                target_date_str,
                expected_prev_trading_day,
                candle_date,
            )
            return None

        friday_close = Decimal(str(latest_candle["close"]))
        thursday_close = Decimal(str(price_history.iloc[-2]["close"]))
        return min(friday_close, thursday_close), 0.0, friday_close, thursday_close

    def _resolve_postmarket_setup_prices(
        self,
        price_history: pd.DataFrame,
        target_date_str: str,
        expected_prev_trading_day: datetime.date,
    ) -> tuple[Decimal, float, Decimal, Decimal] | None:
        """Validates post-market history continuity and evaluates Monday setup condition."""
        if len(price_history) < self.configuration.min_postmarket_history_bars:
            return None

        prev_candle_date = self._extract_candle_date(price_history.iloc[-2]["date"])
        if prev_candle_date != expected_prev_trading_day:
            logger.warning(
                "Stale or gapped prior candle for %s on %s: expected %s, got %s.",
                self.configuration.target_symbol,
                target_date_str,
                expected_prev_trading_day,
                prev_candle_date,
            )
            return None

        current_close = Decimal(str(price_history.iloc[-1]["close"]))
        friday_close = Decimal(str(price_history.iloc[-2]["close"]))
        thursday_close = Decimal(str(price_history.iloc[-3]["close"]))
        setup_result = evaluate_tgim_setup(
            current_close=current_close,
            friday_close=friday_close,
            thursday_close=thursday_close,
        )
        if not setup_result.is_signal:
            logger.debug(
                "TGIM setup condition failed for %s on %s: close=%s not < min(%s, %s).",
                self.configuration.target_symbol,
                target_date_str,
                current_close,
                friday_close,
                thursday_close,
            )
            return None

        return (
            setup_result.threshold_price,
            float(current_close),
            friday_close,
            thursday_close,
        )

    @staticmethod
    def _extract_candle_date(raw_date: object) -> datetime.date:
        """Normalizes a DataFrame date cell into a datetime.date instance."""
        if isinstance(raw_date, datetime.datetime):
            return raw_date.date()
        if isinstance(raw_date, datetime.date):
            return raw_date
        return datetime.date.fromisoformat(str(raw_date)[:10])

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
    "TGIMConfiguration",
    "TGIMSetupResult",
    "TGIMStrategy",
    "TGIMStrategyContext",
    "evaluate_tgim_setup",
]
