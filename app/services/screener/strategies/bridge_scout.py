"""Bridge Scout Screener Strategy.

End-of-Month Mean-Reversion strategy for QQQ that buys short-term dips
in the final trading days of the calendar month and exits on the first
trading day of the next month.
"""

import datetime
import logging
from dataclasses import dataclass
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
from ...telegram import TelegramBot
from ..models import SignalReportItem
from .base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BridgeScoutParameters:
    """Configuration parameters for Bridge Scout setup evaluation."""

    is_live_same_day: bool = False
    rsi_threshold: float = 40.0
    max_atr_pct: float = 3.5


@dataclass(frozen=True)
class BridgeScoutSetupResult:
    """Immutable outcome of pure Bridge Scout setup evaluation."""

    is_signal: bool
    setup_close: float
    entry_price: float
    rsi_2: float | None
    atr_pct: float
    req_close_rsi40: float


class BridgeScoutStrategyContext(TypedDict, total=False):
    """Context payload stored with Bridge Scout signals."""

    date: str
    setup_date: str
    setup_close: float
    rsi_2: float | None
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


def evaluate_bridge_scout_setup(
    close_series: pd.Series,
    high_series: pd.Series,
    low_series: pd.Series,
    params: BridgeScoutParameters | None = None,
) -> BridgeScoutSetupResult | None:
    """Pure calculation: Evaluates Bridge Scout setup conditions without side effects."""
    cfg = params or BridgeScoutParameters()
    if close_series.empty or len(close_series) < 15:
        return None

    atr_series = calculate_atr(high_series, low_series, close_series, 10)
    current_atr = float(atr_series.iloc[-1])

    if cfg.is_live_same_day:
        current_close = float(close_series.iloc[-1])
        if current_close <= 0:
            return None
        rsi_series = calculate_rsi(close_series, 2)
        current_rsi = float(rsi_series.iloc[-1])
        atr_pct = (current_atr / current_close) * 100.0

        if current_rsi >= cfg.rsi_threshold or atr_pct >= cfg.max_atr_pct:
            return BridgeScoutSetupResult(
                is_signal=False,
                setup_close=current_close,
                entry_price=current_close,
                rsi_2=round(current_rsi, 2),
                atr_pct=round(atr_pct, 2),
                req_close_rsi40=0.0,
            )

        req_close_rsi40 = calculate_max_close_for_rsi(
            close_series.iloc[:-1],
            window=2,
            rsi_target=cfg.rsi_threshold,
        )
        return BridgeScoutSetupResult(
            is_signal=True,
            setup_close=current_close,
            entry_price=current_close,
            rsi_2=round(current_rsi, 2),
            atr_pct=round(atr_pct, 2),
            req_close_rsi40=round(req_close_rsi40, 2),
        )

    # Pre-market / live screening: candle for target_date is not in DB yet
    last_close = float(close_series.iloc[-1])
    if last_close <= 0:
        return None
    atr_pct = (current_atr / last_close) * 100.0

    if atr_pct >= cfg.max_atr_pct:
        return BridgeScoutSetupResult(
            is_signal=False,
            setup_close=last_close,
            entry_price=0.0,
            rsi_2=None,
            atr_pct=round(atr_pct, 2),
            req_close_rsi40=0.0,
        )

    req_close_rsi40 = calculate_max_close_for_rsi(
        close_series,
        window=2,
        rsi_target=cfg.rsi_threshold,
    )
    rsi_series = calculate_rsi(close_series, 2)
    rsi_2_val = round(float(rsi_series.iloc[-1]), 2)

    return BridgeScoutSetupResult(
        is_signal=True,
        setup_close=last_close,
        entry_price=float(req_close_rsi40),
        rsi_2=rsi_2_val,
        atr_pct=round(atr_pct, 2),
        req_close_rsi40=round(req_close_rsi40, 2),
    )


class BridgeScoutStrategy(BaseStrategy[int]):
    """Implementation of the Bridge Scout trading strategy.

    Rules:
    - Asset: QQQ exclusively.
    - Timing: End-of-Month window (last 5 trading days of calendar month).
    - Entry: Market on Close (MOC) on day when Close <= req_close_rsi40 (RSI(2) < 40).
    - Exit: Market on Close (MOC) on 1st trading day of new month.
    """

    STRATEGY_IDENTIFIER = Strategies.BridgeScout
    name = Strategies.BridgeScout
    TARGET_SYMBOL = "QQQ"
    DEFAULT_ENTRY_DAYS_BEFORE = 4
    DEFAULT_RSI_THRESHOLD = 40.0
    DEFAULT_MAX_ATR_PCT = 3.5
    DEFAULT_LOOKBACK_PERIOD = 60

    def __init__(
        self,
        trade_repository: TradeRepository,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None,
        holiday_checker: MarketHolidayChecker | None = None,
    ) -> None:
        """Initializes Bridge Scout screener strategy."""
        super().__init__(data_provider=data_provider, telegram_bot=telegram_bot)
        self.trade_repository = trade_repository
        self.holiday_checker = holiday_checker or MarketHolidayChecker()

    @override
    def run(self, days: int = 0, analysis_date: str | None = None) -> int:
        """Executes Bridge Scout screening logic for the specified date."""
        target_date = self._resolve_analysis_date(days, analysis_date)
        target_date_str = target_date.strftime("%Y-%m-%d")

        if not is_in_end_of_month_window(
            target_date,
            days_before=self.DEFAULT_ENTRY_DAYS_BEFORE,
            holiday_checker=self.holiday_checker,
        ):
            logger.debug(
                "Date %s is outside Bridge Scout entry window.", target_date_str
            )
            return 0

        history_map = self.data_provider.get_batch_history(
            symbols=[self.TARGET_SYMBOL],
            days=self.DEFAULT_LOOKBACK_PERIOD,
            end_date=target_date_str,
        )
        price_history = history_map.get(self.TARGET_SYMBOL, pd.DataFrame())

        if price_history.empty or len(price_history) < 15:
            logger.warning(
                "Insufficient price history for %s on %s.",
                self.TARGET_SYMBOL,
                target_date_str,
            )
            return 0

        # Strict single position check (MaxPositions = 1)
        if self._has_existing_trade_or_position(
            self.trade_repository,
            self.TARGET_SYMBOL,
            self.STRATEGY_IDENTIFIER,
            target_date_str,
        ):
            logger.info(
                "Bridge Scout trade or active position already exists for %s on %s.",
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

        close_series = price_history["close"].astype(float)
        high_series = price_history["high"].astype(float)
        low_series = price_history["low"].astype(float)

        params = BridgeScoutParameters(
            is_live_same_day=(candle_date == target_date),
            rsi_threshold=self.DEFAULT_RSI_THRESHOLD,
            max_atr_pct=self.DEFAULT_MAX_ATR_PCT,
        )
        setup_result = evaluate_bridge_scout_setup(
            close_series=close_series,
            high_series=high_series,
            low_series=low_series,
            params=params,
        )

        if setup_result is None or not setup_result.is_signal:
            return 0

        context: BridgeScoutStrategyContext = {
            "date": target_date_str,
            "setup_date": target_date_str,
            "setup_close": setup_result.setup_close,
            "rsi_2": setup_result.rsi_2,
            "atr_pct": setup_result.atr_pct,
            "req_close_rsi40": setup_result.req_close_rsi40,
            "source": "ScreenerEngine",
        }

        trade_id = self.trade_repository.create_trade(
            symbol=self.TARGET_SYMBOL,
            strategy=self.STRATEGY_IDENTIFIER,
            size=0.0,
            entry=setup_result.entry_price,
            stop_loss=0.0,
            target=0.0,
            context=dict(context),
        )

        logger.info(
            "Generated Bridge Scout signal for %s on %s (Trade ID: %s, Threshold: <= %.2f).",
            self.TARGET_SYMBOL,
            target_date_str,
            trade_id,
            setup_result.req_close_rsi40,
        )

        if self.telegram_bot:
            self._send_telegram_report(
                "Bridge Scout",
                [
                    SignalReportItem(
                        symbol=self.TARGET_SYMBOL,
                        action="BUY MOC",
                        entry_price=setup_result.entry_price,
                        details={
                            "Max Close (RSI<40)": round(
                                setup_result.req_close_rsi40, 2
                            ),
                            "ATR%": round(setup_result.atr_pct, 2),
                        },
                    )
                ],
                target_date_str,
            )

        return 1
