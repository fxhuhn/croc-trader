"""Bridge Scout Screener Strategy.

End-of-Month Mean-Reversion strategy for QQQ that buys short-term dips
in the final trading days of the calendar month and exits on the first
trading day of the next month.
"""

import datetime
import logging
from dataclasses import dataclass, replace
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
from ....tools.trading_calendar import (
    get_next_trading_day,
    get_remaining_trading_days_in_month,
    is_in_end_of_month_window,
)
from ...telegram import TelegramBot
from ..models import SignalReportItem
from .base import BaseStrategy

logger = logging.getLogger(__name__)

MIN_BRIDGE_HISTORY_BARS: int = 15
DEFAULT_ATR_WINDOW: int = 10
DEFAULT_RSI_WINDOW: int = 2


@dataclass(frozen=True)
class BridgeScoutConfiguration:
    """Central configuration parameters for Bridge Scout trading strategy."""

    target_symbol: str = "QQQ"
    entry_days_before: int = 4
    rsi_threshold: float = 40.0
    max_atr_pct: float = 3.5
    lookback_period: int = 60
    min_history_bars: int = 15
    atr_window: int = 10
    rsi_window: int = 2
    is_live_same_day: bool = False


BridgeScoutParameters = BridgeScoutConfiguration


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


# Calendar functions get_remaining_trading_days_in_month and is_in_end_of_month_window
# are centrally defined in app.tools.trading_calendar and imported above.


def _evaluate_live_same_day(
    close_series: pd.Series,
    current_atr: float,
    cfg: BridgeScoutConfiguration,
) -> BridgeScoutSetupResult | None:
    current_close = float(close_series.iloc[-1])
    if current_close <= 0:
        return None
    rsi_series = calculate_rsi(close_series, cfg.rsi_window)
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
        window=cfg.rsi_window,
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


def _evaluate_premarket(
    close_series: pd.Series,
    current_atr: float,
    cfg: BridgeScoutConfiguration,
) -> BridgeScoutSetupResult | None:
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
        window=cfg.rsi_window,
        rsi_target=cfg.rsi_threshold,
    )
    rsi_series = calculate_rsi(close_series, cfg.rsi_window)
    rsi_2_val = round(float(rsi_series.iloc[-1]), 2)

    return BridgeScoutSetupResult(
        is_signal=True,
        setup_close=last_close,
        entry_price=float(req_close_rsi40),
        rsi_2=rsi_2_val,
        atr_pct=round(atr_pct, 2),
        req_close_rsi40=round(req_close_rsi40, 2),
    )


def evaluate_bridge_scout_setup(
    close_series: pd.Series,
    high_series: pd.Series,
    low_series: pd.Series,
    params: BridgeScoutConfiguration | None = None,
) -> BridgeScoutSetupResult | None:
    """Pure calculation: Evaluates Bridge Scout setup conditions without side effects."""
    cfg = params or BridgeScoutConfiguration()
    if close_series.empty or len(close_series) < cfg.min_history_bars:
        return None

    atr_series = calculate_atr(high_series, low_series, close_series, cfg.atr_window)
    current_atr = float(atr_series.iloc[-1])

    if cfg.is_live_same_day:
        return _evaluate_live_same_day(close_series, current_atr, cfg)

    return _evaluate_premarket(close_series, current_atr, cfg)


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
        *,
        configuration: BridgeScoutConfiguration | None = None,
    ) -> None:
        """Initializes Bridge Scout screener strategy."""
        super().__init__(data_provider=data_provider, telegram_bot=telegram_bot)
        self.trade_repository = trade_repository
        self.holiday_checker = holiday_checker or MarketHolidayChecker()
        self.configuration = configuration or BridgeScoutConfiguration()

    @override
    def run(self, days: int = 0, analysis_date: str | None = None) -> int:
        """Executes Bridge Scout screening logic for the specified date."""
        target_date = self._resolve_target_date(days, analysis_date)
        target_date_str = target_date.strftime("%Y-%m-%d")

        if not is_in_end_of_month_window(
            target_date,
            days_before=self.configuration.entry_days_before,
            holiday_checker=self.holiday_checker,
        ):
            logger.debug(
                "Date %s is outside Bridge Scout entry window.", target_date_str
            )
            return 0

        history_map = self.data_provider.get_batch_history(
            symbols=[self.configuration.target_symbol],
            days=self.configuration.lookback_period,
            end_date=target_date_str,
        )
        price_history = history_map.get(
            self.configuration.target_symbol, pd.DataFrame()
        )

        if (
            price_history.empty
            or len(price_history) < self.configuration.min_history_bars
        ):
            logger.warning(
                "Insufficient price history for %s on %s.",
                self.configuration.target_symbol,
                target_date_str,
            )
            return 0

        # Strict single position check (MaxPositions = 1)
        if self._has_existing_trade_or_position(
            self.trade_repository,
            self.configuration.target_symbol,
            self.STRATEGY_IDENTIFIER,
            target_date_str,
        ):
            logger.info(
                "Bridge Scout trade or active position already exists for %s on %s.",
                self.configuration.target_symbol,
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

        params = replace(
            self.configuration,
            is_live_same_day=(candle_date == target_date),
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
            symbol=self.configuration.target_symbol,
            strategy=self.STRATEGY_IDENTIFIER,
            size=0.0,
            entry=setup_result.entry_price,
            stop_loss=0.0,
            target=0.0,
            context=dict(context),
        )

        logger.info(
            "Generated Bridge Scout signal for %s on %s (Trade ID: %s, Threshold: <= %.2f).",
            self.configuration.target_symbol,
            target_date_str,
            trade_id,
            setup_result.req_close_rsi40,
        )

        if self.telegram_bot:
            self._send_telegram_report(
                "Bridge Scout",
                [
                    SignalReportItem(
                        symbol=self.configuration.target_symbol,
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

    def _resolve_target_date(
        self, days: int = 0, analysis_date: str | None = None
    ) -> datetime.date:
        """Resolves target analysis date into a clean datetime.date object.

        In live execution (days=0), if the analysis date matches the latest available
        EOD bar in market data (yesterday), the target setup date rolls forward to
        the next active trading session.
        """
        parsed_date = self._resolve_analysis_date(days, analysis_date)

        if days == 0:
            latest_market_date = self.data_provider.get_latest_date()
            if (
                latest_market_date
                and parsed_date.strftime("%Y-%m-%d") == latest_market_date
            ):
                return get_next_trading_day(parsed_date, self.holiday_checker)

        return parsed_date


__all__ = [
    "DEFAULT_ATR_WINDOW",
    "DEFAULT_RSI_WINDOW",
    "MIN_BRIDGE_HISTORY_BARS",
    "BridgeScoutConfiguration",
    "BridgeScoutParameters",
    "BridgeScoutSetupResult",
    "BridgeScoutStrategy",
    "BridgeScoutStrategyContext",
    "evaluate_bridge_scout_setup",
    "get_remaining_trading_days_in_month",
    "is_in_end_of_month_window",
]
