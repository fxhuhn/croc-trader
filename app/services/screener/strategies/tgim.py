"""TGIM (Thank God It's Monday) Screener Strategy.

Identifies Monday setup bars for SPY where Monday's close is strictly
lower than both Friday's close and Thursday's close.
"""

import datetime
import logging
from typing import TypedDict, override

import pandas as pd

from ....const import Strategies
from ....database.repositories.market_data_provider import MarketDataProvider
from ....database.repositories.trade import TradeRepository
from ...telegram import TelegramBot
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class TGIMStrategyContext(TypedDict, total=False):
    """Context data payload for the TGIM signal."""

    date: str
    setup_date: str
    setup_close: float
    threshold_price: float
    friday_close: float
    thursday_close: float
    day: str
    target_symbol: str
    max_holding_bars: int
    source: str


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
    TARGET_SYMBOL = "SPY"
    DEFAULT_MAX_HOLDING_BARS = 2
    DEFAULT_LOOKBACK_PERIOD = 30

    def __init__(
        self,
        trade_repository: TradeRepository,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None,
    ) -> None:
        """Initializes the TGIM screener strategy with required dependencies."""
        super().__init__(data_provider=data_provider, telegram_bot=telegram_bot)
        self.trade_repository = trade_repository

    @override
    def run(self, days: int = 0, analysis_date: str | None = None) -> int:
        """Executes the TGIM screening logic for the specified date."""
        target_date = self._resolve_target_date(days, analysis_date)

        if target_date.weekday() != 0:
            logger.debug("Skipping TGIM screening for %s (not a Monday).", target_date)
            return 0

        target_date_str = target_date.strftime("%Y-%m-%d")

        history_map = self.data_provider.get_batch_history(
            symbols=[self.TARGET_SYMBOL],
            days=self.DEFAULT_LOOKBACK_PERIOD,
            end_date=target_date_str,
        )
        price_history = history_map.get(self.TARGET_SYMBOL, pd.DataFrame())

        if price_history.empty or len(price_history) < 2:
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
            if len(price_history) < 3:
                return 0
            current_close = float(latest_candle["close"])
            friday_candle = price_history.iloc[-2]
            friday_close = float(friday_candle["close"])
            thursday_close = float(price_history.iloc[-3]["close"])
            threshold_price = min(friday_close, thursday_close)

            if not (current_close < threshold_price):
                logger.debug(
                    "TGIM setup condition failed for SPY on %s: close=%.2f not < min(%.2f, %.2f).",
                    target_date_str,
                    current_close,
                    friday_close,
                    thursday_close,
                )
                return 0
            setup_close = current_close
        else:
            # Pre-market / pre-close screening: Monday candle is not in DB yet.
            # Use latest available candles (Friday and Thursday).
            friday_close = float(latest_candle["close"])
            thursday_close = float(price_history.iloc[-2]["close"])
            threshold_price = min(friday_close, thursday_close)
            setup_close = threshold_price

        # entry_price is the strategy reference price (threshold_price) per entry-price semantics
        entry_price = threshold_price

        if self.trade_repository.exists(
            self.TARGET_SYMBOL, self.STRATEGY_IDENTIFIER, target_date_str
        ):
            logger.info(
                "TGIM trade already exists for %s on %s.",
                self.TARGET_SYMBOL,
                target_date_str,
            )
            return 0

        context: TGIMStrategyContext = {
            "date": target_date_str,
            "setup_date": target_date_str,
            "setup_close": setup_close,
            "threshold_price": threshold_price,
            "friday_close": friday_close,
            "thursday_close": thursday_close,
            "day": "Monday",
            "target_symbol": self.TARGET_SYMBOL,
            "max_holding_bars": self.DEFAULT_MAX_HOLDING_BARS,
            "source": "ScreenerEngine",
        }

        trade_id = self.trade_repository.create_trade(
            symbol=self.TARGET_SYMBOL,
            strategy=self.STRATEGY_IDENTIFIER.value,
            size=0.0,
            entry=entry_price,
            stop_loss=0.0,
            target=0.0,
            context=context,
        )

        logger.info(
            "Generated TGIM signal for %s on %s (Trade ID: %s).",
            self.TARGET_SYMBOL,
            target_date_str,
            trade_id,
        )
        return 1

    def _resolve_target_date(
        self, days: int, analysis_date: str | None
    ) -> datetime.date:
        """Resolves the target analysis date as a datetime.date object."""
        if analysis_date:
            return datetime.datetime.strptime(analysis_date, "%Y-%m-%d").date()
        target_datetime = datetime.datetime.now() - datetime.timedelta(days=days)
        return target_datetime.date()
