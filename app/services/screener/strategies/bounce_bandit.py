"""Bounce Bandit Screener Strategy.

Identifies sharp, low-volatility pullbacks in QQQ within an established long-term uptrend:
- Uptrend Regime: Close > SMA_200
- Volatility: ATR(10) / Close * 100 < 2.5%
- Pullback: Close < min(Close[t-1], Close[t-2])
- Oversold: RSI(2) < 20
- Entry: Market On Open (MOO) on next day's open
- Exits: Market On Close (MOC) when Close > SMA_8 OR RSI(2) > 75
"""

import datetime
import logging
from typing import TypedDict, override

import pandas as pd

from ....const import Strategies
from ....database.repositories.market_data_provider import MarketDataProvider
from ....database.repositories.trade import TradeRepository
from ....tools.indicators import calculate_atr, calculate_rsi, calculate_sma
from ....types import TradeStatus
from ...telegram import TelegramBot
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class BounceBanditStrategyContext(TypedDict, total=False):
    """Context data payload for the Bounce Bandit signal."""

    date: str
    setup_close: float
    sma_200: float
    sma_8: float
    target: float
    target_price: float
    required_sma_exit: float
    required_rsi_exit: float
    atr_10: float
    atr_pct: float
    rsi_2: float
    prev_close_1: float
    prev_close_2: float
    source: str


class BounceBanditStrategy(BaseStrategy[int]):
    """Implementation of the Bounce Bandit trading strategy.

    Strategy Logic:
    - Asset: QQQ exclusively.
    - Uptrend: Close > SMA(200).
    - Volatility: ATR(10) / Close * 100 < 2.5%.
    - Pullback: Close < min(Close[t-1], Close[t-2]).
    - Oversold: RSI(2) < 20.
    - Entry: Market On Open (MOO) on next day's open.
    - Exits: Close > SMA(8) OR RSI(2) > 75 (MOC exit).
    """

    STRATEGY_IDENTIFIER = Strategies.BounceBandit
    TARGET_SYMBOL = "QQQ"
    DEFAULT_LOOKBACK_PERIOD = 350
    TREND_SMA_LEN = 200
    ATR_LEN = 10
    MAX_ATR_PCT = 2.5
    RSI_ENTRY_THRESHOLD = 20.0

    def __init__(
        self,
        trade_repository: TradeRepository,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None,
    ) -> None:
        """Initializes the Bounce Bandit screener strategy with required dependencies."""
        super().__init__(data_provider=data_provider, telegram_bot=telegram_bot)
        self.trade_repository = trade_repository

    @override
    def run(self, days: int = 0, analysis_date: str | None = None) -> int:
        """Executes the Bounce Bandit screening logic for the specified date."""
        target_date = self._resolve_target_date(days, analysis_date)
        target_date_str = target_date.strftime("%Y-%m-%d")

        history_map = self.data_provider.get_batch_history(
            symbols=[self.TARGET_SYMBOL],
            days=self.DEFAULT_LOOKBACK_PERIOD,
            end_date=target_date_str,
        )
        price_history = history_map.get(self.TARGET_SYMBOL, pd.DataFrame())

        if price_history.empty or len(price_history) < self.TREND_SMA_LEN + 2:
            logger.warning(
                "Insufficient price history for %s on %s (required >= %d bars).",
                self.TARGET_SYMBOL,
                target_date_str,
                self.TREND_SMA_LEN + 2,
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
                "Market data date %s does not match target date %s (likely holiday or weekend).",
                candle_date,
                target_date,
            )
            return 0

        # Calculate indicators
        close_series = price_history["close"].astype(float)
        high_series = price_history["high"].astype(float)
        low_series = price_history["low"].astype(float)

        sma_200_series = calculate_sma(close_series, self.TREND_SMA_LEN)
        atr_10_series = calculate_atr(
            high_series, low_series, close_series, self.ATR_LEN
        )
        atr_pct_series = (atr_10_series / close_series) * 100.0
        rsi_2_series = calculate_rsi(close_series, 2)

        current_close = float(close_series.iloc[-1])
        current_sma_200 = float(sma_200_series.iloc[-1])
        current_atr_10 = float(atr_10_series.iloc[-1])
        current_atr_pct = float(atr_pct_series.iloc[-1])
        current_rsi_2 = float(rsi_2_series.iloc[-1])

        prev_close_1 = float(close_series.iloc[-2])
        prev_close_2 = float(close_series.iloc[-3])

        regime_ok = (
            current_close > current_sma_200 and current_atr_pct < self.MAX_ATR_PCT
        )
        pullback_ok = current_close < min(prev_close_1, prev_close_2)
        rsi_ok = current_rsi_2 < self.RSI_ENTRY_THRESHOLD

        if not (regime_ok and pullback_ok and rsi_ok):
            logger.debug(
                "Bounce Bandit setup conditions not met for %s on %s: "
                "close=%.2f, sma200=%.2f, atr_pct=%.2f%%, rsi2=%.2f, prev1=%.2f, prev2=%.2f.",
                self.TARGET_SYMBOL,
                target_date_str,
                current_close,
                current_sma_200,
                current_atr_pct,
                current_rsi_2,
                prev_close_1,
                prev_close_2,
            )
            return 0

        # Strict single position check (MaxPositions = 1 / S.Positions == 0)
        active_or_created = self.trade_repository.get_by_status(
            [
                TradeStatus.CREATED,
                TradeStatus.ACTIVE,
            ]
        )
        if any(
            t.get("symbol") == self.TARGET_SYMBOL
            and "bounce_bandit" in str(t.get("strategy")).lower()
            for t in active_or_created
        ) or self.trade_repository.exists(
            self.TARGET_SYMBOL, self.STRATEGY_IDENTIFIER, target_date_str
        ):
            logger.info(
                "Bounce Bandit trade or active position already exists for %s on %s.",
                self.TARGET_SYMBOL,
                target_date_str,
            )
            return 0

        sma_8_series = calculate_sma(close_series, 8)
        current_sma_8 = float(sma_8_series.iloc[-1])

        last_7_closes = close_series.iloc[-7:]
        required_sma_exit = float(last_7_closes.mean()) + 0.01

        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain_series = gain.ewm(alpha=0.5, adjust=False).mean()
        avg_loss_series = loss.ewm(alpha=0.5, adjust=False).mean()
        last_avg_gain = float(avg_gain_series.iloc[-1])
        last_avg_loss = float(avg_loss_series.iloc[-1])
        last_close = float(close_series.iloc[-1])

        required_delta_rsi = max(0.0, (3.0 * last_avg_loss) - last_avg_gain)
        required_rsi_exit = last_close + required_delta_rsi + 0.01
        target_price = min(required_sma_exit, required_rsi_exit)

        context: BounceBanditStrategyContext = {
            "date": target_date_str,
            "setup_close": current_close,
            "sma_200": round(current_sma_200, 2),
            "sma_8": round(current_sma_8, 2),
            "target": round(target_price, 2),
            "target_price": round(target_price, 2),
            "required_sma_exit": round(required_sma_exit, 2),
            "required_rsi_exit": round(required_rsi_exit, 2),
            "atr_10": round(current_atr_10, 2),
            "atr_pct": round(current_atr_pct, 2),
            "rsi_2": round(current_rsi_2, 2),
            "prev_close_1": prev_close_1,
            "prev_close_2": prev_close_2,
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
            "Generated Bounce Bandit signal for %s on %s (Trade ID: %s).",
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
