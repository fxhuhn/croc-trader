"""Bounce Bandit Screener Strategy.

Identifies sharp, low-volatility pullbacks in QQQ within an established long-term uptrend:
- Uptrend Regime: Close > SMA_200
- Volatility: ATR(10) / Close * 100 < 2.5%
- Pullback: Close < min(Close[t-1], Close[t-2])
- Oversold: RSI(2) < 20
- Entry: Market On Open (MOO) on next day's open
- Exits: Market On Close (MOC) when Close > SMA_8 OR RSI(2) > 75
"""

import logging
from dataclasses import dataclass
from typing import TypedDict, override

import pandas as pd

from ....const import Strategies
from ....database.repositories.market_data_provider import MarketDataProvider
from ....database.repositories.trade import TradeRepository
from ....tools.indicators import (
    calculate_atr,
    calculate_rsi,
    calculate_rsi_exit_target,
    calculate_sma,
)
from ...telegram import TelegramBot
from ..models import SignalReportItem
from .base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BounceBanditConfiguration:
    """Central configuration parameters for Bounce Bandit trading strategy."""

    target_symbol: str = "QQQ"
    trend_sma_len: int = 200
    atr_len: int = 10
    max_atr_pct: float = 2.5
    rsi_entry_threshold: float = 20.0
    exit_sma_len: int = 8
    rsi_window: int = 2
    rsi_exit_target: float = 75.0
    lookback_period: int = 350
    lookback_buffer_bars: int = 2


BounceBanditParameters = BounceBanditConfiguration


@dataclass(frozen=True)
class BounceBanditSetupResult:
    """Immutable result of pure Bounce Bandit setup condition evaluation."""

    is_signal: bool
    current_close: float
    current_sma_200: float
    current_sma_8: float
    current_atr_10: float
    current_atr_pct: float
    current_rsi_2: float
    prev_close_1: float
    prev_close_2: float
    target_price: float
    required_sma_exit: float
    required_rsi_exit: float


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


def evaluate_bounce_bandit_setup(
    close_series: pd.Series,
    high_series: pd.Series,
    low_series: pd.Series,
    params: BounceBanditConfiguration | None = None,
) -> BounceBanditSetupResult | None:
    """Pure calculation: Evaluates Bounce Bandit setup and exit price targets without side effects."""
    cfg = params or BounceBanditConfiguration()
    if len(close_series) < cfg.trend_sma_len + cfg.lookback_buffer_bars:
        return None

    sma_200_series = calculate_sma(close_series, cfg.trend_sma_len)
    atr_10_series = calculate_atr(high_series, low_series, close_series, cfg.atr_len)
    safe_close = close_series.replace(0.0, float("nan"))
    atr_pct_series = (atr_10_series / safe_close) * 100.0
    rsi_2_series = calculate_rsi(close_series, cfg.rsi_window)

    current_close = float(close_series.iloc[-1])
    current_sma_200 = float(sma_200_series.iloc[-1])
    current_atr_10 = float(atr_10_series.iloc[-1])
    current_atr_pct = float(atr_pct_series.iloc[-1])
    current_rsi_2 = float(rsi_2_series.iloc[-1])

    prev_close_1 = float(close_series.iloc[-2])
    prev_close_2 = float(close_series.iloc[-3])

    regime_ok = current_close > current_sma_200 and current_atr_pct < cfg.max_atr_pct
    pullback_ok = current_close < min(prev_close_1, prev_close_2)
    rsi_ok = current_rsi_2 < cfg.rsi_entry_threshold

    is_signal = bool(regime_ok and pullback_ok and rsi_ok)

    sma_8_series = calculate_sma(close_series, cfg.exit_sma_len)
    current_sma_8 = float(sma_8_series.iloc[-1])

    last_7_closes = close_series.iloc[-(cfg.exit_sma_len - 1) :]
    required_sma_exit = float(last_7_closes.mean()) + 0.01

    required_rsi_exit = calculate_rsi_exit_target(
        close_series, window=cfg.rsi_window, rsi_target=cfg.rsi_exit_target
    )
    target_price = min(required_sma_exit, required_rsi_exit)

    return BounceBanditSetupResult(
        is_signal=is_signal,
        current_close=current_close,
        current_sma_200=current_sma_200,
        current_sma_8=current_sma_8,
        current_atr_10=current_atr_10,
        current_atr_pct=current_atr_pct,
        current_rsi_2=current_rsi_2,
        prev_close_1=prev_close_1,
        prev_close_2=prev_close_2,
        target_price=target_price,
        required_sma_exit=required_sma_exit,
        required_rsi_exit=required_rsi_exit,
    )


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
    name = Strategies.BounceBandit
    TARGET_SYMBOL = "QQQ"
    DEFAULT_LOOKBACK_PERIOD = 350
    TREND_SMA_LEN = 200
    ATR_LEN = 10
    MAX_ATR_PCT = 2.5
    RSI_ENTRY_THRESHOLD = 20.0
    EXIT_SMA_LEN = 8
    RSI_WINDOW = 2
    RSI_EXIT_TARGET = 75.0

    def __init__(
        self,
        trade_repository: TradeRepository,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None,
        *,
        configuration: BounceBanditConfiguration | None = None,
    ) -> None:
        """Initializes the Bounce Bandit screener strategy with required dependencies."""
        super().__init__(data_provider=data_provider, telegram_bot=telegram_bot)
        self.trade_repository = trade_repository
        self.configuration = configuration or BounceBanditConfiguration()

    @override
    def run(self, days: int = 0, analysis_date: str | None = None) -> int:
        """Executes the Bounce Bandit screening logic for the specified date."""
        target_date = self._resolve_analysis_date(days, analysis_date)
        target_date_str = target_date.strftime("%Y-%m-%d")

        history_map = self.data_provider.get_batch_history(
            symbols=[self.configuration.target_symbol],
            days=self.configuration.lookback_period,
            end_date=target_date_str,
        )
        price_history = history_map.get(
            self.configuration.target_symbol, pd.DataFrame()
        )

        min_required_bars = (
            self.configuration.trend_sma_len + self.configuration.lookback_buffer_bars
        )
        if price_history.empty or len(price_history) < min_required_bars:
            logger.warning(
                "Insufficient price history for %s on %s (required >= %d bars).",
                self.configuration.target_symbol,
                target_date_str,
                min_required_bars,
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

        close_series = price_history["close"].astype(float)
        high_series = price_history["high"].astype(float)
        low_series = price_history["low"].astype(float)

        setup_result = evaluate_bounce_bandit_setup(
            close_series=close_series,
            high_series=high_series,
            low_series=low_series,
            params=self.configuration,
        )

        if setup_result is None or not setup_result.is_signal:
            if setup_result:
                logger.debug(
                    "Bounce Bandit setup conditions not met for %s on %s: "
                    "close=%.2f, sma200=%.2f, atr_pct=%.2f%%, rsi2=%.2f, prev1=%.2f, prev2=%.2f.",
                    self.configuration.target_symbol,
                    target_date_str,
                    setup_result.current_close,
                    setup_result.current_sma_200,
                    setup_result.current_atr_pct,
                    setup_result.current_rsi_2,
                    setup_result.prev_close_1,
                    setup_result.prev_close_2,
                )
            return 0

        # Strict single position check (MaxPositions = 1 / S.Positions == 0)
        if self._has_existing_trade_or_position(
            self.trade_repository,
            self.configuration.target_symbol,
            self.STRATEGY_IDENTIFIER,
            target_date_str,
        ):
            logger.info(
                "Bounce Bandit trade or active position already exists for %s on %s.",
                self.configuration.target_symbol,
                target_date_str,
            )
            return 0

        context: BounceBanditStrategyContext = {
            "date": target_date_str,
            "setup_close": setup_result.current_close,
            "sma_200": round(setup_result.current_sma_200, 2),
            "sma_8": round(setup_result.current_sma_8, 2),
            "target": round(setup_result.target_price, 2),
            "target_price": round(setup_result.target_price, 2),
            "required_sma_exit": round(setup_result.required_sma_exit, 2),
            "required_rsi_exit": round(setup_result.required_rsi_exit, 2),
            "atr_10": round(setup_result.current_atr_10, 2),
            "atr_pct": round(setup_result.current_atr_pct, 2),
            "rsi_2": round(setup_result.current_rsi_2, 2),
            "prev_close_1": setup_result.prev_close_1,
            "prev_close_2": setup_result.prev_close_2,
            "source": "ScreenerEngine",
        }

        trade_id = self.trade_repository.create_trade(
            symbol=self.configuration.target_symbol,
            strategy=self.STRATEGY_IDENTIFIER,
            size=0.0,
            entry=setup_result.current_close,
            stop_loss=0.0,
            target=round(setup_result.target_price, 2),
            context=dict(context),
        )

        logger.info(
            "Generated Bounce Bandit signal for %s on %s (Trade ID: %s).",
            self.configuration.target_symbol,
            target_date_str,
            trade_id,
        )

        if self.telegram_bot:
            self._send_telegram_report(
                "Bounce Bandit",
                [
                    SignalReportItem(
                        symbol=self.configuration.target_symbol,
                        action="BUY MKT",
                        entry_price=setup_result.current_close,
                        target_profit=round(setup_result.target_price, 2),
                        details={
                            "RSI(2)": round(setup_result.current_rsi_2, 2),
                            "ATR%": round(setup_result.current_atr_pct, 2),
                        },
                    )
                ],
                target_date_str,
            )

        return 1


__all__ = [
    "BounceBanditConfiguration",
    "BounceBanditParameters",
    "BounceBanditSetupResult",
    "BounceBanditStrategy",
    "BounceBanditStrategyContext",
    "evaluate_bounce_bandit_setup",
]
