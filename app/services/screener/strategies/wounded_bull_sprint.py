"""Wounded Bull Sprint Screener Strategy.

Mean-reversion strategy targeting liquid S&P 100 equities in structural uptrends
during defensive macroeconomic environments (TLT > SPY MTD) in the second half of the month:
- Macro Regime: SPY Month-to-Date return < TLT Month-to-Date return
- Calendar Window: TradingDayOfMonth > 14 AND remaining trading days in month >= 3
- Universe: S&P 100, Close > 5.0, 20-day Volume SMA > 1,000,000
- Uptrend: Close > SMA(200)
- Volatility: ATR(3) / Close > 2.5%
- Short-Term Pullback: ROC(10) < 0
- Trend Quality: Cross-sectional Frog-in-the-Pan (UpDays over 225 bars) percentile rank >= 51%
- Candidate Ranking: Descending SetupScore = -1.0 * ROC(5)
- Capacity: Maximum 10 positions
- Entry: Next trading session open (NextOpen)
"""

import datetime
import logging
from dataclasses import dataclass
from typing import Any, override

import pandas as pd

from ....const import Strategies
from ....database.repositories.market_data_provider import MarketDataProvider
from ....database.repositories.trade import TradeRepository
from ....tools.indicators import (
    calculate_atr,
    calculate_mtd_return,
    calculate_roc,
    calculate_sma,
    calculate_up_days_count,
)
from ....tools.market_holidays import MarketHolidayChecker
from ....tools.trading_calendar import (
    get_remaining_trading_days_in_month,
    get_trading_day_of_month,
    is_trading_day,
)
from ....types import TradeStatus
from ...telegram import TelegramBot
from ..models import SignalReportItem
from .base import BaseStrategy

logger = logging.getLogger(__name__)

DEFAULT_SIGNAL_DAY_THRESHOLD: int = 14
DEFAULT_TREND_SMA_WINDOW: int = 200
DEFAULT_MAX_POSITIONS: int = 10
DEFAULT_TIME_EXIT_BARS: int = 6
DEFAULT_MIN_DAYS_LEFT_IN_MONTH: int = 3
DEFAULT_MULTIPLIER: float = 1.0
DEFAULT_MIN_VOLATILITY: float = 0.025
DEFAULT_FIP_PERCENTILE: float = 51.0
DEFAULT_MIN_PRICE: float = 5.0
DEFAULT_MIN_VOLUME_MA: float = 1_000_000.0
DEFAULT_VOLUME_MA_WINDOW: int = 20
DEFAULT_ROC_PULLBACK_WINDOW: int = 10
DEFAULT_ROC_SCORE_WINDOW: int = 5
DEFAULT_UP_DAYS_LOOKBACK: int = 225
DEFAULT_ATR_WINDOW: int = 3
DEFAULT_HISTORY_FETCH_DAYS: int = 400


@dataclass(frozen=True)
class WoundedBullSprintParameters:
    """Configuration parameters for WBS screening and ranking."""

    daysp: int = DEFAULT_SIGNAL_DAY_THRESHOLD
    ma_p: int = DEFAULT_TREND_SMA_WINDOW
    max_positions: int = DEFAULT_MAX_POSITIONS
    bars_p: int = DEFAULT_TIME_EXIT_BARS
    bars_left: int = DEFAULT_MIN_DAYS_LEFT_IN_MONTH
    multi: float = DEFAULT_MULTIPLIER
    vola_p: float = DEFAULT_MIN_VOLATILITY
    days_pct: float = DEFAULT_FIP_PERCENTILE
    min_price: float = DEFAULT_MIN_PRICE
    min_volume_ma: float = DEFAULT_MIN_VOLUME_MA
    volume_ma_window: int = DEFAULT_VOLUME_MA_WINDOW
    roc_pullback_window: int = DEFAULT_ROC_PULLBACK_WINDOW
    roc_score_window: int = DEFAULT_ROC_SCORE_WINDOW
    up_days_lookback: int = DEFAULT_UP_DAYS_LOOKBACK
    atr_window: int = DEFAULT_ATR_WINDOW
    history_fetch_days: int = DEFAULT_HISTORY_FETCH_DAYS


@dataclass(frozen=True)
class WBSCandidateScore:
    """Immutable scoring snapshot for a validated WBS candidate."""

    symbol: str
    setup_close: float
    setup_score: float
    roc_5: float
    roc_10: float
    atr_pct: float
    up_days: int
    up_days_pct: float


@dataclass(frozen=True)
class WBSExecutionContext:
    """Execution and regime snapshot for WBS signal generation."""

    target_date_str: str
    tdom: int
    td_left: int
    spy_mtd: float
    tlt_mtd: float


def _validate_trend_and_liquidity(
    price_history: pd.DataFrame,
    params: WoundedBullSprintParameters,
) -> float | None:
    """Validates history length, minimum price, volume SMA, and 200-day trend SMA."""
    min_required_bars = max(params.up_days_lookback + 1, params.ma_p)
    if price_history.empty or len(price_history) < min_required_bars:
        return None

    close_series = price_history["close"].astype(float)
    volume_series = price_history["volume"].astype(float)
    latest_close = float(close_series.iloc[-1])

    if latest_close <= params.min_price:
        return None

    volume_sma_series = calculate_sma(volume_series, params.volume_ma_window)
    if (
        volume_sma_series.empty
        or float(volume_sma_series.iloc[-1]) <= params.min_volume_ma
    ):
        return None

    trend_sma_series = calculate_sma(close_series, params.ma_p)
    if trend_sma_series.empty or latest_close <= float(trend_sma_series.iloc[-1]):
        return None

    return latest_close


def evaluate_symbol_technicals(
    price_history: pd.DataFrame,
    params: WoundedBullSprintParameters,
) -> dict[str, float | int] | None:
    """Pure Function: Evaluates single-symbol technical criteria for WBS.

    Returns a dict containing technical metrics if all conditions pass,
    or None if any technical filter fails.
    """
    latest_close = _validate_trend_and_liquidity(price_history, params)
    if latest_close is None:
        return None

    close_series = price_history["close"].astype(float)
    high_series = price_history["high"].astype(float)
    low_series = price_history["low"].astype(float)

    atr_series = calculate_atr(high_series, low_series, close_series, params.atr_window)
    atr_ratio = (
        float(atr_series.iloc[-1]) / latest_close if not atr_series.empty else 0.0
    )
    if atr_ratio <= params.vola_p:
        return None

    roc_10_series = calculate_roc(close_series, params.roc_pullback_window)
    if roc_10_series.empty or float(roc_10_series.iloc[-1]) >= 0.0:
        return None

    up_days = calculate_up_days_count(close_series, params.up_days_lookback)
    roc_5_series = calculate_roc(close_series, params.roc_score_window)
    if up_days is None or roc_5_series.empty:
        return None

    latest_roc_5 = float(roc_5_series.iloc[-1])
    return {
        "setup_close": latest_close,
        "setup_score": -1.0 * latest_roc_5,
        "roc_5": latest_roc_5,
        "roc_10": float(roc_10_series.iloc[-1]),
        "atr_pct": atr_ratio * 100.0,
        "up_days": up_days,
    }


def rank_and_filter_fip(
    qualifying_technicals: list[dict[str, Any]],
    all_universe_up_days: dict[str, int],
    params: WoundedBullSprintParameters,
) -> list[WBSCandidateScore]:
    """Pure Function: Applies cross-sectional FIP rank filter and ranks by SetupScore.

    Calculates percentile rank of UpDays across all qualifying universe stocks,
    filters by UpDaysPct >= days_pct, and returns descending sorted candidates.
    """
    if not qualifying_technicals or not all_universe_up_days:
        return []

    up_days_series = pd.Series(all_universe_up_days, dtype=float)
    percentile_ranks = up_days_series.rank(pct=True, method="average") * 100.0

    scored_candidates: list[WBSCandidateScore] = []
    for item in qualifying_technicals:
        sym = str(item["symbol"])
        fip_rank = float(percentile_ranks.get(sym, 0.0))
        if fip_rank >= params.days_pct:
            scored_candidates.append(
                WBSCandidateScore(
                    symbol=sym,
                    setup_close=float(item["setup_close"]),
                    setup_score=float(item["setup_score"]),
                    roc_5=float(item["roc_5"]),
                    roc_10=float(item["roc_10"]),
                    atr_pct=float(item["atr_pct"]),
                    up_days=int(item["up_days"]),
                    up_days_pct=fip_rank,
                )
            )

    scored_candidates.sort(key=lambda c: c.setup_score, reverse=True)
    return scored_candidates


class WoundedBullSprintStrategy(BaseStrategy[int]):
    """Screener strategy implementation for Wounded Bull Sprint."""

    STRATEGY_IDENTIFIER = Strategies.WoundedBullSprint
    name = Strategies.WoundedBullSprint

    def __init__(
        self,
        trade_repository: TradeRepository,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None,
        holiday_checker: MarketHolidayChecker | None = None,
        *,
        configuration: WoundedBullSprintParameters | None = None,
    ) -> None:
        """Initializes Wounded Bull Sprint screener strategy."""
        super().__init__(data_provider=data_provider, telegram_bot=telegram_bot)
        self.trade_repository = trade_repository
        self.holiday_checker = holiday_checker or MarketHolidayChecker()
        self.configuration = configuration or WoundedBullSprintParameters()

    def _is_calendar_window_active(
        self, target_date: datetime.date
    ) -> tuple[bool, int, int]:
        """Evaluates trading day and calendar window preconditions."""
        if not is_trading_day(target_date, self.holiday_checker):
            return False, 0, 0

        tdom = get_trading_day_of_month(target_date, self.holiday_checker)
        if tdom <= self.configuration.daysp:
            return False, tdom, 0

        td_left = get_remaining_trading_days_in_month(target_date, self.holiday_checker)
        if td_left < self.configuration.bars_left:
            return False, tdom, td_left

        return True, tdom, td_left

    def _is_macro_regime_defensive(
        self, target_date: datetime.date, target_date_str: str
    ) -> tuple[bool, float, float]:
        """Verifies whether the macro regime is defensive (SPY MTD < TLT MTD)."""
        macro_map = self.data_provider.get_batch_history(
            symbols=["SPY", "TLT"],
            days=60,
            end_date=target_date_str,
        )
        spy_df = macro_map.get("SPY", pd.DataFrame())
        tlt_df = macro_map.get("TLT", pd.DataFrame())

        if spy_df.empty or tlt_df.empty:
            logger.warning("Missing SPY or TLT market data on %s.", target_date_str)
            return False, 0.0, 0.0

        spy_mtd = calculate_mtd_return(
            spy_df["close"].astype(float), spy_df["date"], target_date
        )
        tlt_mtd = calculate_mtd_return(
            tlt_df["close"].astype(float), tlt_df["date"], target_date
        )
        return (spy_mtd < tlt_mtd), spy_mtd, tlt_mtd

    def _get_available_portfolio_slots(self) -> int:
        """Computes remaining capacity for WBS trade positions."""
        active_trades = self.trade_repository.get_by_status(
            [TradeStatus.ACTIVE, TradeStatus.CREATED]
        )
        active_wbs_count = sum(
            1
            for t in active_trades
            if str(t.get("strategy", "")).lower() == self.STRATEGY_IDENTIFIER.lower()
        )
        return max(0, self.configuration.max_positions - active_wbs_count)

    def _evaluate_universe(
        self,
        history_map: dict[str, pd.DataFrame],
        target_date_str: str,
    ) -> list[WBSCandidateScore]:
        """Evaluates all universe symbols and returns sorted, qualifying candidates."""
        qualifying_technicals: list[dict[str, Any]] = []
        all_universe_up_days: dict[str, int] = {}

        for symbol, price_history in history_map.items():
            if price_history.empty:
                continue

            if len(price_history) >= self.configuration.up_days_lookback + 1:
                symbol_up_days = calculate_up_days_count(
                    price_history["close"].astype(float),
                    self.configuration.up_days_lookback,
                )
                if symbol_up_days is not None:
                    all_universe_up_days[symbol] = symbol_up_days

            if self._has_existing_trade_or_position(
                self.trade_repository, symbol, self.STRATEGY_IDENTIFIER, target_date_str
            ):
                continue

            eval_result = evaluate_symbol_technicals(price_history, self.configuration)
            if eval_result is not None:
                qualifying_technicals.append({"symbol": symbol, **eval_result})

        return rank_and_filter_fip(
            qualifying_technicals, all_universe_up_days, self.configuration
        )

    def _persist_and_report_signals(
        self,
        candidates: list[WBSCandidateScore],
        context: WBSExecutionContext,
    ) -> None:
        """Persists trade candidates and dispatches Telegram notification."""
        report_items: list[SignalReportItem] = []
        for candidate in candidates:
            trade_context = {
                "date": context.target_date_str,
                "setup_date": context.target_date_str,
                "TradingDayOfMonth": context.tdom,
                "bars_left": context.td_left,
                "setup_close": candidate.setup_close,
                "setup_score": round(candidate.setup_score, 4),
                "roc_5": round(candidate.roc_5, 2),
                "roc_10": round(candidate.roc_10, 2),
                "atr_pct": round(candidate.atr_pct, 2),
                "up_days": candidate.up_days,
                "up_days_pct": round(candidate.up_days_pct, 2),
                "spy_mtd": round(context.spy_mtd * 100.0, 2),
                "tlt_mtd": round(context.tlt_mtd * 100.0, 2),
                "source": "ScreenerEngine",
            }
            self.trade_repository.create_trade(
                symbol=candidate.symbol,
                strategy=self.STRATEGY_IDENTIFIER,
                size=0.0,
                entry=candidate.setup_close,
                stop_loss=0.0,
                target=0.0,
                context=trade_context,
            )
            report_items.append(
                SignalReportItem(
                    symbol=candidate.symbol,
                    action="BUY NEXT OPEN",
                    entry_price=candidate.setup_close,
                    details={
                        "SetupScore": f"{candidate.setup_score:.2f}",
                        "FIP Rank": f"{candidate.up_days_pct:.1f}%",
                        "ROC(5)": f"{candidate.roc_5:.2f}%",
                        "ATR%": f"{candidate.atr_pct:.2f}%",
                    },
                )
            )

        if self.telegram_bot and report_items:
            self._send_telegram_report(
                "Wounded Bull Sprint", report_items, context.target_date_str
            )

    @override
    def run(self, days: int = 0, analysis_date: str | None = None) -> int:
        """Executes Wounded Bull Sprint screening logic for the specified date."""
        target_date = self._resolve_analysis_date(days, analysis_date)
        target_date_str = target_date.strftime("%Y-%m-%d")

        is_cal_active, tdom, td_left = self._is_calendar_window_active(target_date)
        if not is_cal_active:
            return 0

        is_defensive, spy_mtd, tlt_mtd = self._is_macro_regime_defensive(
            target_date, target_date_str
        )
        if not is_defensive:
            return 0

        available_slots = self._get_available_portfolio_slots()
        if available_slots <= 0 or not self.exchange_symbols.sp_100:
            return 0

        history_map = self.data_provider.get_batch_history(
            symbols=self.exchange_symbols.sp_100,
            days=self.configuration.history_fetch_days,
            end_date=target_date_str,
        )

        ranked_candidates = self._evaluate_universe(history_map, target_date_str)
        if not ranked_candidates:
            return 0

        selected_candidates = ranked_candidates[:available_slots]
        exec_context = WBSExecutionContext(
            target_date_str=target_date_str,
            tdom=tdom,
            td_left=td_left,
            spy_mtd=spy_mtd,
            tlt_mtd=tlt_mtd,
        )
        self._persist_and_report_signals(selected_candidates, exec_context)
        return len(selected_candidates)


__all__ = [
    "WBSCandidateScore",
    "WoundedBullSprintParameters",
    "WoundedBullSprintStrategy",
    "evaluate_symbol_technicals",
    "rank_and_filter_fip",
]
