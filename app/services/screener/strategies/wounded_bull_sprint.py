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
from ....tools.symbol_filter import SymbolFilter
from ....tools.trading_calendar import (
    get_next_trading_day,
    get_remaining_trading_days_in_month,
    get_trading_day_of_month,
    is_trading_day,
)
from ...telegram import TelegramBot
from ..models import (
    SignalReportItem,
    SingleSymbolAnalysisResult,
    SingleSymbolStatus,
)
from .base import BaseStrategy

logger = logging.getLogger(__name__)

DEFAULT_SIGNAL_DAY_THRESHOLD: int = 14
DEFAULT_TREND_SMA_WINDOW: int = 200
DEFAULT_MAX_POSITIONS: int = 5
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


def _evaluate_single_wbs_technicals(
    history: pd.DataFrame,
    config: WoundedBullSprintParameters,
) -> tuple[dict[str, bool], dict[str, Any], list[str], bool, float]:
    """Pure Function: Evaluates technical indicators for a single stock."""
    min_required_bars = max(config.up_days_lookback + 1, config.ma_p)
    has_enough_bars = len(history) >= min_required_bars

    close_series = history["close"].astype(float)
    volume_series = history["volume"].astype(float)
    high_series = history["high"].astype(float)
    low_series = history["low"].astype(float)
    latest_close = float(close_series.iloc[-1])

    volume_sma_series = calculate_sma(volume_series, config.volume_ma_window)
    latest_vol_sma = (
        float(volume_sma_series.iloc[-1]) if not volume_sma_series.empty else 0.0
    )

    trend_sma_series = calculate_sma(close_series, config.ma_p)
    latest_trend_sma = (
        float(trend_sma_series.iloc[-1]) if not trend_sma_series.empty else 0.0
    )

    atr_series = calculate_atr(high_series, low_series, close_series, config.atr_window)
    atr_ratio = (
        float(atr_series.iloc[-1]) / latest_close
        if not atr_series.empty and latest_close > 0.0
        else 0.0
    )

    roc_10_series = calculate_roc(close_series, config.roc_pullback_window)
    latest_roc_10 = float(roc_10_series.iloc[-1]) if not roc_10_series.empty else 0.0

    roc_5_series = calculate_roc(close_series, config.roc_score_window)
    latest_roc_5 = float(roc_5_series.iloc[-1]) if not roc_5_series.empty else 0.0

    up_days = calculate_up_days_count(close_series, config.up_days_lookback)

    min_price_ok = latest_close > config.min_price
    volume_ok = latest_vol_sma > config.min_volume_ma
    uptrend_ok = latest_close > latest_trend_sma if latest_trend_sma > 0.0 else False
    volatility_ok = atr_ratio > config.vola_p
    pullback_ok = latest_roc_10 < 0.0

    technicals_ok = (
        has_enough_bars
        and min_price_ok
        and volume_ok
        and uptrend_ok
        and volatility_ok
        and pullback_ok
        and (up_days is not None)
    )

    rejection_reasons: list[str] = []
    if not min_price_ok:
        rejection_reasons.append(f"Close {latest_close:.2f} <= {config.min_price:.2f}")
    if not volume_ok:
        rejection_reasons.append(
            f"Volume SMA {latest_vol_sma:.0f} <= {config.min_volume_ma:.0f}"
        )
    if not uptrend_ok:
        rejection_reasons.append(
            f"Close {latest_close:.2f} <= SMA200 {latest_trend_sma:.2f}"
        )
    if not volatility_ok:
        rejection_reasons.append(
            f"ATR% {atr_ratio * 100:.2f}% <= {config.vola_p * 100:.1f}%"
        )
    if not pullback_ok:
        rejection_reasons.append(f"ROC10 {latest_roc_10:.2f}% >= 0.0%")

    intrinsic_filters = {
        "min_price": min_price_ok,
        "volume": volume_ok,
        "uptrend": uptrend_ok,
        "volatility": volatility_ok,
        "pullback": pullback_ok,
        "data_sufficient": has_enough_bars,
    }
    metric_values = {
        "close": round(latest_close, 2),
        "sma_200": round(latest_trend_sma, 2),
        "atr_pct": round(atr_ratio * 100.0, 2),
        "roc_10": round(latest_roc_10, 2),
        "roc_5": round(latest_roc_5, 2),
        "up_days_225": up_days,
    }
    setup_score = round(atr_ratio * 100.0, 4)
    return (
        intrinsic_filters,
        metric_values,
        rejection_reasons,
        technicals_ok,
        setup_score,
    )


def _resolve_wbs_status(
    technicals_ok: bool,
    is_defensive: bool,
    fip_rank: float | None,
    min_required_fip: float,
    rejection_reasons: list[str],
) -> str:
    """Pure Function: Resolves execution status for WBS single symbol inspection."""
    if not is_defensive:
        rejection_reasons.append("Macro regime not defensive (TLT <= SPY)")

    if fip_rank is not None:
        if fip_rank < min_required_fip:
            rejection_reasons.append(
                f"FIP Rank {fip_rank:.1f}% < {min_required_fip:.1f}%"
            )
            return SingleSymbolStatus.REJECTED
        if technicals_ok and is_defensive:
            return SingleSymbolStatus.QUALIFIED
        return SingleSymbolStatus.REJECTED

    if technicals_ok:
        rejection_reasons.append(
            "FIP rank requires full universe calculation (unverified in single mode)"
        )
        return SingleSymbolStatus.INTRINSIC_PASSED_RELATIVE_UNVERIFIED

    return SingleSymbolStatus.REJECTED


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
        float(atr_series.iloc[-1]) / latest_close
        if not atr_series.empty and latest_close > 0.0
        else 0.0
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
    vola_pct = atr_ratio * 100.0
    return {
        "setup_close": latest_close,
        "setup_score": round(vola_pct, 4),
        "roc_5": latest_roc_5,
        "roc_10": float(roc_10_series.iloc[-1]),
        "atr_pct": round(vola_pct, 2),
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
        self.latest_qualifying_setups: set[str] = set()
        self._daily_universe_fip_cache: dict[str, float] = {}

    def _is_calendar_window_active(
        self, target_date: datetime.date
    ) -> tuple[bool, int, int]:
        """Evaluates trading day and calendar window preconditions."""
        if not is_trading_day(target_date, self.holiday_checker):
            return False, 0, 0

        tdom = get_trading_day_of_month(target_date, self.holiday_checker)
        if tdom <= self.configuration.daysp:
            return False, tdom, 0

        td_left = (
            get_remaining_trading_days_in_month(target_date, self.holiday_checker) - 1
        )
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
        """Computes remaining forward capacity for WBS trade positions, factoring in pending exits."""
        effective_trades = self.trade_repository.get_effective_active_trades(
            self.STRATEGY_IDENTIFIER
        )
        return max(0, self.configuration.max_positions - len(effective_trades))

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

            eval_result = evaluate_symbol_technicals(price_history, self.configuration)
            if eval_result is not None:
                qualifying_technicals.append({"symbol": symbol, **eval_result})

        ranked_candidates = rank_and_filter_fip(
            qualifying_technicals, all_universe_up_days, self.configuration
        )
        self._daily_universe_fip_cache = {
            c.symbol: c.up_days_pct for c in ranked_candidates
        }
        return ranked_candidates

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
            self.latest_qualifying_setups = set()
            return 0

        # Filter out secondary share classes (e.g. keep GOOGL, drop GOOG)
        candidate_symbols = [c.symbol for c in ranked_candidates]
        preferred_symbols = set(SymbolFilter().filter_symbols(candidate_symbols))
        if len(preferred_symbols) < len(candidate_symbols):
            removed = set(candidate_symbols) - preferred_symbols
            logger.info(
                "[%s] Filtered out duplicate company symbols: %s",
                self.name,
                removed,
            )
            ranked_candidates = [
                c for c in ranked_candidates if c.symbol in preferred_symbols
            ]

        self.latest_qualifying_setups = {c.symbol for c in ranked_candidates}

        new_candidates = [
            c
            for c in ranked_candidates
            if not self._has_existing_trade_or_position(
                self.trade_repository,
                c.symbol,
                self.STRATEGY_IDENTIFIER,
                target_date_str,
            )
        ]
        if not new_candidates:
            return 0

        selected_candidates = new_candidates[:available_slots]
        exec_context = WBSExecutionContext(
            target_date_str=target_date_str,
            tdom=tdom,
            td_left=td_left,
            spy_mtd=spy_mtd,
            tlt_mtd=tlt_mtd,
        )
        self._persist_and_report_signals(selected_candidates, exec_context)
        return len(selected_candidates)

    def get_precheck_candidates(self, limit: int = 5) -> list[dict[str, Any]]:
        """Calculates candidate setups across S&P 100 universe on current EOD data."""
        latest_date_str = self.data_provider.get_latest_date()
        target_date_str = (
            latest_date_str
            if isinstance(latest_date_str, str) and latest_date_str
            else datetime.date.today().strftime("%Y-%m-%d")
        )

        history_map = self.data_provider.get_batch_history(
            symbols=self.exchange_symbols.sp_100,
            days=self.configuration.history_fetch_days,
            end_date=target_date_str,
        )
        ranked_candidates = self._evaluate_universe(history_map, target_date_str)
        if not ranked_candidates:
            return []

        # Filter out secondary share classes (e.g. keep GOOGL, drop GOOG)
        candidate_symbols = [c.symbol for c in ranked_candidates]
        preferred_symbols = set(SymbolFilter().filter_symbols(candidate_symbols))
        filtered = [c for c in ranked_candidates if c.symbol in preferred_symbols]

        self.latest_qualifying_setups = {c.symbol for c in filtered}

        return [
            {
                "symbol": c.symbol,
                "setup_close": round(c.setup_close, 2),
                "setup_score": round(c.setup_score, 2),
                "atr_pct": round(c.atr_pct, 2),
                "roc_10": round(c.roc_10, 2),
                "roc_5": round(c.roc_5, 2),
                "fip_rank": round(c.up_days_pct, 1),
                "status": "QUALIFIED",
            }
            for c in filtered[:limit]
        ]

    def analyze_single_symbol(self, symbol: str) -> SingleSymbolAnalysisResult:
        """Calculates indicators and filter states for a single symbol for debug and MCP inspection."""
        clean_symbol = symbol.upper().strip()
        history = self.data_provider.get_symbol_history(
            clean_symbol, days=self.configuration.history_fetch_days
        )
        if history.empty:
            return SingleSymbolAnalysisResult(
                symbol=clean_symbol,
                strategy=self.STRATEGY_IDENTIFIER.value,
                as_of_market_date="",
                upcoming_eod_date="",
                status=SingleSymbolStatus.INSUFFICIENT_DATA,
                overall_passed=False,
                requires_universe_ranking=True,
                calendar_active_today=None,
                macro_regime_passed=None,
                intrinsic_filters={},
                metric_values={},
                relative_ranking=None,
                rejection_reasons=[f"No market data found for {clean_symbol}"],
                checks={},
                values={},
                technical_filters={},
                all_technical_filters_pass=False,
                data_valid=False,
                error=f"No market data found for {clean_symbol}",
            )

        latest_date_raw = self.data_provider.get_latest_date()
        if isinstance(latest_date_raw, str) and latest_date_raw:
            as_of_market_date = latest_date_raw
        elif not history.empty and "date" in history.columns:
            last_val = history["date"].iloc[-1]
            if isinstance(last_val, datetime.date):
                as_of_market_date = last_val.strftime("%Y-%m-%d")
            else:
                as_of_market_date = str(last_val)[:10]
        else:
            as_of_market_date = datetime.date.today().strftime("%Y-%m-%d")

        try:
            target_date = datetime.date.fromisoformat(as_of_market_date)
        except (ValueError, TypeError):
            target_date = datetime.date.today()
            as_of_market_date = target_date.strftime("%Y-%m-%d")

        next_trading_day = get_next_trading_day(target_date, self.holiday_checker)
        upcoming_eod_date_str = (
            next_trading_day.strftime("%Y-%m-%d")
            if isinstance(next_trading_day, datetime.date)
            else as_of_market_date
        )

        is_cal_active, tdom, td_left = self._is_calendar_window_active(target_date)
        is_cal_active_upcoming, tdom_upcoming, td_left_upcoming = (
            self._is_calendar_window_active(
                next_trading_day
                if isinstance(next_trading_day, datetime.date)
                else target_date
            )
        )
        is_defensive, spy_mtd, tlt_mtd = self._is_macro_regime_defensive(
            target_date, as_of_market_date
        )

        (
            intrinsic_filters,
            metric_values,
            rejection_reasons,
            technicals_ok,
            setup_score,
        ) = _evaluate_single_wbs_technicals(history, self.configuration)

        fip_rank: float | None = self._daily_universe_fip_cache.get(clean_symbol)
        status = _resolve_wbs_status(
            technicals_ok=technicals_ok,
            is_defensive=is_defensive,
            fip_rank=fip_rank,
            min_required_fip=self.configuration.days_pct,
            rejection_reasons=rejection_reasons,
        )

        return SingleSymbolAnalysisResult(
            symbol=clean_symbol,
            strategy=self.STRATEGY_IDENTIFIER.value,
            as_of_market_date=as_of_market_date,
            upcoming_eod_date=upcoming_eod_date_str,
            status=status,
            overall_passed=bool(status == SingleSymbolStatus.QUALIFIED),
            requires_universe_ranking=True,
            calendar_active_today=is_cal_active_upcoming,
            macro_regime_passed=is_defensive,
            intrinsic_filters=intrinsic_filters,
            metric_values=metric_values,
            relative_ranking={
                "fip_rank": round(fip_rank, 1) if fip_rank is not None else None,
                "min_required_fip": self.configuration.days_pct,
                "setup_score": setup_score,
            },
            rejection_reasons=rejection_reasons,
            # Backwards-compatibility
            checks={
                k: v for k, v in intrinsic_filters.items() if k != "data_sufficient"
            },
            values={
                "close": metric_values["close"],
                "sma_200": metric_values["sma_200"],
                "atr_pct": metric_values["atr_pct"],
                "roc_10": metric_values["roc_10"],
            },
            technical_filters={
                "close": metric_values["close"],
                "min_price_ok": intrinsic_filters["min_price"],
                "volume_sma_20": 0.0,
                "volume_ok": intrinsic_filters["volume"],
                "sma_200": metric_values["sma_200"],
                "uptrend_ok": intrinsic_filters["uptrend"],
                "atr_3": 0.0,
                "atr_pct": metric_values["atr_pct"],
                "volatility_ok": intrinsic_filters["volatility"],
                "roc_10": metric_values["roc_10"],
                "pullback_ok": intrinsic_filters["pullback"],
                "roc_5": metric_values["roc_5"],
                "up_days_225": metric_values["up_days_225"],
                "history_bars": len(history),
                "history_ok": intrinsic_filters["data_sufficient"],
            },
            all_technical_filters_pass=technicals_ok,
            calendar={
                "trading_day_of_month": tdom,
                "bars_left": td_left,
                "is_window_active": is_cal_active,
                "next_active_day": self.configuration.daysp + 1,
            },
            macro_regime={
                "spy_mtd": round(spy_mtd * 100.0, 2),
                "tlt_mtd": round(tlt_mtd * 100.0, 2),
                "is_defensive": is_defensive,
            },
            scoring={
                "setup_score": setup_score,
                "fip_rank": fip_rank,
            },
            data_valid=True,
            error=None,
        )


__all__ = [
    "WBSCandidateScore",
    "WoundedBullSprintParameters",
    "WoundedBullSprintStrategy",
    "evaluate_symbol_technicals",
    "rank_and_filter_fip",
]
