import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict, override

import pandas as pd

from ....const import Strategies
from ....database.repositories.market_data_provider import MarketDataProvider
from ....database.repositories.trade import TradeRepository
from ....services.telegram import TelegramBot
from ....tools.indicators import calculate_roc, calculate_sma
from ....tools.market_holidays import MarketHolidayChecker
from ....tools.symbol_lists import ExchangeSymbol
from ..models import SignalReportItem
from .base import BaseStrategy

logger = logging.getLogger(__name__)

SATURDAY_WEEKDAY: int = 5
MONTH_END_LOOKAHEAD_MAX_DAYS: int = 5

ROC_WINDOW_1M: int = 21
ROC_WINDOW_3M: int = 63
ROC_WINDOW_6M: int = 126
ROC_WINDOW_12M: int = 252

ROC_WINDOWS: tuple[int, ...] = (
    ROC_WINDOW_1M,
    ROC_WINDOW_3M,
    ROC_WINDOW_6M,
    ROC_WINDOW_12M,
)

ROC_CONTEXT_KEYS: dict[int, str] = {
    ROC_WINDOW_1M: "roc_1",
    ROC_WINDOW_3M: "roc_3",
    ROC_WINDOW_6M: "roc_6",
    ROC_WINDOW_12M: "roc_12",
}

QQQ_TREND_SMA_WINDOW: int = 200
BREADTH_SMA_WINDOW: int = 100
BREADTH_FAST_SMA_WINDOW: int = 10
BREADTH_SLOW_SMA_WINDOW: int = 50
HISTORY_FETCH_DAYS: int = 450


@dataclass(frozen=True)
class MomentumTradeContext:
    """Context and analytical snapshot for momentum trade generation."""

    symbols: list[str]
    momentum_scores: pd.Series
    roc_matrices: dict[int, pd.DataFrame]
    analysis_date: pd.Timestamp
    price_data: dict[str, pd.DataFrame]
    regime_indicators: dict[str, float | bool]


@dataclass(frozen=True)
class NDXMomentumConfiguration:
    """Configuration settings for the NDX Momentum strategy."""

    maximum_ticker_count: int = 5


class NDXAnalysisResult(TypedDict, total=False):
    """Typed result container for the NDX momentum analysis pipeline.

    All fields are optional (total=False) because early-return branches
    populate only a subset of keys.
    """

    triggered: bool
    date: str
    requested_date: str
    is_rebalance_day: bool
    top_symbols: list[str]
    momentum_scores: pd.Series
    roc_matrices: dict[int, pd.DataFrame]
    price_data: dict[str, pd.DataFrame]
    regime_indicators: dict[str, float | bool]
    error: str


def _build_trade_context(
    symbol: str,
    context: MomentumTradeContext,
    date_iso_string: str,
) -> dict[str, object]:
    """Constructs the JSON-serializable signal context dictionary for a trade."""
    total_momentum_score = float(context.momentum_scores.at[symbol])
    roc_context_values = {
        context_key: round(
            float(context.roc_matrices[window].at[context.analysis_date, symbol]),
            2,
        )
        for window, context_key in ROC_CONTEXT_KEYS.items()
    }
    qqq_val = context.regime_indicators["qqq"]
    qqq_sma_val = context.regime_indicators["qqq_sma"]
    breadth_fast = context.regime_indicators["breadth_fast"]
    breadth_slow = context.regime_indicators["breadth_slow"]
    bull_flag = bool(context.regime_indicators["bull"])

    return {
        "source": "screener",
        "date": date_iso_string,
        **roc_context_values,
        "momentum_score": round(total_momentum_score, 2),
        "qqq_regime": "BULL" if qqq_val > qqq_sma_val else "BEAR",
        "breadth_regime": "BULL" if breadth_fast > breadth_slow else "BEAR",
        "regime": "BULL" if bull_flag else "BEAR",
        "qqq_abs": qqq_val,
        "qqq_sma": qqq_sma_val,
        "breadth_fast": breadth_fast,
        "breadth_slow": breadth_slow,
    }


class NDXMomentumScreener(BaseStrategy[int]):
    """
    Screener for the NASDAQ-100 Momentum strategy.

    This strategy identifies top momentum stocks in the NASDAQ-100 index
    based on a combined Rate of Change (ROC) score across multiple windows.
    It executes rebalances exclusively on the last trading day of the month.
    """

    name: str = str(Strategies.NDXMomentum)

    def __init__(
        self,
        trade_repository: TradeRepository,
        market_data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None,
        configuration: NDXMomentumConfiguration | None = None,
    ) -> None:
        """
        Initializes the NDX Momentum screener.

        Args:
            trade_repository: Repository for trade persistence.
            market_data_provider: Provider for historical price data.
            telegram_bot: Optional bot for notifications.
            configuration: Specific strategy configuration parameters.
        """
        super().__init__(market_data_provider, telegram_bot)
        self.trade_repository = trade_repository
        self.configuration = configuration or NDXMomentumConfiguration()
        self.holiday_checker = MarketHolidayChecker()

    @override
    def run(
        self,
        days: int = 0,
        analysis_date: str | None = None,
        specific_symbols: list[str] | None = None,
    ) -> int:
        """
        Executes the monthly momentum screening process.

        This method checks if the analysis date is the last trading day of the
        month. If so, it calculates momentum scores, determines the market
        regime, and creates trade entries for the top performers.

        Args:
            days: Ignored for this strategy (uses fixed lookback).
            analysis_date: The date to perform analysis for.
            specific_symbols: Optional override for symbols to screen.

        Returns:
            The number of trades successfully created.
        """
        analysis_result = self.calculate_analysis(analysis_date)

        if not analysis_result["triggered"]:
            return 0

        logger.info(
            "[%s] Monthly rebalance triggered for %s",
            self.name,
            analysis_result["date"],
        )

        # 3. Create Trades
        context = MomentumTradeContext(
            symbols=analysis_result["top_symbols"],
            momentum_scores=analysis_result["momentum_scores"],
            roc_matrices=analysis_result["roc_matrices"],
            analysis_date=pd.Timestamp(analysis_result["date"]),
            price_data=analysis_result["price_data"],
            regime_indicators=analysis_result["regime_indicators"],
        )
        return self._create_trades_direct(context)

    def calculate_analysis(
        self, analysis_date: str | None = None, force_run: bool = False
    ) -> NDXAnalysisResult:
        """Performs the momentum analysis without creating trades.

        This method is used both by run() and by the API for status reporting.
        Follows the Step-down rule by delegating data loading and calculation.
        """
        if not analysis_date:
            analysis_date = datetime.now().strftime("%Y-%m-%d")

        target_date = pd.Timestamp(analysis_date)
        is_rebalance_day = self._is_last_trading_day(target_date)

        if not force_run and not is_rebalance_day:
            return {
                "triggered": False,
                "date": analysis_date,
                "is_rebalance_day": False,
            }

        load_result = self._load_analysis_market_data(
            target_date, analysis_date, force_run
        )
        if isinstance(load_result, dict):
            return load_result

        pivoted_data, effective_date, nasdaq_100_symbols = load_result
        return self._compute_analysis_payload(
            pivoted_data,
            effective_date,
            analysis_date,
            is_rebalance_day,
            nasdaq_100_symbols,
        )

    def _load_analysis_market_data(
        self,
        target_date: pd.Timestamp,
        analysis_date: str,
        force_run: bool,
    ) -> tuple[dict[str, pd.DataFrame], pd.Timestamp, list[str]] | NDXAnalysisResult:
        """Fetches and validates universe market data for the target date."""
        exchange_symbol_provider = ExchangeSymbol()
        nasdaq_100_symbols = exchange_symbol_provider.nasdaq_100
        if not nasdaq_100_symbols:
            logger.error("[%s] NASDAQ-100 symbols list is empty.", self.name)
            return {
                "triggered": False,
                "date": analysis_date,
                "error": "NDX universe empty",
            }

        universe_symbols = list(set(nasdaq_100_symbols + ["QQQ"]))
        pivoted_data = self._prepare_pivoted_history(universe_symbols, analysis_date)
        if pivoted_data is None:
            return {
                "triggered": False,
                "date": analysis_date,
                "error": "No market data",
            }

        if "QQQ" not in pivoted_data["close"].columns:
            logger.warning("[%s] QQQ data missing.", self.name)
            return {
                "triggered": False,
                "date": analysis_date,
                "error": "QQQ data missing",
            }

        effective_date_result = self._resolve_effective_date(
            pivoted_data["close"].index, target_date, force_run, analysis_date
        )
        if not isinstance(effective_date_result, pd.Timestamp):
            return effective_date_result

        return pivoted_data, effective_date_result, nasdaq_100_symbols

    def _compute_analysis_payload(
        self,
        pivoted_data: dict[str, pd.DataFrame],
        effective_date: pd.Timestamp,
        analysis_date: str,
        is_rebalance_day: bool,
        nasdaq_100_symbols: list[str],
    ) -> NDXAnalysisResult:
        """Executes indicator scoring and builds the final analysis result payload."""
        qqq_close_series = pivoted_data["close"]["QQQ"]
        current_qqq_price = float(qqq_close_series.at[effective_date])
        qqq_sma_series = calculate_sma(
            qqq_close_series.loc[:effective_date], QQQ_TREND_SMA_WINDOW
        )
        index_moving_average = float(qqq_sma_series.iloc[-1])

        valid_nasdaq_symbols = [
            symbol
            for symbol in nasdaq_100_symbols
            if symbol in pivoted_data["close"].columns
        ]
        nasdaq_closes = pivoted_data["close"].loc[:effective_date][valid_nasdaq_symbols]

        is_bull_regime, regime_indicators = self._calculate_market_regime(
            qqq_close_series,
            nasdaq_closes,
            effective_date,
            current_qqq_price,
            index_moving_average,
        )

        momentum_result = self._calculate_momentum_leaders(
            nasdaq_closes, effective_date
        )
        if isinstance(momentum_result, dict):
            return momentum_result

        target_date_momentum_sum, rolling_roc_results, selected_leaders = (
            momentum_result
        )

        return {
            "triggered": True,
            "date": effective_date.strftime("%Y-%m-%d"),
            "requested_date": analysis_date,
            "is_rebalance_day": is_rebalance_day,
            "top_symbols": selected_leaders,
            "momentum_scores": target_date_momentum_sum,
            "roc_matrices": rolling_roc_results,
            "price_data": pivoted_data,
            "regime_indicators": {
                "bull": bool(is_bull_regime),
                "qqq": round(float(current_qqq_price), 1),
                "qqq_sma": round(float(index_moving_average), 1),
                "breadth_fast": round(float(regime_indicators["breadth_fast"]), 1),
                "breadth_slow": round(float(regime_indicators["breadth_slow"]), 1),
            },
        }

    def _prepare_pivoted_history(
        self, universe_symbols: list[str], end_date: str
    ) -> dict[str, pd.DataFrame] | None:
        """Fetches history and pivots it into aligned DataFrames."""
        full_history_map = self.data_provider.get_batch_history(
            universe_symbols, days=HISTORY_FETCH_DAYS, end_date=end_date
        )
        if not full_history_map:
            logger.warning("[%s] No data found for universe.", self.name)
            return None

        history_dataframes = []
        for symbol, dataframe in full_history_map.items():
            dataframe["symbol"] = symbol
            history_dataframes.append(dataframe)

        universe_dataframe = pd.concat(history_dataframes, ignore_index=True)
        return {
            "close": universe_dataframe.pivot(
                index="date", columns="symbol", values="close"
            ),
            "high": universe_dataframe.pivot(
                index="date", columns="symbol", values="high"
            ),
            "low": universe_dataframe.pivot(
                index="date", columns="symbol", values="low"
            ),
        }

    def _resolve_effective_date(
        self,
        close_index: pd.Index,
        target_date: pd.Timestamp,
        force_run: bool,
        analysis_date: str,
    ) -> pd.Timestamp | NDXAnalysisResult:
        """Resolves the last available trading day at or before the target date.

        Args:
            close_index: Aligned DataFrame date Index.
            target_date: Requested target Timestamp.
            force_run: Override rebalance date constraint flag.
            analysis_date: Date parameter string.

        Returns:
            pd.Timestamp or NDXAnalysisResult error dict.
        """
        try:
            effective_date = self._resolve_effective_trading_date(
                close_index, target_date
            )

            if effective_date is None or pd.isna(effective_date):
                logger.warning(
                    "[%s] No price data available at or before %s.",
                    self.name,
                    analysis_date,
                )
                return {
                    "triggered": False,
                    "date": analysis_date,
                    "error": "No price data found",
                }

            if target_date > close_index[-1]:
                logger.info(
                    "[%s] Requested %s but data ends at %s. Using last available.",
                    self.name,
                    analysis_date,
                    effective_date,
                )

            if not force_run and effective_date != target_date:
                logger.warning(
                    "[%s] %s not in data and not forced. Skip.",
                    self.name,
                    analysis_date,
                )
                return {
                    "triggered": False,
                    "date": analysis_date,
                    "error": f"{analysis_date} not in data",
                }

            return effective_date

        except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
            raise RuntimeError(
                f"[{self.name}] Database unavailable during date resolution: {database_error}"
            ) from database_error
        except (ValueError, KeyError, TypeError) as resolution_error:
            logger.error("[%s] Date resolution error: %s", self.name, resolution_error)
            return {
                "triggered": False,
                "date": analysis_date,
                "error": str(resolution_error),
            }

    def _calculate_market_regime(
        self,
        qqq_close_series: pd.Series,
        nasdaq_closes: pd.DataFrame,
        effective_date: pd.Timestamp,
        current_qqq_price: float,
        index_moving_average: float,
    ) -> tuple[bool, dict[str, float]]:
        """Calculates QQQ trend and NASDAQ breadth regime parameters.

        Args:
            qqq_close_series: QQQ close price series.
            nasdaq_closes: Historical index stock closing prices.
            effective_date: Date of calculation.
            current_qqq_price: QQQ close on effective date.
            index_moving_average: SMA200 of QQQ on effective date.

        Returns:
            tuple of (is_bull_regime, metrics_dict).
        """
        # Breadth
        sma100_matrix = calculate_sma(nasdaq_closes, BREADTH_SMA_WINDOW)
        percentage_above_sma100 = (nasdaq_closes > sma100_matrix).mean(axis=1) * 100
        breadth_fast_average = float(
            calculate_sma(percentage_above_sma100, BREADTH_FAST_SMA_WINDOW).at[
                effective_date
            ]
        )
        breadth_slow_average = float(
            calculate_sma(percentage_above_sma100, BREADTH_SLOW_SMA_WINDOW).at[
                effective_date
            ]
        )

        is_bull_regime = (current_qqq_price > index_moving_average) and (
            breadth_fast_average > breadth_slow_average
        )

        return is_bull_regime, {
            "breadth_fast": breadth_fast_average,
            "breadth_slow": breadth_slow_average,
        }

    def _calculate_momentum_leaders(
        self, nasdaq_closes: pd.DataFrame, effective_date: pd.Timestamp
    ) -> tuple[pd.Series, dict[int, pd.DataFrame], list[str]] | NDXAnalysisResult:
        """Calculates Rate of Change (ROC) and identifies the top momentum leaders.

        Args:
            nasdaq_closes: Nasdaq close prices.
            effective_date: Target calculation date.

        Returns:
            tuple or NDXAnalysisResult error dict.
        """
        rolling_roc_results: dict[int, pd.DataFrame] = {
            window: calculate_roc(nasdaq_closes, window) for window in ROC_WINDOWS
        }

        combined_momentum_matrix = sum(
            rolling_roc_results[window] for window in ROC_WINDOWS
        )

        target_date_momentum_sum = combined_momentum_matrix.loc[effective_date].dropna()
        if target_date_momentum_sum.empty:
            logger.warning("[%s] No momentum scores for %s.", self.name, effective_date)
            return {
                "triggered": False,
                "date": effective_date.strftime("%Y-%m-%d"),
                "error": "No momentum scores",
            }

        selected_leaders = target_date_momentum_sum.nlargest(
            self.configuration.maximum_ticker_count
        ).index.tolist()

        return target_date_momentum_sum, rolling_roc_results, selected_leaders

    def _is_last_trading_day(self, date: pd.Timestamp) -> bool:
        """Checks if the date represents the last trading day of its month."""
        current_month = date.month
        lookahead_date = date + pd.Timedelta(days=1)
        for _ in range(MONTH_END_LOOKAHEAD_MAX_DAYS):
            if lookahead_date.month != current_month:
                return True
            if (
                lookahead_date.dayofweek < SATURDAY_WEEKDAY
                and not self.holiday_checker.is_holiday(lookahead_date)
            ):
                return False
            lookahead_date += pd.Timedelta(days=1)
        return True

    def _create_trades_direct(
        self,
        context: MomentumTradeContext,
    ) -> int:
        """Writes the selected leaders to the trades table as CREATED status."""
        date_iso_string = context.analysis_date.strftime("%Y-%m-%d")
        self.trade_repository.clear_created_trades(self.name)

        created_count = 0

        created_trades: list[SignalReportItem] = []
        for symbol in context.symbols:
            try:
                closing_price = float(
                    context.price_data["close"].at[context.analysis_date, symbol]
                )
                entry_price = self._create_single_momentum_trade(
                    symbol,
                    context,
                    date_iso_string,
                    closing_price=closing_price,
                )
                created_count += 1
                created_trades.append(
                    SignalReportItem(
                        symbol=symbol,
                        action="BUY MKT",
                        entry_price=entry_price,
                    )
                )
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
                raise RuntimeError(
                    f"[{self.name}] Database unavailable saving trade for {symbol}: {database_error}"
                ) from database_error
            except (ValueError, KeyError, TypeError) as data_error:
                logger.warning(
                    "[%s] Error creating trade for %s: %s",
                    self.name,
                    symbol,
                    data_error,
                )

        logger.info("[%s] Created %d CREATED trades.", self.name, created_count)

        if self.telegram_bot and created_trades:
            self._send_telegram_report("NDX Momentum", created_trades, date_iso_string)

        return created_count

    def _create_single_momentum_trade(
        self,
        symbol: str,
        context: MomentumTradeContext,
        date_iso_string: str,
        closing_price: float | None = None,
    ) -> float:
        """Helper to compute context and save a single momentum trade.

        Calculates individual indicator components and persists trade to repository.

        Returns:
            The rounded entry price.
        """
        if closing_price is None:
            closing_price = float(
                context.price_data["close"].at[context.analysis_date, symbol]
            )

        entry_price = round(closing_price, 2)
        trade_context = _build_trade_context(symbol, context, date_iso_string)

        self.trade_repository.create_trade(
            symbol=symbol,
            strategy=Strategies.NDXMomentum,
            size=0,
            entry=entry_price,
            stop_loss=0.0,
            target=0.0,
            context=trade_context,
        )
        return entry_price
