import logging
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import override

from ....database.repositories.market_data_provider import MarketDataProvider
from ....database.repositories.trade import TradeRepository
from ....services.telegram import TelegramBot
from ....tools.symbol_lists import ExchangeSymbol
from ....tools.market_holidays import MarketHolidayChecker
from ....const import Strategies
from .base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NDXMomentumConfiguration:
    """Configuration settings for the NDX Momentum strategy."""

    maximum_ticker_count: int = 5
    maximum_lookback_period: int = 252


class NDXMomentumScreener(BaseStrategy):
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
        return self._create_trades_direct(
            analysis_result["top_symbols"],
            analysis_result["momentum_scores"],
            analysis_result["roc_matrices"],
            pd.Timestamp(analysis_result["date"]),
            analysis_result["price_data"],
            regime_indicators=analysis_result["regime_indicators"],
        )

    def calculate_analysis(
        self, analysis_date: str | None = None, force_run: bool = False
    ) -> dict[str, object]:
        """
        Performs the momentum analysis without creating trades.

        This method is used both by run() and by the API for status reporting.
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

        # 1. Fetch Universe (NDX)
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

        # Use get_batch_history with 450 calendar days to safely cover 252 trading days
        full_history_map = self.data_provider.get_batch_history(
            universe_symbols, days=450, end_date=analysis_date
        )

        if not full_history_map:
            logger.warning("[%s] No data found for universe.", self.name)
            return {
                "triggered": False,
                "date": analysis_date,
                "error": "No market data",
            }

        # Data Pre-processing: Concatenate and Pivot
        history_dataframes = []
        for symbol, dataframe in full_history_map.items():
            dataframe["symbol"] = symbol
            history_dataframes.append(dataframe)

        universe_dataframe = pd.concat(history_dataframes, ignore_index=True)

        pivoted_data = {
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

        if "QQQ" not in pivoted_data["close"].columns:
            logger.warning("[%s] QQQ data missing.", self.name)
            return {
                "triggered": False,
                "date": analysis_date,
                "error": "QQQ data missing",
            }

        try:
            # Resolve the effective date from available data
            if target_date > pivoted_data["close"].index[-1]:
                effective_date = pivoted_data["close"].index[-1]
                logger.info(
                    "[%s] Requested %s but data ends at %s. Using last available.",
                    self.name,
                    analysis_date,
                    effective_date,
                )
            else:
                effective_date = pivoted_data["close"].index.asof(target_date)

            if pd.isna(effective_date):
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

            # If not forcing, we still require exact match for rebalance logic
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

        except Exception as exception:
            logger.error("[%s] Date resolution error: %s", self.name, exception)
            return {"triggered": False, "date": analysis_date, "error": str(exception)}

        # 2. Calculation Pipeline
        qqq_close_series = pivoted_data["close"]["QQQ"]
        current_qqq_price = qqq_close_series.at[effective_date]
        index_moving_average = qqq_close_series.loc[:effective_date].tail(200).mean()

        valid_nasdaq_symbols = [
            s for s in nasdaq_100_symbols if s in pivoted_data["close"].columns
        ]
        nasdaq_closes = pivoted_data["close"].loc[:effective_date][valid_nasdaq_symbols]

        # Breadth
        sma100_matrix = nasdaq_closes.rolling(window=100).mean()
        percentage_above_sma100 = (nasdaq_closes > sma100_matrix).mean(axis=1) * 100
        breadth_fast_average = (
            percentage_above_sma100.rolling(window=10).mean().at[effective_date]
        )
        breadth_slow_average = (
            percentage_above_sma100.rolling(window=50).mean().at[effective_date]
        )

        is_bull_regime = (current_qqq_price > index_moving_average) and (
            breadth_fast_average > breadth_slow_average
        )

        # Momentum
        rolling_roc_results = {}
        for window in [21, 63, 126, 252]:
            rolling_roc_results[window] = nasdaq_closes.pct_change(periods=window) * 100

        combined_momentum_matrix = (
            rolling_roc_results[21]
            + rolling_roc_results[63]
            + rolling_roc_results[126]
            + rolling_roc_results[252]
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
                "breadth_fast": round(float(breadth_fast_average), 1),
                "breadth_slow": round(float(breadth_slow_average), 1),
            },
        }

    def _is_last_trading_day(self, date: pd.Timestamp) -> bool:
        """Checks if the date represents the last trading day of its month."""
        current_month = date.month
        lookahead_date = date + pd.Timedelta(days=1)
        for _ in range(5):
            if lookahead_date.month != current_month:
                return True
            if lookahead_date.dayofweek < 5 and not self.holiday_checker.is_holiday(
                lookahead_date
            ):
                return False
            lookahead_date += pd.Timedelta(days=1)
        return True

    def _create_trades_direct(
        self,
        symbols: list[str],
        momentum_scores: pd.Series,
        roc_matrices: dict[int, pd.DataFrame],
        analysis_date: pd.Timestamp,
        price_data: dict[str, pd.DataFrame],
        regime_indicators: dict,
    ) -> int:
        """Writes the selected leaders to the trades table as CREATED status."""
        date_iso_string = analysis_date.strftime("%Y-%m-%d")
        self.trade_repository.execute(
            "DELETE FROM trades WHERE strategy = ? AND status = 'CREATED'", (self.name,)
        )

        created_count = 0
        for symbol in symbols:
            try:
                total_momentum_score = float(momentum_scores.at[symbol])
                closing_price = float(price_data["close"].at[analysis_date, symbol])

                trade_context = {
                    "source": "screener",
                    "date": date_iso_string,
                    "roc_1": round(
                        float(roc_matrices[21].at[analysis_date, symbol]), 2
                    ),
                    "roc_3": round(
                        float(roc_matrices[63].at[analysis_date, symbol]), 2
                    ),
                    "roc_6": round(
                        float(roc_matrices[126].at[analysis_date, symbol]), 2
                    ),
                    "roc_12": round(
                        float(roc_matrices[252].at[analysis_date, symbol]), 2
                    ),
                    "momentum_score": round(total_momentum_score, 2),
                    "qqq_regime": "BULL"
                    if (regime_indicators["qqq"] > regime_indicators["qqq_sma"])
                    else "BEAR",
                    "breadth_regime": "BULL"
                    if (
                        regime_indicators["breadth_fast"]
                        > regime_indicators["breadth_slow"]
                    )
                    else "BEAR",
                    "regime": "BULL" if regime_indicators["bull"] else "BEAR",
                    "qqq_abs": regime_indicators["qqq"],
                    "qqq_sma": regime_indicators["qqq_sma"],
                    "breadth_fast": regime_indicators["breadth_fast"],
                    "breadth_slow": regime_indicators["breadth_slow"],
                }

                self.trade_repository.create_trade(
                    symbol=symbol,
                    strategy=self.name,
                    size=0,
                    entry=round(closing_price, 2),
                    stop_loss=0.0,
                    target=0.0,
                    context=trade_context,
                )
                created_count += 1
            except Exception as exception:
                logger.error(
                    "[%s] Error creating trade for %s: %s", self.name, symbol, exception
                )

        logger.info("[%s] Created %d CREATED trades.", self.name, created_count)
        return created_count
