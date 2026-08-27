import logging
from dataclasses import dataclass, field
from typing import TypedDict, override

import pandas as pd

from ....const import Strategies
from ....database.repositories.market_data_provider import MarketDataProvider
from ....database.repositories.trade import TradeRepository
from ....tools import indicators
from ....tools.market_holidays import MarketHolidayChecker
from ....tools.symbol_filter import SymbolFilter
from ....tools.symbol_lists import ExchangeSymbol
from ...telegram import TelegramBot
from ..models import SignalReportItem
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class TurnoverCandidate(TypedDict):
    """Metadata for potential trading candidates."""

    symbol: str
    close: float
    sma_price: float
    sma_turnover: float
    atr: float
    indices: str


class TurnoverSignalContext(TypedDict):
    """Metadata for the Turnover Timing signal stored in the database."""

    setup_date: str
    setup_close: float
    setup_candle_green: bool
    green_candle_count: int
    setup_sma_price: float
    setup_turnover_sma: float
    setup_atr: float
    factor: float
    indices: str
    source: str
    date: str


@dataclass(frozen=True)
class TurnoverConfiguration:
    """Configuration for technical analysis thresholds in Turnover Timing strategy."""

    atr_window: int = 3
    # Entry Factors: 0.5 * ATR and 1.0 * ATR below Close
    entry_factors: list[float] = field(default_factory=lambda: [0.5, 1.0])
    sma_window: int = 200
    minimum_lookback_days: int = 800


class TurnoverTimingStrategy(BaseStrategy[int]):
    name: str = str(Strategies.TurnOverTiming)

    def __init__(
        self,
        trade_repository: TradeRepository,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None,
        configuration: TurnoverConfiguration | None = None,
    ) -> None:
        """
        Initializes the Turnover Timing strategy.

        Args:
            trade_repository: Repository for trade persistence.
            data_provider: Provider for market historical data.
            telegram_bot: Optional bot for reporting signals.
            configuration: Configuration parameters for technical indicators.
                Defaults to TurnoverConfiguration() when not provided.
        """
        super().__init__(data_provider, telegram_bot)
        self.trade_repository = trade_repository
        self.configuration = configuration or TurnoverConfiguration()
        self.holiday_checker = MarketHolidayChecker()

    @override
    def run(
        self,
        days: int = 0,
        analysis_date: str | None = None,
        specific_symbols: list[str] | None = None,
    ) -> int:
        """
        Orchestrates the Turnover signals scanning process.

        Logic:
        - Setup: Weekly Close (Friday or Holiday-Thursday).
        - Filter: Close > SMA150 and high relative turnover.
        - Entry: Limit order at 0.5 * ATR or 1.0 * ATR below Close.
        """
        analysis_timestamp = self._get_analysis_timestamp(days, analysis_date)

        if not self._is_setup_day(analysis_timestamp):
            return 0

        # Compile Universe
        symbol_loader = ExchangeSymbol()
        indices_map = {
            "NDX": symbol_loader.nasdaq_100,
            "SPX": symbol_loader.sp_500,
            "RUS": symbol_loader.russell_1000,
        }
        target_universe = self._compile_target_universe(indices_map, specific_symbols)

        # Load Data
        lookback = self._calculate_required_lookback(analysis_timestamp)
        data_frames = self.data_provider.get_universe_daily_data(
            target_universe, days=lookback
        )
        if not data_frames or "close" not in data_frames or data_frames["close"].empty:
            return 0

        # Find last available trading day
        closes = data_frames["close"].ffill()
        setup_date = self._resolve_target_date(closes, analysis_timestamp)
        if setup_date is None:
            return 0

        if not self._is_setup_day(setup_date):
            return 0

        # Compute Indicators & Find Candidates
        candidates = self._identify_strategy_candidates(
            data_frames, setup_date, indices_map
        )
        if not candidates:
            return 0

        # Create Signals
        count = self._process_and_store_signals(candidates, data_frames, setup_date)

        # Report
        self._report_signals_to_telegram(candidates, setup_date)

        return count

    def _resolve_target_date(
        self,
        historical_closes: pd.DataFrame,
        analysis_timestamp: pd.Timestamp,
    ) -> pd.Timestamp | None:
        """Resolves the correct analysis date based on data availability."""
        if historical_closes.empty:
            return None

        if analysis_timestamp in historical_closes.index:
            return analysis_timestamp

        # 1. Post-Data Case (Data lag)
        if analysis_timestamp > historical_closes.index[-1]:
            logger.warning(
                "[%s] Requested analysis timestamp %s but data ends "
                "%s. Falling back to last available data.",
                self.name,
                analysis_timestamp.date(),
                historical_closes.index[-1].date(),
            )
            return historical_closes.index[-1]

        # 2. Gap Case (Holiday)
        available_dates = historical_closes.index[
            historical_closes.index < analysis_timestamp
        ]
        if available_dates.empty:
            logger.error(
                "[%s] No data found before %s",
                self.name,
                analysis_timestamp.date(),
            )
            return None

        fallback_date = available_dates[-1]
        logger.info(
            "[%s] %s is holiday/missing. Using: %s",
            self.name,
            analysis_timestamp.date(),
            fallback_date.date(),
        )
        return fallback_date

    def _get_analysis_timestamp(
        self, days: int, analysis_date: str | None
    ) -> pd.Timestamp:
        """Determines the effective timestamp for signal generation."""
        if analysis_date:
            return pd.Timestamp(analysis_date).normalize()
        return pd.Timestamp.now().normalize() - pd.Timedelta(days=days)

    def _is_setup_day(self, analysis_timestamp: pd.Timestamp) -> bool:
        """Verifies if today is Friday or a holiday-adjusted Thursday."""
        day_of_week = analysis_timestamp.dayofweek  # Monday=0, Sunday=6

        if day_of_week == 4:  # Friday
            return True

        if day_of_week == 3:  # Thursday
            tomorrow = analysis_timestamp + pd.Timedelta(days=1)
            if self.holiday_checker.is_holiday(tomorrow.date()):
                return True

        return False

    def _compile_target_universe(
        self,
        indices_map: dict[str, list[str]],
        specific_symbols: list[str] | None,
    ) -> list[str]:
        """Combines index symbols into a single list for screening."""
        all_symbols = set()
        for symbol_list in indices_map.values():
            all_symbols.update(symbol_list)

        target_universe = list(all_symbols)
        if specific_symbols:
            target_universe = list(set(target_universe) & set(specific_symbols))

        return target_universe

    def _calculate_required_lookback(self, analysis_timestamp: pd.Timestamp) -> int:
        """Calculates the necessary data lookback for indicator calculations."""
        base_lookback = self.configuration.minimum_lookback_days
        days_diff = (pd.Timestamp.now() - analysis_timestamp).days

        if days_diff > (base_lookback - 100):
            return int(days_diff + 500)  # Buffer for historical runs
        return base_lookback

    def _identify_strategy_candidates(
        self,
        data_frames: dict[str, pd.DataFrame],
        setup_date: pd.Timestamp,
        indices_map: dict[str, list[str]],
    ) -> list[TurnoverCandidate]:
        """Identifies potential trading candidates across all indices.

        Uses vectorized Pandas operations to rank by turnover and filter by trend.
        """
        closes = data_frames["close"].ffill()
        highs = data_frames["high"].ffill()
        lows = data_frames["low"].ffill()
        volumes = data_frames["volume"].fillna(0)

        try:
            # Slice inputs for calculation window
            required_window = self.configuration.sma_window
            start_location = max(0, closes.index.get_loc(setup_date) - required_window)
            end_location = closes.index.get_loc(setup_date) + 1

            closes_slice = closes.iloc[start_location:end_location]
            highs_slice = highs.iloc[start_location:end_location]
            lows_slice = lows.iloc[start_location:end_location]
            volumes_slice = volumes.iloc[start_location:end_location]

            # Calculate Indicators (Vectorized)
            indicators_dict = self._calculate_indicators_slice(
                closes_slice, highs_slice, lows_slice, volumes_slice
            )

            current_close = closes.loc[setup_date]
            current_sma_price = indicators_dict["sma_price"].loc[setup_date]
            current_sma_turnover = indicators_dict["sma_turnover"].loc[setup_date]
            current_atr = indicators_dict["atr"].loc[setup_date]
        except KeyError as missing_data_error:
            logger.warning(
                "[%s] Setup date missing from indicators: %s",
                self.name,
                missing_data_error,
            )
            return []

        return self._rank_and_filter_candidates(
            closes,
            indices_map,
            current_close,
            current_sma_price,
            current_sma_turnover,
            current_atr,
        )

    def _calculate_indicators_slice(
        self,
        closes: pd.DataFrame,
        highs: pd.DataFrame,
        lows: pd.DataFrame,
        volumes: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """Calculates indicators on sliced data."""
        turnover = closes * volumes
        sma_turnover_20 = indicators.calculate_sma(turnover, 20)
        sma_price_150 = indicators.calculate_sma(closes, 150)
        atr_series = indicators.calculate_atr(
            highs, lows, closes, self.configuration.atr_window
        )
        return {
            "sma_turnover": sma_turnover_20,
            "sma_price": sma_price_150,
            "atr": atr_series,
        }

    def _rank_and_filter_candidates(
        self,
        closes: pd.DataFrame,
        indices_map: dict[str, list[str]],
        current_close: pd.Series,
        current_sma_price: pd.Series,
        current_sma_turnover: pd.Series,
        current_atr: pd.Series,
    ) -> list[TurnoverCandidate]:
        """Ranks and filters candidates based on turnover and trend metrics."""
        candidates = []
        symbol_filter = SymbolFilter()

        for index_name, symbols_in_index in indices_map.items():
            # Create a mask for valid symbols in this index
            index_mask = [s for s in symbols_in_index if s in closes.columns]
            if not index_mask:
                continue

            # Consolidate data for vectorized filtering
            current_data = pd.DataFrame(
                {
                    "close": current_close.reindex(index_mask),
                    "sma_price": current_sma_price.reindex(index_mask),
                    "turnover_sma": current_sma_turnover.reindex(index_mask),
                    "atr": current_atr.reindex(index_mask),
                }
            ).dropna()

            if current_data.empty:
                continue

            # Rank by Turnover SMA 20 (Highest first)
            ranked_candidates = current_data.sort_values(
                "turnover_sma", ascending=False
            )

            # Filter out secondary share classes (Internal tool is not yet vectorized)
            ranked_symbols = symbol_filter.filter_symbols(
                ranked_candidates.index.tolist()
            )
            top_20_candidates = ranked_candidates.loc[ranked_symbols].head(20)

            # Filter: Close > SMA 150 (Uptrend) and ATR > 0
            final_selection = top_20_candidates.query(
                "close > sma_price and atr > 0"
            ).head(4)

            candidates.extend(
                final_selection.rename_axis("symbol")
                .reset_index()
                .apply(
                    lambda row, index_name=index_name: TurnoverCandidate(
                        symbol=str(row["symbol"]),
                        close=float(row["close"]),
                        sma_price=float(row["sma_price"]),
                        sma_turnover=float(row["turnover_sma"]),
                        atr=float(row["atr"]),
                        indices=index_name,
                    ),
                    axis=1,
                )
                .tolist()
            )

        return candidates

    def _process_and_store_signals(
        self,
        candidates: list[TurnoverCandidate],
        data_frames: dict[str, pd.DataFrame],
        setup_date: pd.Timestamp,
    ) -> int:
        """Deduplicates candidates and stores signals in the database.

        If a symbol is in multiple indices, their names are concatenated.
        """
        signal_count = 0
        setup_date_str = str(setup_date.date())

        # Deduplicate candidates across indices
        merged_candidates: dict[str, TurnoverCandidate] = {}
        for candidate in candidates:
            symbol = str(candidate["symbol"])
            if symbol not in merged_candidates:
                merged_candidates[symbol] = candidate
            else:
                existing_indices = str(merged_candidates[symbol]["indices"]).split(", ")
                if str(candidate["indices"]) not in existing_indices:
                    merged_candidates[symbol]["indices"] = (
                        f"{merged_candidates[symbol]['indices']}, {candidate['indices']}"
                    )

        unique_candidates = merged_candidates.values()

        for candidate in unique_candidates:
            signal_count += self._store_turnover_signals_for_candidate(
                candidate, data_frames, setup_date, setup_date_str
            )

        return signal_count

    def _store_turnover_signals_for_candidate(
        self,
        candidate: TurnoverCandidate,
        data_frames: dict[str, pd.DataFrame],
        setup_date: pd.Timestamp,
        setup_date_str: str,
    ) -> int:
        """Stores entry limit order signal variants for a candidate in database.

        Args:
            candidate: Aggregated turnover candidate data.
            data_frames: Universe daily dataframes.
            setup_date: Setup timestamp.
            setup_date_str: Setup ISO date string.

        Returns:
            int: Number of signals created.
        """
        symbol = str(candidate["symbol"])
        close_price = float(candidate["close"])
        atr_value = float(candidate["atr"])
        created_signals = 0

        for factor in self.configuration.entry_factors:
            strategy_name = f"{self.name}_{factor}"

            # Calculate Limit Entry: Close - (Factor * ATR)
            limit_price = round(close_price - (atr_value * factor), 2)

            if self.trade_repository.exists(symbol, strategy_name, setup_date_str):
                continue

            is_green = bool(close_price > data_frames["open"].loc[setup_date][symbol])
            context: TurnoverSignalContext = {
                "setup_date": setup_date_str,
                "setup_close": close_price,
                "setup_candle_green": is_green,
                "green_candle_count": 1 if is_green else 0,
                "setup_sma_price": float(candidate["sma_price"]),
                "setup_turnover_sma": round(float(candidate["sma_turnover"]), 0),
                "setup_atr": round(atr_value, 2),
                "factor": factor,
                "indices": str(candidate["indices"]),
                "source": "screener",
                "date": setup_date_str,
            }

            self.trade_repository.create_trade(
                symbol=symbol,
                strategy=strategy_name,
                size=0,
                entry=limit_price,
                stop_loss=0.0,
                target=0.0,
                context=dict(context),
            )
            created_signals += 1

        return created_signals

    def _report_signals_to_telegram(
        self, candidates: list[TurnoverCandidate], setup_date: pd.Timestamp
    ) -> None:
        """Sends a consolidated signal report to Telegram."""
        if not self.telegram_bot or not candidates:
            return

        # Deduplicate for reporting
        unique_symbols: dict[str, TurnoverCandidate] = {}
        for candidate in candidates:
            symbol = str(candidate["symbol"])
            if symbol not in unique_symbols:
                unique_symbols[symbol] = candidate

        report_items: list[SignalReportItem] = []
        for item in unique_symbols.values():
            close_price = float(item["close"])
            atr_value = float(item["atr"])

            for factor in self.configuration.entry_factors:
                limit_price = round(close_price - (atr_value * factor), 2)
                report_items.append(
                    SignalReportItem(
                        symbol=str(item["symbol"]),
                        action=f"BUY LMT ({factor} ATR)",
                        entry_price=limit_price,
                        details={
                            "Close": round(close_price, 2),
                            "ATR": round(atr_value, 2),
                        },
                    )
                )

        self._send_telegram_report(
            "Turnover Signals", report_items, str(setup_date.date())
        )

    def analyze_single_symbol(self, symbol: str) -> dict[str, object]:
        """
        Calculates indicators for a single symbol for debugging purposes.
        This represents the Imperative Shell: it fetches data and delegates to the Functional Core.
        """
        days_lookback = 400
        dataframe = self.data_provider.get_symbol_history(symbol, days=days_lookback)
        if dataframe.empty:
            return {"symbol": symbol, "error": "No data found"}

        return self._compute_single_symbol_analysis(symbol, dataframe)

    def _compute_single_symbol_analysis(
        self, symbol: str, dataframe: pd.DataFrame
    ) -> dict[str, object]:
        """
        Functional Core: Computes the indicators for the analysis output without side effects.
        """
        try:
            closes = pd.Series(dataframe["close"].values, index=dataframe["date"])
            highs = pd.Series(dataframe["high"].values, index=dataframe["date"])
            lows = pd.Series(dataframe["low"].values, index=dataframe["date"])
            volumes = pd.Series(dataframe["volume"].values, index=dataframe["date"])
        except (KeyError, ValueError, TypeError) as error:
            return {"symbol": symbol, "error": f"Data Frame Error: {str(error)}"}

        if len(closes) < 200:
            return {
                "symbol": symbol,
                "error": f"Not enough data (Found {len(closes)}, Need 200+)",
            }

        turnover = closes * volumes
        sma_turnover_20 = indicators.calculate_sma(turnover, 20)
        sma_price_150 = indicators.calculate_sma(closes, 150)

        # ATR Calculation
        atr_series = indicators.calculate_atr(
            highs, lows, closes, self.configuration.atr_window
        )

        # Extract Last Values
        current_close = closes.iloc[-1]
        current_sma150 = sma_price_150.iloc[-1]
        current_turnover_sma = sma_turnover_20.iloc[-1]
        current_atr = atr_series.iloc[-1]

        # Trend Logic Check
        uptrend = bool(current_close > current_sma150)

        return {
            "symbol": symbol,
            "last_date": str(dataframe["date"].iloc[-1].date()),
            "data_valid": True,
            "checks": {"uptrend_sma150": uptrend, "data_sufficient": True},
            "values": {
                "close": round(self._extract_safe_float_value(current_close), 2),
                "sma150": round(self._extract_safe_float_value(current_sma150), 2),
                "turnover_sma20": int(
                    self._extract_safe_float_value(current_turnover_sma, 0)
                ),
                "atr": round(self._extract_safe_float_value(current_atr), 2),
            },
            "note": "Rank logic (Top 20) requires full market scan.",
        }

    def _extract_safe_float_value(self, value: object, default: float = 0.0) -> float:
        """Safely extracts a float value from a potentially null/NaN object."""
        if pd.isna(value):
            return default
        return float(str(value))
