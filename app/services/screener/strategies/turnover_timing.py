import logging
from dataclasses import dataclass, field
from typing import override, TypedDict

import pandas as pd

from ....database.repositories.market_data_provider import MarketDataProvider
from ....database.repositories.trade import TradeRepository
from ....tools.symbol_lists import ExchangeSymbol
from ....tools.market_holidays import MarketHolidayChecker
from ....tools import indicators
from ....tools.symbol_filter import SymbolFilter
from ...telegram import TelegramBot
from .base import BaseStrategy
from ....const import Strategies

logger = logging.getLogger(__name__)


class TurnoverSignalContext(TypedDict):
    """Metadata for the Turnover Timing signal."""
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


class TurnoverTimingStrategy(BaseStrategy):
    name: str = str(Strategies.TurnOverTiming)

    def __init__(
        self,
        trade_repository: TradeRepository,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None,
        config: TurnoverConfiguration = TurnoverConfiguration(),
    ) -> None:
        """
        Initializes the Turnover Timing strategy.

        Args:
            trade_repository: Repository for trade persistence.
            data_provider: Provider for market historical data.
            telegram_bot: Optional bot for reporting signals.
            config: Configuration parameters for technical indicators.
        """
        super().__init__(data_provider, telegram_bot)
        self.trade_repository = trade_repository
        self.configuration = config
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
        if not data_frames:
            return 0

        # Find last available trading day
        closes = data_frames["close"].ffill()
        if closes.empty:
            return 0

        available_dates = closes.index[closes.index <= analysis_timestamp]
        if available_dates.empty:
            return 0
        last_trading_day = available_dates[-1]

        if last_trading_day != analysis_timestamp:
            logger.warning(
                "[%s] Analysis date %s requested, but latest data is %s.",
                self.name,
                analysis_timestamp.date(),
                last_trading_day.date(),
            )
            return 0

        setup_date = last_trading_day

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
            return days_diff + 500  # Buffer for historical runs
        return base_lookback

    def _identify_strategy_candidates(
        self,
        data_frames: dict[str, pd.DataFrame],
        setup_date: pd.Timestamp,
        indices_map: dict[str, list[str]],
    ) -> list[dict[str, object]]:
        """Identifies potential trading candidates across all indices."""
        closes = data_frames["close"].ffill()
        highs = data_frames["high"].ffill()
        lows = data_frames["low"].ffill()
        volumes = data_frames["volume"].fillna(0)

        required_window = self.configuration.sma_window
        start_location = max(0, closes.index.get_loc(setup_date) - required_window)

        # Slice inputs for the calculation window
        closes_slice = closes.iloc[start_location : closes.index.get_loc(setup_date) + 1]
        highs_slice = highs.iloc[start_location : highs.index.get_loc(setup_date) + 1]
        lows_slice = lows.iloc[start_location : lows.index.get_loc(setup_date) + 1]
        volumes_slice = volumes.iloc[
            start_location : volumes.index.get_loc(setup_date) + 1
        ]

        # Calculate Indicators
        turnover_slice = closes_slice * volumes_slice
        sma_turnover_20 = indicators.calculate_sma(turnover_slice, 20)
        sma_price_150 = indicators.calculate_sma(closes_slice, 150)
        atr_series = indicators.calculate_atr(
            highs_slice, lows_slice, closes_slice, self.configuration.atr_window
        )

        # Extract values for Setup Day
        try:
            current_close = closes.loc[setup_date]
            current_sma_price = sma_price_150.loc[setup_date]
            current_sma_turnover = sma_turnover_20.loc[setup_date]
            current_atr = atr_series.loc[setup_date]
        except KeyError:
            return []

        candidates: list[dict[str, object]] = []

        for index_name, symbols_in_index in indices_map.items():
            valid_symbols = [
                symbol
                for symbol in symbols_in_index
                if symbol in closes.columns
                and not pd.isna(current_close.get(symbol))
                and not pd.isna(current_sma_turnover.get(symbol))
                and not pd.isna(current_sma_price.get(symbol))
                and not pd.isna(current_atr.get(symbol))
            ]

            if not valid_symbols:
                continue

            # Rank by Turnover SMA 20 (Highest first)
            ranked_by_turnover = sorted(
                valid_symbols, key=lambda s: current_sma_turnover[s], reverse=True
            )

            # Filter out secondary share classes
            ranked_by_turnover = SymbolFilter().filter_symbols(ranked_by_turnover)

            # Take Top 20 (Liquid candidates)
            top_20_liquid = ranked_by_turnover[:20]

            # Filter: Close > SMA 150 (Uptrend) and ATR > 0
            trend_filtered = [
                symbol
                for symbol in top_20_liquid
                if current_close[symbol] > current_sma_price[symbol]
                and current_atr[symbol] > 0
            ]

            # Select Top 4 from the remaining list
            final_selection = trend_filtered[:4]

            for symbol in final_selection:
                candidates.append(
                    {
                        "symbol": symbol,
                        "close": float(current_close[symbol]),
                        "sma_price": float(current_sma_price[symbol]),
                        "sma_turnover": float(current_sma_turnover[symbol]),
                        "atr": float(current_atr[symbol]),
                        "indices": index_name,
                    }
                )

        return candidates

    def _process_and_store_signals(
        self,
        candidates: list[dict[str, object]],
        data_frames: dict[str, pd.DataFrame],
        setup_date: pd.Timestamp,
    ) -> int:
        """Deduplicates candidates and stores signals in the database."""
        signal_count = 0
        setup_date_str = str(setup_date.date())

        # Deduplicate candidates across indices
        merged_candidates: dict[str, dict[str, object]] = {}
        for candidate in candidates:
            symbol = str(candidate["symbol"])
            if symbol not in merged_candidates:
                merged_candidates[symbol] = candidate
            else:
                existing_indices = str(merged_candidates[symbol]["indices"]).split(", ")
                if str(candidate["indices"]) not in existing_indices:
                    merged_candidates[symbol][
                        "indices"
                    ] = f"{merged_candidates[symbol]['indices']}, {candidate['indices']}"

        unique_candidates = merged_candidates.values()

        for candidate in unique_candidates:
            symbol = str(candidate["symbol"])
            close_price = float(candidate["close"])
            atr_value = float(candidate["atr"])

            for factor in self.configuration.entry_factors:
                strategy_name = f"{self.name}_{factor}"

                # Calculate Limit Entry: Close - (Factor * ATR)
                limit_price = round(close_price - (atr_value * factor), 2)

                if self.trade_repository.exists(
                    symbol, strategy_name, setup_date_str
                ):
                    continue

                context: TurnoverSignalContext = {
                    "setup_date": setup_date_str,
                    "setup_close": close_price,
                    "setup_candle_green": bool(
                        close_price > data_frames["open"].loc[setup_date][symbol]
                    ),
                    "green_candle_count": (
                        1
                        if close_price > data_frames["open"].loc[setup_date][symbol]
                        else 0
                    ),
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
                    context=dict(context),  # type: ignore
                )
                signal_count += 1

        return signal_count

    def _report_signals_to_telegram(
        self, candidates: list[dict[str, object]], setup_date: pd.Timestamp
    ) -> None:
        """Sends a consolidated signal report to Telegram."""
        if not self.telegram_bot or not candidates:
            return

        # Deduplicate for reporting
        unique_symbols: dict[str, dict[str, object]] = {}
        for candidate in candidates:
            symbol = str(candidate["symbol"])
            if symbol not in unique_symbols:
                unique_symbols[symbol] = candidate

        report_rows = []
        for item in unique_symbols.values():
            close_price = float(item["close"])
            atr_value = float(item["atr"])

            report_rows.append(
                {
                    "Symbol": item["symbol"],
                    "Entry 0.5 ATR": round(close_price - (atr_value * 0.5), 2),
                    "Entry 1.0 ATR": round(close_price - (atr_value * 1.0), 2),
                    "Close": round(close_price, 2),
                    "ATR": round(atr_value, 2),
                }
            )

        report_dataframe = pd.DataFrame(report_rows)
        self._send_telegram_report(
            "Turnover Signals", str(setup_date.date()), report_dataframe
        )

        return count

    def analyze_single_symbol(self, symbol: str) -> dict[str, object]:
        """
        Calculates indicators for a single symbol for debugging purposes.

        Note:
            Ranking (Top 20 / Top 4) cannot be fully verified in isolation
            as it depends on other stocks. This checks absolute criteria only.
        """
        days_lookback = 400
        dataframe = self.data_provider.get_symbol_history(symbol, days=days_lookback)
        if dataframe.empty:
            return {"symbol": symbol, "error": "No data found"}

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
                "turnover_sma20": int(self._extract_safe_float_value(current_turnover_sma, 0)),
                "atr": round(self._extract_safe_float_value(current_atr), 2),
            },
            "note": "Rank logic (Top 20) requires full market scan.",
        }

    def _extract_safe_float_value(self, value: object, default: float = 0.0) -> float:
        """Safely extracts a float value from a potentially null/NaN object."""
        if pd.isna(value):
            return default
        return float(value)
