import logging
import pandas as pd
from dataclasses import dataclass
from typing import override, TypedDict

from ....database.repositories.market_data_provider import MarketDataProvider
from ....database.repositories.trade import TradeRepository
from ....services.telegram import TelegramBot
from ....tools.symbol_filter import SymbolFilter
from ....tools.symbol_lists import ExchangeSymbol
from .base import BaseStrategy
from ....const import Strategies

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DipBuyerConfig:
    """Configuration for the DipBuyer strategy logic."""

    # 1. Basic Filters
    MIN_VOLUME: int = 1_000_000
    MIN_PRICE: float = 5.0

    # 2. Indicator Parameters
    ATR_WINDOW: int = 5
    ENTRY_FACTOR: float = 1.0  # Entry = Close - (ATR * ENTRY_FACTOR)
    SMA_TREND_WINDOW: int = 200

    # 3. Logic Thresholds
    MIN_VOLATILITY_RATIO: float = 0.03  # ATR must be > 3% of Price
    MAX_IBS: float = 0.2  # Close in bottom 20% of High-Low range
    MAX_ATR_RATIO_3DAY: float = -1.0  # 3-Day drop > 1 ATR (negative value)

    # 4. Exit Parameters
    EXIT_TP_FACTOR: float = 0.8  # Target = Entry + (ATR * EXIT_TP_FACTOR)

    # Data Fetching
    LOOKBACK_DAYS: int = 600


class DipBuyerMarketState(TypedDict):
    """Represents the market state for a single symbol at a specific point in time."""

    close: float
    open: float
    high: float
    high_next_target: float | None  # Optional, for future use or verification
    volume: float
    sma200: float
    volume_sma: float
    atr: float
    atr_ratio_3day: float
    ibs: float
    volatility_ratio: float
    setup_score: float


class SymbolAnalysisResult(TypedDict):
    """Return type for single symbol analysis debugging."""

    symbol: str
    indices: list[str]
    last_date: str
    data_valid: bool
    checks: dict[str, bool]
    values: dict[str, float]
    result: str
    error: str | None


class DipBuyerStrategy(BaseStrategy):
    """
    Identifies mean-reversion opportunities in uptrending stocks.

    Logic:
    1. Trend: Price > SMA 200
    2. Dip: 3-day drop significant relative to ATR
    3. Setup: Close near daily low (Low IBS)
    """

    name: str = str(Strategies.DipBuyer)

    # Pre-calculated set mappings for index lookup (Instance variables for thread safety)
    _dow_set: set[str] | frozenset[str] = frozenset()
    _sp500_set: set[str] | frozenset[str] = frozenset()
    _ndx_set: set[str] | frozenset[str] = frozenset()

    def __init__(
        self,
        trade_repository: TradeRepository,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None,
        config: DipBuyerConfig | None = None,
    ) -> None:
        super().__init__(data_provider, telegram_bot)
        self.trade_repository = trade_repository
        self.config = config or DipBuyerConfig()

        self._initialize_symbol_sets()

    def _initialize_symbol_sets(self) -> None:
        """Initializes index membership sets if not already loaded."""
        if not self._dow_set:
            exchange_symbols = ExchangeSymbol()
            self._dow_set = frozenset(exchange_symbols.dow_30)
            self._sp500_set = frozenset(exchange_symbols.sp_500)
            self._ndx_set = frozenset(exchange_symbols.nasdaq_100)

    @override
    def run(
        self,
        days: int = 0,
        analysis_date: str | None = None,
        specific_symbols: list[str] | None = None,
    ) -> int:
        """
        Executes the strategy pipeline.

        Args:
            days: Not strictly used here, overriden by config.LOOKBACK_DAYS logic.
            analysis_date: Optional date string (YYYY-MM-DD) for backtesting.
            specific_symbols: Optional list of symbols to filter.

        Returns:
            int: Number of trades created.
        """
        # 1. Determine Lookback
        historical_lookback_days = self.config.LOOKBACK_DAYS

        # 2. Determine Universe
        self._initialize_symbol_sets()
        valid_universe = self._dow_set | self._sp500_set | self._ndx_set

        target_symbols: list[str] = []
        if specific_symbols:
            target_symbols = list(set(specific_symbols) & valid_universe)
            if not target_symbols:
                logger.warning(
                    f"[{self.name}] No valid symbols found in request (must be in indices)."
                )
                return 0
        else:
            target_symbols = list(valid_universe)

        # 3. Load Data
        market_data = self.data_provider.get_universe_daily_data(
            target_symbols, days=historical_lookback_days
        )

        if not market_data or "close" not in market_data or market_data["close"].empty:
            logger.warning(f"[{self.name}] No market data available for universe.")
            return 0

        # 4. Resolve Date
        target_date = self._resolve_target_date(market_data["close"], analysis_date)
        if target_date is None:
            return 0

        # 5. Execute Pipeline
        return self._execute_screening_pipeline(market_data, target_date)

    def _resolve_target_date(
        self, closes: pd.DataFrame, analysis_date: str | None
    ) -> pd.Timestamp | None:
        """Resolves the correct analysis date based on data availability."""
        if not analysis_date:
            last_date = closes.index[-1]
            logger.info(
                f"[{self.name}] Analysis Date not provided. Using last DB date: {last_date.date()}"
            )
            return last_date

        try:
            target_date = pd.Timestamp(analysis_date)
        except ValueError:
            logger.error(f"Invalid analysis date: {analysis_date}")
            return None

        if target_date in closes.index:
            return target_date

        # 1. Post-Data Case (Data lag)
        if target_date > closes.index[-1]:
            logger.warning(
                f"[{self.name}] Requested {target_date.date()} but data ends "
                f"{closes.index[-1].date()}. Falling back to last available data."
            )
            return closes.index[-1]

        # 2. Gap Case (Holiday)
        available_dates = closes.index[closes.index < target_date]
        if available_dates.empty:
            logger.error(f"[{self.name}] No data found before {target_date.date()}")
            return None

        fallback_date = available_dates[-1]
        logger.info(
            f"[{self.name}] {target_date.date()} is holiday/missing. "
            f"Using: {fallback_date.date()}"
        )
        return fallback_date

    def _execute_screening_pipeline(
        self, market_data: dict[str, pd.DataFrame], target_date: pd.Timestamp
    ) -> int:
        """Calculates signals and processes valid trades."""
        signals_dataframe = self._calculate_signals(market_data, target_date)

        if signals_dataframe.empty:
            logger.debug(f"[{self.name}] No signals found for {target_date.date()}.")
            return 0

        # --- Symbol Filtering Integration ---
        candidates = signals_dataframe.index.tolist()
        filtered_candidates = SymbolFilter().filter_symbols(candidates)

        if len(candidates) != len(filtered_candidates):
            removed = set(candidates) - set(filtered_candidates)
            logger.info(f"[{self.name}] Filtered out secondary symbols: {removed}")
            signals_dataframe = signals_dataframe.loc[filtered_candidates]

        logger.debug(f"[{self.name}] Total signals: {len(signals_dataframe)}")
        return self._process_signals(signals_dataframe, target_date)

    def _calculate_signals(
        self, market_data: dict[str, pd.DataFrame], target_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Computes indicators and applies logic filters to find valid signals."""
        closes = market_data["close"]

        # FIX: Clean the Index (Remove holidays where all US stocks are NaN)
        valid_trading_days_index = closes.dropna(how="all").index
        closes = closes.loc[valid_trading_days_index]
        highs = market_data["high"].loc[valid_trading_days_index]
        lows = market_data["low"].loc[valid_trading_days_index]
        opens = market_data["open"].loc[valid_trading_days_index]
        volumes = market_data["volume"].loc[valid_trading_days_index]

        # Note: _resolve_target_date already ensured the date exists in the raw data or selected a fallback.
        if target_date not in closes.index:
            logger.warning(
                f"[{self.name}] Target date {target_date.date()} dropped after cleaning (no valid data for universe)."
            )
            return pd.DataFrame()

        try:
            current_index_location = closes.index.get_loc(target_date)
        except KeyError:
            logger.error(f"Date {target_date} not found in market data.")
            return pd.DataFrame()

        # --- Data Slicing ---
        slices = self._slice_data_for_window(
            closes, highs, lows, volumes, opens, current_index_location
        )
        if not slices:
            return pd.DataFrame()

        return self._slice_and_compute_market_state(slices)

    def _slice_and_compute_market_state(
        self,
        slices: tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
        ],
    ) -> pd.DataFrame:
        """Computes technical states and filters candidates from sliced price blocks."""
        closes_slice, highs_slice, lows_slice, volumes_slice, opens_slice = slices

        # --- Indicator Calculation ---
        indicators_dict = self._compute_indicators(
            closes_slice, highs_slice, lows_slice, volumes_slice
        )

        # --- Market State Construction ---
        # Current Row Index (Relative to slice)
        index = len(closes_slice) - 1
        previous_index = index - 1

        if previous_index < 0:
            return pd.DataFrame()

        current_market_state = {
            "close": closes_slice.iloc[index],
            "open": opens_slice.iloc[index],
            "high": highs_slice.iloc[index],
            "low": lows_slice.iloc[index],
            "volume": volumes_slice.iloc[index],
            "sma200": indicators_dict["sma200"].iloc[index],
            "volume_sma": indicators_dict["volume_sma"].iloc[index],
            "atr": indicators_dict["atr"].iloc[index],
            "atr_ratio_3day": indicators_dict["atr_ratio_3day"].iloc[index],
            "ibs": indicators_dict["ibs"].iloc[index],
            "volatility_ratio": indicators_dict["volatility_ratio"].iloc[index],
            "setup_score": indicators_dict["atr_ratio_3day"].iloc[index] * -1,
        }

        previous_market_state = {
            "close": closes_slice.iloc[previous_index],
            "open": opens_slice.iloc[previous_index],
        }

        # --- Filter Application ---
        return self._filter_market_state(current_market_state, previous_market_state)

    def _slice_data_for_window(
        self,
        closes: pd.DataFrame,
        highs: pd.DataFrame,
        lows: pd.DataFrame,
        volumes: pd.DataFrame,
        opens: pd.DataFrame,
        current_location: int,
    ) -> (
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        | None
    ):
        """Slices the dataframes to the required calculation window."""
        required_window = 250
        start_location = max(0, current_location - required_window)
        # Python slicing excludes upper bound, so +1
        end_location = current_location + 1

        if start_location >= end_location:
            return None

        return (
            closes.iloc[start_location:end_location],
            highs.iloc[start_location:end_location],
            lows.iloc[start_location:end_location],
            volumes.iloc[start_location:end_location],
            opens.iloc[start_location:end_location],
        )

    def _compute_indicators(
        self,
        closes: pd.DataFrame,
        highs: pd.DataFrame,
        lows: pd.DataFrame,
        volumes: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """Calculates all technical indicators efficiently."""
        from ....tools import indicators

        # A) Trend: SMA 200
        sma200 = indicators.calculate_sma(closes, self.config.SMA_TREND_WINDOW)

        # B) Volume: SMA 20
        volume_sma20 = indicators.calculate_volume_sma(volumes, 20)

        # C) ATR
        atr = indicators.calculate_atr(highs, lows, closes, self.config.ATR_WINDOW)

        # D) Dip Metrics
        price_drop_3day = closes - closes.shift(3)
        atr_ratio_3day = price_drop_3day / atr

        # E) IBS
        ibs = indicators.calculate_ibs(highs, lows, closes)

        # F) Volatility Ratio
        volatility_ratio = atr / closes.replace(0.0, float("nan"))

        return {
            "sma200": sma200,
            "volume_sma": volume_sma20,
            "atr": atr,
            "atr_ratio_3day": atr_ratio_3day,
            "ibs": ibs,
            "volatility_ratio": volatility_ratio,
        }

    def _filter_market_state(
        self,
        current: dict[str, pd.Series],
        previous: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Applies configuration rules to filter candidates."""

        # 1. Liquidity & Price
        mask = (current["volume_sma"] > self.config.MIN_VOLUME) & (
            current["close"] > self.config.MIN_PRICE
        )

        # 2. Trend (Above SMA200)
        mask &= current["close"] > current["sma200"]

        # 3. Dip Conditions
        mask &= current["atr_ratio_3day"] < self.config.MAX_ATR_RATIO_3DAY
        mask &= current["volatility_ratio"] > self.config.MIN_VOLATILITY_RATIO
        mask &= current["ibs"] < self.config.MAX_IBS

        # 4. Candles (Today Red, Yesterday Red)
        mask &= current["close"] < current["open"]
        mask &= previous["close"] < previous["open"]

        # Apply Mask
        hits_symbols = mask[mask].index

        # Assemble Result
        df_results = pd.DataFrame(
            {
                "close": current["close"],
                "high": current["high"],
                "volume": current["volume"],
                "atr": current["atr"],
                "sma200": current["sma200"],
                "atr_ratio_3day": current["atr_ratio_3day"],
                "ibs": current["ibs"],
                "setup_score": current["setup_score"],
            },
            index=current["close"].index,
        ).loc[hits_symbols]

        return df_results.sort_values(by="setup_score", ascending=False)

    def _process_signals(self, signals: pd.DataFrame, date_obj: pd.Timestamp) -> int:
        """Creates trade objects and reports results."""
        created_trades = []
        saved_count = 0
        date_str = date_obj.strftime("%Y-%m-%d")

        for symbol, signal_row in signals.iterrows():
            try:
                saved = self._create_single_dip_buyer_trade(
                    str(symbol), signal_row, date_str, created_trades
                )
                if saved:
                    saved_count += 1
            except (ValueError, TypeError, KeyError) as data_error:
                logger.warning(
                    "[%s] Failed to save trade for %s: %s",
                    self.name,
                    symbol,
                    data_error,
                )
            except Exception as database_error:
                raise RuntimeError(
                    "[%s] Database unavailable saving trade for %s: %s"
                    % (self.name, symbol, database_error)
                ) from database_error

        # Reporting
        if self.telegram_bot and created_trades:
            prefix = "LIVE"
            df = pd.DataFrame(created_trades)
            self._send_telegram_report(f"{self.name} ({prefix})", date_str, df)

        return saved_count

    def _create_single_dip_buyer_trade(
        self,
        symbol: str,
        signal_row: pd.Series,
        date_str: str,
        created_trades: list[dict[str, object]],
    ) -> bool:
        """Helper to create a single database trade for a candidate symbol.

        Args:
            symbol: Ticker symbol.
            signal_row: Candidate metrics row.
            date_str: Target date string.
            created_trades: List of successfully created trades to append to.

        Returns:
            bool: True if trade was successfully written.
        """
        entry_price = signal_row["close"] - (
            signal_row["atr"] * self.config.ENTRY_FACTOR
        )
        target_price = entry_price + (signal_row["atr"] * self.config.EXIT_TP_FACTOR)
        high_next_target = signal_row["high"] + 0.01

        # Use centralized BaseStrategy index helper (DRY resolution)
        indices = self._get_indices_for_symbol(symbol)

        context = {
            "source": "screener",
            "date": date_str,
            "setup_score": round(signal_row["setup_score"], 2),
            "close": round(signal_row["close"], 2),
            "volume": float(signal_row["volume"]),
            "atr5": round(signal_row["atr"], 2),
            "atr_r3": round(signal_row["atr_ratio_3day"], 2),
            "ibs": round(signal_row["ibs"], 2),
            "sma200": round(signal_row["sma200"], 2),
            "indices": ",".join(indices),
            "threshold_loc": round(high_next_target, 2),
        }

        self.trade_repository.create_trade(
            symbol=symbol,
            strategy=self.name,
            size=0,
            entry=round(entry_price, 2),
            stop_loss=0.0,
            target=round(target_price, 2),
            context=context,
        )

        created_trades.append(
            {
                "Symbol": symbol,
                "Entry": round(entry_price, 2),
                "LOC": round(high_next_target, 2),
                "Score": round(signal_row["setup_score"], 2),
                "Close": round(signal_row["close"], 2),
                "ATR": round(signal_row["atr"], 2),
            }
        )
        return True

    def analyze_single_symbol(self, symbol: str) -> SymbolAnalysisResult:
        """Debug method to analyze a single symbol step-by-step.

        Returns detailed check results.
        """
        historical_lookback_days = self.config.LOOKBACK_DAYS

        # 1. Fetch Data
        df = self.data_provider.get_symbol_history(
            symbol, days=historical_lookback_days
        )
        if df.empty:
            return {
                "symbol": symbol,
                "indices": [],
                "last_date": "",
                "data_valid": False,
                "checks": {},
                "values": {},
                "result": "FAIL",
                "error": "No data found",
            }

        # 2. Pivot to match _compute_indicators expectation
        closes = pd.DataFrame({symbol: df["close"].values}, index=df["date"])
        highs = pd.DataFrame({symbol: df["high"].values}, index=df["date"])
        lows = pd.DataFrame({symbol: df["low"].values}, index=df["date"])
        volumes = pd.DataFrame({symbol: df["volume"].values}, index=df["date"])

        # 3. Indicators
        indicators_dict = self._compute_indicators(closes, highs, lows, volumes)

        if len(closes) < 2:
            return {
                "symbol": symbol,
                "indices": [],
                "last_date": "",
                "data_valid": False,
                "checks": {},
                "values": {},
                "result": "FAIL",
                "error": "Not enough data",
            }

        return self._build_symbol_analysis_result(symbol, df, closes, indicators_dict)

    def _build_symbol_analysis_result(
        self,
        symbol: str,
        df: pd.DataFrame,
        closes: pd.DataFrame,
        indicators_dict: dict[str, pd.DataFrame],
    ) -> SymbolAnalysisResult:
        """Builds the structured analysis report for a single debugged symbol."""
        index = -1
        previous_index = -2

        # Extract values
        current_close = closes.iloc[index][symbol]
        current_open = df["open"].iloc[index]
        previous_close = closes.iloc[previous_index][symbol]
        previous_open = df["open"].iloc[previous_index]

        sma200 = indicators_dict["sma200"].iloc[index][symbol]
        volume_sma = indicators_dict["volume_sma"].iloc[index][symbol]
        atr = indicators_dict["atr"].iloc[index][symbol]
        atr_ratio_3day = indicators_dict["atr_ratio_3day"].iloc[index][symbol]
        ibs = indicators_dict["ibs"].iloc[index][symbol]
        volatility_ratio = indicators_dict["volatility_ratio"].iloc[index][symbol]

        # Use centralized BaseStrategy index helper (DRY resolution)
        indices = self._get_indices_for_symbol(symbol)

        checks = self._run_analysis_checks(
            current_close=current_close,
            current_open=current_open,
            previous_close=previous_close,
            previous_open=previous_open,
            sma200=sma200,
            volume_sma=volume_sma,
            atr_ratio_3day=atr_ratio_3day,
            volatility_ratio=volatility_ratio,
            ibs=ibs,
            has_indices=bool(len(indices) > 0),
        )

        passed = all(checks.values())

        values_dict = self._build_analysis_values(
            current_close=current_close,
            sma200=sma200,
            volume_sma=volume_sma,
            atr=atr,
            atr_ratio_3day=atr_ratio_3day,
            ibs=ibs,
            volatility_ratio=volatility_ratio,
        )

        return {
            "symbol": symbol,
            "indices": indices,
            "last_date": str(df["date"].iloc[index].date()),
            "data_valid": True,
            "checks": checks,
            "values": values_dict,
            "result": "PASS" if passed else "FAIL",
            "error": None,
        }

    def _run_analysis_checks(
        self,
        current_close: float,
        current_open: float,
        previous_close: float,
        previous_open: float,
        sma200: float,
        volume_sma: float,
        atr_ratio_3day: float,
        volatility_ratio: float,
        ibs: float,
        has_indices: bool,
    ) -> dict[str, bool]:
        """Runs the validation checks for a single symbol analysis."""
        return {
            "min_volume": bool(volume_sma > self.config.MIN_VOLUME),
            "min_price": bool(current_close > self.config.MIN_PRICE),
            "uptrend_sma200": bool(current_close > sma200),
            "dip_atr_ratio_3day": bool(atr_ratio_3day < self.config.MAX_ATR_RATIO_3DAY),
            "volatility_ratio": bool(
                volatility_ratio > self.config.MIN_VOLATILITY_RATIO
            ),
            "low_ibs": bool(ibs < self.config.MAX_IBS),
            "red_candle_today": bool(current_close < current_open),
            "red_candle_yesterday": bool(previous_close < previous_open),
            "in_universe": has_indices,
        }

    def _build_analysis_values(
        self,
        current_close: float,
        sma200: float,
        volume_sma: float,
        atr: float,
        atr_ratio_3day: float,
        ibs: float,
        volatility_ratio: float,
    ) -> dict[str, float]:
        """Builds the values dictionary for a single symbol analysis."""

        def _extract_safe_numeric_value(
            raw_metric_value: object, default_fallback_value: float = 0.0
        ) -> float:
            """Safely extracts a float from a potentially NaN value."""
            return (
                default_fallback_value
                if pd.isna(raw_metric_value)
                else float(raw_metric_value)
            )

        return {
            "close": round(_extract_safe_numeric_value(current_close), 2),
            "sma200": round(_extract_safe_numeric_value(sma200), 2),
            "volume_sma": float(int(_extract_safe_numeric_value(volume_sma, 0))),
            "atr": round(_extract_safe_numeric_value(atr), 2),
            "atr_ratio_3day": round(_extract_safe_numeric_value(atr_ratio_3day), 2),
            "ibs": round(_extract_safe_numeric_value(ibs), 2),
            "volatility_ratio": round(_extract_safe_numeric_value(volatility_ratio), 3),
        }
