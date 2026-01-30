import logging
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, override

from ....database.repositories.market_data_provider import MarketDataProvider
from ....database.repositories.trade import TradeRepository
from ....services.telegram import TelegramBot
from ....tools.symbol_lists import ExchangeSymbol
from .base import BaseStrategy

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
    MIN_VOLA_RATIO: float = 0.03  # ATR must be > 3% of Price
    MAX_IBS: float = 0.2          # Close in bottom 20% of High-Low range
    MAX_ATR_R3: float = -1.0      # 3-Day drop > 1 ATR (negative value)

    # Data Fetching
    LOOKBACK_DAYS: int = 600


class DipBuyerStrategy(BaseStrategy):
    """
    Identifies mean-reversion opportunities in uptrending stocks.

    Logic:
    1. Trend: Price > SMA 200
    2. Dip: 3-day drop significant relative to ATR
    3. Setup: Close near daily low (Low IBS)
    """

    name: str = "DipBuyer"

    # Pre-calculated set mappings for index lookup usually static per run
    _dow_set: ClassVar[set[str]] = set()
    _sp500_set: ClassVar[set[str]] = set()
    _ndx_set: ClassVar[set[str]] = set()

    def __init__(
        self,
        trade_repo: TradeRepository,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None,
        config: DipBuyerConfig | None = None,
    ) -> None:
        super().__init__(data_provider, telegram_bot)
        self.trade_repo = trade_repo
        self.config = config or DipBuyerConfig()

        self._initialize_symbol_sets()

    def _initialize_symbol_sets(self) -> None:
        """Initializes index membership sets if not already loaded."""
        if not self._dow_set:
            exchange_symbols = ExchangeSymbol()
            DipBuyerStrategy._dow_set = set(exchange_symbols.dow_30)
            DipBuyerStrategy._sp500_set = set(exchange_symbols.sp_500)
            DipBuyerStrategy._ndx_set = set(exchange_symbols.nasdaq_100)

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
        # 1. Determine Analysis Date and Data Window
        try:
            target_date, lookback = self._get_analysis_parameters(analysis_date)
        except ValueError as error:
            logger.error(f"[{self.name}] Invalid parameters: {error}")
            return 0

        # 2. Load Data
        data = self.data_provider.get_all_daily_data(days=lookback)
        if not data or "close" not in data or data["close"].empty:
            logger.warning(f"[{self.name}] No market data available.")
            return 0

        # 3. Calculate Indicators & Filters
        signals_dataframe = self._calculate_signals(data, target_date)
        if signals_dataframe.empty:
            logger.info(f"[{self.name}] No signals found for {target_date.date()}.")
            return 0

        # 4. Filter by specific symbols if requested
        if specific_symbols:
            allowed = set(specific_symbols)
            signals_dataframe = signals_dataframe[signals_dataframe.index.isin(allowed)]
            logger.info(f"[{self.name}] Specific filter active. Hits: {len(signals_dataframe)}")
        else:
            logger.info(f"[{self.name}] Total market hits: {len(signals_dataframe)}")

        if signals_dataframe.empty:
            return 0

        # 5. Execute & Persist
        return self._process_signals(signals_dataframe, target_date)

    def _get_analysis_parameters(
        self, analysis_date: str | None
    ) -> tuple[pd.Timestamp, int]:
        """Calculates target date and required lookback window."""
        lookback = self.config.LOOKBACK_DAYS

        if not analysis_date:
            logger.info("Analyse für HEUTE")
            return pd.Timestamp.now(), lookback

        try:
            target_dt = pd.Timestamp(analysis_date)
            # Ensure we fetch enough history if the date is far in the past
            days_diff = (pd.Timestamp.now() - target_dt).days
            if days_diff > (lookback - 250):
                lookback = days_diff + 250

            logger.info(f"Analyse für historisches Datum: {analysis_date}")
            return target_dt, lookback
        except (ValueError, TypeError) as error:
            raise ValueError(f"Could not parse analysis_date '{analysis_date}': {error}") from error

    def _calculate_signals(
        self, data: dict[str, pd.DataFrame], target_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Computes indicators and applies logic filters to find valid signals."""
        closes = data["close"]
        
        # Validate Date
        if target_date not in closes.index:
            if target_date > closes.index[-1]:
                pass # Acceptable for 'today' if data is slightly lagging or holiday
            elif target_date < closes.index[0]:
                logger.error(f"Analysis date {target_date} is before start of data.")
                return pd.DataFrame()

        # --- Index Filtering ---
        # Only consider symbols in DOW30, SPX, or NDX
        valid_universe = self._dow_set | self._sp500_set | self._ndx_set
        available_symbols = set(closes.columns)
        target_symbols = list(valid_universe & available_symbols)
        
        if not target_symbols:
            logger.warning(f"[{self.name}] No symbols from major indices found in data.")
            return pd.DataFrame()
            
        # Filter raw dataframes to target set for processing
        closes = closes[target_symbols]
        highs = data["high"][target_symbols]
        lows = data["low"][target_symbols]
        opens = data["open"][target_symbols]
        volumes = data["volume"][target_symbols]

        try:
            # Map target_date to integer location
            index_location = closes.index.get_loc(target_date)
        except KeyError:
            logger.error(f"Date {target_date} not found in market data.")
            return pd.DataFrame()

        # --- Vectorized Indicator Calculation ---
        indicators = self._compute_indicators(closes, highs, lows, volumes)

        # --- Apply Filtering ---
        # Capture current slice
        current_data = {
            "close": closes.iloc[index_location],
            "open": opens.iloc[index_location],
            "high": highs.iloc[index_location],
            "low": lows.iloc[index_location],
            "volume": volumes.iloc[index_location],
            "sma200": indicators["sma200"].iloc[index_location],
            "volume_sma": indicators["volume_sma"].iloc[index_location],
            "atr": indicators["atr"].iloc[index_location],
            "atr_r3": indicators["atr_r3"].iloc[index_location],
            "ibs": indicators["ibs"].iloc[index_location],
            "vola_ratio": indicators["vola_ratio"].iloc[index_location]
        }
        
        # Capture previous slice (for candle comparison)
        if index_location == 0:
            return pd.DataFrame()
            
        previous_data = {
            "close": closes.iloc[index_location - 1],
            "open": opens.iloc[index_location - 1]
        }

        # Apply Logic
        df_results = self._apply_entry_filter(current_data, previous_data)
        
        return df_results

    def _compute_indicators(
        self, 
        closes: pd.DataFrame, 
        highs: pd.DataFrame, 
        lows: pd.DataFrame, 
        volumes: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """Calculates all technical indicators efficiently."""
        
        # A) Trend: SMA 200
        sma200 = closes.rolling(window=self.config.SMA_TREND_WINDOW, min_periods=150).mean()

        # B) Volume: SMA 20
        volume_sma20 = volumes.rolling(window=20).mean()

        # C) ATR
        true_range_1 = highs - lows
        true_range_2 = (highs - closes.shift(1)).abs()
        true_range_3 = (lows - closes.shift(1)).abs()
        
        # Vectorized Max
        true_range = true_range_1.where(true_range_1 > true_range_2, true_range_2).where(lambda x: x > true_range_3, true_range_3)
        
        rma_span = (2 * self.config.ATR_WINDOW) - 1
        atr = true_range.ewm(span=rma_span, adjust=False).mean()

        # D) Dip Metrics
        diff_3day = closes - closes.shift(3)
        atr_r3 = diff_3day / atr

        # E) IBS
        high_low_range = (highs - lows).replace(0, 0.01)
        ibs = (closes - lows) / high_low_range

        # F) Volatility Ratio
        vola_ratio = atr / closes

        return {
            "sma200": sma200,
            "volume_sma": volume_sma20,
            "atr": atr,
            "atr_r3": atr_r3,
            "ibs": ibs,
            "vola_ratio": vola_ratio
        }

    def _apply_entry_filter(self, current: dict, previous: dict) -> pd.DataFrame:
        """Applies configuration rules to filter candidates."""
        
        # 1. Liquidity & Price
        mask = (current["volume_sma"] > self.config.MIN_VOLUME) & (current["close"] > self.config.MIN_PRICE)

        # 2. Trend (Above SMA200)
        mask &= current["close"] > current["sma200"]

        # 3. Dip Conditions
        mask &= current["atr_r3"] < self.config.MAX_ATR_R3
        mask &= current["vola_ratio"] > self.config.MIN_VOLA_RATIO
        mask &= current["ibs"] < self.config.MAX_IBS

        # 4. Candles (Today Red, Yesterday Red)
        mask &= current["close"] < current["open"]
        mask &= previous["close"] < previous["open"]

        # Apply Mask
        hits_symbols = mask[mask].index

        # Assemble Result
        return pd.DataFrame(
            {
                "close": current["close"],
                "high": current["high"],
                "volume": current["volume"],
                "atr": current["atr"],
                "sma200": current["sma200"],
                "atr_r3": current["atr_r3"],
                "ibs": current["ibs"],
                "setup_score": current["atr_r3"] * -1,
            },
            index=current["close"].index,
        ).loc[hits_symbols]

    def _process_signals(self, signals: pd.DataFrame, date_obj: pd.Timestamp) -> int:
        """Creates trade objects and reports results."""
        created_trades = []
        saved_count = 0
        date_str = date_obj.strftime("%Y-%m-%d")

        for symbol, row in signals.iterrows():
            try:
                # Calculations
                entry_price = row["close"] - (row["atr"] * self.config.ENTRY_FACTOR)
                # Target & LOC Basis: High of the signal day
                high_next_target = row["high"] + 0.01

                # Identify Indices
                indices = self._get_index_membership(str(symbol))

                context = {
                    "source": "screener",
                    "date": date_str,
                    "setup_score": round(row["setup_score"], 2),
                    "close": round(row["close"], 2),
                    "volume": float(row["volume"]),
                    "atr5": round(row["atr"], 2),
                    "atr_r3": round(row["atr_r3"], 2),
                    "ibs": round(row["ibs"], 2),
                    "sma200": round(row["sma200"], 2),
                    "indices": ",".join(indices),
                    "threshold_loc": round(high_next_target, 2),
                }

                self.trade_repo.create_trade(
                    symbol=str(symbol),
                    strategy=self.name,
                    size=0,
                    entry=round(entry_price, 2),
                    sl=0.0,
                    target=row["high"],
                    context=context,
                )
                saved_count += 1
                created_trades.append(
                    {
                        "symbol": symbol,
                        "date": date_str,
                        "close": row["close"],
                        "score": row["setup_score"],
                    }
                )

            except Exception as error:
                logger.error(f"[{self.name}] Failed to save trade for {symbol}: {error}")

        # Reporting
        if self.telegram_bot and created_trades:
            prefix = (
                "BACKTEST"
                if date_str != datetime.now().strftime("%Y-%m-%d")
                else "LIVE"
            )
            self._send_telegram_report(
                f"{self.name} ({prefix})", date_str, created_trades
            )

        return saved_count

    def _get_index_membership(self, symbol: str) -> list[str]:
        """Identifies which major indices the symbol belongs to."""
        indices = []
        if symbol in self._dow_set:
            indices.append("DOW")
        if symbol in self._sp500_set:
            indices.append("SPX")
        if symbol in self._ndx_set:
            indices.append("NDX")
        return indices