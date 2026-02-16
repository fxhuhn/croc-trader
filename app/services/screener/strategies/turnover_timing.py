import logging
from dataclasses import dataclass, field
from typing import override

import pandas as pd

from ....database.repositories.market_data_provider import MarketDataProvider
from ....database.repositories.trade import TradeRepository
from ....tools.symbol_lists import ExchangeSymbol
from ....tools.market_holidays import MarketHolidayChecker
from ...telegram import TelegramBot
from .base import BaseStrategy
from ....const import Strategies

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnoverConfig:
    ATR_WINDOW: int = 3
    # Entry Factors: 0.5 * ATR and 1.0 * ATR below Close
    ENTRY_FACTORS: list[float] = field(default_factory=lambda: [0.5, 1.0])
    SMA_WINDOW: int = 200


class TurnoverTimingStrategy(BaseStrategy):
    name: str = str(Strategies.TurnOverTiming)

    def __init__(
        self,
        trade_repository: TradeRepository,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None,
        config: TurnoverConfig = TurnoverConfig(),
    ):
        super().__init__(data_provider, telegram_bot)

        self.name = Strategies.TurnOverTiming
        self.trade_repository = trade_repository
        self.config = config

    @override
    def run(
        self,
        days: int = 0,
        analysis_date: str | None = None,
        specific_symbols: list[str] | None = None,
    ) -> int:
        """
        Scans for Turnover signals (Weekly Close).
        Logic: Close < SMA200. Entry = Close - (ATR(3) * Factor).
        """
        # 1. Determine Analysis Date (Today)
        if analysis_date:
            analysis_datetime = pd.Timestamp(analysis_date)
        else:
            analysis_datetime = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)

        # 2. Strict Weekly Close Check: Today must be Friday or Holiday-Thursday
        holiday_checker = MarketHolidayChecker()
        day_of_week = analysis_datetime.dayofweek # Monday=0, Sunday=6

        is_setup_day = False
        if day_of_week == 4: # Friday
            is_setup_day = True
        elif day_of_week == 3: # Thursday
            # If Friday is a holiday, Thursday is the setup day
            tomorrow = analysis_datetime + pd.Timedelta(days=1)
            if holiday_checker.is_holiday(tomorrow.date()):
                is_setup_day = True

        if not is_setup_day:
            return 0

        # Adjust Lookback for Backtesting
        lookback = 800
        if analysis_date:
             days_diff = (pd.Timestamp.now() - analysis_datetime).days
             if days_diff > (lookback - 100):
                 lookback = days_diff + 500 # Buffer

        # 3. Compile Universe (Pre-Filtering)
        symbol_loader = ExchangeSymbol()
        indices_map = {
            "NDX": symbol_loader.nasdaq_100,
            "SPX": symbol_loader.sp_500,
            "RUS": symbol_loader.russell_1000,
        }
        all_symbols = set()
        for symbol_list in indices_map.values():
            all_symbols.update(symbol_list)

        target_universe = list(all_symbols)
        if specific_symbols:
            target_universe = list(set(target_universe) & set(specific_symbols))

        # Load Data (Optimized)
        data_frames = self.data_provider.get_universe_daily_data(
            target_universe, days=lookback
        )
        if not data_frames:
            return 0

        # Handle missing data strictly but gracefully:
        # Use local variables to avoid mutating shared cache
        closes = data_frames["close"].ffill()
        highs = data_frames["high"].ffill()
        lows = data_frames["low"].ffill()
        volumes = data_frames["volume"].fillna(0)

        # 4. Find last available dataset (Avoid Look-Ahead in Backtest)
        if closes.empty:
            return 0

        available_dates = closes.index[closes.index <= analysis_datetime]
        if available_dates.empty:
            return 0
        last_trading_day = available_dates[-1]
        
        # Ensure we are not running on stale data (unless it's exactly the setup day)
        if last_trading_day != analysis_datetime:
            # If we are in the middle of a Friday, it's possible 
            # we don't have today's Close yet.
            # But the screener usually runs EOD.
            logger.warning(
                "[%s] Analysis date %s requested, but latest data is %s.",
                self.name, analysis_datetime.date(), last_trading_day.date()
            )
            return 0

        setup_date = last_trading_day

        # 5. Compute Indicators (Vectorized)
        required_window = 200
        start_location = max(0, closes.index.get_loc(setup_date) - required_window)
        
        # Slice inputs
        closes_slice = closes.iloc[start_location : closes.index.get_loc(setup_date) + 1]
        highs_slice = highs.iloc[start_location : highs.index.get_loc(setup_date) + 1]
        lows_slice = lows.iloc[start_location : lows.index.get_loc(setup_date) + 1]
        volumes_slice = volumes.iloc[start_location : volumes.index.get_loc(setup_date) + 1]

        from ....tools import indicators
        
        # Turnover = Close * Volume
        turnover_slice = closes_slice * volumes_slice

        # Indicators
        sma_turnover_20 = indicators.calculate_sma(turnover_slice, 20)
        sma_price_150 = indicators.calculate_sma(closes_slice, 150)

        # ATR Calculation (Wilder's Smoothing) w/ centralized logic
        atr_series = indicators.calculate_atr(
            highs_slice, lows_slice, closes_slice, self.config.ATR_WINDOW
        )

        # Extract values for the Setup Day
        try:
            current_close = closes.loc[setup_date]
            current_sma_price = sma_price_150.loc[setup_date]
            current_sma_turnover = sma_turnover_20.loc[setup_date]
            current_atr = atr_series.loc[setup_date]
        except KeyError:
            return 0

        # 6. Index & Selection Logic (Formerly Buckets)
        candidates = []

        for index_name, symbols_in_index in indices_map.items():
            # Filter symbols that exist in our data and have valid data
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

            # Filter out secondary share classes (e.g., GOOGL) if primary (GOOG) is present
            from ....tools.symbol_filter import SymbolFilter
            ranked_by_turnover = SymbolFilter().filter_symbols(ranked_by_turnover)

            # Take Top 20 (Liquid)
            top_20_liquid = ranked_by_turnover[:20]

            # Filter: Close > SMA 150 (Trend) and ATR > 0
            trend_filtered = []
            for symbol in top_20_liquid:
                if current_close[symbol] > current_sma_price[symbol] and current_atr[symbol] > 0:
                    trend_filtered.append(symbol)

            # Select Top 4 from the remaining list (Already sorted by Turnover)
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

        count = 0
        signal_date_string = str(setup_date.date())

        # Deduplicate candidates and merge indices
        merged_candidates = {}
        for candidate in candidates:
            symbol = candidate["symbol"]
            if symbol not in merged_candidates:
                merged_candidates[symbol] = candidate
            else:
                existing_indices = merged_candidates[symbol]["indices"].split(", ")
                if candidate["indices"] not in existing_indices:
                    merged_candidates[symbol]["indices"] += f", {candidate['indices']}"

        unique_candidates = merged_candidates.values()

        for candidate in unique_candidates:
            for factor in self.config.ENTRY_FACTORS:  # [0.5, 1.0]
                strategy_name = f"{self.name}_{factor}"

                # Calculate Limit Entry: Close - (Factor * ATR)
                limit_price = candidate["close"] - (candidate["atr"] * factor)
                limit_price = round(limit_price, 2)

                # Exists Check
                if self.trade_repository.exists(
                    candidate["symbol"], strategy_name, signal_date_string
                ):
                    continue

                context = {
                    "setup_date": signal_date_string,
                    "setup_close": candidate["close"],
                    "setup_candle_green": bool(
                        candidate["close"]
                        > data_frames["open"].loc[setup_date][candidate["symbol"]]
                    ),
                    "green_candle_count": 1 if bool(
                        candidate["close"]
                        > data_frames["open"].loc[setup_date][candidate["symbol"]]
                    ) else 0,
                    "setup_sma150": candidate["sma_price"],
                    "setup_turnover_sma20": round(candidate["sma_turnover"], 0),
                    "setup_atr": round(candidate["atr"], 2),
                    "factor": factor,
                    "indices": candidate["indices"],
                    "source": "screener",
                    "date": signal_date_string,
                }

                self.trade_repository.create_trade(
                    symbol=candidate["symbol"],
                    strategy=strategy_name,
                    size=0,
                    entry=limit_price,
                    stop_loss=0.0,
                    target=0.0,
                    context=context,
                )
                count += 1

        # Telegram Report
        if self.telegram_bot and unique_candidates:
            report_rows = []
            for item in unique_candidates:
                entry_1 = item["close"] - (item["atr"] * 0.5)
                entry_2 = item["close"] - (item["atr"] * 1.0)

                report_rows.append(
                    {
                        "symbol": item["symbol"],
                        "e1": round(entry_1, 2),
                        "e2": round(entry_2, 2),
                        "close": round(item["close"], 2),
                        "atr": round(item["atr"], 2),
                    }
                )

            if report_rows:
                report_dataframe = pd.DataFrame(report_rows)
                report_dataframe.columns = ["Symbol", "Entry 1", "Entry 2", "Close", "ATR"]
                self._send_telegram_report(
                    "Turnover Signals", str(setup_date.date()), report_dataframe
                )

        return count

    def analyze_single_symbol(self, symbol: str) -> dict[str, object]:
        """
        Debug method to analyze a single symbol.
        Note: Ranking (Top 20 / Top 4) cannot be fully verified in isolation
        as it depends on other stocks. This checks absolute criteria only.
        """
        days = 400
        dataframe = self.data_provider.get_symbol_history(symbol, days=days)
        if dataframe.empty:
            return {"symbol": symbol, "error": "No data found"}

        # Safe extraction helper
        def safe_value(value_expr, default=0.0) -> float:
            return default if pd.isna(value_expr) else float(value_expr)

        try:
            closes = pd.Series(dataframe["close"].values, index=dataframe["date"])
            highs = pd.Series(dataframe["high"].values, index=dataframe["date"])
            lows = pd.Series(dataframe["low"].values, index=dataframe["date"])
            volumes = pd.Series(dataframe["volume"].values, index=dataframe["date"])
        except Exception as error:
            return {"symbol": symbol, "error": f"Data Frame Error: {str(error)}"}

        if len(closes) < 200:
            return {
                "symbol": symbol,
                "error": f"Not enough data (Found {len(closes)}, Need 200+)",
            }

        from ....tools import indicators
        
        turnover = closes * volumes
        sma_turnover_20 = indicators.calculate_sma(turnover, 20)
        sma_price_150 = indicators.calculate_sma(closes, 150)

        # ATR (3)
        atr_series = indicators.calculate_atr(
            highs, lows, closes, self.config.ATR_WINDOW
        )

        # Last Values
        index = -1
        current_close = closes.iloc[index]
        current_sma150 = sma_price_150.iloc[index]
        current_turnover_sma = sma_turnover_20.iloc[index]
        current_atr = atr_series.iloc[index]

        # Check
        uptrend = bool(current_close > current_sma150)

        return {
            "symbol": symbol,
            "last_date": str(dataframe["date"].iloc[index].date()),
            "data_valid": True,
            "checks": {"uptrend_sma150": uptrend, "data_sufficient": True},
            "values": {
                "close": round(safe_value(current_close), 2),
                "sma150": round(safe_value(current_sma150), 2),
                "turnover_sma20": int(safe_value(current_turnover_sma, 0)),
                "atr": round(safe_value(current_atr), 2),
            },
            "note": "Rank logic (Top 20) requires full market scan.",
        }
