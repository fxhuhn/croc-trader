import logging
from dataclasses import dataclass, field
from typing import override

import pandas as pd

from ....database.repositories.market_data_provider import MarketDataProvider
from ....database.repositories.trade import TradeRepository
from ....tools.symbol_lists import ExchangeSymbol
from ...telegram import TelegramBot
from .base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class TurnoverConfig:
    ATR_WINDOW: int = 3
    # Entry Factors: 0.5 * ATR and 1.0 * ATR below Close
    ENTRY_FACTORS: list[float] = field(default_factory=lambda: [0.5, 1.0])
    SMA_WINDOW: int = 200


class TurnoverTimingStrategy(BaseStrategy):
    def __init__(
        self,
        trade_repo: TradeRepository,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None,
        config: TurnoverConfig = TurnoverConfig(),
    ):
        super().__init__(data_provider, telegram_bot)

        self.name = "TurnoverTiming"
        self.trade_repo = trade_repo
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
        # 0. Adjust Lookback for Backtesting
        lookback = 800
        if analysis_date:
             target_dt = pd.Timestamp(analysis_date)
             days_diff = (pd.Timestamp.now() - target_dt).days
             if days_diff > (lookback - 100):
                 lookback = days_diff + 500 # Buffer

        # 0. Compile Universe (Pre-Filtering)
        symbol_loader = ExchangeSymbol()
        buckets = {
            "NASDAQ_100": symbol_loader.nasdaq_100,
            "SP_500": symbol_loader.sp_500,
            "RUSSELL_1000": symbol_loader.russell_1000,
        }
        all_symbols = set()
        for s_list in buckets.values():
            all_symbols.update(s_list)

        target_universe = list(all_symbols)
        if specific_symbols:
            target_universe = list(set(target_universe) & set(specific_symbols))

        # Load Data (Optimized)
        data_frames = self.data_provider.get_universe_daily_data(
            target_universe, days=lookback
        )
        if not data_frames:
            return 0

        # Data Cleaning: Remove days where > 80% of universe is NaN (e.g. Holidays with partial data)
        # This fixes strict rolling window failures caused by sparse data rows.
        raw_closes = data_frames["close"]
        valid_counts = raw_closes.notna().sum(axis=1)
        universe_size = raw_closes.shape[1]
        
        # Keep days where at least 20% of universe traded (generous filter for major holidays)
        valid_days_mask = (valid_counts / universe_size) > 0.2
        
        # Log dropped days for verification
        dropped_days = raw_closes.index[~valid_days_mask]
        if not dropped_days.empty:
            logger.warning(f"[{self.name}] Dropped {len(dropped_days)} sparse data days (Holidays/Bad Data): {dropped_days.tolist()}")
            
        # Apply mask to all frames
        for key in data_frames:
             data_frames[key] = data_frames[key].loc[valid_days_mask]

        closes = data_frames["close"]
        highs = data_frames["high"]
        lows = data_frames["low"]

        # 1. Determine Analysis Date
        if analysis_date:
            today = pd.Timestamp(analysis_date)
        else:
            today = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)

        # Find last available dataset (Avoid Look-Ahead in Backtest)
        # Ensure we have data
        if closes.empty:
            return 0

        available_dates = closes.index[closes.index <= today]
        if available_dates.empty:
            return 0
        last_trading_day = available_dates[-1]

        # Weekly Close Check: Only allow Thursday (3) or Friday (4) (or if specific date requested)
        if last_trading_day.dayofweek < 3:
            return 0

        setup_date = last_trading_day
        # logger.info(f"[{self.name}] Analyzing {setup_date.date()} (Weekday {last_trading_day.dayofweek})")

        # 2. Compute Indicators (Vectorized)
        # Turnover = Close * Volume
        # Note: Volume can be 0 or NaN, handle safely
        turnover = closes * data_frames["volume"]

        # Indicators
        sma_turnover_20 = turnover.rolling(20).mean()
        sma_price_150 = closes.rolling(150).mean()

        # ATR Calculation (Wilder's Smoothing)
        prev_close = closes.shift(1)
        true_range_1 = highs - lows
        true_range_2 = (highs - prev_close).abs()
        true_range_3 = (lows - prev_close).abs()

        true_range = true_range_1.where(
            true_range_1 > true_range_2, true_range_2
        ).where(lambda x: x > true_range_3, true_range_3)
        rma_span = (2 * self.config.ATR_WINDOW) - 1
        atr = true_range.ewm(span=rma_span, adjust=False).mean()

        # Extract values for the Setup Day
        try:
            current_close = closes.loc[setup_date]
            current_sma_price = sma_price_150.loc[setup_date]
            current_sma_turnover = sma_turnover_20.loc[setup_date]
            current_atr = atr.loc[setup_date]
        except KeyError:
            return 0

        # 3. Bucketing & Selection Logic
        # (buckets dict is already defined above, reuse)

        candidates = []

        for bucket_name, symbols_in_bucket in buckets.items():
            # 1. Filter symbols that exist in our data and have valid data
            valid_symbols = [
                s
                for s in symbols_in_bucket
                if s in closes.columns
                and not pd.isna(current_close.get(s))
                and not pd.isna(current_sma_turnover.get(s))
                and not pd.isna(current_sma_price.get(s))
                and not pd.isna(current_atr.get(s))
            ]

            if not valid_symbols:
                continue

            # 2. Rank by Turnover SMA 20 (Highest first)
            # Create a list of (symbol, turnover_sma) tuples
            ranked_by_turnover = sorted(
                valid_symbols, key=lambda s: current_sma_turnover[s], reverse=True
            )

            # 3. Take Top 20 (Liquid)
            top_20_liquid = ranked_by_turnover[:20]

            # 4. Filter: Close > SMA 150 (Trend) and ATR > 0
            trend_filtered = []
            for s in top_20_liquid:
                if current_close[s] > current_sma_price[s] and current_atr[s] > 0:
                    trend_filtered.append(s)

            # 5. Select Top 4 from the remaining list (Already sorted by Turnover)
            final_selection = trend_filtered[:4]

            for symbol in final_selection:
                candidates.append(
                    {
                        "symbol": symbol,
                        "close": float(current_close[symbol]),
                        "sma_price": float(current_sma_price[symbol]),
                        "sma_turnover": float(current_sma_turnover[symbol]),
                        "atr": float(current_atr[symbol]),
                        "bucket": bucket_name,
                    }
                )

        count = 0
        signal_date_str = str(setup_date.date())

        # Deduplicate candidates and merge buckets
        merged_candidates = {}
        for c in candidates:
            sym = c["symbol"]
            if sym not in merged_candidates:
                merged_candidates[sym] = c
            else:
                # Append bucket if not already present
                existing_buckets = merged_candidates[sym]["bucket"].split(", ")
                if c["bucket"] not in existing_buckets:
                    merged_candidates[sym]["bucket"] += f", {c['bucket']}"

        unique_candidates = merged_candidates.values()

        for candidate in unique_candidates:
            for factor in self.config.ENTRY_FACTORS:  # [0.5, 1.0]
                strat_name = f"{self.name}_{factor}"

                # Calculate Limit Entry: Close - (Factor * ATR)
                limit_price = candidate["close"] - (candidate["atr"] * factor)
                limit_price = round(limit_price, 2)

                # Exists Check
                if self.trade_repo.exists(
                    candidate["symbol"], strat_name, signal_date_str
                ):
                    continue

                context = {
                    "setup_date": signal_date_str,
                    "setup_close": candidate["close"],
                    "setup_candle_green": bool(
                        candidate["close"]
                        > data_frames["open"].loc[setup_date][candidate["symbol"]]
                    ),
                    "setup_sma150": candidate["sma_price"],
                    "setup_turnover_sma20": round(candidate["sma_turnover"], 0),
                    "setup_atr": round(candidate["atr"], 2),
                    "factor": factor,
                    "bucket": candidate["bucket"],
                }

                self.trade_repo.create_trade(
                    symbol=candidate["symbol"],
                    strategy=strat_name,
                    size=0,
                    entry=limit_price,
                    sl=0.0,
                    target=0.0,
                    context=context,
                )
                count += 1

        # Telegram Report
        if self.telegram_bot and unique_candidates:
            report_rows = []
            for cand in unique_candidates:
                # Calculate Entries
                # Factor 0.5
                e1 = cand["close"] - (cand["atr"] * 0.5)
                # Factor 1.0
                e2 = cand["close"] - (cand["atr"] * 1.0)

                report_rows.append(
                    {
                        "symbol": cand["symbol"],
                        "e1": round(e1, 2),
                        "e2": round(e2, 2),
                        "close": round(cand["close"], 2),
                        "atr": round(cand["atr"], 2),
                    }
                )

            if report_rows:
                df = pd.DataFrame(report_rows)
                df.columns = ["Symbol", "Entry 1", "Entry 2", "Close", "ATR"]
                self._send_telegram_report(
                    "Turnover Signals", str(setup_date.date()), df
                )

        return count

    def analyze_single_symbol(self, symbol: str) -> dict[str, any]:
        """
        Debug method to analyze a single symbol.
        Note: Ranking (Top 20 / Top 4) cannot be fully verified in isolation
        as it depends on other stocks. This checks absolute criteria only.
        """
        days = 400
        df = self.data_provider.get_symbol_history(symbol, days=days)
        if df.empty:
            return {"symbol": symbol, "error": "No data found"}

        # Safe extraction helper
        def safe_val(val, default=0.0):
            return default if pd.isna(val) else val

        try:
            # FIX: Use values to avoid index mismatch
            closes = pd.Series(df["close"].values, index=df["date"])
            highs = pd.Series(df["high"].values, index=df["date"])
            lows = pd.Series(df["low"].values, index=df["date"])
            volumes = pd.Series(df["volume"].values, index=df["date"])
        except Exception as e:
            return {"symbol": symbol, "error": f"Data Frame Error: {str(e)}"}

        if len(closes) < 200:
            return {
                "symbol": symbol,
                "error": f"Not enough data (Found {len(closes)}, Need 200+)",
            }

        # Indicators
        turnover = closes * volumes
        sma_turnover_20 = turnover.rolling(20).mean()
        sma_price_150 = closes.rolling(150).mean()

        # ATR (3)
        prev_close = closes.shift(1)
        tr1 = highs - lows
        tr2 = (highs - prev_close).abs()
        tr3 = (lows - prev_close).abs()
        tr = tr1.where(tr1 > tr2, tr2).where(lambda x: x > tr3, tr3)
        atr = tr.ewm(span=(2 * self.config.ATR_WINDOW) - 1, adjust=False).mean()

        # Last Values
        idx = -1
        current_close = closes.iloc[idx]
        current_sma150 = sma_price_150.iloc[idx]
        current_turnover_sma = sma_turnover_20.iloc[idx]
        current_atr = atr.iloc[idx]

        # Check
        uptrend = bool(current_close > current_sma150)

        return {
            "symbol": symbol,
            "last_date": str(df["date"].iloc[idx].date()),
            "data_valid": True,
            "checks": {"uptrend_sma150": uptrend, "data_sufficient": True},
            "values": {
                "close": round(safe_val(current_close), 2),
                "sma150": round(safe_val(current_sma150), 2),
                "turnover_sma20": int(safe_val(current_turnover_sma, 0)),
                "atr": round(safe_val(current_atr), 2),
            },
            "note": "Rank logic (Top 20) requires full market scan.",
        }
