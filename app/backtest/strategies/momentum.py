"""NASDAQ momentum strategy implementations."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
"""NASDAQ momentum strategies with regime filters."""

from app.backtest.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MomentumConfig:
    """Configuration for momentum strategies."""

    start_date: str = "2022-01-01"
    top_n: int = 5
    regime_symbol: str = "QQQ"


class NasdaqMomentumStrategy(BaseStrategy):
    """
    NASDAQ Momentum with single-symbol regime filter.

    Ranking: Composite momentum (1M + 3M + 6M + 12M ROC)
    Regime: Only buy when QQQ > SMA(200)
    Rebalance: Monthly
    """

    def __init__(self, config: MomentumConfig, universe: list[str]) -> None:
        self.config = config
        self.universe = list(universe)
        if config.regime_symbol not in self.universe:
            self.universe.append(config.regime_symbol)
            logger.info("Added %s to universe for regime filter", config.regime_symbol)

    @property
    def name(self) -> str:
        return f"NASDAQ Momentum ({self.config.regime_symbol} Regime)"

    @property
    def lookback_days(self) -> int:
        return 500

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing momentum scores...")

        # Reset index to work with columns
        working = df.reset_index()

        def calc_score(g: pd.DataFrame) -> pd.DataFrame:
            close = g["close"]
            roc1 = close.pct_change(21)
            roc3 = close.pct_change(63)
            roc6 = close.pct_change(126)
            roc12 = close.pct_change(252)
            sma200 = close.rolling(200).mean()
            score = roc1 + roc3 + roc6 + roc12

            return pd.DataFrame(
                {"score": score, "sma200": sma200},
                index=g.index,
            )

        # Calculate features per symbol
        features = working.groupby("symbol", group_keys=False).apply(
            calc_score, include_groups=False
        )

        # Join back to original data
        working = pd.concat([working, features], axis=1)

        # Restore MultiIndex
        working = working.set_index(["date", "symbol"]).sort_index()

        # Verify QQQ
        symbols = working.index.get_level_values("symbol").unique()
        if self.config.regime_symbol in symbols:
            logger.info(
                "✓ %s found in features (%d symbols total)",
                self.config.regime_symbol,
                len(symbols),
            )
        else:
            logger.error(
                "❌ %s NOT in features! Have: %s",
                self.config.regime_symbol,
                list(symbols[:10]),
            )

        return working

    def generate_signals(
        self, features: pd.DataFrame, signal_date: pd.Timestamp
    ) -> list[str]:
        try:
            daily = features.loc[signal_date]
        except KeyError:
            return []

        if self.config.regime_symbol not in daily.index:
            logger.warning(
                "%s missing on %s", self.config.regime_symbol, signal_date.date()
            )
            return []

        regime_row = daily.loc[self.config.regime_symbol]

        if pd.isna(regime_row["sma200"]):
            return []

        bull_market = regime_row["close"] > regime_row["sma200"]

        if not bull_market:
            logger.debug("Bear regime on %s", signal_date.date())
            return []

        candidates = daily[daily.index != self.config.regime_symbol].copy()
        candidates = candidates.dropna(subset=["score"])

        if candidates.empty:
            return []

        candidates = candidates.sort_values("score", ascending=False)
        selected = candidates.head(self.config.top_n).index.tolist()

        logger.info("Generated %d signals on %s", len(selected), signal_date.date())
        return selected

    def get_rebalance_schedule(self, data: pd.DataFrame) -> pd.DataFrame:
        dates = self._date_index(data, self.config.start_date)
        cal = pd.DataFrame({"date": dates})
        cal["ym"] = cal["date"].dt.to_period("M")
        grouped = (
            cal.groupby("ym")["date"]
            .agg(["min", "max"])
            .rename(columns={"min": "month_start", "max": "month_end"})
        )
        periods = grouped.index.sort_values()
        rows: list[dict[str, pd.Timestamp]] = []
        for i in range(len(periods) - 1):
            rows.append(
                {
                    "signal_date": grouped.loc[periods[i], "month_end"],
                    "trade_date": grouped.loc[periods[i + 1], "month_start"],
                }
            )
        return pd.DataFrame(rows)


class BreadthMomentumStrategy(BaseStrategy):
    """Market Breadth regime filter."""

    def __init__(self, config: MomentumConfig, universe: list[str]) -> None:
        self.config = config
        self.universe = universe

    @property
    def name(self) -> str:
        return "NASDAQ Momentum (Breadth Regime)"

    @property
    def lookback_days(self) -> int:
        return 500

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing momentum scores + market breadth...")

        working = df.reset_index()

        def calc_score(g: pd.DataFrame) -> pd.DataFrame:
            close = g["close"]
            roc1 = close.pct_change(21)
            roc3 = close.pct_change(63)
            roc6 = close.pct_change(126)
            roc12 = close.pct_change(252)
            sma200 = close.rolling(200).mean()
            score = roc1 + roc3 + roc6 + roc12

            return pd.DataFrame(
                {"score": score, "sma200": sma200},
                index=g.index,
            )

        features = working.groupby("symbol", group_keys=False).apply(
            calc_score, include_groups=False
        )
        working = pd.concat([working, features], axis=1)
        working = working.set_index(["date", "symbol"]).sort_index()

        # Calculate breadth
        above_sma200 = working["close"] > working["sma200"]
        breadth_count = above_sma200.groupby(level="date").sum()
        breadth_ma21 = breadth_count.rolling(21).mean()
        breadth_ma63 = breadth_count.rolling(63).mean()

        regime_df = pd.DataFrame(
            {"breadth_ma21": breadth_ma21, "breadth_ma63": breadth_ma63}
        )
        working = working.join(regime_df, on="date")

        logger.info(
            "Breadth computed (avg MA21=%.1f, MA63=%.1f)",
            breadth_ma21.mean(),
            breadth_ma63.mean(),
        )
        return working

    def generate_signals(
        self, features: pd.DataFrame, signal_date: pd.Timestamp
    ) -> list[str]:
        try:
            daily = features.loc[signal_date]
        except KeyError:
            return []

        first_row = daily.iloc[0] if not daily.empty else None
        if first_row is None:
            return []

        bull_market = (
            not pd.isna(first_row["breadth_ma21"])
            and not pd.isna(first_row["breadth_ma63"])
            and first_row["breadth_ma21"] > first_row["breadth_ma63"]
        )

        if not bull_market:
            return []

        candidates = daily.dropna(subset=["score"])
        candidates = candidates.sort_values("score", ascending=False)
        selected = candidates.head(self.config.top_n).index.tolist()

        logger.info("Generated %d signals on %s", len(selected), signal_date.date())
        return selected

    def get_rebalance_schedule(self, data: pd.DataFrame) -> pd.DataFrame:
        dates = self._date_index(data, self.config.start_date)
        cal = pd.DataFrame({"date": dates})
        cal["ym"] = cal["date"].dt.to_period("M")
        grouped = (
            cal.groupby("ym")["date"]
            .agg(["min", "max"])
            .rename(columns={"min": "month_start", "max": "month_end"})
        )
        periods = grouped.index.sort_values()
        rows: list[dict[str, pd.Timestamp]] = []
        for i in range(len(periods) - 1):
            rows.append(
                {
                    "signal_date": grouped.loc[periods[i], "month_end"],
                    "trade_date": grouped.loc[periods[i + 1], "month_start"],
                }
            )
        return pd.DataFrame(rows)
