"""Minervini Trend Template strategy implementation."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


from app.backtest.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MinerviniConfig:
    start_date: str = "2022-01-01"
    max_positions: int = 10
    min_rs_rank: float = 70.0
    rs_window: int = 252
    sma_trend_window: int = 20


class MinerviniStrategy(BaseStrategy):
    def __init__(self, config: MinerviniConfig) -> None:
        self.config = config
        self.universe: list[str] = []  # ← Wird von main.py gesetzt

    @property
    def name(self) -> str:
        return "Minervini Trend (Weekly)"

    @property
    def lookback_days(self) -> int:
        return 400

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing Minervini indicators...")
        df = df.sort_index()
        working = df.reset_index()

        def compute_group(g: pd.DataFrame) -> pd.DataFrame:
            close = g["close"]

            sma50 = close.rolling(50).mean()
            sma150 = close.rolling(150).mean()
            sma200 = close.rolling(200).mean()
            high52 = close.rolling(252).max()
            low52 = close.rolling(252).min()

            sma200_trending = sma200 > sma200.shift(self.config.sma_trend_window)
            rs_raw = close.pct_change(self.config.rs_window)

            return pd.DataFrame(
                {
                    "sma50": sma50,
                    "sma150": sma150,
                    "sma200": sma200,
                    "high52": high52,
                    "low52": low52,
                    "sma200_trending": sma200_trending,
                    "rs_raw": rs_raw,
                },
                index=g.index,
            )

        feats = working.groupby("symbol", group_keys=False).apply(
            compute_group, include_groups=False
        )
        working = pd.concat([working, feats], axis=1)
        working["rs_rank"] = working.groupby("date")["rs_raw"].rank(pct=True) * 100

        c1 = (
            (working["close"] > working["sma50"])
            & (working["close"] > working["sma150"])
            & (working["close"] > working["sma200"])
        )
        c2 = (working["sma50"] > working["sma150"]) & (
            working["sma150"] > working["sma200"]
        )
        c3 = working["sma200_trending"]
        c4 = working["close"] >= (1.3 * working["low52"])
        c5 = working["close"] >= (0.75 * working["high52"])
        c6 = working["rs_rank"] >= self.config.min_rs_rank

        working["is_valid"] = c1 & c2 & c3 & c4 & c5 & c6
        working["dist_to_high"] = working["close"] / working["high52"]

        out = working.set_index(["date", "symbol"]).sort_index()
        logger.debug("Feature frame rows=%d cols=%d", out.shape[0], out.shape[1])
        return out

    def get_rebalance_schedule(self, data: pd.DataFrame) -> pd.DataFrame:
        dates = self._date_index(data, self.config.start_date)
        cal = pd.DataFrame({"date": dates})
        cal["week"] = cal["date"].dt.to_period("W-SUN")

        grouped = (
            cal.groupby("week")["date"]
            .agg(["min", "max"])
            .rename(columns={"min": "monday", "max": "friday"})
        )

        weeks = grouped.index.sort_values()
        rows: list[dict[str, pd.Timestamp]] = []
        for i in range(1, len(weeks)):
            rows.append(
                {
                    "signal_date": grouped.loc[weeks[i - 1], "friday"],
                    "trade_date": grouped.loc[weeks[i], "monday"],
                }
            )
        return pd.DataFrame(rows)

    def generate_signals(
        self, features: pd.DataFrame, signal_date: pd.Timestamp
    ) -> list[str]:
        try:
            daily = features.loc[signal_date]
        except KeyError:
            return []

        candidates = daily[daily["is_valid"]].copy()
        candidates = candidates.sort_values(
            by=["rs_rank", "dist_to_high"], ascending=[False, False]
        )
        selected = candidates.index.tolist()
        logger.debug("Signals on %s: %d", signal_date.date(), len(selected))
        return selected
