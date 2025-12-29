"""NASDAQ momentum strategy implementations."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
from nasdaq_100_ticker_history import tickers_as_of

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
"""NASDAQ momentum strategies with regime filters."""

from app.backtest.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

"""NASDAQ momentum strategies with regime filters and dynamic monthly universe."""


@dataclass(frozen=True, slots=True)
class MomentumConfig:
    """Configuration for momentum strategies with dynamic Nasdaq-100 universe."""

    start_date: str = "2022-01-01"
    top_n: int = 5
    regime_symbol: str = "QQQ"
    # Which day of month to use for membership snapshot (None = use trade_date's day)
    membership_day: int | None = None


@lru_cache(maxsize=1024)
def _tickers_as_of_cached(year: int, month: int, day: int) -> tuple[str, ...]:
    """
    Cached wrapper around tickers_as_of to avoid repeated API calls.

    Returns Nasdaq-100 constituents as of the given date.
    Coverage: accurate from Jan 1, 2016 through at least Jan 31, 2025.
    """
    return tuple(sorted(set(tickers_as_of(year, month, day))))


def ndx_universe_asof(ts: pd.Timestamp, cfg: MomentumConfig) -> list[str]:
    """
    Get Nasdaq-100 constituents as-of a specific date.

    Args:
        ts: Timestamp to query (typically trade_date)
        cfg: Strategy configuration

    Returns:
        List of ticker symbols including regime_symbol if needed
    """
    ts = pd.to_datetime(ts)
    day = (
        ts.day
        if cfg.membership_day is None
        else min(cfg.membership_day, ts.days_in_month)
    )

    base = list(_tickers_as_of_cached(ts.year, ts.month, day))

    logger.info(
        "NDX as-of %s: %d symbols (e.g., %s)",
        ts.strftime("%Y-%m-%d"),
        len(base),
        ", ".join(base[:3]),
    )

    if cfg.regime_symbol and cfg.regime_symbol not in base:
        base.append(cfg.regime_symbol)

    return base


class NasdaqMomentumStrategy(BaseStrategy):
    """
    NASDAQ Momentum with single-symbol regime filter.

    Universe: Nasdaq-100 constituents as-of each trade_date (monthly rebalance)
    Ranking: Composite momentum (1M + 3M + 6M + 12M ROC)
    Regime: Only buy when QQQ > SMA(200)
    Rebalance: Monthly (signal on month-end, trade on next month-start)
    """

    def __init__(self, config: MomentumConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return "NASDAQ Momentum (QQQ Regime, monthly NDX universe)"

    @property
    def lookback_days(self) -> int:
        return 500

    def universe_for_date(self, asof: pd.Timestamp) -> list[str]:
        """
        Return Nasdaq-100 universe as-of the given date.

        Called by engine for each rebalance event (trade_date).
        """
        u = ndx_universe_asof(asof, self.config)
        logger.debug(
            "NDX universe as-of %s: %d symbols", pd.to_datetime(asof).date(), len(u)
        )
        return u

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum score and SMA200 for all symbols in universe."""
        logger.info("Computing momentum scores...")

        working = df.reset_index()

        def calc_score(g: pd.DataFrame) -> pd.DataFrame:
            close = g["close"]
            roc1 = close.pct_change(21)
            roc3 = close.pct_change(63)
            roc6 = close.pct_change(126)
            roc12 = close.pct_change(252)
            sma200 = close.rolling(200).mean()
            score = roc1 + roc3 + roc6 + roc12

            return pd.DataFrame({"score": score, "sma200": sma200}, index=g.index)

        feats = working.groupby("symbol", group_keys=False).apply(
            calc_score, include_groups=False
        )
        working = pd.concat([working, feats], axis=1)
        working = working.set_index(["date", "symbol"]).sort_index()

        return working

    def generate_signals(
        self, features: pd.DataFrame, signal_date: pd.Timestamp
    ) -> list[str]:
        """Generate buy signals if QQQ regime filter passes."""
        try:
            daily = features.loc[signal_date]
        except KeyError:
            return []

        # Regime check: QQQ > SMA(200)
        if self.config.regime_symbol not in daily.index:
            logger.debug(
                "Regime symbol %s missing on %s",
                self.config.regime_symbol,
                signal_date.date(),
            )
            return []

        regime_row = daily.loc[self.config.regime_symbol]

        if pd.isna(regime_row["sma200"]):
            logger.debug(
                "SMA200 is NaN for %s on %s",
                self.config.regime_symbol,
                signal_date.date(),
            )
            return []

        bull_market = regime_row["close"] > regime_row["sma200"]

        if not bull_market:
            logger.debug(
                "Bear regime on %s (close=%.2f < sma200=%.2f)",
                signal_date.date(),
                regime_row["close"],
                regime_row["sma200"],
            )
            return []

        # Rank candidates (exclude regime symbol)
        candidates = daily.drop(
            index=self.config.regime_symbol, errors="ignore"
        ).dropna(subset=["score"])

        if candidates.empty:
            logger.debug("No valid candidates on %s", signal_date.date())
            return []

        candidates = candidates.sort_values("score", ascending=False)
        selected = candidates.head(self.config.top_n).index.tolist()

        logger.info("Generated %d signals on %s", len(selected), signal_date.date())
        return selected

    def get_rebalance_schedule(self, data: pd.DataFrame) -> pd.DataFrame:
        """Monthly rebalance: signal on month-end, trade on next month-start."""
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
    """
    NASDAQ Momentum with Market Breadth regime filter.

    Universe: Nasdaq-100 constituents as-of each trade_date (monthly rebalance)
    Ranking: Composite momentum (1M + 3M + 6M + 12M ROC)
    Regime: MA(21) of breadth > MA(63) of breadth
    Breadth: Number of stocks in universe with close > SMA(200)
    Rebalance: Monthly
    """

    def __init__(self, config: MomentumConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return "NASDAQ Momentum (Breadth Regime, monthly NDX universe)"

    @property
    def lookback_days(self) -> int:
        return 500

    def universe_for_date(self, asof: pd.Timestamp) -> list[str]:
        """Return Nasdaq-100 universe as-of the given date."""
        u = ndx_universe_asof(asof, self.config)
        logger.debug(
            "NDX universe as-of %s: %d symbols", pd.to_datetime(asof).date(), len(u)
        )
        return u

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum scores + market breadth for universe."""
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

            return pd.DataFrame({"score": score, "sma200": sma200}, index=g.index)

        feats = working.groupby("symbol", group_keys=False).apply(
            calc_score, include_groups=False
        )
        working = pd.concat([working, feats], axis=1)
        working = working.set_index(["date", "symbol"]).sort_index()

        # Calculate breadth (count of stocks above SMA200)
        above_sma200 = working["close"] > working["sma200"]
        breadth_count = above_sma200.groupby(level="date").sum()

        breadth_ma21 = breadth_count.rolling(21).mean()
        breadth_ma63 = breadth_count.rolling(63).mean()

        # Join breadth metrics to features
        regime_df = pd.DataFrame(
            {"breadth_ma21": breadth_ma21, "breadth_ma63": breadth_ma63}
        )
        working = working.join(regime_df, on="date")

        logger.debug(
            "Breadth computed (avg MA21=%.1f, MA63=%.1f)",
            breadth_ma21.mean(),
            breadth_ma63.mean(),
        )
        return working

    def generate_signals(
        self, features: pd.DataFrame, signal_date: pd.Timestamp
    ) -> list[str]:
        """Generate buy signals if breadth regime filter passes."""
        try:
            daily = features.loc[signal_date]
        except KeyError:
            return []

        if daily.empty:
            return []

        # Regime check: Breadth MA21 > MA63
        first_row = daily.iloc[0]

        bull_market = (
            not pd.isna(first_row["breadth_ma21"])
            and not pd.isna(first_row["breadth_ma63"])
            and first_row["breadth_ma21"] > first_row["breadth_ma63"]
        )

        if not bull_market:
            logger.debug(
                "Bear breadth on %s (MA21=%.1f < MA63=%.1f)",
                signal_date.date(),
                first_row.get("breadth_ma21", 0),
                first_row.get("breadth_ma63", 0),
            )
            return []

        # Rank by momentum score
        candidates = daily.dropna(subset=["score"]).sort_values(
            "score", ascending=False
        )
        selected = candidates.head(self.config.top_n).index.tolist()

        logger.info(
            "Generated %d signals on %s (breadth bull)",
            len(selected),
            signal_date.date(),
        )
        return selected

    def get_rebalance_schedule(self, data: pd.DataFrame) -> pd.DataFrame:
        """Monthly rebalance: signal on month-end, trade on next month-start."""
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
