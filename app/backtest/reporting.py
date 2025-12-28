from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")  # Non-GUI backend for server environments

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PerformanceStats:
    """Performance metrics for a strategy or benchmark."""

    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    end_equity: float

    def to_dict(self) -> dict[str, float]:
        """Convert to plain dict (for YAML/JSON serialization)."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TradeStats:
    """Trade statistics summary."""

    count: int
    win_rate: float
    profit_factor: float
    avg_return_pct: float
    avg_hold_days: float


class BacktestReporter:
    """
    Generate performance reports, charts, and export results.

    Expects config with:
    - strategy_name: str
    - out_dir: Path
    - initial_capital: float
    """

    BENCHMARKS = ["QQQ", "SPY"]
    CHART_DPI = 300

    def __init__(
        self,
        bt_repo: Any,  # BacktestRepository
        market_repo: Any,  # OHLCVRepository
        config: Any,  # BacktestRunConfig
    ) -> None:
        self.bt_repo = bt_repo
        self.market_repo = market_repo
        self.cfg = config
        self.benchmark_curves: dict[str, pd.Series] = {}

        self.out_dir = Path(self.cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.file_prefix = self._sanitize_filename(self.cfg.strategy_name)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Remove problematic characters from strategy name for filenames."""
        return (
            name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        )

    def generate(self) -> None:
        """Generate full report: stats, charts, exports."""
        logger.info("Generating report for %s", self.cfg.strategy_name)

        equity = self.bt_repo.get_equity_curve(self.cfg.strategy_name)
        trades = self.bt_repo.get_trades(self.cfg.strategy_name)

        if equity.empty:
            logger.warning("No equity data to analyze.")
            return

        # Compute metrics
        strategy_stats = self._compute_stats(equity["total_equity"])
        trade_stats = self._compute_trade_stats(trades)
        bench_stats = self._load_benchmarks(equity.index)
        monthly_table = self._get_monthly_returns_table(equity["total_equity"])

        # Console output
        self._print_summary(strategy_stats, bench_stats, trade_stats)
        self._print_monthly_table(monthly_table)

        # Save outputs
        self._save_metrics(strategy_stats, bench_stats, trade_stats)
        self._save_monthly_csv(monthly_table)
        self._plot_results(equity)

        logger.info("Report saved to %s", self.out_dir)

    def _compute_stats(self, equity_series: pd.Series) -> PerformanceStats:
        """Calculate performance statistics from equity curve."""
        if equity_series.empty:
            logger.warning("Empty equity series for stats computation.")
            return PerformanceStats(0.0, 0.0, 0.0, 0.0, 0.0)

        start_val = float(equity_series.iloc[0])
        end_val = float(equity_series.iloc[-1])
        days = (equity_series.index[-1] - equity_series.index[0]).days
        years = days / 365.25

        # CAGR
        cagr = ((end_val / start_val) ** (1 / years)) - 1 if years > 0 else 0.0

        # Max Drawdown
        rolling_max = equity_series.cummax()
        dd = (equity_series - rolling_max) / rolling_max
        max_dd = abs(float(dd.min())) * 100

        # Sharpe Ratio (annualized, assuming daily returns)
        returns = equity_series.pct_change().dropna()
        sharpe = 0.0
        if len(returns) > 0 and returns.std() > 0:
            trading_days_per_year = 252
            sharpe = (returns.mean() * trading_days_per_year - 0.02) / (
                returns.std() * np.sqrt(trading_days_per_year)
            )

        return PerformanceStats(
            total_return_pct=round((end_val / start_val - 1) * 100, 2),
            cagr_pct=round(cagr * 100, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(float(sharpe), 2),
            end_equity=round(end_val, 2),
        )

    def _compute_trade_stats(self, trades: pd.DataFrame) -> TradeStats:
        """Calculate trade statistics."""
        if trades.empty:
            return TradeStats(0, 0.0, 0.0, 0.0, 0.0)

        n_trades = len(trades)
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]

        win_rate = (len(wins) / n_trades) * 100 if n_trades > 0 else 0.0

        gross_win = float(wins["pnl"].sum())
        gross_loss = abs(float(losses["pnl"].sum()))

        if gross_loss > 0:
            profit_factor = gross_win / gross_loss
        else:
            profit_factor = float("inf") if gross_win > 0 else 0.0

        avg_return = float(trades["return_pct"].mean()) if not trades.empty else 0.0
        avg_hold = float(trades["hold_days"].mean()) if not trades.empty else 0.0

        return TradeStats(
            count=n_trades,
            win_rate=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            avg_return_pct=round(avg_return, 2),
            avg_hold_days=round(avg_hold, 1),
        )

    def _load_benchmarks(
        self, strategy_index: pd.DatetimeIndex
    ) -> dict[str, PerformanceStats]:
        """Load and compute benchmark statistics."""
        bench_stats: dict[str, PerformanceStats] = {}

        for symbol in self.BENCHMARKS:
            curve = self._get_benchmark_equity(symbol, strategy_index)
            if not curve.empty:
                self.benchmark_curves[symbol] = curve
                bench_stats[symbol] = self._compute_stats(curve)
                logger.debug("Loaded benchmark %s", symbol)

        return bench_stats

    def _get_benchmark_equity(
        self, symbol: str, strategy_index: pd.DatetimeIndex
    ) -> pd.Series:
        """Calculate buy-and-hold equity curve for benchmark symbol."""
        try:
            df = self.market_repo.get_data_after_date(
                [symbol], str(strategy_index[0].date()), inclusive=True
            )
        except Exception as e:
            logger.warning("Failed to load benchmark %s: %s", symbol, e)
            return pd.Series(dtype=float)

        if df.empty:
            return pd.Series(dtype=float)

        # Extract close prices
        try:
            if isinstance(df.index, pd.MultiIndex) and "symbol" in df.index.names:
                closes = df.xs(symbol, level="symbol")["close"]
            else:
                closes = df["close"]
        except KeyError:
            logger.debug("No close prices for benchmark %s", symbol)
            return pd.Series(dtype=float)

        # Align with strategy dates
        closes.index = pd.to_datetime(closes.index)
        closes = closes[~closes.index.duplicated(keep="first")]
        closes = closes.reindex(strategy_index, method="ffill")

        if closes.empty or pd.isna(closes.iloc[0]):
            return pd.Series(dtype=float)

        # Scale to initial capital
        start_price = float(closes.iloc[0])
        multiplier = self.cfg.initial_capital / start_price
        return closes * multiplier

    def _get_monthly_returns_table(self, equity_curve: pd.Series) -> pd.DataFrame:
        """Pivot monthly returns into year x month table."""
        monthly = equity_curve.resample("ME").last().pct_change() * 100
        monthly = monthly.dropna()

        if monthly.empty:
            return pd.DataFrame()

        df_m = pd.DataFrame({"return": monthly})
        df_m["year"] = df_m.index.year
        df_m["month"] = df_m.index.month

        pivot = df_m.pivot(index="year", columns="month", values="return")

        month_names = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }

        pivot = pivot.rename(columns=month_names)
        pivot["Avg"] = pivot.mean(axis=1)
        return pivot.round(2)

    def _print_summary(
        self,
        strategy_stats: PerformanceStats,
        bench_stats: dict[str, PerformanceStats],
        trade_stats: TradeStats,
    ) -> None:
        """Print formatted performance summary to console."""
        logger.info("=" * 60)
        logger.info("PERFORMANCE REPORT: %s", self.cfg.strategy_name)
        logger.info("=" * 60)

        headers = ["Metric", "Strategy"] + list(bench_stats.keys())
        header_line = f"{headers[0]:<15} {headers[1]:<12} " + " ".join(
            [f"{h:<10}" for h in headers[2:]]
        )
        logger.info(header_line)
        logger.info("-" * 60)

        metrics_map = [
            ("CAGR %", "cagr_pct"),
            ("Drawdown %", "max_drawdown_pct"),
            ("Return %", "total_return_pct"),
            ("Sharpe", "sharpe_ratio"),
            ("End Equity", "end_equity"),
        ]

        strategy_dict = strategy_stats.to_dict()
        for label, key in metrics_map:
            val = strategy_dict.get(key, "N/A")
            row = [label, val]
            for bench in bench_stats.values():
                row.append(bench.to_dict().get(key, "N/A"))

            row_line = f"{row[0]:<15} {row[1]:<12} " + " ".join(
                [f"{str(x):<10}" for x in row[2:]]
            )
            logger.info(row_line)

        logger.info("-" * 60)
        logger.info(
            "Trades: %d | Win Rate: %.1f%% | PF: %.2f | Avg Return: %.2f%% | Avg Hold: %.1f days",
            trade_stats.count,
            trade_stats.win_rate,
            trade_stats.profit_factor,
            trade_stats.avg_return_pct,
            trade_stats.avg_hold_days,
        )

    def _print_monthly_table(self, monthly_table: pd.DataFrame) -> None:
        """Print monthly returns table."""
        if monthly_table.empty:
            return

        logger.info("\nMONTHLY RETURNS (%%)")
        logger.info("\n%s", monthly_table.to_string(na_rep="-"))
        logger.info("=" * 60)

    def _save_metrics(
        self,
        strategy_stats: PerformanceStats,
        bench_stats: dict[str, PerformanceStats],
        trade_stats: TradeStats,
    ) -> None:
        """Save metrics to YAML file."""
        metrics = {
            "strategy_name": self.cfg.strategy_name,
            "performance": strategy_stats.to_dict(),
            "benchmarks": {k: v.to_dict() for k, v in bench_stats.items()},
            "trades": asdict(trade_stats),
        }

        yaml_path = self.out_dir / f"{self.file_prefix}_metrics.yaml"
        try:
            with open(yaml_path, "w") as f:
                yaml.safe_dump(metrics, f, sort_keys=False, default_flow_style=False)
            logger.debug("Saved metrics to %s", yaml_path)
        except Exception as e:
            logger.error("Failed to save metrics: %s", e)

    def _save_monthly_csv(self, monthly_table: pd.DataFrame) -> None:
        """Save monthly returns table to CSV."""
        if monthly_table.empty:
            return

        csv_path = self.out_dir / f"{self.file_prefix}_monthly_returns.csv"
        try:
            monthly_table.to_csv(csv_path)
            logger.debug("Saved monthly returns to %s", csv_path)
        except Exception as e:
            logger.error("Failed to save monthly CSV: %s", e)

    def _plot_results(self, equity: pd.DataFrame) -> None:
        """Generate equity curve and drawdown chart."""
        try:
            dates = pd.to_datetime(equity.index)
            equity_values = equity["total_equity"].values
            drawdown_values = equity["drawdown_pct"].values * -1

            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(12, 8),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )
            plt.subplots_adjust(hspace=0.05)

            # Equity curve
            ax1.plot(
                dates, equity_values, label="Strategy", color="#1f77b4", linewidth=2
            )

            colors = {"QQQ": "orange", "SPY": "gray"}
            for sym, curve in self.benchmark_curves.items():
                if not curve.empty:
                    ax1.plot(
                        dates,
                        curve.values,
                        label=f"{sym} (B&H)",
                        color=colors.get(sym, "black"),
                        linestyle="--",
                        alpha=0.8,
                    )

            ax1.set_ylabel("Equity ($)")
            ax1.set_title(self.cfg.strategy_name, fontweight="bold", fontsize=14)
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

            # Drawdown
            ax2.fill_between(dates, drawdown_values, 0, color="red", alpha=0.3)
            ax2.set_ylabel("Drawdown %")
            ax2.set_xlabel("Date")
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0f}%"))

            chart_path = self.out_dir / f"{self.file_prefix}_chart.png"
            plt.savefig(chart_path, dpi=self.CHART_DPI, bbox_inches="tight")
            plt.close(fig)

            logger.debug("Saved chart to %s", chart_path)

        except Exception as e:
            logger.error("Failed to generate chart: %s", e)
