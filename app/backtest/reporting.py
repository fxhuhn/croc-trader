import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


class BacktestReporter:
    def __init__(self, bt_repo, market_repo, config):
        """
        Args:
            bt_repo: BacktestRepository instance.
            market_repo: OHLCVRepository instance.
            config: Strategy config (needs strategy_name, out_dir, initial_capital).
        """
        self.bt_repo = bt_repo
        self.market_repo = market_repo
        self.cfg = config
        self.benchmark_curves = {}

        # Standardize Output Directory
        self.out_dir = (
            self.cfg.out_dir
        )  # Path("backtest_results")  # <--- Unified Folder
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize strategy name for filenames
        self.file_prefix = (
            self.cfg.strategy_name.replace(" ", "_").replace("(", "").replace(")", "")
        )

    def generate(self):
        eq = self.bt_repo.get_equity_curve()
        trades = self.bt_repo.get_trades()

        if eq.empty:
            print("⚠️ No equity data to analyze.")
            return

        # 1. Strategy Stats
        strategy_stats = self._compute_stats(eq["total_equity"])

        # 2. Benchmark Stats
        benchmarks = ["QQQ", "SPY"]
        bench_stats = {}
        for b in benchmarks:
            b_curve = self._get_benchmark_equity(b, eq.index)
            if not b_curve.empty:
                self.benchmark_curves[b] = b_curve
                bench_stats[b] = self._compute_stats(b_curve)

        # 3. Trade Stats
        n_trades = len(trades)
        win_rate = 0.0
        profit_factor = 0.0
        if n_trades > 0:
            wins = trades[trades["pnl"] > 0]
            losses = trades[trades["pnl"] <= 0]
            win_rate = (len(wins) / n_trades) * 100
            gross_win = wins["pnl"].sum()
            gross_loss = abs(losses["pnl"].sum())
            if gross_loss > 0:
                profit_factor = gross_win / gross_loss
            else:
                profit_factor = float("inf") if gross_win > 0 else 0.0

        # 4. Monthly Returns Table
        monthly_table = self._get_monthly_returns_table(eq["total_equity"])

        # --- Console Output ---
        print("\n" + "=" * 60)
        print(f"📊 PERFORMANCE REPORT: {self.cfg.strategy_name}")
        print("=" * 60)

        headers = ["Metric", "Strategy"] + list(bench_stats.keys())
        print(
            f"{headers[0]:<15} {headers[1]:<12} "
            + " ".join([f"{h:<10}" for h in headers[2:]])
        )
        print("-" * 60)

        keys_map = [
            ("CAGR %", "cagr_pct"),
            ("Drawdown %", "max_drawdown_pct"),
            ("Return %", "total_return_pct"),
            ("Sharpe", "sharpe_ratio"),
            ("End Equity", "end_equity"),
        ]

        for label, key in keys_map:
            val = strategy_stats.get(key, "N/A")
            row = [label, val]
            for b in bench_stats:
                row.append(bench_stats[b].get(key, "N/A"))
            print(
                f"{row[0]:<15} {row[1]:<12} "
                + " ".join([f"{str(x):<10}" for x in row[2:]])
            )

        print("-" * 60)
        print(
            f"Trades: {n_trades} | Win Rate: {win_rate:.1f}% | PF: {profit_factor:.2f}"
        )
        print("\n📅 MONTHLY RETURNS (%):")
        print(monthly_table.to_string(na_rep="-"))
        print("=" * 60 + "\n")

        # --- Save Outputs with Prefix ---

        # YAML Metrics
        metrics = {
            "strategy_name": self.cfg.strategy_name,
            "performance": strategy_stats,
            "benchmarks": bench_stats,
            "trades": {
                "count": n_trades,
                "win_rate": round(win_rate, 2),
                "profit_factor": round(profit_factor, 2),
            },
        }
        yaml_path = self.out_dir / f"{self.file_prefix}_metrics.yaml"  # <--- New Name
        with open(yaml_path, "w") as f:
            yaml.safe_dump(self._clean_for_yaml(metrics), f, sort_keys=False)

        # CSV Monthly Table
        csv_path = (
            self.out_dir / f"{self.file_prefix}_monthly_returns.csv"
        )  # <--- New Name
        monthly_table.to_csv(csv_path)

        # Chart
        self._plot_results(eq)
        print(f"✅ Report saved to {self.out_dir}")

    def _plot_results(self, eq: pd.DataFrame):
        dates = pd.to_datetime(eq.index)
        equity = eq["total_equity"]
        drawdown = eq["drawdown_pct"] * -1

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        plt.subplots_adjust(hspace=0.05)

        ax1.plot(dates, equity, label="Strategy", color="#1f77b4", linewidth=2)

        colors = {"QQQ": "orange", "SPY": "gray"}
        for sym, curve in self.benchmark_curves.items():
            if not curve.empty:
                ax1.plot(
                    dates,
                    curve,
                    label=f"{sym} (B&H)",
                    color=colors.get(sym, "black"),
                    linestyle="--",
                    alpha=0.8,
                )

        ax1.set_ylabel("Equity ($)")
        ax1.set_title(self.cfg.strategy_name, fontweight="bold")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        ax2.fill_between(dates, drawdown, 0, color="red", alpha=0.3)
        ax2.set_ylabel("Drawdown %")
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0f}%"))

        chart_path = self.out_dir / f"{self.file_prefix}_chart.png"  # <--- New Name
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _compute_stats(self, equity_series: pd.Series) -> dict:
        if equity_series.empty:
            return {}
        start_val = equity_series.iloc[0]
        end_val = equity_series.iloc[-1]
        days = (equity_series.index[-1] - equity_series.index[0]).days
        years = days / 365.25
        cagr = ((end_val / start_val) ** (1 / years)) - 1 if years > 0 else 0
        rolling_max = equity_series.cummax()
        dd = (equity_series - rolling_max) / rolling_max
        max_dd = abs(dd.min()) * 100
        ret = equity_series.pct_change().dropna()
        sharpe = 0.0
        if ret.std() > 0:
            sharpe = (ret.mean() * 52 - 0.02) / (ret.std() * np.sqrt(52))
        return {
            "total_return_pct": round((end_val / start_val - 1) * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "end_equity": round(end_val, 2),
        }

    def _get_benchmark_equity(
        self, symbol: str, strategy_index: pd.DatetimeIndex
    ) -> pd.Series:
        df = self.market_repo.get_data_after_date(
            [symbol], str(strategy_index[0].date()), inclusive=True
        )
        if df.empty:
            return pd.Series()
        try:
            if isinstance(df.index, pd.MultiIndex):
                if "symbol" in df.index.names:
                    closes = df.xs(symbol, level="symbol")["close"]
                else:
                    closes = df["close"]
            else:
                closes = df["close"]
        except KeyError:
            return pd.Series()
        closes.index = pd.to_datetime(closes.index)
        closes = closes[~closes.index.duplicated(keep="first")]
        closes = closes.reindex(strategy_index, method="ffill")
        if closes.empty or pd.isna(closes.iloc[0]):
            return pd.Series()
        start_price = closes.iloc[0]
        multiplier = self.cfg.initial_capital / start_price
        return closes * multiplier

    def _get_monthly_returns_table(self, equity_curve: pd.Series) -> pd.DataFrame:
        monthly = equity_curve.resample("M").last().pct_change() * 100
        monthly = monthly.dropna()
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

    def _clean_for_yaml(self, obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: self._clean_for_yaml(v) for k, v in obj.items()}
        return obj
