import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

# Add project root to sys.path so we can import 'app', 'core', 'services'
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


from pathlib import Path

from app.backtest.backtest_core import BacktestRepository, BacktestTrade
from app.backtest.reporting import BacktestReporter

# --- Import from your existing files ---
from app.config import settings
from app.core.database import OHLCVRepository

# Reuse the BacktestRepository/Trade classes from minervini_backtester.py
# (You can move them to a shared 'backtest_core.py' to avoid duplication,
# but for now I will redefine them here for a standalone script).


@dataclass
class MomentumConfig:
    strategy_name: str = "NASDAQ Momentum (Monthly)"
    start_date: str = "2022-01-01"
    initial_capital: float = 100_000.0
    top_n: int = 5
    out_dir: str = "backtest_results"  # Unified


class NasdaqMomentumBacktester:
    def __init__(self, config: MomentumConfig, universe: list[str]):
        self.cfg = config
        self.universe = universe
        # Ensure QQQ is loaded for Regime Filter
        if "QQQ" not in self.universe:
            self.universe.append("QQQ")

        self.market_repo = OHLCVRepository(str(settings.database.market_data_path))
        self.bt_repo = BacktestRepository(str(settings.database.backtest_path))
        self.bt_repo.init_tables(clear_existing=True)

        self.cash = config.initial_capital
        self.positions: dict[str, dict] = {}
        self.peak_equity = config.initial_capital

        # Initialize Reusable Reporter
        self.reporter = BacktestReporter(self.bt_repo, self.market_repo, self.cfg)

    def _load_data(self):
        start_dt = pd.to_datetime(self.cfg.start_date) - timedelta(days=400)
        print(f"Loading data from {start_dt.date()}...")
        return self.market_repo.get_data_after_date(
            self.universe, str(start_dt.date()), inclusive=True
        )

    def prepare_features(self, df: pd.DataFrame):
        print("Computing Momentum Scores...")
        df = df.sort_index()

        def calc_score(g):
            c = g["close"]
            roc1 = c.pct_change(21)
            roc3 = c.pct_change(63)
            roc6 = c.pct_change(126)
            roc12 = c.pct_change(252)

            # Regime Filter Indicator (Only needed for QQQ)
            sma200 = c.rolling(200).mean()

            score = roc1 + roc3 + roc6 + roc12
            return pd.DataFrame(
                {"score": score, "close": c, "sma200": sma200}, index=g.index
            )

        features = df.groupby("symbol", group_keys=False).apply(
            calc_score, include_groups=False
        )

        # Ensure correct index naming
        if features.index.names != ["symbol", "date"]:
            features.index.names = ["symbol", "date"]

        return features.reorder_levels(["date", "symbol"]).sort_index()

    def get_monthly_schedule(self, df: pd.DataFrame):
        # FIX: Robust date handling for comparison
        date_level = df.index.get_level_values("date").unique()
        dates = pd.to_datetime(date_level).sort_values()

        start_dt = pd.to_datetime(self.cfg.start_date)
        dates = dates[dates >= start_dt]

        cal = pd.DataFrame({"date": dates})
        cal["ym"] = cal["date"].dt.to_period("M")

        grouped = (
            cal.groupby("ym")["date"]
            .agg(["min", "max"])
            .rename(columns={"min": "month_start", "max": "month_end"})
        )

        schedule = []
        periods = grouped.index.sort_values()

        for i in range(len(periods) - 1):
            curr_month = periods[i]
            next_month = periods[i + 1]

            schedule.append(
                {
                    "signal_date": grouped.loc[curr_month, "month_end"],
                    "trade_date": grouped.loc[next_month, "month_start"],
                }
            )

        return pd.DataFrame(schedule)

    def run(self):
        # 1. Load & Prep
        raw_df = self._load_data()
        if raw_df.empty:
            print("❌ No data.")
            return

        data = self.prepare_features(raw_df)

        try:
            qqq_data = data.xs("QQQ", level="symbol")
        except KeyError:
            print("❌ QQQ data missing for regime filter!")
            return

        schedule = self.get_monthly_schedule(data)
        print(f"Running Monthly Backtest: {len(schedule)} periods.")

        # 2. Simulation
        for _, row in schedule.iterrows():
            sig_date = row["signal_date"]
            trade_date = row["trade_date"]

            # --- REGIME CHECK (on Signal Date) ---
            try:
                qqq_row = qqq_data.loc[sig_date]
                bull_market = qqq_row["close"] > qqq_row["sma200"]
            except KeyError:
                bull_market = False

            # --- RANKING (on Signal Date) ---
            try:
                daily = data.loc[sig_date]
            except KeyError:
                continue

            candidates = daily[daily.index != "QQQ"].copy()
            candidates = candidates.dropna(subset=["score"])
            candidates = candidates.sort_values("score", ascending=False)
            top_n = candidates.head(self.cfg.top_n).index.tolist()
            top_n_set = set(top_n)

            # --- TRADING (on Trade Date) ---
            try:
                trade_prices = data.loc[trade_date]
            except KeyError:
                continue

            # 1. SELL Logic
            for sym in list(self.positions.keys()):
                if sym not in top_n_set:
                    self._close_position(sym, trade_prices, trade_date)

            # 2. BUY Logic (Regime Filtered)
            if bull_market:
                current_holdings = set(self.positions.keys())
                to_buy = [s for s in top_n if s not in current_holdings]
                slots_needed = len(to_buy)

                if slots_needed > 0 and self.cash > 1000:
                    alloc = self.cash / slots_needed
                    for sym in to_buy:
                        self._open_position(sym, alloc, trade_prices, trade_date)

            # 3. Mark to Market
            self._update_equity(trade_prices, trade_date)

        # Finalize
        last_date = schedule.iloc[-1]["trade_date"]
        self._force_close_all(data, last_date)

        # --- REPORTING ---
        self.reporter.generate()
        print(f"✅ Backtest finished. Results in {self.cfg.out_dir}")

    # --- Helper Methods ---
    def _open_position(self, sym, alloc, df, date):
        if sym not in df.index:
            return
        price = df.loc[sym, "close"]
        if pd.isna(price) or price <= 0:
            return
        shares = int(alloc // price)
        if shares == 0:
            return
        cost = shares * price
        self.cash -= cost
        self.positions[sym] = {
            "shares": shares,
            "entry_price": price,
            "entry_date": date,
            "cost": cost,
        }

    def _close_position(self, sym, df, date):
        if sym not in df.index:
            return
        price = df.loc[sym, "close"]
        pos = self.positions.pop(sym)
        proceeds = pos["shares"] * price
        self.cash += proceeds
        pnl = proceeds - pos["cost"]
        ret = (pnl / pos["cost"]) * 100
        hold_days = (date - pos["entry_date"]).days
        t = BacktestTrade(
            symbol=sym,
            entry_date=str(pos["entry_date"].date()),
            exit_date=str(date.date()),
            entry_price=pos["entry_price"],
            exit_price=price,
            shares=pos["shares"],
            pnl=pnl,
            return_pct=ret,
            hold_days=hold_days,
        )

        # FIX: Pass self.cfg.strategy_name
        self.bt_repo.log_trade(t, self.cfg.strategy_name)

    def _update_equity(self, df, date):
        pos_val = sum(
            pos["shares"] * df.loc[sym, "close"] if sym in df.index else pos["cost"]
            for sym, pos in self.positions.items()
        )
        total = self.cash + pos_val
        self.peak_equity = max(self.peak_equity, total)
        dd = (
            ((self.peak_equity - total) / self.peak_equity) * 100
            if self.peak_equity > 0
            else 0
        )
        self.bt_repo.log_equity(str(date.date()), total, self.cash, pos_val, dd)

    def _force_close_all(self, df, date):
        try:
            daily = df.loc[date]
            for sym in list(self.positions.keys()):
                self._close_position(sym, daily, date)
        except KeyError:
            pass
