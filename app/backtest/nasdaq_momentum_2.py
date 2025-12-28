import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from app.backtest.backtest_core import BacktestRepository, BacktestTrade
from app.backtest.reporting import BacktestReporter
from app.config import settings
from app.core.database import OHLCVRepository


@dataclass
class BreadthMomentumConfig:
    strategy_name: str = "NASDAQ Momentum (Breadth Regime)"
    start_date: str = "2022-01-01"
    initial_capital: float = 100_000.0
    top_n: int = 5
    out_dir: str = settings.backtest.report_path


class NasdaqBreadthMomentumBacktester:
    def __init__(self, config: BreadthMomentumConfig, universe: list[str]):
        self.cfg = config
        self.universe = universe

        self.market_repo = OHLCVRepository(str(settings.database.market_data_path))
        self.bt_repo = BacktestRepository(str(settings.database.backtest_path))

        self.bt_repo.init_tables()
        self.bt_repo.cleanup_strategy(config.strategy_name)

        self.cash = config.initial_capital
        self.positions: dict[str, dict] = {}
        self.peak_equity = config.initial_capital

        self.reporter = BacktestReporter(self.bt_repo, self.market_repo, self.cfg)

    def _load_data(self):
        # Load extra buffer for 200 SMA + Breadth MAs
        start_dt = pd.to_datetime(self.cfg.start_date) - timedelta(days=500)
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

            # SMA 200 is needed for every stock for the breadth calculation
            sma200 = c.rolling(200).mean()

            score = roc1 + roc3 + roc6 + roc12
            return pd.DataFrame(
                {"score": score, "close": c, "sma200": sma200}, index=g.index
            )

        features = df.groupby("symbol", group_keys=False).apply(
            calc_score, include_groups=False
        )

        if features.index.names != ["symbol", "date"]:
            features.index.names = ["symbol", "date"]

        return features.reorder_levels(["date", "symbol"]).sort_index()

    def get_monthly_schedule(self, df: pd.DataFrame):
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

        # --- NEW: Calculate Market Breadth Regime ---
        print("Computing Market Breadth Regime...")

        # Boolean Series: Is close > sma200?
        # Note: If sma200 is NaN, result is False, which is correct (not above).
        above_sma200 = data["close"] > data["sma200"]

        # Count number of True values per date (Market Breadth)
        breadth_count = above_sma200.groupby(level="date").sum()

        # Apply MA 21 and MA 63 on the count
        breadth_ma21 = breadth_count.rolling(21).mean()
        breadth_ma63 = breadth_count.rolling(63).mean()

        # Combine into a regime dataframe for easy lookup
        regime_df = pd.DataFrame(
            {"breadth_ma21": breadth_ma21, "breadth_ma63": breadth_ma63}
        )
        # --------------------------------------------

        schedule = self.get_monthly_schedule(data)
        print(f"Running Monthly Backtest: {len(schedule)} periods.")

        # 2. Simulation
        for _, row in schedule.iterrows():
            sig_date = row["signal_date"]
            trade_date = row["trade_date"]

            # --- REGIME CHECK (on Signal Date) ---
            # Condition: MA 21 of breadth > MA 63 of breadth
            try:
                regime_row = regime_df.loc[sig_date]
                bull_market = regime_row["breadth_ma21"] > regime_row["breadth_ma63"]
            except KeyError:
                # If data is missing (e.g. at very start), default to False
                bull_market = False

            # --- RANKING (on Signal Date) ---
            try:
                daily = data.loc[sig_date]
            except KeyError:
                continue

            # Exclude QQQ from candidates if present, though logic handles it via universe usually
            candidates = daily.copy()
            if "QQQ" in candidates.index:
                candidates = candidates[candidates.index != "QQQ"]

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

    # --- Helper Methods (Identical to previous) ---
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

        self.bt_repo.log_equity(
            str(date.date()), total, self.cash, pos_val, dd, self.cfg.strategy_name
        )

    def _force_close_all(self, df, date):
        try:
            daily = df.loc[date]
            for sym in list(self.positions.keys()):
                self._close_position(sym, daily, date)
        except KeyError:
            pass


# Example Usage Block (can be adjusted)
if __name__ == "__main__":
    # Example: Define universe manually or load from file
    universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "QQQ"]
    # NOTE: In a real scenario, 'universe' should be the full NASDAQ 100 list
    # to make the breadth calculation meaningful.

    cfg = BreadthMomentumConfig()
    bt = NasdaqBreadthMomentumBacktester(cfg, universe)
    bt.run()
